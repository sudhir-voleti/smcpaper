# =============================================================================
# SMC_HMM_BEMMAOR.PY
# Bemmaor & Glady (2012) inspired HMM with correlated NBD-Gamma
# Hybrid: Gemini's anchoring + Grok's numerical stability
# =============================================================================

import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'

import numpy as np
import pytensor.tensor as pt
import pytensor
from pytensor import scan

import argparse
import time
import pickle
import warnings
from pathlib import Path

import pandas as pd
import pymc as pm
import arviz as az
from patsy import dmatrix
from sklearn.metrics import adjusted_rand_score, confusion_matrix

warnings.filterwarnings('ignore')
RANDOM_SEED = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_rfm_features(y, mask):
    """Compute RFM features."""
    N, T = y.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        last_purchase = -1
        cum_freq = 0
        cum_spend = 0.0

        for t in range(T):
            if mask[i, t]:
                if y[i, t] > 0:
                    last_purchase = t
                    cum_freq += 1
                    cum_spend += y[i, t]

                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cum_freq
                    M[i, t] = cum_spend / cum_freq if cum_freq > 0 else 0.0
                else:
                    R[i, t] = t + 1
                    F[i, t] = 0
                    M[i, t] = 0.0

    return R, F, M


def load_simulation_data_from_csv(csv_path, T=104, N=None, seed=RANDOM_SEED):
    """Load simulation data from CSV."""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    print(f"  Reading CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    
    world = "unknown"
    for w in ['Harbor', 'Breeze', 'Fog', 'Cliff']:
        if w.lower() in csv_path.name.lower():
            world = w
            break
    
    col_mapping = {
        'customer_id': ['customer_id', 'cust_id', 'id', 'customer', 'cust'],
        't': ['t', 'time', 'period', 'time_period', 'week', 'Time'],
        'y': ['y', 'spend', 'purchase', 'value', 'spend_value', 'amount'],
        'true_state': ['true_state', 'state', 'true_latent', 'latent', 'truestate']
    }
    
    actual_cols = {}
    for std_name, variants in col_mapping.items():
        for v in variants:
            if v in df.columns:
                actual_cols[std_name] = v
                break
    
    if len(actual_cols) < 4:
        missing = set(col_mapping.keys()) - set(actual_cols.keys())
        raise ValueError(f"CSV missing columns. Found: {list(df.columns)}, need: {missing}")
    
    df = df.rename(columns={v: k for k, v in actual_cols.items()})
    
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()
    
    y_full = df.pivot(index='customer_id', columns='t', values='y').values
    true_states_full = df.pivot(index='customer_id', columns='t', values='true_state').values
    
    if T_actual < T:
        pad_width = ((0, 0), (0, T - T_actual))
        y_full = np.pad(y_full, pad_width, mode='constant', constant_values=0)
        true_states_full = np.pad(true_states_full, pad_width, mode='constant', constant_values=-1)
        T_effective = T
    elif T_actual > T:
        y_full = y_full[:, :T]
        true_states_full = true_states_full[:, :T]
        T_effective = T
    else:
        T_effective = T_actual
    
    if N is not None and N < N_actual:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_actual, N, replace=False)
        y_full = y_full[idx, :]
        true_states_full = true_states_full[idx, :]
        N_effective = N
        print(f"  Subsampled: N={N} (from {N_actual})")
    else:
        N_effective = N_actual
    
    mask = (true_states_full >= 0) & (~np.isnan(y_full))
    y_full = np.where(mask, y_full, 0.0)
    
    R, F, M = compute_rfm_features(y_full, mask)
    
    M_log = np.log1p(M)
    R_valid, F_valid, M_valid = R[mask], F[mask], M_log[mask]
    
    if len(R_valid) > 0 and R_valid.std() > 0:
        R = (R - R_valid.mean()) / (R_valid.std() + 1e-6)
    if len(F_valid) > 0 and F_valid.std() > 0:
        F = (F - F_valid.mean()) / (F_valid.std() + 1e-6)
    if len(M_valid) > 0 and M_valid.std() > 0:
        M_scaled = (M_log - M_valid.mean()) / (M_valid.std() + 1e-6)
    else:
        M_scaled = M_log
    
    data = {
        'N': N_effective,
        'T': T_effective,
        'y': y_full.astype(np.float32),
        'mask': mask.astype(bool),
        'R': R.astype(np.float32),
        'F': F.astype(np.float32),
        'M': M_scaled.astype(np.float32),
        'true_states': true_states_full.astype(np.int32),
        'world': world,
        'M_raw': M.astype(np.float32),
        'source_file': str(csv_path.name)
    }
    
    y_valid = y_full[mask]
    zero_rate = np.mean(y_valid == 0) if len(y_valid) > 0 else 0.0
    mean_spend = np.mean(y_valid[y_valid > 0]) if (y_valid > 0).any() else 0.0
    
    print(f"  Data: N={N_effective}, T={T_effective}, zeros={zero_rate:.1%}, mean=${mean_spend:.2f}")
    
    return data


# =============================================================================
# FORWARD ALGORITHM
# =============================================================================

def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """Batched forward algorithm."""
    N, T, K = log_emission.shape
    
    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]
    log_Z_init = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)
    log_alpha_norm_init = log_alpha_init - log_Z_init
    
    def forward_step(log_emit_t, log_alpha_prev, log_Z_prev, log_Gamma):
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)
        log_alpha_norm = log_alpha_new - log_Z_t
        return log_alpha_norm, log_Z_t
    
    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)
    
    (log_alpha_norm_seq, log_Z_seq), _ = scan(
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
    )
    
    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],
        log_alpha_norm_seq
    ], axis=0)
    
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    
    return log_marginal, log_alpha_norm_full


# =============================================================================
# BEMMAOR HMM MODEL
# =============================================================================

def make_bemmaor_hmm(data, K, pilot=False):
    """
    Bemmaor & Glady (2012) HMM with correlated NBD-Gamma.
    Hybrid: Gemini's anchoring + Grok's numerical stability.
    """
    y = data['y']
    mask = data['mask']
    N, T = data['N'], data['T']

    if pilot:
        print(f"  [PILOT] Building Bemmaor model: N={N}, T={T}, K={K}")

    with pm.Model(coords={
        "customer": np.arange(N),
        "time": np.arange(T),
        "state": np.arange(K)
    }) as model:

        # =====================================================================
        # 1. LATENT DYNAMICS
        # =====================================================================
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype=np.float32))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.eye(K)*5 + 1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            log_Gamma = pt.log(Gamma)

        # =====================================================================
        # 2. SHARED LATENT FACTOR (Anchored)
        # =====================================================================
        theta = pm.Normal("theta", mu=0, sigma=1, shape=(N, 1))
        
        # gamma_m anchored positive: theta>0 → higher spend
        gamma_m = pm.HalfNormal("gamma_m", sigma=1.0)
        # gamma_h free: theta>0 → ? frequency (can be negative)
        gamma_h = pm.Normal("gamma_h", mu=0, sigma=1.0)

        # =====================================================================
        # 3. NBD PART (Zero/Frequency, Correlated)
        # =====================================================================
        # Log-space parameterization for numerical stability (Grok)
        log_r = pm.Normal("log_r", 0, 1, shape=K if K > 1 else None)
        r_nbd = pt.exp(log_r)
        
        # Lambda (mean frequency) parameterization (Gemini)
        if K == 1:
            alpha_h = pm.Normal("alpha_h", 0, 1)
            log_lam = alpha_h + gamma_h * theta
        else:
            alpha_h = pm.Normal("alpha_h", 0, 1, shape=K)
            log_lam = alpha_h[None, None, :] + gamma_h * theta[:, :, None]
        
        lam = pt.exp(pt.clip(log_lam, -10, 10))
        
        # NBD P(y=0) = (r/(r+lam))^r
        if K == 1:
            log_p_zero_nbd = r_nbd * (pt.log(r_nbd) - pt.log(r_nbd + lam.squeeze()))
        else:
            r_exp = r_nbd[None, None, :]
            lam_exp = lam
            log_p_zero_nbd = r_exp * (pt.log(r_exp) - pt.log(r_exp + lam_exp))

        # =====================================================================
        # 4. GAMMA PART (Spend, Correlated)
        # =====================================================================
        # Log-space shape for numerical stability (Grok)
        if K == 1:
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1)
            beta_m = beta_m_raw
            log_alpha_gamma = pm.Normal("log_alpha_gamma", 0, 1)
        else:
            # Ordered intercepts for identifiability (Gemini)
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1, shape=K)
            beta_m = pm.Deterministic("beta_m", pt.sort(beta_m_raw))
            log_alpha_gamma = pm.Normal("log_alpha_gamma", 0, 1, shape=K)
        
        # mu parameterized with anchored gamma_m
        if K == 1:
            log_mu = beta_m + gamma_m * theta.squeeze()
        else:
            log_mu = beta_m[None, None, :] + gamma_m * theta[:, :, None]
        
        mu = pt.exp(pt.clip(log_mu, -10, 10))
        alpha_gamma = pt.exp(log_alpha_gamma)
        beta_gamma = alpha_gamma / mu

        # =====================================================================
        # 5. EMISSION LIKELIHOOD
        # =====================================================================
        if K == 1:
            log_zero = log_p_zero_nbd
            
            y_clipped = pt.clip(y, 1e-10, 1e10)
            log_gamma = ((alpha_gamma - 1) * pt.log(y_clipped) - 
                        beta_gamma * y + 
                        alpha_gamma * pt.log(beta_gamma) - 
                        pt.gammaln(alpha_gamma))
            
            # P(y>0) = 1 - P(y=0)
            log_pos = pt.log1p(-pt.exp(log_zero) + 1e-10) + log_gamma
            log_emission = pt.where(y == 0, log_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
            logp_cust = pt.sum(log_emission, axis=1)

        else:
            y_exp = y[..., None]
            mask_exp = mask[..., None]
            
            log_zero = log_p_zero_nbd
            
            # P(y>0) = 1 - P(y=0)
            log_p_pos = pt.log1p(-pt.exp(log_zero) + 1e-10)
            
            y_clipped = pt.clip(y_exp, 1e-10, 1e10)
            alpha_exp = alpha_gamma[None, None, :]
            beta_exp = beta_gamma
            
            log_gamma = ((alpha_exp - 1) * pt.log(y_clipped) - 
                        beta_exp * y_exp + 
                        alpha_exp * pt.log(beta_exp) - 
                        pt.gammaln(alpha_exp))
            
            log_pos = log_p_pos + log_gamma
            log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)
            log_emission = pt.where(mask_exp, log_emission, 0.0)

            if pilot:
                print(f"  [PILOT] Running forward algorithm...")

            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)

            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered,
                           dims=("customer", "time", "state"))

        # =====================================================================
        # 6. LIKELIHOOD
        # =====================================================================
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        pm.Potential("loglike", pt.sum(logp_cust))

        return model


# =============================================================================
# SMC RUNNER
# =============================================================================

def run_smc_bemmaor(data, K, draws, chains, seed, out_dir):
    """Run SMC with Bemmaor model."""
    cores = min(chains, 4)
    t0 = time.time()
    
    try:
        with make_bemmaor_hmm(data, K) as model:
            print(f"\nModel: K={K}, BEMMAOR, world={data['world']}")
            print(f"SMC: draws={draws}, chains={chains}, cores={cores}")
            
            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                return_inferencedata=True
            )
            
            elapsed = (time.time() - t0) / 60
            
            log_ev = np.nan
            try:
                lm = idata.sample_stats.log_marginal_likelihood.values
                if hasattr(lm, 'flatten'):
                    flat = lm.flatten()
                    valid = flat[np.isfinite(flat)]
                    log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
            except:
                pass
            
            print(f"  log_ev={log_ev:.2f}, time={elapsed:.1f}min")
            
            diagnostics = {}
            try:
                ess = az.ess(idata)
                rhat = az.rhat(idata)
                ess_vals = [ess[v].values for v in ess.data_vars if hasattr(ess[v].values, 'size')]
                rhat_vals = [rhat[v].values for v in rhat.data_vars if hasattr(rhat[v].values, 'size')]
                diagnostics['ess_min'] = float(min([v.min() for v in ess_vals])) if ess_vals else np.nan
                diagnostics['rhat_max'] = float(max([v.max() for v in rhat_vals])) if rhat_vals else np.nan
                print(f"  ESS: min={diagnostics['ess_min']:.0f}")
                print(f"  R-hat: max={diagnostics['rhat_max']:.3f}")
            except:
                diagnostics = {'ess_min': np.nan, 'rhat_max': np.nan}
            
            res = {
                'K': K,
                'model_type': 'BEMMAOR',
                'world': data['world'],
                'N': data['N'],
                'T': data['T'],
                'log_evidence': log_ev,
                'draws': draws,
                'chains': chains,
                'time_min': elapsed,
                **diagnostics
            }
            
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            pkl_name = f"smc_K{K}_BEMMAOR_N{data['N']}_T{data['T']}_D{draws}.pkl"
            pkl_path = out_dir / pkl_name
            
            with open(pkl_path, 'wb') as f:
                pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)
            
            print(f"  Saved PKL: {pkl_path}")
            return pkl_path, res, idata
            
    except Exception as e:
        elapsed = (time.time() - t0) / 60
        print(f"  FAILED after {elapsed:.1f}min: {str(e)}")
        raise


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bemmaor HMM: Correlated NBD-Gamma')
    
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument('--T', type=int, default=104)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bemmaor HMM: Correlated NBD-Gamma")
    print("=" * 70)
    
    data = load_simulation_data_from_csv(Path(args.csv_path), args.T, args.N, seed=args.seed)
    
    print(f"\nConfiguration: K={args.K}, N={data['N']}, T={data['T']}")
    print("=" * 70)
    
    out_dir = Path(args.out_dir) / data['world'].lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pkl_path, res, idata = run_smc_bemmaor(
        data=data,
        K=args.K,
        draws=args.draws,
        chains=args.chains,
        seed=args.seed,
        out_dir=out_dir
    )
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Log-Evidence: {res['log_evidence']:.2f}")
    print(f"Runtime: {res['time_min']:.1f} minutes")
    print(f"Output: {pkl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
