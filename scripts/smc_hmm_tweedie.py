#!/usr/bin/env python3
"""
smc_hmm_tweedie.py
==================
HMM-Tweedie model with saddlepoint approximation.
Optimized for Apple Silicon (M1/M2/M3)

Usage:
    python smc_hmm_tweedie.py --dataset simulation --sim_path ./data/hmm_Breeze_N200_T104.csv --K 3 --state_specific_p --no_gam --draws 1000
"""

# =============================================================================
# 0. APPLE SILICON OPTIMIZATION (MUST BE FIRST)
# =============================================================================
import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'

import pytensor
import numpy as np
import pytensor.tensor as pt

print(f"PyTensor config: floatX={pytensor.config.floatX}, optimizer={pytensor.config.optimizer}")
os.environ['PYTENSOR_METAL'] = '0'

# =============================================================================
# STANDARD IMPORTS
# =============================================================================
import argparse
import time
import pathlib
import pickle
import warnings
import platform

import pandas as pd
import pymc as pm
import arviz as az
from patsy import dmatrix
from scipy.special import logsumexp

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'
if IS_APPLE_SILICON:
    print(f"Detected Apple Silicon ({platform.machine()}). Float32 optimization enabled.")


# =============================================================================
# 1. B-SPLINE BASIS FUNCTION
# =============================================================================
def create_bspline_basis(x, df=3, degree=3):
    """Create B-spline basis matrix for GAM."""
    x = np.asarray(x, dtype=np.float32).flatten()
    n_knots = df - degree + 1
    if n_knots > 1:
        knots = np.quantile(x, np.linspace(0, 1, n_knots)[1:-1]).tolist()
    else:
        knots = []
    formula = f"bs(x, knots={list(knots)}, degree={degree}, include_intercept=False)"
    basis = dmatrix(formula, {"x": x}, return_type='matrix')
    return np.asarray(basis, dtype=np.float32)


# =============================================================================
# 2. TWEEDIE MODEL BUILDER
# =============================================================================
def make_model(data, K=3, state_specific_p=True, p_fixed=1.5, use_gam=True, gam_df=3, use_covariates=True):
    """Build HMM-Tweedie (K>=2) or Static Tweedie (K=1) with saddlepoint approximation."""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    if use_gam and K >= 1 and use_covariates:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        n_basis_R = basis_R.shape[1]
        n_basis_F = basis_F.shape[1]
        n_basis_M = basis_M.shape[1]
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={"customer": np.arange(N), "time": np.arange(T), "state": np.arange(K)}) as model:
        # ---- 1. LATENT DYNAMICS ----
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            # Sticky prior to encourage state persistence
            Gamma = pm.Dirichlet("Gamma", a=np.eye(K)*4 + 1, shape=(K, K))

        # ---- 2. INTERCEPTS & DISPERSION ----
        beta0_prior_mean = 2.0
        beta0_prior_sd = 2.0
        
        if K == 1:
            beta0 = pm.Normal("beta0", beta0_prior_mean, beta0_prior_sd)
            phi = pm.Exponential("phi", lam=0.5)
        else:
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0_sorted = pt.sort(beta0_raw)
            beta0 = pm.Deterministic("beta0", beta0_sorted * beta0_prior_sd + beta0_prior_mean)
            phi = pm.Exponential("phi", lam=0.5, shape=K)

        # ---- 3. SLOPES / BASIS WEIGHTS ----
        if use_covariates:
            if use_gam:
                if K == 1:
                    w_R = pm.Normal("w_R", 0, 1, shape=n_basis_R)
                    w_F = pm.Normal("w_F", 0, 1, shape=n_basis_F)
                    w_M = pm.Normal("w_M", 0, 1, shape=n_basis_M)
                else:
                    w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R))
                    w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F))
                    w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M))
            else:
                if K == 1:
                    betaR = pm.Normal("betaR", 0, 1)
                    betaF = pm.Normal("betaF", 0, 1)
                    betaM = pm.Normal("betaM", 0, 1)
                else:
                    betaR = pm.Normal("betaR", 0, 1, shape=K)
                    betaF = pm.Normal("betaF", 0, 1, shape=K)
                    betaM = pm.Normal("betaM", 0, 1, shape=K)
        else:
            if use_gam:
                w_R = pt.zeros(n_basis_R) if K == 1 else pt.zeros((K, n_basis_R))
                w_F = pt.zeros(n_basis_F) if K == 1 else pt.zeros((K, n_basis_F))
                w_M = pt.zeros(n_basis_M) if K == 1 else pt.zeros((K, n_basis_M))
            else:
                betaR = pt.as_tensor_variable(0.0) if K == 1 else pt.zeros(K)
                betaF = pt.as_tensor_variable(0.0) if K == 1 else pt.zeros(K)
                betaM = pt.as_tensor_variable(0.0) if K == 1 else pt.zeros(K)

        # ---- 4. POWER PARAMETER P ----
        if K == 1:
            p_raw = pm.Beta("p_raw", alpha=2, beta=2)
            p = pm.Deterministic("p", 1.1 + p_raw * 0.8)
        elif state_specific_p:
            p_raw = pm.Beta("p_raw", alpha=2, beta=2, shape=K)
            p_sorted = pt.sort(p_raw)
            p = pm.Deterministic("p", 1.1 + p_sorted * 0.8)
        else:
            p = pt.as_tensor_variable(np.array([p_fixed] * K, dtype=np.float32))

        # ---- 5. MU CALCULATION ----
        if use_covariates:
            if use_gam:
                if K == 1:
                    eff_R = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                    eff_F = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                    eff_M = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                    mu = pt.exp(beta0 + eff_R + eff_F + eff_M)
                else:
                    eff_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                    eff_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                    eff_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                    mu = pt.exp(beta0 + eff_R + eff_F + eff_M)
            else:
                if K == 1:
                    mu = pt.exp(beta0 + betaR * R + betaF * F + betaM * M)
                else:
                    mu = pt.exp(beta0 + betaR * R[..., None] + 
                                betaF * F[..., None] + betaM * M[..., None])
        else:
            mu = pt.exp(beta0)

        mu = pt.clip(mu, 1e-3, 1e6)

        # ---- 6. TWEEDIE EMISSION (PROPER SADDLEPOINT APPROXIMATION) ----
        if K == 1:
            p_exp, phi_exp = p, phi
            mu_exp = mu
            y_in, mask_in = y, mask
        else:
            p_exp = p[None, None, :]
            phi_exp = phi[None, None, :]
            mu_exp = mu[..., None] if mu.ndim == 2 else mu
            y_in, mask_in = y[..., None], mask[:, :, None]

        # Common parameters
        exponent = 2.0 - p_exp          # 2-p
        kappa = exponent / (p_exp - 1.0)  # (2-p)/(p-1)

        # Zero mass: exact compound Poisson
        lambda_param = pt.pow(mu_exp, exponent) / (phi_exp * exponent)
        log_zero = -lambda_param

        # Saddlepoint approximation for y > 0
        y_safe = pt.clip(y_in, 1e-10, None)
        u = y_safe / mu_exp
        log_u = pt.log(u)

        log_f = (
            -0.5 * pt.log(2 * np.pi * phi_exp * y_safe * (p_exp - 1.0)) +
            kappa * (u**(1.0 - kappa) - 1.0 - (1.0 - kappa) * log_u) -
            0.5 * pt.log(1.0 + kappa * (p_exp - 1.0)**2 * pt.pow(u, p_exp - 2.0) / phi_exp)
        )

        # Combine hurdle + saddlepoint
        log_emission = pt.switch(
            pt.eq(y_in, 0),
            log_zero,
            log_f
        )

        # Apply mask (strong negative log-prob for masked entries)
        log_emission = pt.where(mask_in, log_emission, -1e12)

        # ---- 7. FORWARD ALGORITHM ----
        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
            alpha_filtered_val = pt.ones((N, T, 1))
        else:
            pi0_safe = (pi0 + 1e-10) / pt.sum(pi0 + 1e-10)
            log_alpha = pt.log(pi0_safe) + log_emission[:, 0, :]
            log_norm_0 = pt.logsumexp(log_alpha, axis=1, keepdims=True)
            log_alpha_norm = log_alpha - log_norm_0
            
            alpha_seq = [log_alpha_norm]
            
            eps = 1e-10
            Gamma_safe = Gamma + eps
            Gamma_safe = Gamma_safe / pt.sum(Gamma_safe, axis=1, keepdims=True)
            log_Gamma = pt.log(Gamma_safe)[None, :, :]
            
            log_cumulant = log_norm_0  
            
            for t in range(1, T):
                temp = log_alpha_norm[:, :, None] + log_Gamma
                curr_em = pt.clip(log_emission[:, t, :], -1e6, 0.0)
                log_alpha = curr_em + pt.logsumexp(temp, axis=1)
                
                log_norm_t = pt.logsumexp(log_alpha, axis=1, keepdims=True)
                log_alpha_norm = log_alpha - log_norm_t
                
                alpha_seq.append(log_alpha_norm)
                log_cumulant = log_cumulant + log_norm_t  
            
            log_alpha_stacked = pt.stack(alpha_seq, axis=1)
            alpha_filtered_val = pt.exp(log_alpha_stacked) 
            logp_cust = pt.squeeze(log_cumulant, axis=1)

        # ---- 8. DETERMINISTICS & POTENTIAL ----
        if K > 1:
            pm.Deterministic('alpha_filtered', alpha_filtered_val, dims=('customer', 'time', 'state'))

        pm.Potential('loglike', pt.sum(logp_cust))
        pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))

        return model


# =============================================================================
# 3. DATA LOADING
# =============================================================================
def compute_rfm_features(y, mask):
    """Compute Recency, Frequency, Monetary features from panel data."""
    N, T = y.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    
    for i in range(N):
        last_purchase = -1
        cumulative_freq = 0
        cumulative_spend = 0.0
        
        for t in range(T):
            if mask[i, t]:
                if y[i, t] > 0:
                    last_purchase = t
                    cumulative_freq += 1
                    cumulative_spend += y[i, t]
                
                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cumulative_freq
                    M[i, t] = cumulative_spend / cumulative_freq if cumulative_freq > 0 else 0.0
                else:
                    R[i, t] = t + 1
                    F[i, t] = 0
                    M[i, t] = 0.0
            else:
                R[i, t] = 0
                F[i, t] = 0
                M[i, t] = 0.0
    
    return R, F, M


def load_simulation_data(data_path, n_cust=None, seed=42, train_frac=1.0):
    """Load simulation from CSV or PKL file."""
    import pickle
    
    data_path = pathlib.Path(data_path)
    
    if data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            sim = pickle.load(f)
        N_full, T = sim['N'], sim['T']
        obs = sim['observations']
        source = 'pkl'
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        N_full = df['customer_id'].nunique()
        T = df['time_period'].nunique() if 'time_period' in df.columns else df['t'].nunique()
        obs_col = 'y' if 'y' in df.columns else 'observations'
        id_col = 'customer_id'
        time_col = 'time_period' if 'time_period' in df.columns else 't'
        obs = df.pivot(index=id_col, columns=time_col, values=obs_col).values
        source = 'csv'
    else:
        raise ValueError(f"Unknown format: {data_path.suffix}")
    
    if n_cust is not None and n_cust < N_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_full, n_cust, replace=False)
        obs = obs[idx, :]
        N = n_cust
    else:
        N = N_full
        idx = np.arange(N)
    
    T_train = int(T * train_frac)
    
    if train_frac < 1.0:
        obs_train = obs[:, :T_train]
        obs_test = obs[:, T_train:]
    else:
        obs_train = obs
        obs_test = None
    
    mask_train = np.ones((N, T_train), dtype=bool)
    R_train, F_train, M_train = compute_rfm_features(obs_train, mask_train)
    
    # Standardize features
    M_train = np.log1p(M_train)
    R_train = (R_train - np.mean(R_train)) / (np.std(R_train) + 1e-6)
    F_train = (F_train - np.mean(F_train)) / (np.std(F_train) + 1e-6)
    M_train = (M_train - np.mean(M_train)) / (np.std(M_train) + 1e-6)

    data = {
        'N': N, 'T': T_train, 'y': obs_train.astype(np.float32),
        'mask': mask_train, 'R': R_train.astype(np.float32),
        'F': F_train.astype(np.float32), 'M': M_train.astype(np.float32),
        'customer_id': idx, 'time': np.arange(T_train), 'T_full': T,
    }
    
    if obs_test is not None:
        data['y_test'] = obs_test.astype(np.float32)
        data['mask_test'] = np.ones((N, T - T_train), dtype=bool)
        data['T_test'] = T - T_train
    
    print(f"  Loaded: N={N}, T_train={T_train}, zeros={np.mean(obs_train==0):.1%} ({source})")
    return data


# =============================================================================
# 4. SMC RUNNER
# =============================================================================
def run_smc(data, K, state_specific_p, p_fixed, use_gam, gam_df,
            draws, chains, seed, out_dir, use_covariates=True):
    """Run SMC with Tweedie model."""
    
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()

    try:
        with make_model(data, K=K, state_specific_p=state_specific_p,
                       p_fixed=p_fixed, use_gam=use_gam, gam_df=gam_df,
                       use_covariates=use_covariates) as model:

            print(f" Model: K={K}, Tweedie-{'GAM' if use_gam else 'GLM'}, "
                  f"p={'state-specific' if state_specific_p else p_fixed}")
            
            idata = pm.sample_smc(draws=draws, chains=chains, cores=cores,
                                  random_seed=seed, return_inferencedata=True,
                                  threshold=0.5 if K > 2 else 0.8)

        # Extract log-evidence
        log_ev = np.nan
        try:
            lm = idata.sample_stats.log_marginal_likelihood.values
            if isinstance(lm, np.ndarray) and lm.dtype == object:
                chain_vals = []
                for c in range(lm.shape[1] if lm.ndim > 1 else 1):
                    if lm.ndim > 1:
                        chain_list = lm[-1, c] if lm.shape[0] > 0 else lm[0, c]
                    else:
                        chain_list = lm[c] if lm.ndim == 1 else lm[0]
                    if isinstance(chain_list, list):
                        valid = [float(x) for x in chain_list 
                                if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                        if valid:
                            chain_vals.append(valid[-1])
                    elif isinstance(chain_list, (int, float, np.floating)) and np.isfinite(chain_list):
                        chain_vals.append(float(chain_list))
                log_ev = float(np.mean(chain_vals)) if chain_vals else np.nan
            else:
                flat = np.array(lm).flatten()
                valid = flat[np.isfinite(flat)]
                log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
        except Exception as e:
            print(f"  Warning: log-ev extraction failed: {e}")

        if log_ev > 0:
            print("  WARNING: Positive Log-Ev detected. Numerical instability likely.")

        elapsed = (time.time() - t0) / 60

        res = {
            'K': K, 'model_type': 'tweedie', 'N': data['N'], 'T': data['T'],
            'use_gam': use_gam, 'gam_df': gam_df if use_gam else None,
            'state_specific_p': state_specific_p, 'p_fixed': p_fixed,
            'log_evidence': log_ev, 'draws': draws, 'chains': chains,
            'time_min': elapsed, 'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'ess_min': float(az.ess(idata).to_array().min()) if 'posterior' in idata else np.nan,
            'rhat_max': float(az.rhat(idata).to_array().max()) if 'posterior' in idata else np.nan,
        }

        p_tag = "statep" if state_specific_p else f"p{p_fixed}"
        pkl_path = out_dir / f"smc_K{K}_TWEEDIE_{'GAM' if use_gam else 'GLM'}_{p_tag}_N{data['N']}_T{data['T']}_D{draws}.pkl"

        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)

        print(f"  log_ev={log_ev:.2f}, time={elapsed:.1f}min")
        print(f"  Saved: {pkl_path.name}")
        return pkl_path, res

    except Exception as e:
        print(f"  CRASH: {str(e)[:60]}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='HMM-Tweedie Model with Saddlepoint Approximation')
    parser.add_argument('--dataset', required=True, choices=['uci', 'cdnow', 'simulation'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--sim_path', type=str, default=None,
                       help='Path to simulation file (.csv or .pkl). Required if --dataset simulation')
    parser.add_argument('--n_cust', type=int, default=None)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--state_specific_p', action='store_true')
    parser.add_argument('--p_fixed', type=float, default=1.5)
    parser.add_argument('--no_gam', action='store_true')
    parser.add_argument('--gam_df', type=int, default=3)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--train_frac', type=float, default=1.0)
    parser.add_argument('--no_covariates', action='store_true')
    
    args = parser.parse_args()

    if args.gam_df is None:
        args.gam_df = 2 if args.K > 2 else 3

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"HMM-Tweedie: {args.dataset.upper()} | K={args.K} | "
          f"{'state-p' if args.state_specific_p else f'p={args.p_fixed}'}")
    print(f"{'='*70}")

    if args.dataset == 'simulation':
        if not args.sim_path:
            raise ValueError("--sim_path required for simulation dataset")
        data = load_simulation_data(args.sim_path, n_cust=args.n_cust, 
                                    seed=args.seed, train_frac=args.train_frac)
    else:
        raise NotImplementedError("Only --dataset simulation supported in this version")

    print(f"\nRunning SMC...")
    pkl_path, res = run_smc(data, args.K, args.state_specific_p, args.p_fixed,
                           not args.no_gam, args.gam_df, args.draws, args.chains,
                           args.seed, out_dir, use_covariates=not args.no_covariates)

    print(f"\n{'='*70}")
    print("RESULTS")
    for key, val in res.items():
        print(f"  {key}: {val}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
