#!/usr/bin/env python3
"""
smc_hmm_nbd_only.py - NBD-only HMM (frequency only, no monetary)
Benchmarks loss of "M" in RFM-SMC
"""

import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'

import numpy as np
import pytensor.tensor as pt
import pymc as pm
import arviz as az
import argparse
import time
import pickle
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')
RANDOM_SEED = 42


# =============================================================================
# FORWARD ALGORITHM (pt.scan)
# =============================================================================

def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """Batched forward algorithm with pt.scan."""
    N, T, K = log_emission.shape
    
    # Initial step
    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]
    log_Z_init = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)
    log_alpha_norm_init = log_alpha_init - log_Z_init
    
    # Scan step
    def forward_step(log_emit_t, log_alpha_prev, log_Z_prev, log_Gamma):
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)
        log_alpha_norm = log_alpha_new - log_Z_t
        return log_alpha_norm, log_Z_t
    
    # Emission sequence from t=1
    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)
    
    (log_alpha_norm_seq, log_Z_seq), _ = pt.scan(
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
    )
    
    # Full sequence
    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],
        log_alpha_norm_seq
    ], axis=0)
    
    # Marginal likelihood
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    
    return log_marginal, log_alpha_norm_full


# =============================================================================
# NBD-ONLY MODEL
# =============================================================================

def make_nbd_only_hmm(data, K):
    """NBD-only HMM: frequency only, no monetary."""
    # Binary incidence
    y_binary = (data['y'] > 0).astype(np.float32)
    mask = data['mask'].astype(bool)
    N, T = data['N'], data['T']
    
    with pm.Model(coords={
        "customer": np.arange(N),
        "time": np.arange(T),
        "state": np.arange(K)
    }) as model:
        
        # 1. Latent Dynamics
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype="float32"))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype="float32"))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.eye(K)*5 + 1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype="float32"))
            log_Gamma = pt.log(Gamma)
        
        # 2. NBD Parameters with heterogeneity
        theta = pm.Normal("theta", mu=0, sigma=1, shape=(N, 1))
        gamma_h = pm.HalfNormal("gamma_h", sigma=0.5)
        
        alpha_h_raw = pm.Normal("alpha_h_raw", 0, 1, shape=K if K > 1 else None)
        if K > 1:
            alpha_h = pt.sort(alpha_h_raw)
            log_lambda_base = alpha_h[None, None, :] + gamma_h * theta[:, :, None]
        else:
            alpha_h = alpha_h_raw
            log_lambda_base = alpha_h + gamma_h * theta
        
        lambda_nbd = pt.exp(pt.clip(log_lambda_base, -10, 10))
        
        # Dispersion (shared)
        log_r = pm.Normal("log_r", 0, 1)
        r_nbd = pt.exp(log_r)
        
        # 3. NBD P(0)
        log_p_zero = r_nbd * (pt.log(r_nbd) - pt.log(r_nbd + lambda_nbd))
        
        # 4. Binary Emission
        log_zero = log_p_zero
        log_pos = pt.log(1 - pt.exp(log_p_zero) + 1e-10)
        
        y_exp = y_binary[..., None] if K > 1 else y_binary
        mask_exp = mask[..., None] if K > 1 else mask
        
        log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)
        log_emission = pt.where(mask_exp, log_emission, 0.0)
        
        # 5. Forward Algorithm
        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)
            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered, dims=("customer", "time", "state"))
        
        # 6. Likelihood
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        pm.Potential("loglike", pt.sum(logp_cust))
    
    return model


# =============================================================================
# DATA LOADING (reuse from Bemmaor)
# =============================================================================

def load_simulation_data_from_csv(csv_path, T=104, N=None, train_ratio=1.0, seed=RANDOM_SEED):
    """Load data - simplified for NBD-only (no RFM features needed)."""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    print(f"  Reading: {csv_path.name}")
    df = pd.read_csv(csv_path)
    
    # Detect world
    world = "unknown"
    for w in ['Harbor', 'Breeze', 'Fog', 'Cliff']:
        if w.lower() in csv_path.name.lower():
            world = w
            break
    
    # Column mapping
    col_mapping = {
        'customer_id': ['customer_id', 'cust_id', 'id', 'customer'],
        't': ['t', 'time', 'period', 'week'],
        'y': ['y', 'spend', 'purchase', 'value'],
        'true_state': ['true_state', 'state', 'true_latent', 'latent']
    }
    
    actual_cols = {}
    for std, variants in col_mapping.items():
        for v in variants:
            if v in df.columns:
                actual_cols[std] = v
                break
    
    df = df.rename(columns={v: k for k, v in actual_cols.items()})
    
    # Reshape
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()
    
    y_full = df.pivot(index='customer_id', columns='t', values='y').values
    true_states_full = df.pivot(index='customer_id', columns='t', values='true_state').values
    
    # Pad/truncate
    if T_actual < T:
        pad = ((0, 0), (0, T - T_actual))
        y_full = np.pad(y_full, pad, mode='constant', constant_values=0)
        true_states_full = np.pad(true_states_full, pad, mode='constant', constant_values=-1)
    elif T_actual > T:
        y_full = y_full[:, :T]
        true_states_full = true_states_full[:, :T]
    
    # Subsample
    if N is not None and N < N_actual:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_actual, N, replace=False)
        y_full = y_full[idx, :]
        true_states_full = true_states_full[idx, :]
        N_effective = N
    else:
        N_effective = N_actual
    
    # Train/test split
    if train_ratio < 1.0:
        T_train = int(T * train_ratio)
        y_train = y_full[:, :T_train]
        y_test = y_full[:, T_train:]
        true_train = true_states_full[:, :T_train]
        
        mask = (true_train >= 0) & (~np.isnan(y_train))
        y_train = np.where(mask, y_train, 0.0)
        
        data = {
            'N': N_effective, 'T': T_train, 'y': y_train.astype(np.float32),
            'mask': mask.astype(bool), 'true_states': true_train.astype(np.int32),
            'world': world, 'T_total': T, 'train_ratio': train_ratio,
            'y_test': y_test.astype(np.float32),
            'mask_test': ((true_states_full[:, T_train:] >= 0) & (~np.isnan(y_full[:, T_train:]))).astype(bool),
            'true_states_test': true_states_full[:, T_train:].astype(np.int32),
            'T_test': T - T_train
        }
    else:
        mask = (true_states_full >= 0) & (~np.isnan(y_full))
        y_full = np.where(mask, y_full, 0.0)
        
        data = {
            'N': N_effective, 'T': T, 'y': y_full.astype(np.float32),
            'mask': mask.astype(bool), 'true_states': true_states_full.astype(np.int32),
            'world': world
        }
    
    y_valid = data['y'][data['mask']]
    print(f"  Data: N={N_effective}, T={data['T']}, zeros={np.mean(y_valid==0):.1%}")
    
    return data


# =============================================================================
# SMC RUNNER
# =============================================================================

def run_smc_nbd_only(data, K, draws, chains, seed, out_dir):
    """Run SMC for NBD-only model."""
    cores = min(chains, 4)
    t0 = time.time()
    
    try:
        with make_nbd_only_hmm(data, K) as model:
            print(f"\nModel: K={K}, NBD-ONLY, world={data['world']}")
            print(f"SMC: draws={draws}, chains={chains}, cores={cores}")
            
            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                return_inferencedata=True
            )
            
            elapsed = (time.time() - t0) / 60
            
            # Log-evidence (fixed extraction)
            log_ev = np.nan
            try:
                lm = idata.sample_stats.log_marginal_likelihood.values
                if hasattr(lm, 'dtype') and lm.dtype == object:
                    chain_finals = []
                    for chain_data in np.array(lm).flatten():
                        if isinstance(chain_data, (list, np.ndarray)):
                            valid = [float(x) for x in chain_data 
                                    if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                            if valid:
                                chain_finals.append(valid[-1])
                        elif isinstance(chain_data, (int, float, np.floating)) and np.isfinite(chain_data):
                            chain_finals.append(float(chain_data))
                    log_ev = float(np.mean(chain_finals)) if chain_finals else np.nan
                else:
                    flat = np.array(lm).flatten()
                    valid = flat[np.isfinite(flat)]
                    log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
            except Exception as e:
                print(f"  Log-ev warning: {e}")
            
            print(f"  log_ev={log_ev:.2f}, time={elapsed:.1f}min")
            
            # Diagnostics
            diagnostics = {}
            try:
                ess = az.ess(idata)
                rhat = az.rhat(idata)
                ess_vals = [ess[v].values for v in ess.data_vars if hasattr(ess[v].values, 'size')]
                rhat_vals = [rhat[v].values for v in rhat.data_vars if hasattr(rhat[v].values, 'size')]
                diagnostics['ess_min'] = float(min([v.min() for v in ess_vals])) if ess_vals else np.nan
                diagnostics['rhat_max'] = float(max([v.max() for v in rhat_vals])) if rhat_vals else np.nan
                print(f"  ESS: min={diagnostics['ess_min']:.0f}, R-hat: max={diagnostics['rhat_max']:.3f}")
            except:
                diagnostics = {'ess_min': np.nan, 'rhat_max': np.nan}
            
            # Compile results
            res = {
                'K': K,
                'model_type': 'NBD_ONLY',
                'world': data['world'],
                'N': data['N'],
                'T': data['T'],
                'log_evidence': log_ev,
                'draws': draws,
                'chains': chains,
                'time_min': elapsed,
                **diagnostics
            }
            
            # Save
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            pkl_name = f"smc_K{K}_NBDONLY_N{data['N']}_T{data['T']}_D{draws}.pkl"
            pkl_path = out_dir / pkl_name
            
            with open(pkl_path, 'wb') as f:
                pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)
            
            print(f"  Saved: {pkl_path}")
            return pkl_path, res, idata
            
    except Exception as e:
        elapsed = (time.time() - t0) / 60
        print(f"  FAILED after {elapsed:.1f}min: {str(e)}")
        raise


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='NBD-only HMM: Frequency only')
    
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument('--T', type=int, default=104)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NBD-Only HMM: Frequency Only (No Monetary)")
    print("=" * 70)
    
    data = load_simulation_data_from_csv(
        Path(args.csv_path), args.T, args.N, args.train_ratio, args.seed
    )
    
    print(f"\nConfig: K={args.K}, N={data['N']}, T={data['T']}, world={data['world']}")
    print("=" * 70)
    
    out_dir = Path(args.out_dir) / data['world'].lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pkl_path, res, idata = run_smc_nbd_only(
        data=data, K=args.K, draws=args.draws, chains=args.chains,
        seed=args.seed, out_dir=out_dir
    )
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Log-Evidence: {res['log_evidence']:.2f}")
    print(f"Runtime: {res['time_min']:.1f} minutes")
    print(f"Output: {pkl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
