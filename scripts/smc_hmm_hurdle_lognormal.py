#!/usr/bin/env python3
"""
smc_hmm_hurdle.py - Production HMM-Hurdle-Gamma with Batched Forward Algorithm
GitHub: https://github.com/sudhir-voleti/smcpaper/blob/main/scripts/smc_hmm_hurdle.py

Usage:
    # PILOT MODE: Quick validation (N=50, T=52, fast)
    python smc_hmm_hurdle.py --world Cliff --K 3 --N 50 --T 52 --draws 200 --pilot

    # PRODUCTION: Paper results (N=200, T=104)
    python smc_hmm_hurdle.py --world Harbor --K 2 --N 200 --T 104
    python smc_hmm_hurdle.py --world Cliff --K 3 --N 200 --T 104

    # EXTRACTION: Post-process results
    python smc_hmm_hurdle.py --extract --pkl_file results/smc_Cliff_K3_GAM_N200_T104_D1000.pkl
"""

# =============================================================================
# 0. ENVIRONMENT
# =============================================================================
import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'

import numpy as np
import pytensor.tensor as pt
import pytensor
from pytensor import scan

print(f"PyTensor: floatX={pytensor.config.floatX}, device={pytensor.config.device}")

# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import time
import pickle
import warnings
from pathlib import Path
import json

import pandas as pd
import pymc as pm
import arviz as az
from patsy import dmatrix
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score, confusion_matrix

warnings.filterwarnings('ignore')
RANDOM_SEED = 42

# =============================================================================
# 2. DATA LOADING (with flexible subsetting)
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


def load_simulation_data(world: str, data_dir: Path, T: int = 104, N: int = None, 
                        pilot: bool = False, seed: int = RANDOM_SEED):
    """
    Load simulation CSV with flexible subsetting for pilot mode.

    Parameters:
    -----------
    world : str - Simulation world name
    data_dir : Path - Directory containing CSV files
    T : int - Target time periods
    N : int - Target customers (None = all, or subsample to this)
    pilot : bool - If True, use faster defaults and verbose output
    seed : int - Random seed for reproducible subsampling
    """
    if pilot:
        print(f"  [PILOT MODE] Fast settings, verbose output")

    # Try multiple filename patterns
    patterns = [
        data_dir / f"hmm_{world}_N{N}_T{T}.csv",
        data_dir / f"hmm_{world}_T{T}.csv",
        data_dir / f"{world.lower()}_N{N or 200}_T{T}.csv",
        data_dir / f"sim_{world.lower()}_N{N or 200}_T{T}_seed42.csv",
        data_dir / f"{world.lower()}.csv"
    ]

    csv_path = None
    for p in patterns:
        if p.exists():
            csv_path = p
            if pilot:
                print(f"  [PILOT] Found data: {p.name}")
            break

    if csv_path is None:
        raise FileNotFoundError(f"Simulation data not found. Tried: {[str(p) for p in patterns]}")

    df = pd.read_csv(csv_path)

    # Handle column name variations
    col_mapping = {
        'customer_id': ['customer_id', 'cust_id', 'id', 'customer'],
        't': ['t', 'time', 'period', 'time_period', 'week'],
        'y': ['y', 'spend', 'purchase', 'value'],
        'true_state': ['true_state', 'state', 'true_latent', 'latent']
    }

    actual_cols = {}
    for std_name, variants in col_mapping.items():
        for v in variants:
            if v in df.columns:
                actual_cols[std_name] = v
                break

    if len(actual_cols) < 4:
        missing = set(col_mapping.keys()) - set(actual_cols.keys())
        raise ValueError(f"CSV missing columns. Found: {df.columns.tolist()}, need: {missing}")

    # Rename to standard
    df = df.rename(columns={v: k for k, v in actual_cols.items()})

    # Reshape to balanced panel
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()

    if pilot:
        print(f"  [PILOT] Raw data: N={N_actual}, T={T_actual}")

    y_full = df.pivot(index='customer_id', columns='t', values='y').values
    true_states_full = df.pivot(index='customer_id', columns='t', values='true_state').values

    # Pad or truncate to target T
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

    # Subsample customers if requested
    if N is not None and N < N_actual:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_actual, N, replace=False)
        y_full = y_full[idx, :]
        true_states_full = true_states_full[idx, :]
        N_effective = N
        if pilot:
            print(f"  [PILOT] Subsampled to N={N} (seed={seed})")
    else:
        N_effective = N_actual

    # Create mask
    mask = (true_states_full >= 0) & (~np.isnan(y_full))
    y_full = np.where(mask, y_full, 0.0)

    # Compute RFM
    R, F, M = compute_rfm_features(y_full, mask)

    # Standardize
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

    # Summary
    y_valid = y_full[mask]
    zero_rate = np.mean(y_valid == 0) if len(y_valid) > 0 else 0.0
    mean_spend = np.mean(y_valid[y_valid > 0]) if (y_valid > 0).any() else 0.0

    print(f"  Loaded: N={N_effective}, T={T_effective}, zeros={zero_rate:.1%}, mean_spend=${mean_spend:.2f}")

    return data

## ----

def load_simulation_data_from_csv(csv_path: Path, T: int = 104, N: int = None, 
                                  seed: int = RANDOM_SEED):
    """
    Load simulation data from explicit CSV file path.
    
    Parameters:
    -----------
    csv_path : Path - Direct path to CSV file
    T : int - Target time periods (will pad/truncate to this)
    N : int - Number of customers to subsample (None = use all)
    seed : int - Random seed for reproducible subsampling
    
    Returns:
    --------
    data : dict with keys 'N', 'T', 'y', 'mask', 'R', 'F', 'M', 'true_states', etc.
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"  Reading CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    
    # Detect world name from filename
    world = "unknown"
    for w in ['Harbor', 'Breeze', 'Fog', 'Cliff']:
        if w.lower() in csv_path.name.lower():
            world = w
            break
    
    # Flexible column name mapping
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
    
    # Rename to standard
    df = df.rename(columns={v: k for k, v in actual_cols.items()})
    
    # Reshape to panel
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()
    
    y_full = df.pivot(index='customer_id', columns='t', values='y').values
    true_states_full = df.pivot(index='customer_id', columns='t', values='true_state').values
    
    # Pad or truncate to target T
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
    
    # Subsample customers if requested
    if N is not None and N < N_actual:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_actual, N, replace=False)
        y_full = y_full[idx, :]
        true_states_full = true_states_full[idx, :]
        N_effective = N
        print(f"  Subsampled: N={N} (from {N_actual})")
    else:
        N_effective = N_actual
    
    # Create mask and compute RFM
    mask = (true_states_full >= 0) & (~np.isnan(y_full))
    y_full = np.where(mask, y_full, 0.0)
    
    R, F, M = compute_rfm_features(y_full, mask)
    
    # Standardize RFM
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
    
    # Summary stats
    y_valid = y_full[mask]
    zero_rate = np.mean(y_valid == 0) if len(y_valid) > 0 else 0.0
    mean_spend = np.mean(y_valid[y_valid > 0]) if (y_valid > 0).any() else 0.0
    
    print(f"  Data: N={N_effective}, T={T_effective}, zeros={zero_rate:.1%}, mean=${mean_spend:.2f}")
    
    return data

# =============================================================================
# 3. GAM BASIS
# =============================================================================

def create_bspline_basis(x, df=3, degree=3):
    """Create B-spline basis matrix."""
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
# 4. FORWARD ALGORITHM (Grok-Corrected)
# =============================================================================

def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """
    Batched forward algorithm with proper scaling.
    """
    N, T, K = log_emission.shape
    
    # Initial step (t=0)
    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]
    log_Z_init     = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)
    log_alpha_norm_init = log_alpha_init - log_Z_init
    
    # Scan step
    def forward_step(log_emit_t, log_alpha_prev, log_Z_prev, log_Gamma):
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)
        
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)
        log_alpha_norm = log_alpha_new - log_Z_t
        
        return log_alpha_norm, log_Z_t
    
    # Emission sequence from t=1 onward
    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)  # (T-1, N, K)
    
    # Run scan
    (log_alpha_norm_seq, log_Z_seq), updates = scan(
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
    )
    
    # Full normalized alpha sequence (T, N, K)
    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],
        log_alpha_norm_seq
    ], axis=0)
    
    # Correct marginal likelihood: sum all normalization constants
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    
    return log_marginal, log_alpha_norm_full


# =============================================================================
# 5. HMM MODEL
# =============================================================================

def make_hurdle_hmm(data, K, use_gam=True, gam_df=3, pilot=False):
    """HMM-Hurdle-Gamma with batched forward algorithm."""
    y = data['y']
    R, F, M = data['R'], data['F'], data['M']
    mask = data['mask']
    N, T = data['N'], data['T']

    if pilot:
        print(f"  [PILOT] Building model: N={N}, T={T}, K={K}")

    # GAM basis
    if use_gam:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()

        basis_R = create_bspline_basis(R_flat, df=gam_df).reshape(N, T, -1)
        basis_F = create_bspline_basis(F_flat, df=gam_df).reshape(N, T, -1)
        basis_M = create_bspline_basis(M_flat, df=gam_df).reshape(N, T, -1)

        n_basis_R = basis_R.shape[2]
        n_basis_F = basis_F.shape[2]
        n_basis_M = basis_M.shape[2]
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={
        "customer": np.arange(N),
        "time": np.arange(T),
        "state": np.arange(K)
    }) as model:

        # Latent dynamics
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype=np.float32))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.eye(K)*5 + 1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            log_Gamma = pt.log(Gamma)

        # Hurdle parameters
        if use_gam:
            alpha0_h = pm.Normal("alpha0_h", 0, 1, shape=K if K > 1 else None)
            w_R_h = pm.Normal("w_R_h", 0, 1, shape=(K, n_basis_R) if K > 1 else n_basis_R)
            w_F_h = pm.Normal("w_F_h", 0, 1, shape=(K, n_basis_F) if K > 1 else n_basis_F)
            w_M_h = pm.Normal("w_M_h", 0, 1, shape=(K, n_basis_M) if K > 1 else n_basis_M)

            if K == 1:
                logit_pi = (alpha0_h + 
                           pt.tensordot(basis_R, w_R_h, axes=([2], [0])) +
                           pt.tensordot(basis_F, w_F_h, axes=([2], [0])) +
                           pt.tensordot(basis_M, w_M_h, axes=([2], [0])))
            else:
                logit_pi = (alpha0_h[None, None, :] +
                           pt.tensordot(basis_R, w_R_h, axes=([2], [1])) +
                           pt.tensordot(basis_F, w_F_h, axes=([2], [1])) +
                           pt.tensordot(basis_M, w_M_h, axes=([2], [1])))
        else:
            alpha0_h = pm.Normal("alpha0_h", 0, 1, shape=K if K > 1 else None)
            alphaR_h = pm.Normal("alphaR_h", 0, 1, shape=K if K > 1 else None)
            alphaF_h = pm.Normal("alphaF_h", 0, 1, shape=K if K > 1 else None)
            alphaM_h = pm.Normal("alphaM_h", 0, 1, shape=K if K > 1 else None)

            if K == 1:
                logit_pi = alpha0_h + alphaR_h * R + alphaF_h * F + alphaM_h * M
            else:
                logit_pi = (alpha0_h + 
                           alphaR_h * R[..., None] + 
                           alphaF_h * F[..., None] + 
                           alphaM_h * M[..., None])

        pi = pt.clip(pt.sigmoid(logit_pi), 1e-6, 1 - 1e-6)

        # Log-Normal parameters (sigma = log-scale SD)
        if K == 1:
            beta0 = pm.Normal("beta0", 0, 1)
            sigma = pm.Exponential("sigma", 1)  # Log-scale SD
        else:
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0 = pm.Deterministic("beta0", pt.sort(beta0_raw))
            sigma = pm.Exponential("sigma", 1, shape=K)


        # Mean spend
        if use_gam:
            w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R) if K > 1 else n_basis_R)
            w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F) if K > 1 else n_basis_F)
            w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M) if K > 1 else n_basis_M)

            if K == 1:
                log_mu = (beta0 + 
                         pt.tensordot(basis_R, w_R, axes=([2], [0])) +
                         pt.tensordot(basis_F, w_F, axes=([2], [0])) +
                         pt.tensordot(basis_M, w_M, axes=([2], [0])))
            else:
                log_mu = (beta0[None, None, :] +
                         pt.tensordot(basis_R, w_R, axes=([2], [1])) +
                         pt.tensordot(basis_F, w_F, axes=([2], [1])) +
                         pt.tensordot(basis_M, w_M, axes=([2], [1])))
        else:
            betaR = pm.Normal("betaR", 0, 1, shape=K if K > 1 else None)
            betaF = pm.Normal("betaF", 0, 1, shape=K if K > 1 else None)
            betaM = pm.Normal("betaM", 0, 1, shape=K if K > 1 else None)

            if K == 1:
                log_mu = beta0 + betaR * R + betaF * F + betaM * M
            else:
                log_mu = (beta0 + 
                         betaR * R[..., None] + 
                         betaF * F[..., None] + 
                         betaM * M[..., None])

        mu = pt.exp(pt.clip(log_mu, -10, 10))

        # Emissions (Log-Normal hurdle)
        if K == 1:
            log_zero = pt.log(1 - pi)
            y_clipped = pt.clip(y, 1e-10, 1e10)
            log_y = pt.log(y_clipped)
            # Log-Normal logpdf: -0.5*((log(y)-mu)^2/sigma^2 + log(2*pi*sigma^2)) - log(y)
            log_lognorm = -0.5 * ((log_y - log_mu)**2 / sigma**2 + pt.log(2 * np.pi * sigma**2)) - log_y
            log_pos = pt.log(pi) + log_lognorm
            log_emission = pt.where(y == 0, log_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
            logp_cust = pt.sum(log_emission, axis=1)

        else:
            pi_exp = pi[..., None] if pi.ndim == 2 else pi
            sigma_exp = sigma[None, None, :] if sigma.ndim == 1 else sigma
            log_mu_exp = log_mu[..., None] if log_mu.ndim == 2 else log_mu

            y_exp = y[..., None]
            mask_exp = mask[:, :, None]

            log_zero = pt.log(1 - pi_exp)
            y_clipped = pt.clip(y_exp, 1e-10, 1e10)
            log_y = pt.log(y_clipped)
            # Log-Normal logpdf
            log_lognorm = -0.5 * ((log_y - log_mu_exp)**2 / sigma_exp**2 + pt.log(2 * np.pi * sigma_exp**2)) - log_y
            log_pos = pt.log(pi_exp) + log_lognorm

            log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)
            log_emission = pt.where(mask_exp, log_emission, 0.0)

            if pilot:
                print(f"  [PILOT] Running batched forward algorithm...")

            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)

            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered,
                           dims=("customer", "time", "state"))

        # Add log_likelihood deterministic for WAIC/LOO computation (both K=1 and K>1)
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        
        # Viterbi MAP states from filtered probabilities (K>1 only)
        if K > 1:
            viterbi_map = pt.argmax(alpha_filtered, axis=2)
            pm.Deterministic("viterbi", viterbi_map, dims=("customer", "time"))
        
        pm.Potential("loglike", pt.sum(logp_cust))

    return model


def compute_oos_prediction(data, idata, use_gam, gam_df, n_draws_use=200):
    """
    Compute OOS predictions for Hurdle model.
    """
    try:
        N, T_test = data['y_test'].shape
        y_test = data['y_test']
        R_test, F_test, M_test = data['R_test'], data['F_test'], data['M_test']
        
        post = idata.posterior
        n_chains, n_draws_total = post.sizes['chain'], post.sizes['draw']
        K = post['Gamma'].shape[-1] if 'Gamma' in post else 1
        
        # Precompute GAM bases
        if use_gam:
            basis_R = create_bspline_basis(R_test.flatten(), df=gam_df).reshape(N, T_test, -1)
            basis_F = create_bspline_basis(F_test.flatten(), df=gam_df).reshape(N, T_test, -1)
            basis_M = create_bspline_basis(M_test.flatten(), df=gam_df).reshape(N, T_test, -1)
        
        draw_idx = np.random.choice(n_chains * n_draws_total, min(n_draws_use, n_chains * n_draws_total), replace=False)
        
        y_pred_samples = []
        
        for idx in draw_idx:
            c = idx // n_draws_total
            d = idx % n_draws_total
            
            # Hurdle params
            alpha0 = post['alpha0'].isel(chain=c, draw=d).values
            beta0 = post['beta0'].isel(chain=c, draw=d).values
            
            if use_gam:
                w_R_h = post['w_R_h'].isel(chain=c, draw=d).values
                w_F_h = post['w_F_h'].isel(chain=c, draw=d).values
                w_M_h = post['w_M_h'].isel(chain=c, draw=d).values
                w_R = post['w_R'].isel(chain=c, draw=d).values
                w_F = post['w_F'].isel(chain=c, draw=d).values
                w_M = post['w_M'].isel(chain=c, draw=d).values
                
                if K == 1:
                    # Zero part
                    eff_R_h = np.tensordot(basis_R, w_R_h, axes=([2], [0]))
                    eff_F_h = np.tensordot(basis_F, w_F_h, axes=([2], [0]))
                    eff_M_h = np.tensordot(basis_M, w_M_h, axes=([2], [0]))
                    logit_p = alpha0 + eff_R_h + eff_F_h + eff_M_h
                    p_pos = 1 / (1 + np.exp(-logit_p))
                    
                    # Positive part
                    eff_R = np.tensordot(basis_R, w_R, axes=([2], [0]))
                    eff_F = np.tensordot(basis_F, w_F, axes=([2], [0]))
                    eff_M = np.tensordot(basis_M, w_M, axes=([2], [0]))
                    mu = np.exp(beta0 + eff_R + eff_F + eff_M)
                    
                    y_pred_d = p_pos * mu
                else:
                    # Zero part
                    eff_R_h = np.tensordot(basis_R, w_R_h, axes=([2], [1]))
                    eff_F_h = np.tensordot(basis_F, w_F_h, axes=([2], [1]))
                    eff_M_h = np.tensordot(basis_M, w_M_h, axes=([2], [1]))
                    logit_p = alpha0[None, None, :] + eff_R_h + eff_F_h + eff_M_h
                    p_pos = 1 / (1 + np.exp(-logit_p))
                    
                    # Positive part
                    eff_R = np.tensordot(basis_R, w_R, axes=([2], [1]))
                    eff_F = np.tensordot(basis_F, w_F, axes=([2], [1]))
                    eff_M = np.tensordot(basis_M, w_M, axes=([2], [1]))
                    mu = np.exp(beta0[None, None, :] + eff_R + eff_F + eff_M)
                    
                    # HMM state propagation
                    if 'alpha_filtered' in post:
                        state_prob = post['alpha_filtered'].isel(chain=c, draw=d).values[:, -1, :]
                    else:
                        state_prob = np.ones((N, K)) / K
                    Gamma = post['Gamma'].isel(chain=c, draw=d).values
                    
                    y_pred_d = np.zeros((N, T_test))
                    for t in range(T_test):
                        state_prob = state_prob @ Gamma
                        y_pred_d[:, t] = np.sum(state_prob * p_pos[:, t, :] * mu[:, t, :], axis=1)
            else:
                # GLM version - add if needed, or raise error
                raise NotImplementedError("OOS for Hurdle-GLM not implemented, use GAM")
            
            y_pred_samples.append(y_pred_d)
        
        y_pred_mean = np.mean(y_pred_samples, axis=0)
        
        # Metrics
        mask = ~np.isnan(y_test)
        if mask.sum() == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'y_pred': None}
        
        residuals = y_test[mask] - y_pred_mean[mask]
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        return {'rmse': float(rmse), 'mae': float(mae), 'y_pred': y_pred_mean}
        
    except Exception as e:
        print(f"  OOS error: {e}")
        return {'rmse': np.nan, 'mae': np.nan, 'y_pred': None}


# =============================================================================
# 6. SMC RUNNER
# =============================================================================

def run_smc_hmm(data, K, use_gam, gam_df, draws, chains, seed, out_dir):
    """
    Run SMC with diagnostics and dual output (PKL + CSV summary).
    
    Saves:
        - .pkl: Full InferenceData for post-processing
        - summary_*.csv: Flat table for easy aggregation
    """
    cores = min(chains, 4)
    t0 = time.time()


    try:
        with make_hurdle_hmm(data, K, use_gam, gam_df) as model:
            glm_gam = "GAM" if use_gam else "GLM"
            print(f"\nModel: K={K}, HURDLE-{glm_gam}, world={data['world']}")
            print(f"SMC: draws={draws}, chains={chains}, cores={cores}")
            
            idata = pm.sample_smc(
                draws=draws, 
                chains=chains, 
                cores=cores,
                random_seed=seed, 
                return_inferencedata=True
            )
                
            # Post-hoc ESS calculation (more accurate)
            try:
                # Use arviz ESS on key parameters
                ess_vals = []
                for var in ['beta0', 'phi', 'Gamma']:
                    if var in idata.posterior:
                        ess = az.ess(idata, var_names=[var])
                        ess_vals.append(ess.to_array().values.flatten())
                if ess_vals:
                    ess_min = float(np.nanmin(np.concatenate(ess_vals)))
                else:
                    ess_min = np.nan
            except:
                ess_min = np.nan
               
            elapsed = (time.time() - t0) / 60
 
            # OOS prediction
            oos_rmse, oos_mae = np.nan, np.nan
            if 'y_test' in data:
                print("  Computing OOS predictions...")
                oos_results = compute_oos_prediction(data, idata, use_gam, gam_df)
                oos_rmse = oos_results.get('rmse', np.nan)
                oos_mae = oos_results.get('mae', np.nan)
                print(f"  OOS RMSE: {oos_rmse:.4f}, MAE: {oos_mae:.4f}")
 
            # Extract log-evidence (works for both Tweedie and Hurdle)
            log_ev = np.nan
            try:
                lm = idata.sample_stats.log_marginal_likelihood.values
                if isinstance(lm, np.ndarray):
                    # Handle object dtype (list of lists)
                    if lm.dtype == object:
                        chain_vals = []
                        for c in range(lm.shape[1] if lm.ndim > 1 else lm.shape[0]):
                            # Get last element from each chain
                            if lm.ndim > 1:
                                chain_data = lm[-1, c] if lm.shape[0] > 0 else lm[0, c]
                            else:
                                chain_data = lm[c]
                            
                            # Extract final value from list/array
                            if isinstance(chain_data, (list, np.ndarray)):
                                valid = [float(x) for x in chain_data 
                                        if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                                if valid:
                                    chain_vals.append(valid[-1])
                            elif isinstance(chain_data, (int, float, np.floating)) and np.isfinite(chain_data):
                                chain_vals.append(float(chain_data))
                        
                        log_ev = float(np.mean(chain_vals)) if chain_vals else np.nan
                    else:
                        # Numeric array
                        flat = np.array(lm).flatten()
                        valid = flat[np.isfinite(flat)]
                        log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
 
            except Exception as e:
                print(f"  Warning: log-ev extraction failed: {e}")
                log_ev = np.nan
 
                       
            print(f"  log_ev={log_ev:.2f}, time={elapsed:.1f}min")
            
            # Diagnostics
            diagnostics = {}
            try:
                ess = az.ess(idata)
                rhat = az.rhat(idata)
                
                ess_vals = [ess[v].values for v in ess.data_vars if hasattr(ess[v].values, 'size')]
                rhat_vals = [rhat[v].values for v in rhat.data_vars if hasattr(rhat[v].values, 'size')]
                
                diagnostics['ess_min'] = float(min([v.min() for v in ess_vals])) if ess_vals else np.nan
                diagnostics['ess_median'] = float(np.median([v.mean() for v in ess_vals])) if ess_vals else np.nan
                diagnostics['rhat_max'] = float(max([v.max() for v in rhat_vals])) if rhat_vals else np.nan
                
                print(f"  ESS: min={diagnostics['ess_min']:.0f}, med={diagnostics['ess_median']:.0f}")
                print(f"  R-hat: max={diagnostics['rhat_max']:.3f}")
                
                if diagnostics['rhat_max'] > 1.1:
                    print(f"WARNING: R-hat > 1.1 indicates possible non-convergence")
                
            except Exception as e:
                diagnostics = {'ess_min': np.nan, 'ess_median': np.nan, 'rhat_max': np.nan}

            # Compile results
            res = {
                'K': K,
                'model_type': 'HURDLE',
                'glm_gam': glm_gam,
                'world': data['world'],
                'N': data['N'],
                'T': data['T'],
                'log_evidence': log_ev,
                'draws': draws,
                'chains': chains,
                'time_min': elapsed,
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'zero_rate': float(np.mean(data['y'][data['mask']] == 0)),
                'mean_spend': float(np.mean(data['y'][data['mask']][data['y'][data['mask']] > 0])) if (data['y'][data['mask']] > 0).any() else 0.0,
                'oos_rmse': oos_rmse,
                'oos_mae': oos_mae,
                **diagnostics
            }

            # Setup output directory
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PKL (full data)
            pkl_name = f"smc_K{K}_{glm_gam}_N{data['N']}_T{data['T']}_D{draws}.pkl"
            pkl_path = out_dir / pkl_name
            
            with open(pkl_path, 'wb') as f:
                pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)
            
            # Save CSV (flat summary for aggregation)
            csv_name = f"summary_K{K}_{glm_gam}_N{data['N']}_T{data['T']}_D{draws}.csv"
            csv_path = out_dir / csv_name
            
            # Flatten for CSV (convert arrays to strings)
            res_flat = {}
            for k, v in res.items():
                if isinstance(v, (np.ndarray, list)):
                    res_flat[k] = str(v)
                elif isinstance(v, (np.integer, np.floating)):
                    res_flat[k] = float(v)
                else:
                    res_flat[k] = v
            
            pd.DataFrame([res_flat]).to_csv(csv_path, index=False)
            
            print(f"  Saved: {pkl_path}")
            print(f"  Saved: {csv_path}")
            
            return pkl_path, res, idata
            
    except Exception as e:
        elapsed = (time.time() - t0) / 60
        print(f"  FAILED after {elapsed:.1f}min: {str(e)}")
        raise

# =============================================================================
# 7. POST-PROCESSING & EXTRACTION
# =============================================================================

def extract_viterbi_states(idata, data):
    """
    Viterbi decoding from posterior means.

    Uses MAP estimate of parameters to find most likely state sequence.
    """
    if 'alpha_filtered' not in idata.posterior:
        print("WARNING: alpha_filtered not found in idata")
        return None

    # Get posterior means
    post = idata.posterior
    Gamma_mean = post['Gamma'].mean(dim=['chain', 'draw']).values
    pi0_mean = post['pi0'].mean(dim=['chain', 'draw']).values

    # Get emission parameters
    alpha_mean = post['alpha_gamma'].mean(dim=['chain', 'draw']).values
    #beta_mean = post['beta_gamma'].mean(dim=['chain', 'draw']).values

    y = data['y']
    N, T = data['N'], data['T']
    K = len(pi0_mean)

    viterbi_states = np.zeros((N, T), dtype=int)

    for i in range(N):
        # Log emission probs for customer i
        log_emit = np.zeros((T, K))

        for t in range(T):
            if not data['mask'][i, t]:
                continue

            y_it = y[i, t]
            for k in range(K):
                # Hurdle emission
                pi_ik = post['pi'].mean(dim=['chain', 'draw']).values[i, t, k] if 'pi' in post else 0.5

                if y_it == 0:
                    log_emit[t, k] = np.log(1 - pi_ik + 1e-10)
                else:
                    # Gamma logpdf
                    from scipy.stats import gamma
                    log_gamma = gamma.logpdf(y_it, alpha_mean[k], scale=1/beta_mean[i,t,k] if beta_mean.ndim > 1 else 1/beta_mean[k])
                    log_emit[t, k] = np.log(pi_ik + 1e-10) + log_gamma

        # Viterbi
        log_delta = np.log(pi0_mean + 1e-10) + log_emit[0]
        psi = np.zeros((T, K), dtype=int)

        for t in range(1, T):
            for k in range(K):
                scores = log_delta + np.log(Gamma_mean[:, k] + 1e-10)
                psi[t, k] = np.argmax(scores)
                log_delta[k] = np.max(scores) + log_emit[t, k]

        # Backtrack
        states_i = np.zeros(T, dtype=int)
        states_i[-1] = np.argmax(log_delta)
        for t in range(T-2, -1, -1):
            states_i[t] = psi[t+1, states_i[t+1]]

        viterbi_states[i] = states_i

    return viterbi_states


def compute_state_recovery_metrics(idata, data):
    """
    Compute state recovery metrics against true states.

    Returns ARI, confusion matrix, per-state accuracy.
    """
    if 'true_states' not in data:
        print("WARNING: No true_states in data")
        return None

    true_states = data['true_states']

    # Try to get Viterbi from model (if stored) or compute post-hoc
    if 'viterbi' in idata.posterior:
        viterbi = idata.posterior['viterbi'].mean(dim=['chain', 'draw']).values.astype(int)
    else:
        print("Computing Viterbi post-hoc...")
        viterbi = extract_viterbi_states(idata, data)

    if viterbi is None:
        return None

    # Flatten
    mask = data['mask']
    true_flat = true_states[mask]
    viterbi_flat = viterbi[mask]

    # ARI
    ari = adjusted_rand_score(true_flat, viterbi_flat)

    # Confusion matrix
    K = len(np.unique(true_flat))
    conf_mat = confusion_matrix(true_flat, viterbi_flat, labels=range(K))

    # Per-state accuracy
    per_state_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)

    metrics = {
        'ari': ari,
        'confusion_matrix': conf_mat.tolist(),
        'per_state_accuracy': per_state_acc.tolist(),
        'overall_accuracy': np.mean(viterbi_flat == true_flat)
    }

    return metrics


def quick_extract(pkl_path):
    """
    Quick extraction and summary of a .pkl result file.

    Usage:
        python smc_hmm_hurdle.py --extract --pkl_file results/smc_...pkl
    """
    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        return

    print(f"\nLoading: {pkl_path.name}")
    print("=" * 60)

    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    idata = result['idata']
    res = result['res']
    data = result.get('data', {})

    # Basic info
    print(f"Model: K={res['K']}, {res['glm_gam']}, world={res['world']}")
    print(f"Data: N={res['N']}, T={res['T']}, zeros={res['zero_rate']:.1%}")
    print(f"SMC: draws={res['draws']}, chains={res['chains']}, time={res['time_min']:.1f}min")
    print(f"Fit: log_ev={res['log_evidence']:.2f}")

    if 'ess_min' in res and not np.isnan(res['ess_min']):
        print(f"Diagnostics: ESS={res['ess_min']:.0f}, R-hat={res['rhat_max']:.3f}")

    # Posterior summary
    print("\nPosterior Summary:")
    print("-" * 60)

    if 'beta0' in idata.posterior:
        beta0 = idata.posterior['beta0']
        print(f"beta0 (state intercepts):")
        for k in range(beta0.shape[-1]):
            mean = float(beta0.mean(dim=['chain', 'draw']).values[k])
            sd = float(beta0.std(dim=['chain', 'draw']).values[k])
            print(f"  State {k}: {mean:.3f} (±{sd:.3f})")

    if 'alpha_gamma' in idata.posterior:
        alpha = idata.posterior['alpha_gamma']
        print(f"\nalpha (Gamma shape):")
        for k in range(alpha.shape[-1]):
            mean = float(alpha.mean(dim=['chain', 'draw']).values[k])
            sd = float(alpha.std(dim=['chain', 'draw']).values[k])
            print(f"  State {k}: {mean:.3f} (±{sd:.3f})")

    if 'Gamma' in idata.posterior:
        Gamma = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
        print(f"\nTransition matrix (mean):")
        print(f"  From\To   Dormant  Lukewarm  Whale")
        for i in range(Gamma.shape[0]):
            row = "  ".join([f"{Gamma[i,j]:.3f}" for j in range(Gamma.shape[1])])
            print(f"  State {i}:  {row}")

    # State recovery
    if 'true_states' in data:
        print("\nState Recovery:")
        print("-" * 60)
        metrics = compute_state_recovery_metrics(idata, data)
        if metrics:
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"  Per-state accuracy: {[f'{a:.2%}' for a in metrics['per_state_accuracy']]}")

    print("=" * 60)

    return result

## ----

def aggregate_results(base_out_dir: str, worlds: list = None):
    """
    Aggregate all summary CSVs across worlds into single table.
    """
    base_path = Path(base_out_dir)
    
    if worlds is None:
        worlds = ['harbor', 'breeze', 'fog', 'cliff']
    
    all_summaries = []
    
    for world in worlds:
        world_dir = base_path / world
        if not world_dir.exists():
            continue
        
        csv_files = list(world_dir.glob("summary_*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['world'] = world
                df['source_file'] = csv_file.name
                all_summaries.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {csv_file}: {e}")
    
    if not all_summaries:
        print("No summary files found")
        return None
    
    combined = pd.concat(all_summaries, ignore_index=True)
    
    if 'log_evidence' in combined.columns:
        combined = combined.sort_values(['world', 'log_evidence'], ascending=[True, False])
    
    master_path = base_path / "all_worlds_summary.csv"
    combined.to_csv(master_path, index=False)
    
    print(f"\nAggregated {len(all_summaries)} runs from {len(set(combined['world']))} worlds")
    print(f"Saved: {master_path}")
    
    # Display
    print("\n" + "="*100)
    print("AGGREGATED RESULTS")
    print("="*100)
    
    display_cols = ['world', 'K', 'glm_gam', 'N', 'T', 'log_evidence', 'time_min', 'ess_min', 'rhat_max']
    display_cols = [c for c in display_cols if c in combined.columns]
    
    for col in ['log_evidence', 'time_min', 'ess_min', 'rhat_max']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    print(combined[display_cols].to_string(index=False))
    print("="*100)
    
    return combined

# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='HMM-Hurdle-Gamma with Pilot Mode & Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PILOT: Quick test (N=50, T=52, fast)
  python smc_hmm_hurdle.py --world Cliff --K 3 --N 50 --T 52 --draws 200 --pilot

  # PRODUCTION: Paper results (N=200, T=104)
  python smc_hmm_hurdle.py --world Cliff --K 3 --N 200 --T 104

  # EXTRACT: Examine results
  python smc_hmm_hurdle.py --extract --pkl_file results/smc_Cliff_K3_GAM_N200_T104_D1000.pkl
        """
    )

    # Mode selection
    parser.add_argument('--extract', action='store_true',
                       help='Extraction mode: analyze existing .pkl file')
    parser.add_argument('--pkl_file', type=str,
                       help='Path to .pkl file for extraction')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate all summary CSVs across worlds')

    # Data input (choose one)
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Direct path to input CSV file (overrides --data_dir)')
    parser.add_argument('--data_dir', type=str, default='./data/simulation',
                       help='Data directory (used if --csv_path not provided)')
    parser.add_argument('--world', type=str,
                       choices=['Harbor', 'Breeze', 'Fog', 'Cliff'],
                       help='Simulation world (used if --csv_path not provided)')

    # Model parameters
    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3, 4],
                       help='Number of latent states')
    parser.add_argument('--T', type=int, default=104,
                       help='Time periods (default: 104 = 2 years weekly)')
    parser.add_argument('--N', type=int, default=None,
                       help='Subsample to N customers (default: use all in CSV)')

    # Model settings
    parser.add_argument('--no_gam', action='store_true',
                       help='Use GLM instead of GAM')
    parser.add_argument('--gam_df', type=int, default=3,
                       help='GAM degrees of freedom')

    # SMC settings
    parser.add_argument('--draws', type=int, default=1000,
                       help='SMC draws per chain')
    parser.add_argument('--chains', type=int, default=4,
                       help='Number of chains')
    parser.add_argument('--out_dir', type=str, default='./results',
                       help='Base output directory (world subfolders auto-created)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')

    args = parser.parse_args()

    # AGGREGATION MODE
    if args.aggregate:
        aggregate_results(args.out_dir)
        return

    # EXTRACTION MODE
    if args.extract:
        if not args.pkl_file:
            print("ERROR: --extract requires --pkl_file")
            return
        quick_extract(args.pkl_file)
        return

    # RUN MODE
    print("=" * 70)
    print("HMM-Hurdle-Gamma: SMC Estimation")
    print("=" * 70)

    # Load data
    if args.csv_path:
        # Direct CSV path provided
        print(f"Loading from: {args.csv_path}")
        csv_file = Path(args.csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        
        # Detect world from filename
        world = "unknown"
        for w in ['Harbor', 'Breeze', 'Fog', 'Cliff']:
            if w.lower() in csv_file.name.lower():
                world = w
                break
        
        data = load_simulation_data_from_csv(csv_file, args.T, args.N, seed=args.seed)
        args.world = world  # Override for output folder naming
        
    else:
        # Use data_dir + world
        if not args.world:
            print("ERROR: --world required when --csv_path not provided")
            return
        print(f"Loading {args.world} from: {args.data_dir}")
        data_dir = Path(args.data_dir)
        data = load_simulation_data(args.world, data_dir, args.T, args.N, seed=args.seed)

    print(f"\nConfiguration:")
    print(f"  World: {args.world} | K={args.K} | N={data['N']} | T={data['T']}")
    print(f"  Model: {'GLM' if args.no_gam else 'GAM'} | Draws: {args.draws}")
    print("=" * 70)

    # Setup output directory with world subfolder
    base_out_dir = Path(args.out_dir)
    world_out_dir = base_out_dir / args.world.lower()
    world_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {world_out_dir}")

    # Run SMC
    pkl_path, res, idata = run_smc_hmm(
        data=data,
        K=args.K,
        use_gam=not args.no_gam,
        gam_df=args.gam_df,
        draws=args.draws,
        chains=args.chains,
        seed=args.seed,
        out_dir=world_out_dir
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Log-Evidence: {res['log_evidence']:.2f}")
    print(f"Runtime: {res['time_min']:.1f} minutes")
    print(f"Output: {pkl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
