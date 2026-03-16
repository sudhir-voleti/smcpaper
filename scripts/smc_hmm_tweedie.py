#!/usr/bin/env python3
"""
smc_hmm_tweedie.py
==================
HMM-Tweedie model with saddlepoint approximation and OOS evaluation.
Optimized for Apple Silicon (M1/M2/M3)


# Recommended: state-varying p with shared phi
python smc_hmm_tweedie.py --sim_path data.csv --K 3 --state_specific_p --shared_phi ...

# Or: fixed p with state-varying phi
python smc_hmm_tweedie.py --sim_path data.csv --K 3 --p_fixed 1.5 ...

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
from pathlib import Path
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
# 2. RFM FEATURE COMPUTATION
# =============================================================================
def compute_rfm_features(y, mask):
    """
    Compute Recency, Frequency, Monetary features from panel data.
    R = time since last purchase
    F = cumulative purchase count  
    M = average spend per purchase
    """
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
                    R[i, t] = t + 1  # Time since "start" if no purchase yet
                    F[i, t] = 0
                    M[i, t] = 0.0
            else:
                R[i, t] = 0
                F[i, t] = 0
                M[i, t] = 0.0
    
    return R, F, M


def compute_rfm_features_oos(y_train, y_test, mask_test):
    """
    Propagate RFM state from training history into test period.
    """
    N, T_test = y_test.shape
    T_train = y_train.shape[1]
    R = np.zeros((N, T_test), dtype=np.float32)
    F = np.zeros((N, T_test), dtype=np.float32)
    M = np.zeros((N, T_test), dtype=np.float32)
    
    for i in range(N):
        # Initialize from training history
        train_purchase_indices = np.where(y_train[i, :] > 0)[0]
        if len(train_purchase_indices) > 0:
            last_p = train_purchase_indices[-1]
            cum_f = len(train_purchase_indices)
            cum_m = np.sum(y_train[i, :])
        else:
            last_p = -1
            cum_f = 0
            cum_m = 0.0
        
        # Roll forward through test period
        for t in range(T_test):
            t_abs = T_train + t
            if mask_test[i, t] and y_test[i, t] > 0:
                last_p = t_abs
                cum_f += 1
                cum_m += y_test[i, t]
            
            if last_p != -1:
                R[i, t] = t_abs - last_p
                F[i, t] = cum_f
                M[i, t] = cum_m / cum_f if cum_f > 0 else 0.0
            else:
                R[i, t] = t_abs + 1
                F[i, t] = 0
                M[i, t] = 0.0
            
    return R.astype(np.float32), F.astype(np.float32), M.astype(np.float32)


def gamma_logp_det(value, mu, phi):
    """
    Deterministic Gamma log-density with numerical stability.
    """
    alpha = mu / phi
    beta = 1.0 / phi
    
    logp = (alpha - 1) * pt.log(value) - value * beta + alpha * pt.log(beta) - pt.gammaln(alpha)
    
    # Clip to prevent extreme values
    return pt.clip(logp, -1e6, 0.0)

# =============================================================================
# 3. TWEEDIE MODEL BUILDER
# =============================================================================
def make_model(data, K=3, state_specific_p=True, p_fixed=None, use_gam=True, gam_df=3, use_covariates=True, shared_phi=False):

    """Build HMM-Tweedie (K>=2) or Static Tweedie (K=1) with saddlepoint approximation."""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    # Build GAM bases if needed
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
            phi = pm.Exponential("phi", lam=2.0)
        else:
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0_sorted = pt.sort(beta0_raw)
            beta0 = pm.Deterministic("beta0", beta0_sorted * beta0_prior_sd + beta0_prior_mean)
            
            # Shared phi across states for identification (recommended when state_specific_p=True)
            if shared_phi:
                phi = pm.Exponential("phi", lam=2.0)
            else:
                phi = pm.Exponential("phi", lam=2.0, shape=K)

        # ---- 3. COVARIATE EFFECTS ----
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
            # Zero effects if no covariates
            if use_gam:
                w_R = pt.zeros(n_basis_R) if K == 1 else pt.zeros((K, n_basis_R))
                w_F = pt.zeros(n_basis_F) if K == 1 else pt.zeros((K, n_basis_F))
                w_M = pt.zeros(n_basis_M) if K == 1 else pt.zeros((K, n_basis_M))
            else:
                betaR = pt.as_tensor_variable(0.0) if K == 1 else pt.zeros(K)
                betaF = pt.as_tensor_variable(0.0) if K == 1 else pt.zeros(K)
                betaM = pt.as_tensor_variable(0.0) if K == 1 else pt.zeros(K)


        # ---- 4. POWER PARAMETER P ----
        # Constrain p to [1.3, 1.7] to prevent boundary collapse (was [1.1, 1.9])
        # ALWAYS wrap p in pm.Deterministic so it appears in posterior
        if p_fixed is not None:
            # Fixed p across all states - still save as deterministic for consistency
            p = pm.Deterministic("p", pt.as_tensor_variable(np.array([p_fixed] * K, dtype=np.float32)))
        elif state_specific_p:
            # State-varying p with tighter bounds [1.3, 1.7]
            p_raw = pm.Beta("p_raw", alpha=2, beta=2, shape=K)
            p_sorted = pt.sort(p_raw)
            p = pm.Deterministic("p", 1.3 + p_sorted * 0.4)  # [1.3, 1.7]
        else:
            # Shared p across states (estimated)
            p_raw = pm.Beta("p_raw", alpha=2, beta=2)
            p_val = pm.Deterministic("p_val", 1.3 + p_raw * 0.4)  # scalar for inspection
            if K > 1:
                p = pm.Deterministic("p", pt.stack([p_val] * K))  # vector for broadcasting
            else:
                p = pm.Deterministic("p", pt.stack([p_val]))  # ensure 1D even for K=1


        # DEBUG: Verify p is registered as deterministic
        assert 'p' in model.named_vars, f"'p' not in model.named_vars! p type: {type(p)}"

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


        # ---- 6. ZIG EMISSION (Zero-Inflated Gamma approximation to Tweedie) ----
        # From manuscript Appendix D: KL ≈ 0.04 nats from true Tweedie
        if K == 1:
            p_exp, phi_exp = p, phi
            mu_exp = mu
            y_in, mask_in = y, mask
        else:
            p_exp = p[None, None, :]
            # Handle scalar phi (shared_phi=True) vs vector phi (state-specific)
            if phi.ndim == 0:
                phi_exp = phi  # scalar, will broadcast automatically
            else:
                phi_exp = phi[None, None, :]
            mu_exp = mu[..., None] if mu.ndim == 2 else mu
            y_in, mask_in = y[..., None], mask[:, :, None]

        # Zero probability from Tweedie thin-plate limit: ψ = exp(-μ^(2-p)/(φ(2-p)))
        exponent = 2.0 - p_exp
        lambda_param = pt.pow(mu_exp, exponent) / (phi_exp * exponent)
        psi = pt.exp(-lambda_param)
        psi = pt.clip(psi, 1e-12, 1.0 - 1e-12)

        # Positive part: Gamma(α=μ/φ, β=1/φ) - matches Tweedie mean=μ, var=φμ^p in limit
        log_zero = pt.log(psi)
        log_pos = pt.log1p(-psi) + gamma_logp_det(y_in, mu_exp, phi_exp)

        # Combine
        log_emission = pt.switch(pt.eq(y_in, 0), log_zero, log_pos)
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

        # Single log_likelihood deterministic for WAIC/LOO (both K=1 and K>1)
        pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))
        
        pm.Potential('loglike', pt.sum(logp_cust))

        return model
            

# =============================================================================
# 4. DATA LOADING
# =============================================================================

def load_uci_data(csv_path, n_cust=None, seed=42, train_frac=0.8, max_week=None):
    """
    Load UCI/CDNOW empirical data (ported from smc_tw_empirics.py).
    """
    import pandas as pd
    from pathlib import Path
    
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Detect columns
    id_col = 'customer_id' if 'customer_id' in df.columns else 'cust_id'
    time_col = 'WeekStart' if 'WeekStart' in df.columns else 't'
    y_col = 'spend' if 'spend' in df.columns else 'WeeklySpend'
    
    N_full = df[id_col].nunique()
    T_full = df[time_col].nunique() if 't' in df.columns else df.groupby(id_col).size().iloc[0]
    
    # Pivot to wide format
    obs = df.pivot(index=id_col, columns=time_col, values=y_col).values
    
    # Subsample if needed
    if n_cust is not None and n_cust < N_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_full, n_cust, replace=False)
        obs = obs[idx, :]
        N = n_cust
    else:
        N = N_full
        idx = np.arange(N)
    
    # Truncate to max_week
    if max_week is not None:
        T_effective = min(obs.shape[1], max_week)
        obs = obs[:, :T_effective]
    else:
        T_effective = obs.shape[1]
    
    # Train/test split
    T_train = int(T_effective * train_frac)
    T_test = T_effective - T_train
    
    obs_train = obs[:, :T_train]
    obs_test = obs[:, T_train:] if T_test > 0 else None
    
    # Masks
    mask_train = ~np.isnan(obs_train)
    mask_test = ~np.isnan(obs_test) if obs_test is not None else None
    
    # RFM features
    R_train, F_train, M_train = compute_rfm_features(obs_train, mask_train)
    
    # Standardize
    if R_train.std() > 0:
        R_train = (R_train - R_train.mean()) / (R_train.std() + 1e-6)
    if F_train.std() > 0:
        F_train = (F_train - F_train.mean()) / (F_train.std() + 1e-6)
    if M_train.std() > 0:
        M_train = np.log1p(M_train)
        M_train = (M_train - M_train.mean()) / (M_train.std() + 1e-6)
    
    data = {
        'N': N, 'T': T_train, 'y': obs_train.astype(np.float32),
        'mask': mask_train, 'R': R_train, 'F': F_train, 'M': M_train,
        'customer_id': idx, 'time': np.arange(T_train), 'T_full': T_effective,
        'T_test': T_test, 'world': 'uci',
    }
    
    if obs_test is not None:
        data['y_test'] = obs_test.astype(np.float32)
        data['mask_test'] = mask_test
        
        # OOS RFM
        R_test, F_test, M_test = compute_rfm_features_oos(obs_train, obs_test, mask_test)
        
        # Standardize using train stats
        if R_train.std() > 0:
            R_test = (R_test - R_train.mean()) / (R_train.std() + 1e-6)
        if F_train.std() > 0:
            F_test = (F_test - F_train.mean()) / (F_train.std() + 1e-6)
        if M_train.std() > 0:
            M_test = np.log1p(M_test)
            M_test = (M_test - M_train.mean()) / (M_train.std() + 1e-6)
        
        data['R_test'] = R_test
        data['F_test'] = F_test
        data['M_test'] = M_test
    
    print(f"  UCI: N={N}, T_train={T_train}, T_test={T_test}, zeros={np.mean(obs_train==0):.1%}")
    return data

## ----


def load_simulation_data(data_path, n_cust=None, seed=42, train_frac=0.8, max_week=None):
    """
    Load simulation data with train/test split.
    Default: 80% train, 20% test for OOS evaluation.
    
    Parameters:
    -----------
    max_week : int, optional
        If provided, truncate data to this many periods before splitting.
        Allows matching T across models (e.g., T=52 for all).
    """
    data_path = pathlib.Path(data_path)
    
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        # Detect column names
        id_col = 'customer_id' if 'customer_id' in df.columns else 'cust_id'
        time_col = 't' if 't' in df.columns else 'time_period'
        y_col = 'y' if 'y' in df.columns else 'observations'
        
        N_full = df[id_col].nunique()
        T_full = df[time_col].nunique()
        obs = df.pivot(index=id_col, columns=time_col, values=y_col).values
        source = 'csv'
        # Detect world from filename
        world = "unknown"
        for w in ['Harbor', 'Breeze', 'Fog', 'Cliff']:
            if w.lower() in str(data_path).lower():
                world = w
                break
    else:
        raise ValueError(f"Only CSV supported, got: {data_path.suffix}")
    


    # Subsample if requested
    if n_cust is not None and n_cust < N_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_full, n_cust, replace=False)
        obs = obs[idx, :]
        N = n_cust
    else:
        N = N_full
        idx = np.arange(N)
    
    # Truncate to max_week if specified
    if max_week is not None:
        T_effective = min(T_full, max_week)
        obs = obs[:, :T_effective]
        print(f"  Truncated to max_week={max_week} (from {T_full})")
    else:
        T_effective = T_full
    
    # Train/test split
    T_train = int(T_effective * train_frac)
    T_test = T_effective - T_train
    
    obs_train = obs[:, :T_train]
    obs_test = obs[:, T_train:] if T_test > 0 else None
    
    # Build masks
    mask_train = ~np.isnan(obs_train)
    mask_test = ~np.isnan(obs_test) if obs_test is not None else None
    
    # Compute RFM features
    R_train, F_train, M_train = compute_rfm_features(obs_train, mask_train)
    
    # Standardize features
    if R_train.std() > 0:
        R_train = (R_train - R_train.mean()) / (R_train.std() + 1e-6)
    if F_train.std() > 0:
        F_train = (F_train - F_train.mean()) / (F_train.std() + 1e-6)
    if M_train.std() > 0:
        M_train = np.log1p(M_train)
        M_train = (M_train - M_train.mean()) / (M_train.std() + 1e-6)

    data = {
        'N': N, 
        'T': T_train, 
        'y': obs_train.astype(np.float32),
        'mask': mask_train, 
        'R': R_train.astype(np.float32),
        'F': F_train.astype(np.float32), 
        'M': M_train.astype(np.float32),
        'customer_id': idx, 
        'time': np.arange(T_train),
        'T_full': T_effective,
        'T_test': T_test,
        'world': world,
    }
    
    if obs_test is not None:
        data['y_test'] = obs_test.astype(np.float32)
        data['mask_test'] = mask_test
        
        # Compute OOS RFM features
        R_test, F_test, M_test = compute_rfm_features_oos(obs_train, obs_test, mask_test)
        
        # Standardize using training stats
        if R_train.std() > 0:
            R_test = (R_test - R_train.mean()) / (R_train.std() + 1e-6)
        if F_train.std() > 0:
            F_test = (F_test - F_train.mean()) / (F_train.std() + 1e-6)
        if M_train.std() > 0:
            M_test = np.log1p(M_test)
            M_test = (M_test - M_train.mean()) / (M_train.std() + 1e-6)
        
        data['R_test'] = R_test.astype(np.float32)
        data['F_test'] = F_test.astype(np.float32)
        data['M_test'] = M_test.astype(np.float32)
    
    print(f"  Loaded: N={N}, T_train={T_train}, T_test={T_test}, "
          f"zeros={np.mean(obs_train==0):.1%} ({source})")
    
    return data

# =============================================================================
# 5. OOS PREDICTION
# =============================================================================

def compute_tweedie_clv(idata, data, discount_rate=0.10, ci_levels=[2.5, 97.5]):
    """
    Compute CLV from Tweedie HMM with empirical spend per state.
    Uses actual observed y values, not model-based psi.
    """
    try:
        post = idata.posterior
        
        # Get state assignments from filtered probabilities
        if 'alpha_filtered' in post:
            alpha = post['alpha_filtered'].mean(dim=['chain', 'draw']).values
            states = np.argmax(alpha, axis=-1)  # (N, T)
        else:
            print("    No alpha_filtered found")
            return None
        
        y = data['y']
        mask = data['mask']
        
        # Determine K correctly from posterior
        if 'beta0' in post:
            beta0_shape = post['beta0'].shape
            if len(beta0_shape) > 2:
                K = beta0_shape[-1]
            else:
                # Check if scalar or array
                beta0_vals = post['beta0'].values
                if np.isscalar(beta0_vals) or beta0_vals.ndim == 0:
                    K = 1
                else:
                    K = beta0_shape[-1] if len(beta0_shape) > 0 else 1
        else:
            K = 1
        
        # Flatten arrays for proper indexing
        states_flat = states.flatten()
        y_flat = y.flatten()
        mask_flat = mask.flatten()
        
        print(f"    DEBUG: K={K}, states shape={states.shape}, y shape={y.shape}")
        
        # Compute empirical spend per state
        empirical_spend = []
        for k in range(K):
            state_mask = (states_flat == k) & mask_flat
            n_obs = state_mask.sum()
            if n_obs > 0:
                spend_k = y_flat[state_mask]
                # Mean of positive spends only
                positive_spend = spend_k[spend_k > 0]
                if len(positive_spend) > 0:
                    mean_spend_k = float(np.mean(positive_spend))
                else:
                    mean_spend_k = 1.0  # Fallback if no positive spends
                print(f"    State {k}: n_obs={n_obs}, positive={len(positive_spend)}, mean=${mean_spend_k:.2f}")
            else:
                mean_spend_k = 1.0
                print(f"    State {k}: NO OBSERVATIONS")
            empirical_spend.append(mean_spend_k)
        
        empirical_spend = np.array(empirical_spend)
        print(f"    Empirical spend by state: {empirical_spend}")
        
        # Churn rate from Gamma diagonal
        if K > 1 and 'Gamma' in post:
            Gamma = post['Gamma'].mean(dim=['chain', 'draw']).values
            churn_rate = 1.0 - np.diag(Gamma)
        else:
            churn_rate = np.zeros(K)
        
        print(f"    Churn rate: {churn_rate}")
        
        # CLV formula: empirical_spend / (discount + churn)
        delta = discount_rate
        clv_k = empirical_spend / (delta + churn_rate + 1e-10)
        
        print(f"    Raw CLV: {clv_k}")
        
        # Soft floor at $0.50 (only if very small)
        clv_k = np.maximum(clv_k, 0.50)
        
        # Sort by CLV magnitude
        order = np.argsort(clv_k)
        clv_sorted = clv_k[order]
        
        result = {
            'clv_by_state_sorted': clv_sorted.tolist(),
            'state_labels': ['Dormant', 'Regular', 'Whale'] if K >= 3 else ['Low', 'High'],
            'clv_total': float(np.sum(clv_k)),
            'clv_ratio': float(np.max(clv_k) / (np.min(clv_k) + 1e-6)),
            'discount_rate': discount_rate,
            'order_indices': order.tolist(),
            'empirical_spend': empirical_spend[order].tolist()
        }
        
        # Posterior CI via sampling
        if 'draw' in post.dims and post.draw.size > 1:
            n_chains, n_draws = post.sizes['chain'], post.sizes['draw']
            clv_draws = []
            
            for c in range(n_chains):
                for d in range(n_draws):
                    # State assignments for this draw
                    if 'alpha_filtered' in post:
                        alpha_d = post['alpha_filtered'].isel(chain=c, draw=d).values
                        states_d = np.argmax(alpha_d, axis=-1).flatten()
                    else:
                        states_d = states_flat
                    
                    # Empirical spend for this draw's state assignments
                    spend_d = []
                    for k in range(K):
                        state_mask_d = (states_d == k) & mask_flat
                        if state_mask_d.sum() > 0:
                            spend_k_d = y_flat[state_mask_d]
                            positive_spend_d = spend_k_d[spend_k_d > 0]
                            if len(positive_spend_d) > 0:
                                spend_d.append(np.mean(positive_spend_d))
                            else:
                                spend_d.append(empirical_spend[k])
                        else:
                            spend_d.append(empirical_spend[k])
                    
                    spend_d = np.array(spend_d)
                    
                    # Gamma and churn
                    if K > 1 and 'Gamma' in post:
                        Gamma_d = post['Gamma'].isel(chain=c, draw=d).values
                        churn_d = 1.0 - np.diag(Gamma_d)
                    else:
                        churn_d = np.zeros(K)
                    
                    clv_d = spend_d / (delta + churn_d + 1e-10)
                    clv_d = np.maximum(clv_d, 0.50)
                    clv_draws.append(clv_d)
            
            clv_draws = np.array(clv_draws)
            ci_low = np.percentile(clv_draws, ci_levels[0], axis=0)
            ci_high = np.percentile(clv_draws, ci_levels[1], axis=0)
            
            result['clv_ci_low'] = ci_low[order].tolist()
            result['clv_ci_high'] = ci_high[order].tolist()
        
        return result
        
    except Exception as e:
        print(f"    CLV computation error: {e}")
        import traceback
        traceback.print_exc()
        return None

## ----

def compute_oos_prediction(data, idata, use_gam, gam_df, n_draws_use=200):
    """
    Compute OOS predictions with proper HMM state propagation.
    Uses filtered state probabilities transitioned via Gamma.
    """
    try:
        N, T_test = data['y_test'].shape
        y_test = data['y_test']
        R_test, F_test, M_test = data['R_test'], data['F_test'], data['M_test']
        
        post = idata.posterior
        n_chains, n_draws_total = post.sizes['chain'], post.sizes['draw']
        
        # Determine K
        if 'Gamma' in post:
            K = post['Gamma'].shape[-1]
        elif 'beta0' in post:
            K = post['beta0'].shape[-1] if len(post['beta0'].shape) > 2 else 1
        else:
            K = 1
        
        # Precompute GAM bases once (Grok's optimization)
        if use_gam:
            basis_R = create_bspline_basis(R_test.flatten(), df=gam_df).reshape(N, T_test, -1)
            basis_F = create_bspline_basis(F_test.flatten(), df=gam_df).reshape(N, T_test, -1)
            basis_M = create_bspline_basis(M_test.flatten(), df=gam_df).reshape(N, T_test, -1)
        
        # Sample draws
        draw_idx = np.random.choice(n_chains * n_draws_total, min(n_draws_use, n_chains * n_draws_total), replace=False)
        
        y_pred_samples = []
        
        for idx in draw_idx:
            c = idx // n_draws_total
            d = idx % n_draws_total
            
            # Use xarray isel (Grok's suggestion)
            beta0_draw = post['beta0'].isel(chain=c, draw=d).values
            
            if use_gam:
                w_R_draw = post['w_R'].isel(chain=c, draw=d).values
                w_F_draw = post['w_F'].isel(chain=c, draw=d).values
                w_M_draw = post['w_M'].isel(chain=c, draw=d).values
                
                if K == 1:
                    eff_R = np.tensordot(basis_R, w_R_draw, axes=([2], [0]))
                    eff_F = np.tensordot(basis_F, w_F_draw, axes=([2], [0]))
                    eff_M = np.tensordot(basis_M, w_M_draw, axes=([2], [0]))
                    mu_draw = np.exp(beta0_draw + eff_R + eff_F + eff_M)
                    y_pred_d = mu_draw  # No state mixing for K=1
                else:
                    eff_R = np.tensordot(basis_R, w_R_draw, axes=([2], [1]))
                    eff_F = np.tensordot(basis_F, w_F_draw, axes=([2], [1]))
                    eff_M = np.tensordot(basis_M, w_M_draw, axes=([2], [1]))
                    mu_draw = np.exp(beta0_draw[None, None, :] + eff_R + eff_F + eff_M)
                    
                    # HMM state propagation (Gemini's approach)
                    # Get initial state prob from last filtered state
                    if 'alpha_filtered' in post:
                        last_alpha = post['alpha_filtered'].isel(chain=c, draw=d).values[:, -1, :]  # (N, K)
                        state_prob = last_alpha
                    else:
                        state_prob = np.ones((N, K)) / K
                    
                    Gamma_draw = post['Gamma'].isel(chain=c, draw=d).values  # (K, K)
                    
                    # Propagate and predict
                    y_pred_d = np.zeros((N, T_test))
                    for t in range(T_test):
                        state_prob = state_prob @ Gamma_draw  # (N, K)
                        y_pred_d[:, t] = np.sum(state_prob * mu_draw[:, t, :], axis=1)
            else:
                # Linear/Intercept only
                if K == 1:
                    mu_draw = np.exp(beta0_draw)
                    y_pred_d = np.full((N, T_test), mu_draw)
                else:
                    # Would need betaR, betaF, betaM here - add if needed
                    mu_draw = np.exp(beta0_draw[None, None, :])  # Broadcast
                    
                    if 'alpha_filtered' in post:
                        last_alpha = post['alpha_filtered'].isel(chain=c, draw=d).values[:, -1, :]
                        state_prob = last_alpha
                    else:
                        state_prob = np.ones((N, K)) / K
                    
                    Gamma_draw = post['Gamma'].isel(chain=c, draw=d).values
                    
                    y_pred_d = np.zeros((N, T_test))
                    for t in range(T_test):
                        state_prob = state_prob @ Gamma_draw
                        # Assume same mu across time for intercept-only
                        y_pred_d[:, t] = np.sum(state_prob * mu_draw[0, 0, :], axis=1)
            
            y_pred_samples.append(y_pred_d)
        
        # Aggregate predictions
        y_pred_mean = np.mean(y_pred_samples, axis=0)
        
        # Compute metrics
        mask = ~np.isnan(y_test)
        if mask.sum() == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'y_pred': None}
        
        residuals = y_test[mask] - y_pred_mean[mask]
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # Log predictive score (approximate)
        # For Tweedie: log p(y|mu,p,phi) - would need p and phi per draw
        # Simplified version: log of predicted mean (crude)
        lps = np.mean(np.log(y_pred_mean[mask] + 1e-10))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'log_pred_score': float(lps),
            'y_pred': y_pred_mean,
            'n_draws': len(draw_idx)
        }
        
    except Exception as e:
        print(f"  OOS prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {'rmse': np.nan, 'mae': np.nan, 'y_pred': None}

## ----


# =============================================================================
# 6. SMC RUNNER
# =============================================================================

def run_smc(data, K, state_specific_p, p_fixed, use_gam, gam_df, draws, chains, seed, out_dir, use_covariates=True, shared_phi=False):
    """Run SMC with Tweedie model - FIXED ESS MEMORY ISSUE."""
    cores = min(chains, 4)
    t0 = time.time()

    # Initialize ALL metrics
    log_ev = np.nan
    oos_rmse_val = np.nan
    oos_mae_val = np.nan
    clv_by_state_list = []
    clv_total_val = np.nan
    clv_ratio_val = np.nan
    clv_ci_low_list = []
    clv_ci_high_list = []
    ess_min_val = np.nan

    try:
        with make_model(data, K, state_specific_p, p_fixed, use_gam, gam_df, use_covariates, shared_phi) as model:
            print(f"\nModel: K={K}, TWEEDIE, p={'state-varying' if state_specific_p else p_fixed}")
            print(f"SMC: draws={draws}, chains={chains}, cores={cores}")

            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                return_inferencedata=True
            )

            elapsed = (time.time() - t0) / 60

            # Log-evidence extraction
            try:
                lm = idata.sample_stats.log_marginal_likelihood.values
                if hasattr(lm, 'dtype') and lm.dtype == object:
                    chain_finals = []
                    for c in range(lm.shape[1] if lm.ndim > 1 else 1):
                        chain_slice = lm[-1, c] if lm.ndim > 1 else lm[c]
                        if isinstance(chain_slice, (list, np.ndarray)):
                            valid = [float(x) for x in chain_slice if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                            if valid:
                                chain_finals.append(valid[-1])
                        elif np.isfinite(float(chain_slice)):
                            chain_finals.append(float(chain_slice))
                    log_ev = float(np.mean(chain_finals)) if chain_finals else np.nan
                else:
                    flat = np.array(lm).flatten()
                    valid = flat[np.isfinite(flat)]
                    log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
                print(f"  EXTRACTED log_ev: {log_ev:.2f}")
            except Exception as e:
                print(f"  LOG-EV ERROR: {e}")
                log_ev = np.nan

            # OOS prediction
            if 'y_test' in data:
                print("  Computing OOS predictions...")
                try:
                    oos_results = compute_oos_prediction(data, idata, use_gam, gam_df)
                    if oos_results and isinstance(oos_results, dict):
                        oos_rmse_val = float(oos_results.get('rmse', np.nan))
                        oos_mae_val = float(oos_results.get('mae', np.nan))
                        print(f"  EXTRACTED OOS: RMSE={oos_rmse_val:.4f}, MAE={oos_mae_val:.4f}")
                except Exception as e:
                    print(f"  OOS ERROR: {e}")

            # CLV computation
            print("  Computing CLV...")
            try:
                clv_results = compute_tweedie_clv(idata, data, discount_rate=0.10)
                if clv_results and isinstance(clv_results, dict):
                    clv_arr = clv_results.get('clv_by_state_sorted', [])
                    clv_by_state_list = [float(x) for x in np.atleast_1d(clv_arr)]
                    clv_total_val = float(clv_results.get('clv_total', np.nan))
                    clv_ratio_val = float(clv_results.get('clv_ratio', np.nan))
                    clv_ci_low_list = np.atleast_1d(clv_results.get('clv_ci_low', [])).tolist()
                    clv_ci_high_list = np.atleast_1d(clv_results.get('clv_ci_high', [])).tolist()
                    print(f"  EXTRACTED CLV: {clv_by_state_list}")
            except Exception as e:
                print(f"  CLV ERROR: {e}")

            # ESS - SKIP TO AVOID MEMORY SPIKE
            print("  Skipping ESS (memory intensive)...")
            ess_min_val = np.nan

            # Build res
            res = {
                'K': K,
                'model_type': 'TWEEDIE',
                'world': data.get('world', 'unknown'),
                'N': data['N'],
                'T': data['T'],
                'log_evidence': log_ev,
                'draws': draws,
                'chains': chains,
                'time_min': elapsed,
                'oos_rmse': oos_rmse_val,
                'oos_mae': oos_mae_val,
                'clv_by_state': clv_by_state_list,
                'clv_total': clv_total_val,
                'clv_ratio': clv_ratio_val,
                'clv_ci_low': clv_ci_low_list,
                'clv_ci_high': clv_ci_high_list,
                'ess_min': ess_min_val,
                'p_fixed': None if state_specific_p else p_fixed,
                'state_specific_p': state_specific_p,
                'use_gam': use_gam,
                'use_covariates': use_covariates,
                'shared_phi': shared_phi
            }

            # SAVE
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            gam_str = 'GAM' if use_gam else 'GLM'
            p_str = 'statep' if state_specific_p else f'p{p_fixed}'
            pkl_name = f"smc_K{K}_TWEEDIE_{gam_str}_{p_str}_N{data['N']}_T{data['T']}_D{draws}.pkl"
            pkl_path = out_dir / pkl_name

            data_light = {
                'N': data.get('N'),
                'T': data.get('T'),
                'T_full': data.get('T_full'),
                'T_test': data.get('T_test'),
                'world': data.get('world'),
                'customer_id': data.get('customer_id'),
                'train_stats': data.get('train_stats')
            }

            print(f"  Saving...")
            with open(pkl_path, 'wb') as f:
                pickle.dump({'idata': idata, 'res': res, 'data': data_light}, f, protocol=4)

            print(f"\n  SAVED: {pkl_path}")
            return pkl_path, res

    except Exception as e:
        print(f"\n  CRASH: {str(e)[:80]}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}

# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='HMM-Tweedie Model with OOS Evaluation')
    parser.add_argument('--dataset', required=True, choices=['simulation', 'uci', 'cdnow'])
    parser.add_argument('--sim_path', type=str, required=True,
                       help='Path to simulation CSV file')
    parser.add_argument('--n_cust', type=int, default=None)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--state_specific_p', action='store_true')
    parser.add_argument('--p_fixed', type=float, default=None)
    parser.add_argument('--no_gam', action='store_true')
    parser.add_argument('--gam_df', type=int, default=3)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--train_frac', type=float, default=0.8,
                       help='Fraction for training (default 0.8 = 80%% train, 20%% test)')
    parser.add_argument('--max_week', type=int, default=None,
                       help='Maximum week to use (truncate T to this value before train/test split)')
    parser.add_argument('--no_covariates', action='store_true',
                       help='Disable RFM covariates (intercept-only)')
    parser.add_argument('--shared_phi', action='store_true',
                       help='Use single phi shared across all states (default: state-specific, recommended with state_specific_p)')
    
    args = parser.parse_args()

    if args.gam_df is None:
        args.gam_df = 2 if args.K > 2 else 3

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"HMM-Tweedie: {args.dataset.upper()} | K={args.K} | "
          f"{'state-p' if args.state_specific_p else f'p={args.p_fixed}'}")
    print(f"Train/Test: {args.train_frac:.0%}/{1-args.train_frac:.0%} | "
          f"Covariates: {'no' if args.no_covariates else 'yes'}")
    print(f"{'='*70}")

    if args.dataset in ['uci', 'cdnow']:
        data = load_uci_data(args.sim_path, n_cust=args.n_cust,
                            seed=args.seed, train_frac=args.train_frac,
                            max_week=args.max_week)
    else:
        data = load_simulation_data(args.sim_path, n_cust=args.n_cust,
                                    seed=args.seed, train_frac=args.train_frac,
                                    max_week=args.max_week)

    print(f"\nRunning SMC...")
    pkl_path, res = run_smc(data, args.K, args.state_specific_p, args.p_fixed,
                           not args.no_gam, args.gam_df, args.draws, args.chains,
                           args.seed, out_dir, use_covariates=not args.no_covariates,
                           shared_phi=args.shared_phi)

    print(f"DEBUG: state_specific_p={args.state_specific_p}, p_fixed={args.p_fixed}")

    print(f"\n{'='*70}")
    print("RESULTS")
    for key, val in res.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}" if abs(val) < 1000 else f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
