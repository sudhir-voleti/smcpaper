#!/usr/bin/env python3
"""
smc_hmm_tweedie.py
==================
HMM-Tweedie model with saddlepoint approximation and OOS evaluation.
Optimized for Apple Silicon (M1/M2/M3)
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
def make_model(data, K=3, state_specific_p=True, p_fixed=None, use_gam=True, gam_df=3, use_covariates=True):

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
            phi = pm.Exponential("phi", lam=0.5)
        else:
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0_sorted = pt.sort(beta0_raw)
            beta0 = pm.Deterministic("beta0", beta0_sorted * beta0_prior_sd + beta0_prior_mean)
            phi = pm.Exponential("phi", lam=0.5, shape=K)

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
        if p_fixed is not None:
            # Fixed p across all states
            p = pt.as_tensor_variable(np.array([p_fixed] * K, dtype=np.float32))
        elif state_specific_p:
            # State-varying p with tighter bounds [1.3, 1.7]
            p_raw = pm.Beta("p_raw", alpha=2, beta=2, shape=K)
            p_sorted = pt.sort(p_raw)
            p = pm.Deterministic("p", 1.3 + p_sorted * 0.4)  # [1.3, 1.7]
        else:
            # Shared p across states (estimated)
            p_raw = pm.Beta("p_raw", alpha=2, beta=2)
            p = pm.Deterministic("p", 1.3 + p_raw * 0.4)  # [1.3, 1.7]
            if K > 1:
                p = pt.stack([p] * K)

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
def load_simulation_data(data_path, n_cust=None, seed=42, train_frac=0.8):
    """
    Load simulation data with train/test split.
    Default: 80% train, 20% test for OOS evaluation.
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
        T = df[time_col].nunique()
        obs = df.pivot(index=id_col, columns=time_col, values=y_col).values
        source = 'csv'
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
    
    # Train/test split
    T_train = int(T * train_frac)
    T_test = T - T_train
    
    obs_train = obs[:, :T_train]
    obs_test = obs[:, T_train:] if T_test > 0 else None
    
    # Build masks
    mask_train = ~np.isnan(obs_train)
    mask_test = ~np.isnan(obs_test) if obs_test is not None else None
    
    # Compute RFM features
    R_train, F_train, M_train = compute_rfm_features(obs_train, mask_train)
    
    # Standardize features (important for numerical stability)
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
        'T_full': T,
        'T_test': T_test,
    }
    
    if obs_test is not None:
        data['y_test'] = obs_test.astype(np.float32)
        data['mask_test'] = mask_test
        
        # Compute OOS RFM features (propagating from training)
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
    print(f"  RFM stats - R: [{R_train.min():.2f}, {R_train.max():.2f}], "
          f"F: [{F_train.min():.2f}, {F_train.max():.2f}], "
          f"M: [{M_train.min():.2f}, {M_train.max():.2f}]")
    
    return data


# =============================================================================
# 5. OOS PREDICTION
# =============================================================================

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


# =============================================================================
# 6. SMC RUNNER
# =============================================================================
def run_smc(
    data,
    K,
    state_specific_p,
    p_fixed,
    use_gam,
    gam_df,
    draws,
    chains,
    seed,
    out_dir,
    use_covariates=True
):
    """Run SMC with Tweedie model and OOS evaluation."""
    
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()
    
    try:
        with make_model(
            data,
            K=K,
            state_specific_p=state_specific_p,
            p_fixed=p_fixed,
            use_gam=use_gam,
            gam_df=gam_df,
            use_covariates=use_covariates
        ) as model:
            
            print(
                f" Model: K={K}, Tweedie-{'GAM' if use_gam else 'GLM'}, "
                f"p={'state-specific' if state_specific_p else p_fixed}, "
                f"covariates={'yes' if use_covariates else 'no'}"
            )


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
                        valid = [
                            float(x) for x in chain_list
                            if isinstance(x, (int, float, np.floating)) and np.isfinite(x)
                        ]
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
            print(f" Warning: log-ev extraction failed: {e}")
        
        if log_ev > 0:
            print(" WARNING: Positive Log-Ev detected. Numerical instability likely.")
        
        elapsed = (time.time() - t0) / 60
        
        # Compute ESS and R-hat
        ess_min = np.nan
        rhat_max = np.nan
        try:
            ess = az.ess(idata)
            rhat = az.rhat(idata)
            ess_min = float(ess.to_array().min())
            rhat_max = float(rhat.to_array().max())
        except:
            pass
        
        res = {
            'K': K,
            'model_type': 'tweedie',
            'N': data['N'],
            'T': data['T'],
            'T_test': data.get('T_test', 0),
            'use_gam': use_gam,
            'gam_df': gam_df if use_gam else None,
            'use_covariates': use_covariates,
            'state_specific_p': state_specific_p,
            'p_fixed': p_fixed,
            'log_evidence': log_ev,
            'draws': draws,
            'chains': chains,
            'time_min': elapsed,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'ess_min': ess_min,
            'rhat_max': rhat_max,
        }
        
        # OOS prediction
        if 'y_test' in data:
            print(" Computing OOS predictions...")
            oos_results = compute_oos_prediction(data, idata, use_gam, gam_df)
            res['oos_rmse'] = oos_results.get('rmse', np.nan)
            res['oos_mae'] = oos_results.get('mae', np.nan)
            res['oos_log_pred'] = oos_results.get('log_pred_score', np.nan)
            res['y_pred_mean'] = oos_results.get('y_pred', None)  # Save actual predictions
            print(f" OOS RMSE: {res['oos_rmse']:.4f}, MAE: {res['oos_mae']:.4f}")
        
        # Compute WAIC and LOO for model comparison
        try:
            waic = az.waic(idata)
            loo = az.loo(idata)
            res['waic'] = float(waic.waic)
            res['waic_se'] = float(waic.waic_se)
            res['loo'] = float(loo.loo)
            res['loo_se'] = float(loo.loo_se)
            print(f" WAIC: {res['waic']:.2f}, LOO: {res['loo']:.2f}")
        except Exception as e:
            print(f" WAIC/LOO computation failed: {e}")
            res['waic'] = np.nan
            res['loo'] = np.nan
        
        p_tag = "statep" if state_specific_p else f"p{p_fixed}"
        pkl_path = out_dir / f"smc_K{K}_TWEEDIE_{'GAM' if use_gam else 'GLM'}_{p_tag}_N{data['N']}_T{data['T']}_D{draws}.pkl"
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)
        
        print(
            f" log_ev={log_ev:.2f}, ess_min={ess_min:.1f}, "
            f"rhat_max={rhat_max:.3f}, time={elapsed:.1f}min"
        )
        print(f" Saved: {pkl_path}")
        
        return pkl_path, res
   
    except Exception as e:
        print(f" CRASH: {str(e)[:60]}")
        import traceback
        traceback.print_exc()
        raise

# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='HMM-Tweedie Model with OOS Evaluation')
    parser.add_argument('--dataset', required=True, choices=['simulation'])
    parser.add_argument('--sim_path', type=str, required=True,
                       help='Path to simulation CSV file')
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
    parser.add_argument('--train_frac', type=float, default=0.8,
                       help='Fraction for training (default 0.8 = 80%% train, 20%% test)')
    parser.add_argument('--no_covariates', action='store_true',
                       help='Disable RFM covariates (intercept-only)')
    
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

    data = load_simulation_data(args.sim_path, n_cust=args.n_cust, 
                                seed=args.seed, train_frac=args.train_frac)

    print(f"\nRunning SMC...")
    pkl_path, res = run_smc(data, args.K, args.state_specific_p, args.p_fixed,
                           not args.no_gam, args.gam_df, args.draws, args.chains,
                           args.seed, out_dir, use_covariates=not args.no_covariates)

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
