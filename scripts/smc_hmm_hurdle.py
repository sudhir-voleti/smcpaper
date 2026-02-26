#!/usr/bin/env python3
"""
smc_hmm_hurdle_final.py - Production HMM-Hurdle-Gamma with Batched Forward Algorithm
For Marketing Science: "SMC Enables Scalable HMM Estimation in High-Sparsity Data"

Key Computational Claim: Vectorized scan provides 10x+ speedup over Python-loop,
enabling HMM inference at N=500+ where HMC/NUTS fails to compile/converge.

Final Council-Reviewed Version:
- Grok-corrected forward algorithm with proper log-space normalization
- Alpha > 0.1 constraint for heavy-tail stability (Cliff world)
- Ordered mu constraint for identifiability
- Diagnostic extraction (ESS, R-hat, SMC stats)
- Balanced panel handling (padding to max_T)

Usage:
    # Debug validation (N=50)
    python smc_hmm_hurdle_final.py --world Cliff --K 3 --N 50 --draws 200 --debug

    # Production runs (N=200, paper tables)
    python smc_hmm_hurdle_final.py --world Harbor --K 2 --N 200 --T 104
    python smc_hmm_hurdle_final.py --world Cliff --K 3 --N 200 --T 104
    python smc_hmm_hurdle_final.py --world Cliff --K 4 --N 200 --T 104  # Overfit demo

    # Scale validation (N=500, computational claim)
    python smc_hmm_hurdle_final.py --world Cliff --K 3 --N 500 --T 104 --draws 1000
"""

# =============================================================================
# 0. ENVIRONMENT & OPTIMIZATION
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
from patsy import dmatrix

warnings.filterwarnings('ignore')
RANDOM_SEED = 42

# =============================================================================
# 2. DATA LOADING & RFM FEATURES (Balanced Panels)
# =============================================================================

def compute_rfm_features(y, mask):
    """
    Compute Recency, Frequency, Monetary features.

    Parameters:
    -----------
    y : (N, T) - spend amounts (0 for no purchase)
    mask : (N, T) - True for valid observations

    Returns:
    --------
    R, F, M : (N, T) - RFM features (0 where mask=False)
    """
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
                    R[i, t] = t + 1  # No purchase yet
                    F[i, t] = 0
                    M[i, t] = 0.0
            # Else: leave as 0 (masked)

    return R, F, M


def load_simulation_data(world: str, data_dir: Path, T: int = 104, N: int = None):
    """
    Load simulation CSV with true states. Assumes balanced panel (all customers have T periods).

    Parameters:
    -----------
    world : str - Simulation world name
    data_dir : Path - Directory containing CSV files
    T : int - Expected time periods (will pad/truncate to this)
    N : int - Number of customers to subsample (None = all)

    Returns:
    --------
    data : dict with keys 'N', 'T', 'y', 'mask', 'R', 'F', 'M', 'true_states', etc.
    """
    # Try multiple filename patterns
    csv_path = data_dir / f"hmm_{world}_N{N or 'all'}_T{T}.csv"
    if not csv_path.exists():
        csv_path = data_dir / f"hmm_{world}_T{T}.csv"
    if not csv_path.exists():
        csv_path = data_dir / f"{world.lower()}_N{N or 200}_T{T}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Simulation data not found. Tried: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    required = ['customer_id', 't', 'y', 'true_state']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {df.columns.tolist()}")

    # Reshape to balanced panel
    N_actual = df['customer_id'].nunique()
    T_actual = df['t'].nunique()

    y_full = df.pivot(index='customer_id', columns='t', values='y').values
    true_states_full = df.pivot(index='customer_id', columns='t', values='true_state').values

    # Pad or truncate to target T
    if T_actual < T:
        # Pad with zeros (masked)
        pad_width = ((0, 0), (0, T - T_actual))
        y_full = np.pad(y_full, pad_width, mode='constant', constant_values=0)
        true_states_full = np.pad(true_states_full, pad_width, mode='constant', constant_values=-1)
        T_effective = T
    elif T_actual > T:
        # Truncate
        y_full = y_full[:, :T]
        true_states_full = true_states_full[:, :T]
        T_effective = T
    else:
        T_effective = T_actual

    # Subsample customers if requested
    if N is not None and N < N_actual:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(N_actual, N, replace=False)
        y_full = y_full[idx, :]
        true_states_full = true_states_full[idx, :]
        N_effective = N
    else:
        N_effective = N_actual

    # Create mask: valid where we have observations (not padded)
    mask = (true_states_full >= 0) & (~np.isnan(y_full))

    # Ensure y is 0 where masked (for numerical safety)
    y_full = np.where(mask, y_full, 0.0)

    # Compute RFM
    R, F, M = compute_rfm_features(y_full, mask)

    # Standardize for numerical stability
    # Only use valid observations for standardization
    M_valid = M[mask]
    if len(M_valid) > 0 and M_valid.std() > 0:
        M_log = np.log1p(M)
        M_valid_log = M_log[mask]
        M_scaled = (M_log - M_valid_log.mean()) / (M_valid_log.std() + 1e-6)
    else:
        M_scaled = M

    R_valid = R[mask]
    F_valid = F[mask]
    if len(R_valid) > 0 and R_valid.std() > 0:
        R = (R - R_valid.mean()) / (R_valid.std() + 1e-6)
    if len(F_valid) > 0 and F_valid.std() > 0:
        F = (F - F_valid.mean()) / (F_valid.std() + 1e-6)

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
        'M_raw': M.astype(np.float32)
    }

    # Summary statistics
    zero_rate = np.mean(y_full[mask] == 0) if mask.any() else 0.0
    mean_spend = np.mean(y_full[mask][y_full[mask] > 0]) if (y_full[mask] > 0).any() else 0.0

    print(f"  Loaded: N={N_effective}, T={T_effective}, zeros={zero_rate:.1%}, mean_spend=${mean_spend:.2f}")

    return data


# =============================================================================
# 3. GAM BASIS
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
# 4. VECTORIZED FORWARD ALGORITHM (Grok-Corrected with Proper Scaling)
# =============================================================================

def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """
    Batched forward algorithm using pytensor.scan with proper log-space normalization.

    This is the corrected version that properly accumulates normalization constants
    for accurate marginal likelihood computation.

    Parameters:
    -----------
    log_emission : (N, T, K)
        Log emission probabilities (already masked, 0 where invalid)
    log_Gamma : (K, K)
        Log transition matrix
    pi0 : (K,)
        Initial state distribution

    Returns:
    --------
    log_marginal : (N,)
        Log marginal likelihood p(y_{1:T} | params) for each customer
    log_alpha_norm : (T, N, K)
        Normalized filtered log-probabilities log p(s_t | y_{1:t}) for Viterbi
    """
    N, T, K = log_emission.shape

    # Initial step (t=0)
    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]  # (N, K)
    log_Z_init = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)  # (N, 1)
    log_alpha_norm_init = log_alpha_init - log_Z_init  # Normalized

    # Scan step: forward propagation with normalization
    def forward_step(log_emit_t, log_alpha_prev, log_Gamma):
        """
        Single forward step.

        log_emit_t: (N, K) - emissions at time t
        log_alpha_prev: (N, K) - normalized log-alpha from t-1
        log_Gamma: (K, K) - transition matrix
        """
        # Transition: (N, K, 1) + (1, K, K) -> (N, K, K)
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]

        # New alpha: emission + sum over previous states
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)  # (N, K)

        # Normalization constant
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)  # (N, 1)

        # Normalized alpha (for numerical stability)
        log_alpha_norm = log_alpha_new - log_Z_t  # (N, K)

        return log_alpha_norm, log_Z_t

    # Prepare emission sequence: swap time to first dimension
    # log_emission[:, 1:, :] is (N, T-1, K), swapaxes -> (T-1, N, K)
    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)

    # Run scan
    (log_alpha_norm_seq, log_Z_seq), _ = scan(
        fn=forward_step,
        sequences=log_emit_seq,
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=log_Gamma,
        strict=True
    )

    # log_alpha_norm_seq: (T-1, N, K)
    # log_Z_seq: (T-1, N, 1)

    # Full normalized alpha sequence (prepend t=0)
    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],  # (1, N, K)
        log_alpha_norm_seq  # (T-1, N, K)
    ], axis=0)  # (T, N, K)

    # Marginal likelihood: sum of all log normalization constants
    # log_Z_init: (N, 1), log_Z_seq: (T-1, N, 1)
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)  # (N,)

    return log_marginal, log_alpha_norm_full


# =============================================================================
# 5. HMM MODEL (Production)
# =============================================================================

def make_hurdle_hmm(data, K, use_gam=True, gam_df=3, debug=False):
    """
    HMM-Hurdle-Gamma with batched forward algorithm.

    Features:
    - Vectorized scan for 10x+ speedup claim
    - Alpha > 0.1 constraint for heavy-tail stability
    - Ordered mu constraint for identifiability
    - Proper masking for balanced panels
    """
    y = data['y']
    R, F, M = data['R'], data['F'], data['M']
    mask = data['mask']
    N, T = data['N'], data['T']

    if debug:
        print(f"  [DEBUG] Model: N={N}, T={T}, K={K}, GAM={use_gam}")

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

        # ---- 1. LATENT DYNAMICS ----
        if K == 1:
            # Static model (no HMM)
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype=np.float32))
        else:
            # Sticky transition prior
            Gamma = pm.Dirichlet("Gamma", a=np.eye(K)*5 + 1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            log_Gamma = pt.log(Gamma)

        # ---- 2. HURDLE PARAMETERS (pi = P(y>0 | state)) ----
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

        # ---- 3. GAMMA PARAMETERS (with stability constraints) ----
        if K == 1:
            beta0 = pm.Normal("beta0", 0, 1)

            # Alpha > 0.1 constraint: prevents gammaln explosion in Cliff world
            alpha_raw = pm.Beta("alpha_raw", alpha=2, beta=2)
            alpha_gamma = pm.Deterministic("alpha_gamma", 0.1 + alpha_raw * 2.0)
        else:
            # Ordered intercepts: ensures mu_1 < mu_2 < ... < mu_K
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0 = pm.Deterministic("beta0", pt.sort(beta0_raw))

            # Alpha > 0.1 for all states
            alpha_raw = pm.Beta("alpha_raw", alpha=2, beta=2, shape=K)
            alpha_gamma = pm.Deterministic("alpha_gamma", 0.1 + alpha_raw * 2.0)

        # Covariate effects for mean spend
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

        # Gamma rate: beta = alpha / mu
        if K == 1:
            beta_gamma = pm.Deterministic("beta_gamma", alpha_gamma / mu)
        else:
            beta_gamma = pm.Deterministic("beta_gamma", alpha_gamma[None, None, :] / mu)

        # ---- 4. EMISSION LOG-PROBABILITIES ----
        if K == 1:
            # Static model: no forward algorithm needed
            log_zero = pt.log(1 - pi)
            y_clipped = pt.clip(y, 1e-10, 1e10)
            log_gamma = ((alpha_gamma - 1) * pt.log(y_clipped) - 
                        beta_gamma * y + 
                        alpha_gamma * pt.log(beta_gamma) - 
                        pt.gammaln(alpha_gamma))
            log_pos = pt.log(pi) + log_gamma
            log_emission = pt.where(y == 0, log_zero, log_pos)

            # Apply mask: zero contribution where masked (balanced panel)
            log_emission = pt.where(mask, log_emission, 0.0)

            logp_cust = pt.sum(log_emission, axis=1)

        else:
            # HMM: Prepare emissions for batched forward algorithm
            # Expand to (N, T, K)
            pi_exp = pi[..., None] if pi.ndim == 2 else pi
            alpha_exp = alpha_gamma[None, None, :] if alpha_gamma.ndim == 1 else alpha_gamma
            beta_exp = beta_gamma

            y_exp = y[..., None]
            mask_exp = mask[..., None]

            # Hurdle log-prob
            log_zero = pt.log(1 - pi_exp)
            y_clipped = pt.clip(y_exp, 1e-10, 1e10)
            log_gamma = ((alpha_exp - 1) * pt.log(y_clipped) - 
                        beta_exp * y_exp + 
                        alpha_exp * pt.log(beta_exp) - 
                        pt.gammaln(alpha_exp))
            log_pos = pt.log(pi_exp) + log_gamma

            log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)

            # Apply mask: zero contribution where masked
            log_emission = pt.where(mask_exp, log_emission, 0.0)

            # ---- 5. BATCHED FORWARD ALGORITHM ----
            if debug:
                print(f"  [DEBUG] Running batched forward algorithm (scan)")

            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)

            # Store filtered probabilities for post-proc Viterbi
            # log_alpha_norm is (T, N, K), transpose to (N, T, K) for consistency
            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered,
                           dims=("customer", "time", "state"))

        # ---- 6. LIKELIHOOD ----
        pm.Potential("loglike", pt.sum(logp_cust))
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))

        return model


# =============================================================================
# 6. SMC RUNNER WITH DIAGNOSTICS
# =============================================================================

def run_smc_hmm(data, K, use_gam, gam_df, draws, chains, seed, out_dir, debug=False):
    """
    Run SMC with comprehensive diagnostics.

    Returns diagnostic metrics to prove "SMC succeeds where HMC fails":
    - ESS (effective sample size) - should be high for SMC
    - R-hat - should be ~1.0 for convergence
    - Runtime scaling - demonstrates efficiency claim
    """
    cores = min(chains, 4)
    t0 = time.time()

    try:
        with make_hurdle_hmm(data, K, use_gam, gam_df, debug) as model:
            model_type = "HURDLE"
            glm_gam = "GAM" if use_gam else "GLM"
            print(f"\nModel: K={K}, {model_type}-{glm_gam}, world={data['world']}")
            print(f"SMC: draws={draws}, chains={chains}, cores={cores}")

            if debug:
                print(f"  [DEBUG] Starting SMC sampling...")

            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                return_inferencedata=True,
                threshold=0.8
            )

            elapsed = (time.time() - t0) / 60

            # Extract log-evidence
            log_ev = np.nan
            try:
                lm = idata.sample_stats.log_marginal_likelihood.values
                if hasattr(lm, 'flatten'):
                    flat = lm.flatten()
                    valid = flat[np.isfinite(flat)]
                    log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
            except:
                pass

            if log_ev > 0:
                print(f"WARNING: Positive log-evidence ({log_ev:.2f}) - instability")

            print(f"  log_ev={log_ev:.2f}, time={elapsed:.1f}min")

            # Extract diagnostics
            diagnostics = {}
            try:
                import arviz as az
                ess = az.ess(idata)
                rhat = az.rhat(idata)

                ess_vals = [ess[v].values for v in ess.data_vars if hasattr(ess[v].values, 'size')]
                rhat_vals = [rhat[v].values for v in rhat.data_vars if hasattr(rhat[v].values, 'size')]

                diagnostics['ess_min'] = float(min([v.min() for v in ess_vals])) if ess_vals else np.nan
                diagnostics['ess_median'] = float(np.median([v.mean() for v in ess_vals])) if ess_vals else np.nan
                diagnostics['rhat_max'] = float(max([v.max() for v in rhat_vals])) if rhat_vals else np.nan

                print(f"  ESS: min={diagnostics['ess_min']:.0f}, med={diagnostics['ess_median']:.0f}")
                print(f"  R-hat: max={diagnostics['rhat_max']:.3f}")

                # Convergence check
                if diagnostics['rhat_max'] > 1.1:
                    print(f"WARNING: R-hat > 1.1 indicates possible non-convergence")

            except Exception as e:
                if debug:
                    print(f"  [DEBUG] Diagnostics failed: {e}")
                diagnostics = {'ess_min': np.nan, 'ess_median': np.nan, 'rhat_max': np.nan}

            # Compile results
            res = {
                'K': K,
                'model_type': model_type,
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
                **diagnostics
            }

            # Save
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            pkl_name = f"smc_{data['world']}_K{K}_{glm_gam}_N{data['N']}_T{data['T']}_D{draws}.pkl"
            pkl_path = out_dir / pkl_name

            with open(pkl_path, 'wb') as f:
                pickle.dump({'idata': idata, 'res': res, 'data': data}, f, protocol=4)

            print(f"  Saved: {pkl_path}")

            return pkl_path, res

    except Exception as e:
        elapsed = (time.time() - t0) / 60
        print(f"  FAILED after {elapsed:.1f}min: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        raise


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='HMM-Hurdle-Gamma with Batched Forward Algorithm (Production)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug: N=50, fast iteration
  python smc_hmm_hurdle_final.py --world Cliff --K 3 --N 50 --draws 200 --debug

  # Production: N=200, paper results
  python smc_hmm_hurdle_final.py --world Harbor --K 2 --N 200 --T 104
  python smc_hmm_hurdle_final.py --world Cliff --K 3 --N 200 --T 104

  # Scale test: N=500, computational claim
  python smc_hmm_hurdle_final.py --world Cliff --K 3 --N 500 --T 104 --draws 1000
        """
    )
    parser.add_argument('--world', required=True,
                       choices=['Harbor', 'Breeze', 'Fog', 'Cliff'],
                       help='Simulation world (DGP)')
    parser.add_argument('--data_dir', type=str, default='./data/simulation',
                       help='Directory containing simulation CSVs')
    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3, 4],
                       help='Number of latent states (K=3 is true DGP)')
    parser.add_argument('--T', type=int, default=104,
                       help='Time periods (default: 104 = 2 years weekly)')
    parser.add_argument('--N', type=int, default=200,
                       help='Number of customers (default: 200, use 50 for debug)')
    parser.add_argument('--no_gam', action='store_true',
                       help='Use GLM instead of GAM (faster, less flexible)')
    parser.add_argument('--gam_df', type=int, default=3,
                       help='GAM degrees of freedom (default: 3)')
    parser.add_argument('--draws', type=int, default=1000,
                       help='SMC draws per chain (default: 1000)')
    parser.add_argument('--chains', type=int, default=4,
                       help='Number of SMC chains (default: 4)')
    parser.add_argument('--out_dir', type=str, default='./results/hmm',
                       help='Output directory for .pkl results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output and verbose error messages')

    args = parser.parse_args()

    print("=" * 70)
    print("HMM-Hurdle-Gamma: Batched Forward Algorithm (Final)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  World: {args.world} | K={args.K} | N={args.N} | T={args.T}")
    print(f"  Model: {'GLM' if args.no_gam else 'GAM'} | Draws: {args.draws} | Chains: {args.chains}")
    if args.debug:
        print(f"  Mode: DEBUG (verbose output)")
    print("=" * 70)

    # Load data
    data_dir = Path(args.data_dir)
    data = load_simulation_data(args.world, data_dir, args.T, args.N)

    # Run SMC
    pkl_path, res = run_smc_hmm(
        data=data,
        K=args.K,
        use_gam=not args.no_gam,
        gam_df=args.gam_df,
        draws=args.draws,
        chains=args.chains,
        seed=args.seed,
        out_dir=args.out_dir,
        debug=args.debug
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Log-Evidence: {res['log_evidence']:.2f}")
    print(f"Runtime: {res['time_min']:.1f} minutes")
    if 'ess_min' in res and not np.isnan(res['ess_min']):
        print(f"ESS (min): {res['ess_min']:.0f}")
        print(f"R-hat (max): {res['rhat_max']:.3f}")
    print(f"Output: {pkl_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
