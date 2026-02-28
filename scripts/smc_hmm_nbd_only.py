import pymc as pm
import pytensor.tensor as pt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Forward algorithm (copied from your smc_hmm_bemmaor.py for self-containment)
# ─────────────────────────────────────────────────────────────────────────────
def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """Batched forward algorithm with proper scaling."""
    N, T, K = log_emission.shape
    
    # Initial step (t=0)
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
    
    # Emission sequence from t=1 onward
    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)
    
    (log_alpha_norm_seq, log_Z_seq), _ = pt.scan(
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
    )
    
    # Full normalized alpha sequence
    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],
        log_alpha_norm_seq
    ], axis=0)
    
    # Marginal log-likelihood
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    
    return log_marginal, log_alpha_norm_full


# ─────────────────────────────────────────────────────────────────────────────
# Improved NBD-only HMM
# ─────────────────────────────────────────────────────────────────────────────
def make_nbd_only_hmm(data, K):
    """
    Simplified HMM focusing exclusively on purchase frequency (NBD).
    Benchmarks the loss of 'Monetary' signal in RFM-SMC papers.
    
    Improvements over Gemini's version:
    - Self-contained: includes forward_algorithm_scan (no external import needed)
    - y_binary explicitly cast to float32 (fixes bool dtype issue)
    - Added customer heterogeneity via theta + gamma_h (consistent with Bemmaor)
    - State ordering via pt.sort on log_lambda_base
    - Deterministics for log_likelihood + filtered states (for diagnostics/ARI)
    - Clipping + small epsilon for numerical stability
    - Clear comments on purpose & limitations
    """
    # Convert to binary incidence (purchase = 1 if y > 0, else 0)
    y_binary = (data['y'] > 0).astype(np.float32)  # Ensures float32, not bool
    mask = data['mask'].astype(bool)               # Ensure bool mask
    N, T = data['N'], data['T']
    
    with pm.Model(coords={
        "customer": np.arange(N),
        "time": np.arange(T),
        "state": np.arange(K)
    }) as model:
        
        # 1. Latent Dynamics (Transition Matrix)
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype="float32"))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype="float32"))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.eye(K)*5 + 1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype="float32"))
            log_Gamma = pt.log(Gamma)
        
        # 2. NBD Incidence Parameters (with customer heterogeneity)
        theta = pm.Normal("theta", mu=0, sigma=1, shape=(N, 1))           # Customer RE
        gamma_h = pm.HalfNormal("gamma_h", sigma=0.5)                     # Positive effect on lambda
        
        alpha_h_raw = pm.Normal("alpha_h_raw", 0, 1, shape=K if K > 1 else None)
        if K > 1:
            alpha_h = pt.sort(alpha_h_raw)                                # Order states by increasing lambda
            log_lambda_base = alpha_h[None, None, :] + gamma_h * theta[:, :, None]
        else:
            alpha_h = alpha_h_raw
            log_lambda_base = alpha_h + gamma_h * theta
        
        lambda_nbd = pt.exp(pt.clip(log_lambda_base, -10, 10))            # NBD mean (lambda)
        
        # Dispersion r (shared across states for identifiability)
        log_r = pm.Normal("log_r", 0, 1)
        r_nbd = pt.exp(log_r)
        
        # 3. NBD P(0) = (r / (r + lambda))^r
        log_p_zero = r_nbd * (pt.log(r_nbd) - pt.log(r_nbd + lambda_nbd))
        
        # 4. Binary Emission Likelihood (incidence only)
        log_zero = log_p_zero
        log_pos = pt.log(1 - pt.exp(log_p_zero) + 1e-10)                 # P(y>0) = 1 - P(0)
        
        y_exp = y_binary[..., None] if K > 1 else y_binary
        mask_exp = mask[..., None] if K > 1 else mask
        
        log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)
        log_emission = pt.where(mask_exp, log_emission, 0.0)
        
        # 5. Forward Algorithm + Likelihood
        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)
            
            # Filtered state probabilities (for ARI / diagnostics)
            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered, dims=("customer", "time", "state"))
        
        # 6. Model Likelihood + Deterministics
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        pm.Potential("loglike", pt.sum(logp_cust))
    
    return model
