"""
PyMC HMM-Hurdle-Gamma for HMC failure demonstration
"""

import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
from pathlib import Path


def create_hmm_hurdle_gamma_model(y, N_cust, T_vec, K=3):
    """
    Create PyMC model for HMM with hurdle-Gamma emissions
    
    Parameters:
    -----------
    y : observed spend (flattened, length N_total)
    N_cust : number of customers
    T_vec : array of time lengths per customer
    K : number of states (default 3)
    """
    N_total = len(y)
    
    with pm.Model() as model:
        # --- Transition matrix (sticky prior) ---
        # Gamma[j, k] = P(z_t=k | z_{t-1}=j)
        Gamma = pm.Dirichlet(
            'Gamma',
            a=np.eye(K) * 4 + np.ones((K, K)),  # sticky diagonal
            shape=(K, K)
        )
        
        # --- Emission parameters with order constraints ---
        # Use ordered transform to prevent label switching
        # Order by alpha (Gamma shape) - higher alpha = higher mean spend
        
        alpha_raw = pm.Normal('alpha_raw', 0, 1, shape=K)
        # Cumulative sum ensures alpha[0] < alpha[1] < alpha[2]
        alpha = pm.Deterministic('alpha', pm.math.exp(alpha_raw.cumsum()))
        
        beta = pm.Exponential('beta', 0.1, shape=K)
        pi0 = pm.Beta('pi0', 2, 2, shape=K)
        
        # --- Forward algorithm (marginal likelihood) ---
        # We need to compute p(y | params) by marginalizing over states
        
        log_Gamma = pm.math.log(Gamma)
        
        # Initialize list to store per-customer log-likelihoods
        log_liks = []
        
        pos = 0
        for c in range(N_cust):
            T_c = int(T_vec[c])
            
            # t=1: uniform initial state
            log_alpha = pm.math.log(pm.math.ones(K) / K)
            
            for t in range(T_c):
                y_obs = y[pos]
                
                # Compute log p(y_t | z_t=k) for each state k
                log_p_y = []
                for k in range(K):
                    # Hurdle-Gamma log-likelihood
                    log_zero = pm.math.log(pi0[k])
                    log_pos = pm.math.log(1 - pi0[k]) + pm.logp(
                        pm.Gamma.dist(alpha=alpha[k], beta=beta[k]),
                        y_obs
                    )
                    log_p_y_k = pm.math.switch(pm.math.eq(y_obs, 0), log_zero, log_pos)
                    log_p_y.append(log_p_y_k)
                
                log_p_y = pm.math.stack(log_p_y)
                
                if t == 0:
                    log_alpha = log_p_y + log_alpha
                else:
                    # Forward step: log_alpha_new[k] = log_sum_exp_j(log_alpha[j] + log_Gamma[j,k]) + log_p_y[k]
                    log_alpha_next = []
                    for k in range(K):
                        # Sum over previous states j
                        temp = log_alpha + log_Gamma[:, k]
                        log_sum_exp_j = pm.math.logsumexp(temp)
                        log_alpha_next.append(log_sum_exp_j + log_p_y[k])
                    log_alpha = pm.math.stack(log_alpha_next)
                
                pos += 1
            
            # Customer likelihood
            log_liks.append(pm.math.logsumexp(log_alpha))
        
        # Total log-likelihood
        pm.Potential('log_lik', pm.math.sum(log_liks))
        
    return model


def run_hmc(y, N_cust, T_vec, K=3, draws=1000, tune=1000, chains=4):
    """Run NUTS on HMM model"""
    
    model = create_hmm_hurdle_gamma_model(y, N_cust, T_vec, K)
    
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=0.8,
            return_inferencedata=True
        )
    
    return idata


def diagnose_hmc(idata):
    """Print HMC diagnostics"""
    print("=" * 60)
    print("HMC DIAGNOSTICS")
    print("=" * 60)
    
    # R-hat
    rhat = az.rhat(idata)
    print("\nR-hat (max):")
    for var in rhat.data_vars:
        max_rhat = float(rhat[var].max())
        print(f"  {var}: {max_rhat:.3f}")
    
    # ESS
    ess = az.ess(idata)
    print("\nESS (min):")
    for var in ess.data_vars:
        min_ess = float(ess[var].min())
        print(f"  {var}: {min_ess:.1f}")
    
    # Divergences
    divergences = idata.sample_stats.diverging.sum().values
    print(f"\nDivergences: {divergences}")
    
    # Overall assessment
    max_rhat_all = max([float(rhat[v].max()) for v in rhat.data_vars])
    min_ess_all = min([float(ess[v].min()) for v in ess.data_vars])
    
    print(f"\nOverall: max R-hat = {max_rhat_all:.3f}, min ESS = {min_ess_all:.1f}")
    if max_rhat_all < 1.1 and min_ess_all > 100 and divergences < 10:
        print("Status: CONVERGED")
    else:
        print("Status: FAILED")
    
    return {'max_rhat': max_rhat_all, 'min_ess': min_ess_all, 'divergences': divergences}


# --- Load your simulation and run ---
if __name__ == "__main__":
    # Load one world (e.g., Cliff)
    df = pd.read_csv("data/simulation_hmm/hmm_Cliff_N200_T52.csv")
    
    y = df['y'].values
    N_cust = df['customer_id'].nunique()
    
    # Get T per customer
    T_vec = df.groupby('customer_id')['t'].count().values
    
    print(f"Running HMC on Cliff world...")
    print(f"N={N_cust}, T_mean={T_vec.mean():.1f}, sparsity={(y==0).mean():.2%}")
    
    idata = run_hmc(y, N_cust, T_vec, K=3, draws=500, tune=500, chains=2)
    diagnose_hmc(idata)
