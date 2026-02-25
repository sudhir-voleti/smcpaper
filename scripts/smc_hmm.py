#!/usr/bin/env python3
"""
smc_hmm.py - SMC for HMM-Hurdle-Gamma on 4-world simulation
State recovery focus: Viterbi decoding vs true states
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pathlib import Path
import pickle
import time
import json
from dataclasses import dataclass
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/Users/sudhirvoleti/research related/SMC paper Feb2026/data/simulation_hmm")
OUT_DIR = Path("results/smc")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_world_data(world_name: str) -> Dict:
    """Load simulation data with true states"""
    csv_file = DATA_DIR / f"hmm_{world_name}_N200_T52.csv"
    states_file = DATA_DIR / f"hmm_{world_name}_true_states.npy"
    
    df = pd.read_csv(csv_file)
    true_states = np.load(states_file)
    
    # Construct data dict for SMC
    N = df['customer_id'].nunique()
    T = df.groupby('customer_id')['t'].count().values
    
    # Simple features: just intercept for now (GLM), add GAM later
    data = {
        'N': N,
        'T': T,
        'y': df['y'].values,
        'true_states': true_states,
        'world': world_name
    }
    
    return data


def make_hurdle_hmm(data: Dict, K: int, use_gam: bool = False) -> pm.Model:
    """
    HMM with hurdle-Gamma emissions
    No covariates for now (intercept only) - matches simulation DGP
    """
    y = data['y']
    N = data['N']
    T_vec = data['T']
    K = int(K)
    
    with pm.Model() as model:
        # Transition matrix (sticky prior)
        Gamma = pm.Dirichlet('Gamma', a=np.eye(K)*5 + np.ones((K,K)), shape=(K,K))
        
        # Emission parameters with order constraints on alpha
        # Ordered transform prevents label switching
        alpha_raw = pm.Normal('alpha_raw', 0, 1, shape=K, 
                              transform=pm.distributions.transforms.ordered)
        alpha = pm.Deterministic('alpha', pm.math.exp(alpha_raw))
        
        beta = pm.Exponential('beta', 0.1, shape=K)
        pi0 = pm.Beta('pi0', 2, 2, shape=K)
        
        # Forward algorithm (marginal likelihood)
        log_Gamma = pm.math.log(Gamma)
        
        # For each customer
        log_liks = []
        pos = 0
        
        for i in range(N):
            T_i = int(T_vec[i])
            
            # Initial state (uniform)
            log_alpha = pm.math.log(pm.math.ones(K) / K)
            
            for t in range(T_i):
                y_it = y[pos]
                
                # Emission log-prob for each state
                log_p_y = []
                for k in range(K):
                    # Hurdle: log P(y=0) or log P(y>0) + log Gamma(y|alpha,beta)
                    log_zero = pm.math.log(pi0[k])
                    log_pos = (pm.math.log(1 - pi0[k]) + 
                               pm.logp(pm.Gamma.dist(alpha=alpha[k], beta=beta[k]), y_it))
                    log_p_y.append(pm.math.switch(pm.math.eq(y_it, 0), log_zero, log_pos))
                
                log_p_y = pm.math.stack(log_p_y)
                
                if t == 0:
                    log_alpha = log_p_y + log_alpha
                else:
                    # Forward step: log_alpha_new[k] = logsumexp_j(log_alpha[j] + log_Gamma[j,k]) + log_p_y[k]
                    log_alpha_next = []
                    for k in range(K):
                        temp = log_alpha + log_Gamma[:, k]
                        log_alpha_next.append(pm.math.logsumexp(temp) + log_p_y[k])
                    log_alpha = pm.math.stack(log_alpha_next)
                
                pos += 1
            
            log_liks.append(pm.math.logsumexp(log_alpha))
        
        pm.Potential('log_lik', pm.math.sum(log_liks))
        
    return model


def run_smc_hmm(data: Dict, K: int, use_gam: bool = False, 
                draws: int = 500, chains: int = 4, seed: int = 42) -> Tuple:
    """Run SMC on HMM-hurdle-Gamma"""
    
    t0 = time.time()
    
    with make_hurdle_hmm(data, K, use_gam) as model:
        print(f"  SMC: K={K}, {'GAM' if use_gam else 'GLM'}, {data['world']}")
        
        idata = pm.sample_smc(
            draws=draws,
            chains=chains,
            cores=min(chains, 4),
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
        
        print(f"    log_ev={log_ev:.2f}, time={elapsed:.1f}min")
        
        return idata, {'log_ev': log_ev, 'time_min': elapsed, 'K': K, 
                      'use_gam': use_gam, 'world': data['world']}


def extract_viterbi_states(idata, data: Dict) -> np.ndarray:
    """
    Viterbi decoding using posterior mean parameters
    Simplified: uses MAP estimate for discrete decoding
    """
    # Get posterior means
    post = idata.posterior
    Gamma_mean = post['Gamma'].mean(dim=['chain', 'draw']).values
    pi0_mean = post['pi0'].mean(dim=['chain', 'draw']).values
    alpha_mean = post['alpha'].mean(dim=['chain', 'draw']).values
    beta_mean = post['beta'].mean(dim=['chain', 'draw']).values
    
    y = data['y']
    N = data['N']
    T_vec = data['T']
    K = len(pi0_mean)
    
    # Viterbi for each customer
    viterbi_states = np.zeros((N, max(T_vec)), dtype=int)
    
    pos = 0
    for i in range(N):
        T_i = int(T_vec[i])
        y_i = y[pos:pos+T_i]
        
        # Log emission probs
        log_emit = np.zeros((T_i, K))
        for t in range(T_i):
            for k in range(K):
                if y_i[t] == 0:
                    log_emit[t, k] = np.log(pi0_mean[k] + 1e-10)
                else:
                    log_pos = np.log(1 - pi0_mean[k] + 1e-10)
                    # Gamma logpdf
                    from scipy.stats import gamma
                    log_pos += gamma.logpdf(y_i[t], alpha_mean[k], scale=1/beta_mean[k])
                    log_emit[t, k] = log_pos
        
        # Viterbi
        log_delta = np.log(1.0/K) + log_emit[0]
        psi = np.zeros((T_i, K), dtype=int)
        
        for t in range(1, T_i):
            for k in range(K):
                scores = log_delta + np.log(Gamma_mean[:, k] + 1e-10)
                psi[t, k] = np.argmax(scores)
                log_delta[k] = np.max(scores) + log_emit[t, k]
        
        # Backtrack
        states_i = np.zeros(T_i, dtype=int)
        states_i[-1] = np.argmax(log_delta)
        for t in range(T_i-2, -1, -1):
            states_i[t] = psi[t+1, states_i[t+1]]
        
        viterbi_states[i, :T_i] = states_i
        pos += T_i
    
    return viterbi_states


def compute_metrics(idata, data: Dict, viterbi_states: np.ndarray) -> Dict:
    """Compute state recovery metrics"""
    true_states = data['true_states']
    N, T_max = true_states.shape
    
    # Flatten for comparison
    true_flat = true_states.flatten()
    viterbi_flat = viterbi_states.flatten()
    
    # Mask valid entries
    mask = true_flat >= 0
    true_valid = true_flat[mask]
    viterbi_valid = viterbi_flat[mask]
    
    # Overall accuracy
    accuracy = np.mean(viterbi_valid == true_valid)
    
    # Per-state metrics
    K = len(np.unique(true_valid))
    per_state = {}
    for k in range(K):
        true_k = (true_valid == k)
        pred_k = (viterbi_valid == k)
        precision = np.sum(pred_k & true_k) / (np.sum(pred_k) + 1e-10)
        recall = np.sum(pred_k & true_k) / (np.sum(true_k) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        per_state[f'state_{k}'] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # Confusion matrix
    confusion = np.zeros((K, K))
    for true_k in range(K):
        for pred_k in range(K):
            confusion[true_k, pred_k] = np.sum((true_valid == true_k) & (viterbi_valid == pred_k))
        confusion[true_k] /= np.sum(true_valid == true_k) + 1e-10
    
    return {
        'accuracy': accuracy,
        'per_state': per_state,
        'confusion': confusion.tolist(),
        'n_states': K,
        'n_obs': len(true_valid)
    }


def run_all_configs():
    """Run all K × world combinations"""
    worlds = ['Harbor', 'Breeze', 'Fog', 'Cliff']
    K_values = [2, 3, 4]
    
    all_results = []
    
    print("=" * 70)
    print("SMC HMM-Hurdle-Gamma: State Recovery")
    print("=" * 70)
    
    for world in worlds:
        print(f"\n{'='*70}")
        print(f"World: {world}")
        print('='*70)
        
        data = load_world_data(world)
        
        for K in K_values:
            print(f"\nK={K}:")
            try:
                # Run SMC
                idata, smc_info = run_smc_hmm(data, K, use_gam=False, 
                                              draws=500, chains=4, seed=42)
                
                # Viterbi decoding
                viterbi_states = extract_viterbi_states(idata, data)
                
                # Metrics
                metrics = compute_metrics(idata, data, viterbi_states)
                
                print(f"  State accuracy: {metrics['accuracy']:.3f}")
                
                # Save
                result = {**smc_info, **metrics, 'viterbi_states': viterbi_states.tolist()}
                all_results.append(result)
                
                pkl_path = OUT_DIR / f"smc_{world}_K{K}.pkl"
                with open(pkl_path, 'wb') as f:
                    pickle.dump({'idata': idata, 'result': result}, f)
                
            except Exception as e:
                print(f"  FAILED: {str(e)[:60]}")
                all_results.append({'world': world, 'K': K, 'error': str(e)})
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    for r in all_results:
        if 'error' not in r:
            print(f"{r['world']:8s} K={r['K']} | Acc={r['accuracy']:.3f} | log_ev={r['log_ev']:.1f}")
        else:
            print(f"{r['world']:8s} K={r['K']} | FAILED")
    
    # Save all
    with open(OUT_DIR / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


if __name__ == "__main__":
    results = run_all_configs()
