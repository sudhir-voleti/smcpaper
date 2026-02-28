#!/usr/bin/env python3
"""
SMC-HMM Bemmaor Post-Processing Utility
Functionalized for GitHub: Handles state recovery, Viterbi decoding, 
and parameter recovery across diverse simulation worlds.
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from itertools import permutations
from sklearn.metrics import adjusted_rand_score, accuracy_score

# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def viterbi_decode(log_emission, log_Gamma, log_pi0):
    """
    Standard Viterbi algorithm for global state sequence recovery.
    Inputs are log-space likelihoods and transition probabilities.
    """
    N, T, K = log_emission.shape
    v = np.zeros((N, T, K))
    ptr = np.zeros((N, T, K), dtype=int)
    
    v[:, 0, :] = log_pi0[None, :] + log_emission[:, 0, :]
    
    for t in range(1, T):
        # Broadcasted scores: previous log-prob + transition log-prob
        scores = v[:, t-1, :, None] + log_Gamma[None, :, :]
        v[:, t, :] = np.max(scores, axis=1) + log_emission[:, t, :]
        ptr[:, t, :] = np.argmax(scores, axis=1)
    
    best_paths = np.zeros((N, T), dtype=int)
    best_paths[:, T-1] = np.argmax(v[:, T-1, :], axis=1)
    
    for t in range(T-2, -1, -1):
        best_paths[:, t] = ptr[np.arange(N), t+1, best_paths[:, t+1]]
    
    return best_paths [cite: 4, 5]

def extract_log_evidence(idata):
    """
    Extracts log-marginal likelihood from PyMC SMC InferenceData.
    Handles both standard numeric arrays and list-of-list object structures.
    """
    try:
        lm = idata.sample_stats.log_marginal_likelihood.values
        if hasattr(lm, 'dtype') and lm.dtype == object:
            chain_finals = []
            for chain_data in np.array(lm).flatten():
                if isinstance(chain_data, (list, np.ndarray)):
                    valid = [float(x) for x in chain_data if np.isfinite(x)]
                    if valid: chain_finals.append(valid[-1])
            return float(np.mean(chain_finals)) if chain_finals else np.nan
        else:
            valid = lm[np.isfinite(lm)]
            return float(np.mean(valid)) if len(valid) > 0 else np.nan
    except Exception as e:
        return np.nan [cite: 6, 7, 8, 9]

def align_states(true_flat, pred_flat, K=3):
    """
    Finds the optimal permutation of state labels to maximize accuracy.
    Crucial for addressing label-switching in Bayesian HMMs.
    """
    best_acc = -1
    best_map = {i: i for i in range(K)}
    for perm in permutations(range(K)):
        remap = {i: perm[i] for i in range(K)}
        aligned = np.vectorize(remap.get)(pred_flat)
        acc = accuracy_score(true_flat, aligned)
        if acc > best_acc:
            best_acc = acc
            best_map = remap
    return best_map [cite: 11]

# =============================================================================
# DATA WRAPPING & PROCESSING
# =============================================================================

def process_single_pkl(pkl_path, true_states_path, true_params):
    """
    Analyzes a single simulation result. Accepts paths and truth parameters.
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    
    idata = result['idata']
    data = result.get('data', {})
    post = idata.posterior
    
    # 1. Log-Evidence
    log_ev = extract_log_evidence(idata) [cite: 17]
    
    # 2. State Recovery
    alpha_filtered = post['alpha_filtered'].mean(dim=['chain', 'draw']).values
    log_pi0 = np.log(post['pi0'].mean(dim=['chain', 'draw']).values + 1e-10)
    log_Gamma = np.log(post['Gamma'].mean(dim=['chain', 'draw']).values + 1e-10)
    
    # Approximate log-emission for Viterbi
    log_emission = np.log(alpha_filtered + 1e-10)
    viterbi_paths = viterbi_decode(log_emission, log_Gamma, log_pi0) [cite: 20]
    
    # 3. Validation against Ground Truth
    metrics = {'World': pkl_path.stem, 'Log_Evidence': log_ev}
    
    if true_states_path and true_states_path.exists():
        true_full = np.load(true_states_path)
        true_train = true_full[:, :alpha_filtered.shape[1]]
        mask = data.get('mask', np.ones_like(true_train, dtype=bool))
        
        true_f = true_train[mask].flatten()
        vit_f = viterbi_paths[mask].flatten()
        
        metrics['ARI_Viterbi'] = adjusted_rand_score(true_f, vit_f)
        metrics['State_Acc'] = accuracy_score(true_f, vit_f) [cite: 23]
        
        # Parameter Recovery (RMSE)
        label_map = align_states(true_f, vit_f)
        if 'log_alpha_gamma' in post:
            alpha_est = np.exp(post['log_alpha_gamma'].mean(dim=['chain', 'draw']).values)
            aligned_alpha = np.array([alpha_est[label_map[i]] for i in range(3)])
            metrics['Alpha_RMSE'] = np.sqrt(np.mean((aligned_alpha - true_params['alpha_base'])**2)) [cite: 25]
            
    return metrics

# =============================================================================
# MAIN EXECUTION (CLI)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-process Bemmaor SMC results.")
    parser.add_argument("--pkl_dir", type=str, required=True, help="Directory containing .pkl files")
    parser.add_argument("--truth_dir", type=str, required=True, help="Directory containing true_states.npy files")
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)
    results = []

    # Example loop logic - in production, you might pass specific world params via a JSON config
    for pkl_path in pkl_dir.glob("*.pkl"):
        # Match world logic (e.g., finding 'harbor' in filename)
        world_key = next((w for w in ["Harbor", "Cliff", "Breeze", "Fog"] if w.lower() in pkl_path.name.lower()), "Harbor")
        
        # Dynamically find truth file
        true_file = Path(args.truth_dir) / f"{world_key.lower()}/true_states_{world_key}_N500_T104.npy"
        
        # Placeholder for world-specific true params
        # (In a real GitHub repo, these would be in a separate config.json)
        world_params = {'alpha_base': np.array([4.0, 4.0, 4.0])} 
        
        metrics = process_single_pkl(pkl_path, true_file, world_params)
        results.append(metrics)

    df = pd.DataFrame(results)
    print(df.to_string()) [cite: 29]

if __name__ == "__main__":
    main()
