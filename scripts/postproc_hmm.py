#!/usr/bin/env python3
"""
postproc_hmm.py - Final hybrid with confusion matrix and LaTeX export
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, confusion_matrix, accuracy_score
from itertools import permutations
import re

# True DGP parameters
TRUE_PARAMS = {
    "Harbor": {
        "pi0_base": np.array([0.90, 0.50, 0.15]),
        "mu_base": np.array([5, 25, 120]),
        "alpha_base": np.array([4.0, 4.0, 4.0]),
        "Gamma": np.array([[0.92, 0.06, 0.02], 
                          [0.10, 0.80, 0.10], 
                          [0.05, 0.15, 0.80]])
    },
    "Breeze": {
        "pi0_base": np.array([0.90, 0.50, 0.15]),
        "mu_base": np.array([5, 25, 120]),
        "alpha_base": np.array([0.8, 0.7, 0.5]),
        "Gamma": np.array([[0.92, 0.06, 0.02], 
                          [0.10, 0.80, 0.10], 
                          [0.05, 0.15, 0.80]])
    },
    "Fog": {
        "pi0_base": np.array([0.98, 0.85, 0.40]),
        "mu_base": np.array([2, 10, 80]),
        "alpha_base": np.array([4.0, 4.0, 4.0]),
        "Gamma": np.array([[0.92, 0.06, 0.02], 
                          [0.10, 0.80, 0.10], 
                          [0.05, 0.15, 0.80]])
    },
    "Cliff": {
        "pi0_base": np.array([0.98, 0.85, 0.40]),
        "mu_base": np.array([2, 10, 80]),
        "alpha_base": np.array([0.8, 0.7, 0.5]),
        "Gamma": np.array([[0.92, 0.06, 0.02], 
                          [0.10, 0.80, 0.10], 
                          [0.05, 0.15, 0.80]])
    }
}


def find_best_alignment(true_flat, pred_flat, K):
    """
    Find label permutation that maximizes ACCURACY (not ARI).
    Returns: label_map (pred->true), accuracy, aligned_predictions
    """
    if K > 6:  # Limit permutations for large K
        return {i: i for i in range(K)}, np.nan, pred_flat
    
    best_acc = -1
    best_map = {i: i for i in range(K)}
    best_aligned = pred_flat
    
    for perm in permutations(range(K)):
        # perm[i] = what true label should mapped pred label i become
        remap = {i: perm[i] for i in range(K)}
        aligned = np.vectorize(remap.get)(pred_flat)
        acc = accuracy_score(true_flat, aligned)
        
        if acc > best_acc:
            best_acc = acc
            best_map = remap
            best_aligned = aligned
    
    return best_map, best_acc, best_aligned


def decode_states(idata):
    """Posterior decoding from alpha_filtered."""
    if 'alpha_filtered' not in idata.posterior:
        return None, 0
    
    # Use xarray mean (cleaner)
    alpha = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values
    return np.argmax(alpha, axis=-1), alpha.shape[-1]  # states, K


def compute_confusion_matrix(true_flat, pred_flat, K):
    """Compute normalized confusion matrix (rows=true, cols=pred)."""
    cm = confusion_matrix(true_flat, pred_flat, labels=range(K))
    # Normalize by row (true state distribution)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    return cm_norm


def load_true_states(world_dir, world, N, T):
    """Load true states from .npy file."""
    patterns = [
        f"true_states_{world}_N{N}_T{T}.npy",
        f"true_states_{world.capitalize()}_N{N}_T{T}.npy",
        f"true_states_{world.lower()}_N{N}_T{T}.npy",
    ]
    
    for pattern in patterns:
        npy_path = world_dir / pattern
        if npy_path.exists():
            return np.load(npy_path)
    return None


def parse_pkl_filename(filename):
    """Extract metadata from PKL filename."""
    pattern = r"smc_K(\d+)_(GLM|GAM)_N(\d+)_T(\d+)_D(\d+)\.pkl"
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if match:
        return {
            'K': int(match.group(1)),
            'glm_gam': match.group(2).upper(),
            'N': int(match.group(3)),
            'T': int(match.group(4)),
            'draws': int(match.group(5))
        }
    return None


def extract_posterior_means(idata):
    """Extract posterior means."""
    means = {}
    
    if 'beta0' in idata.posterior:
        b = idata.posterior['beta0'].mean(dim=['chain', 'draw']).values
        if b.ndim == 0:
            b = np.array([b])
        for i, val in enumerate(b):
            means[f'beta0_state{i}'] = float(val)
    
    alpha_key = 'alpha_gamma' if 'alpha_gamma' in idata.posterior else 'alpha'
    if alpha_key in idata.posterior:
        a = idata.posterior[alpha_key].mean(dim=['chain', 'draw']).values
        if a.ndim == 0:
            a = np.array([a])
        for i, val in enumerate(a):
            means[f'alpha_state{i}'] = float(val)
    
    if 'Gamma' in idata.posterior:
        G = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
        K = G.shape[0]
        for i in range(K):
            for j in range(K):
                means[f'Gamma_{i}{j}'] = float(G[i, j])
            means[f'Gamma_{i}_diag'] = float(G[i, i])
    
    return means


def compute_aligned_rmse(estimates, true_values, label_map):
    """Compute RMSE after aligning states."""
    K = min(len(label_map), len(true_values))
    aligned = np.array([estimates[label_map[i]] for i in range(K) if i in label_map])
    return np.sqrt(np.mean((aligned - true_values[:len(aligned)])**2))


def process_single_pkl(pkl_path, world_dir=None, world=None):
    """Process single PKL with full diagnostics."""
    pkl_path = Path(pkl_path)
    file_info = parse_pkl_filename(pkl_path.name)
    
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    
    idata = result['idata']
    res = result['res']
    
    world = world or res.get('world', pkl_path.parent.name)
    world = world.capitalize()
    world_dir = world_dir or pkl_path.parent
    
    # Basic metrics
    metrics = {
        'file': pkl_path.name,
        'world': world,
        'K': file_info['K'] if file_info else res.get('K', 0),
        'glm_gam': file_info['glm_gam'] if file_info else res.get('glm_gam', 'unknown'),
        'N': file_info['N'] if file_info else res.get('N', 0),
        'T': file_info['T'] if file_info else res.get('T', 0),
        'draws': file_info['draws'] if file_info else res.get('draws', 0),
        'log_evidence': res.get('log_evidence', np.nan),
        'time_min': res.get('time_min', np.nan),
        'ess_min': res.get('ess_min', np.nan),
        'rhat_max': res.get('rhat_max', np.nan),
    }
    
    # Posterior means
    means = extract_posterior_means(idata)
    metrics.update(means)
    
    # State recovery
    true_states = load_true_states(world_dir, world, metrics['N'], metrics['T'])
    
    if true_states is not None:
        est_states, K_est = decode_states(idata)
        
        if est_states is not None:
            true_flat = true_states.flatten()
            est_flat = est_states.flatten()
            
            # ARI (invariant to labels)
            metrics['ari'] = adjusted_rand_score(true_flat, est_flat)
            
            # Alignment for accuracy and parameter recovery
            label_map, acc, aligned_est = find_best_alignment(true_flat, est_flat, K_est)
            metrics['state_accuracy'] = acc
            metrics['label_map'] = str(label_map)
            
            # Confusion matrix (using aligned predictions)
            cm = compute_confusion_matrix(true_flat, aligned_est, K_est)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    metrics[f'cm_{i}{j}'] = float(cm[i, j])
            
            # Parameter recovery (aligned)
            if world in TRUE_PARAMS:
                true = TRUE_PARAMS[world]
                
                if 'alpha_state0' in means:
                    est_alphas = np.array([means[f'alpha_state{i}'] for i in range(K_est)])
                    metrics['alpha_rmse'] = compute_aligned_rmse(est_alphas, true['alpha_base'], label_map)
                
                if 'Gamma_0_diag' in means:
                    est_diag = np.array([means[f'Gamma_{i}_diag'] for i in range(K_est)])
                    true_diag = np.diag(true['Gamma'])
                    metrics['gamma_diag_rmse'] = compute_aligned_rmse(est_diag, true_diag, label_map)
            
            metrics['true_states_found'] = True
        else:
            metrics['ari'] = np.nan
            metrics['state_accuracy'] = np.nan
            metrics['true_states_found'] = False
    else:
        metrics['ari'] = np.nan
        metrics['state_accuracy'] = np.nan
        metrics['true_states_found'] = False
    
    return metrics


def aggregate_all(base_dir, worlds=None):
    """Process all PKLs."""
    base_path = Path(base_dir)
    
    if worlds is None:
        worlds = [d.name for d in base_path.iterdir() if d.is_dir()]
        worlds = [w for w in worlds if w.lower() in ['harbor', 'breeze', 'fog', 'cliff']]
    
    worlds = [w.capitalize() for w in worlds]
    all_metrics = []

    for world in worlds:
        world_dir = base_path / world
        if not world_dir.exists():
            continue

        pkl_files = list(world_dir.glob("smc_*.pkl"))
        print(f"\n{world.upper()}: {len(pkl_files)} PKL files")

        for pkl_file in pkl_files:
            try:
                metrics = process_single_pkl(pkl_file, world_dir, world)
                all_metrics.append(metrics)
                
                ari_str = f", ARI={metrics['ari']:.3f}" if not np.isnan(metrics.get('ari', np.nan)) else ""
                acc_str = f", Acc={metrics['state_accuracy']:.3f}" if 'state_accuracy' in metrics else ""
                print(f"  ✓ {pkl_file.name}: log_ev={metrics['log_evidence']:.1f}{ari_str}{acc_str}")
                
            except Exception as e:
                print(f"  ✗ {pkl_file.name}: {str(e)[:60]}")
                import traceback
                traceback.print_exc()

    if not all_metrics:
        return None

    df = pd.DataFrame(all_metrics)
    df = df.sort_values(['world', 'log_evidence'], ascending=[True, False])
    
    # Save CSV
    master_csv = base_path / "all_models_summary.csv"
    df.to_csv(master_csv, index=False)
    print(f"\n{'='*70}\nSaved CSV: {master_csv} ({len(df)} models)")
    
    return df


def display_results(df, base_path=None):
    """Display and export results."""
    if df is None or df.empty:
        return

    print("\n" + "="*110)
    print("MODEL COMPARISON TABLE")
    print("="*110)
    
    # Core columns
    cols = ['world', 'K', 'glm_gam', 'log_evidence', 'ari', 'state_accuracy', 
            'alpha_rmse', 'gamma_diag_rmse', 'time_min', 'ess_min', 'rhat_max']
    display_cols =
