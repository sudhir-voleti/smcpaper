#!/usr/bin/env python3
"""
Extract Viterbi states and compute ARI for all PKLs
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import sys

sys.path.insert(0, '/Users/sudhirvoleti/rfm-hmm-smc-main/src')

def decode_viterbi(idata):
    """Decode states from alpha_filtered (posterior mean)."""
    if 'alpha_filtered' not in idata.posterior:
        return None
    
    # Mean across chains and draws
    alpha = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values
    # alpha shape: (N, T, K)
    return np.argmax(alpha, axis=-1)  # (N, T)

def compute_ari(true_states, pred_states):
    """Compute ARI, handling label permutation."""
    if true_states is None or pred_states is None:
        return np.nan
    
    true_flat = true_states.flatten()
    pred_flat = pred_states.flatten()
    
    # Only valid states
    mask = (true_flat >= 0) & (pred_flat >= 0)
    if mask.sum() < 10:
        return np.nan
    
    return adjusted_rand_score(true_flat[mask], pred_flat[mask])

def extract_from_pkl(pkl_path):
    """Extract all metrics including ARI."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None
    
    # Basic info from res dict
    res = data.get('res', {})
    idata = data.get('idata', None)
    dgp_data = data.get('data', {})
    
    results = {
        'file': pkl_path.name,
        'world': res.get('world', ''),
        'model_type': res.get('model_type', ''),
        'N': res.get('N', 0),
        'T': res.get('T', 0),
        'K': res.get('K', 0),
        'log_evidence': res.get('log_evidence', np.nan),
        'time_min': res.get('time_min', np.nan),
        'ess_min': res.get('ess_min', np.nan),
        'clv_ratio': res.get('clv_ratio', np.nan),
        'n_whales_true': res.get('n_whales_true', np.nan),
        'n_whales_pred': res.get('n_whales_pred', np.nan),
        'whale_precision': res.get('whale_precision', np.nan),
        'whale_recall': res.get('whale_recall', np.nan),
        'whale_f1': res.get('whale_f1', np.nan),
    }
    
    # Extract state contrast if BEMMAOR
    if idata is not None and 'alpha_h' in idata.posterior:
        alpha_h = idata.posterior['alpha_h'].values  # (chain, draw, K)
        if alpha_h.shape[-1] >= 2:
            contrast = np.abs(alpha_h[..., 0] - alpha_h[..., 1])
            results['state_contrast_mean'] = np.mean(contrast)
            results['state_contrast_std'] = np.std(contrast)
    
    # Compute ARI
    true_states = dgp_data.get('true_states', None)
    if true_states is not None and idata is not None:
        pred_states = decode_viterbi(idata)
        if pred_states is not None:
            results['ari'] = compute_ari(true_states, pred_states)
            results['ari_computed'] = True
        else:
            results['ari'] = np.nan
            results['ari_computed'] = False
    else:
        results['ari'] = np.nan
        results['ari_computed'] = False
    
    return results

def process_directory(base_dir, output_name):
    """Process all PKLs in directory."""
    base_path = Path(base_dir)
    
    # Find all PKLs
    bem_pkls = list(base_path.rglob('*BEMMAOR*.pkl'))
    hur_pkls = list(base_path.rglob('*GAM*.pkl')) if 'N500' in str(base_dir) else list(base_path.rglob('*Hurdle*.pkl'))
    
    all_results = []
    
    print(f"Processing {len(bem_pkls)} BEMMAOR and {len(hur_pkls)} Hurdle PKLs...")
    
    for pkl_path in bem_pkls + hur_pkls:
        print(f"  {pkl_path.name}...")
        result = extract_from_pkl(pkl_path)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("No results extracted!")
        return None
    
    df = pd.DataFrame(all_results)
    output_csv = base_path / output_name
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved: {output_csv}")
    print(f"Total rows: {len(df)}")
    print(f"ARI computed: {df['ari_computed'].sum()}/{len(df)}")
    
    if 'ari' in df.columns:
        print(f"\nARI summary by model:")
        print(df.groupby('model_type')['ari'].describe())
    
    return df

def main():
    # Process N250
    print("="*70)
    print("Processing N=250...")
    print("="*70)
    df_250 = process_directory('/Users/sudhirvoleti/jrssc_april/results_april05_N250', 
                                'metrics_with_ari_N250.csv')
    
    # Process N500
    print("\n" + "="*70)
    print("Processing N=500...")
    print("="*70)
    df_500 = process_directory('/Users/sudhirvoleti/jrssc_april/results_jasa_N500',
                                'metrics_with_ari_N500.csv')

if __name__ == '__main__':
    main()
