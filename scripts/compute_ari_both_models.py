#!/usr/bin/env python3
"""
Post-hoc ARI for BOTH BEMMAOR and Hurdle
BEMMAOR: from alpha_filtered
Hurdle: from viterbi (already computed)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import re

def compute_ari_for_pkl(pkl_path):
    """Compute ARI for either BEMMAOR or Hurdle."""
    try:
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        
        idata = saved['idata']
        res = saved.get('res', {})
        
        # Determine model type
        fname = pkl_path.name
        if 'BEMMAOR' in fname:
            model_type = 'BEMMAOR'
            # Use alpha_filtered for BEMMAOR
            if 'alpha_filtered' not in idata.posterior:
                print(f"  No alpha_filtered in {fname}")
                return None
            alpha_filt = idata.posterior['alpha_filtered'].values
            chains, draws, N, T, K = alpha_filt.shape
            z_samples = np.argmax(alpha_filt, axis=-1)
            z_flat = z_samples.reshape(-1, N, T)
            # Mode across samples
            Z_pred = np.zeros((N, T), dtype=int)
            for i in range(N):
                for t in range(T):
                    states, counts = np.unique(z_flat[:, i, t], return_counts=True)
                    Z_pred[i, t] = states[np.argmax(counts)]
        
        elif 'GAM' in fname:
            model_type = 'Hurdle'
            # Use pre-computed viterbi for Hurdle
            if 'viterbi' not in idata.posterior:
                print(f"  No viterbi in {fname}")
                return None
            # viterbi shape: (chains, draws, N, T) or (N, T)
            viterbi = idata.posterior['viterbi'].values
            if viterbi.ndim == 4:
                # Mode across chains/draws
                N, T = viterbi.shape[-2:]
                Z_pred = np.zeros((N, T), dtype=int)
                for i in range(N):
                    for t in range(T):
                        states, counts = np.unique(viterbi[:, :, i, t].flatten(), return_counts=True)
                        Z_pred[i, t] = states[np.argmax(counts)]
            else:
                Z_pred = viterbi.astype(int)
        else:
            print(f"  Unknown model type: {fname}")
            return None
        
        # Load DGP ground truth
        pkl_dir = pkl_path.parent
        dgp_files = list(pkl_dir.glob("dgp*.pkl"))
        if not dgp_files:
            print(f"  No DGP found for {fname}")
            return None
        
        with open(dgp_files[0], 'rb') as f:
            dgp = pickle.load(f)
        Z_true = dgp['Z']
        
        # Compute ARI
        ari_scores = []
        for i in range(Z_true.shape[0]):
            valid = (Z_true[i, :] >= 0) & (Z_pred[i, :] >= 0)
            if valid.sum() > 1:
                ari = adjusted_rand_score(Z_true[i, valid], Z_pred[i, valid])
                ari_scores.append(ari)
        
        mean_ari = np.mean(ari_scores) if ari_scores else np.nan
        
        # Parse condition
        path_str = str(pkl_path)
        pi0_match = re.search(r'pi0[_]?(\d+\.\d+)', path_str)
        rho_match = re.search(r'rho[_]?(\d+\.\d+)', path_str)
        n_match = re.search(r'N(\d+)', fname)
        
        return {
            'pkl_path': str(pkl_path),
            'model_type': model_type,
            'N': int(n_match.group(1)) if n_match else 250,
            'pi0': float(pi0_match.group(1)) if pi0_match else np.nan,
            'rho': float(rho_match.group(1)) if rho_match else np.nan,
            'ari': mean_ari,
            'n_customers': len(ari_scores),
            'log_evidence': res.get('log_evidence', np.nan)
        }
        
    except Exception as e:
        print(f"  ERROR in {pkl_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='ari_both_models.csv')
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    
    # Find ALL model PKLs (BEMMAOR + Hurdle)
    pkl_files = [p for p in base_path.rglob("*.pkl") 
                 if 'BEMMAOR' in p.name or ('GAM' in p.name and 'dgp' not in p.name)]
    
    print(f"Found {len(pkl_files)} model PKLs")
    
    records = []
    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"\n[{i}/{len(pkl_files)}] {pkl_file.name[:50]}...")
        record = compute_ari_for_pkl(pkl_file)
        if record:
            records.append(record)
            print(f"  {record['model_type']}: ARI = {record['ari']:.3f}")
    
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*60}")
    print(f"SAVED: {args.output}")
    print(f"Total: {len(df)}")
    
    if len(df) > 0:
        print(f"\nSUMMARY BY MODEL:")
        summary = df.groupby(['model_type', 'N', 'pi0', 'rho'])['ari'].agg(['mean', 'std', 'count']).round(3)
        print(summary)

if __name__ == "__main__":
    main()
