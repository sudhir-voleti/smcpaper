#!/usr/bin/env python3
"""
Unified post-processing: extract all metrics from PKL folders and consolidate.
Usage: python post_processing.py --folders folders.txt --output metrics_overnight.csv --n_jobs 4
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import multiprocessing as mp
import re
import warnings
import sys
from tqdm import tqdm

warnings.filterwarnings('ignore')


def parse_condition_from_path(pkl_path):
    """Extract pi0, psi, rho, N, T from folder path."""
    path_str = str(pkl_path)
    match = re.search(r'pi0[_](0\.\d+)_psi[_](\d+)_rho[_](0\.\d+|1\.\d+)_T(\d+)_N(\d+)', path_str)
    if match:
        return {
            'pi0': float(match.group(1)),
            'psi': int(match.group(2)),
            'rho': float(match.group(3)),
            'T': int(match.group(4)),
            'N': int(match.group(5))
        }
    return {}


def extract_ari_for_pkl(pkl_path, dgp_path):
    """Compute ARI for a single PKL."""
    try:
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        
        idata = saved['idata']
        res = saved.get('res', {})
        fname = Path(pkl_path).name
        
        # Load DGP true states
        with open(dgp_path, 'rb') as f:
            dgp = pickle.load(f)
        Z_true = dgp['Z']
        N, T = Z_true.shape
        
        if 'BEMMAOR' in fname:
            model_type = 'BEMMAOR'
            if 'alpha_filtered' not in idata.posterior:
                return None
            af = idata.posterior['alpha_filtered'].values
            n_chains, n_draws, _, _, K = af.shape
            Z_pred = np.zeros((N, T), dtype=int)
            for i in range(min(N, 1000)):
                for t in range(T):
                    states, counts = np.unique(af[:, :, i, t].flatten(), return_counts=True)
                    Z_pred[i, t] = states[np.argmax(counts)]
        elif 'GAM' in fname or 'Hurdle' in fname:
            model_type = 'Hurdle'
            if 'viterbi' in idata.posterior:
                vit = idata.posterior['viterbi'].values
                Z_pred = np.zeros((N, T), dtype=int)
                for i in range(min(N, 1000)):
                    for t in range(T):
                        states, counts = np.unique(vit[:, :, i, t].flatten(), return_counts=True)
                        Z_pred[i, t] = states[np.argmax(counts)]
            else:
                return None
        else:
            return None
        
        ari = adjusted_rand_score(Z_true[:N].flatten(), Z_pred[:N].flatten())
        
        return {
            'pkl_path': str(pkl_path),
            'model_type': model_type,
            'ari': ari,
            'n_customers': N,
            'log_evidence': res.get('log_evidence', res.get('log_ev', np.nan))
        }
    except Exception as e:
        return {'pkl_path': str(pkl_path), 'error': str(e)}


def extract_hessian_for_pkl(pkl_path):
    """Extract Hessian-related metrics."""
    try:
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        
        idata = saved['idata']
        res = saved.get('res', {})
        
        result = {
            'pkl_path': str(pkl_path),
            'model_type': 'BEMMAOR' if 'BEMMAOR' in str(pkl_path) else 'Hurdle'
        }
        
        if hasattr(idata, 'posterior'):
            n_vars = len(idata.posterior.data_vars)
            result['n_posterior_vars'] = n_vars
        
        result['log_evidence'] = res.get('log_evidence', res.get('log_ev', np.nan))
        
        return result
    except Exception as e:
        return {'pkl_path': str(pkl_path), 'error': str(e)}


def extract_eigen_for_pkl(pkl_path):
    """Extract eigenvalue metrics."""
    try:
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        
        idata = saved['idata']
        res = saved.get('res', {})
        
        result = {
            'pkl_path': str(pkl_path),
            'model_type': 'BEMMAOR' if 'BEMMAOR' in str(pkl_path) else 'Hurdle',
            'log_evidence': res.get('log_evidence', res.get('log_ev', np.nan))
        }
        
        try:
            from scipy import linalg
            if 'theta' in idata.posterior:
                theta = idata.posterior['theta'].values
                theta_flat = theta.reshape(-1, theta.shape[-1])
                cov = np.cov(theta_flat.T)
                eigenvalues = linalg.eigvalsh(cov)
                result['lambda_max'] = np.max(eigenvalues)
                result['lambda_min'] = np.min(eigenvalues)
                result['condition_number'] = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
        except:
            pass
        
        return result
    except Exception as e:
        return {'pkl_path': str(pkl_path), 'error': str(e)}


def extract_viterbi_for_pkl(pkl_path):
    """Extract Viterbi-based dynamics metrics."""
    try:
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        
        idata = saved['idata']
        res = saved.get('res', {})
        fname = Path(pkl_path).name
        
        N = res.get('N', 1000)
        T = res.get('T', 104)
        
        if 'BEMMAOR' in fname:
            if 'alpha_filtered' not in idata.posterior:
                return None
            af = idata.posterior['alpha_filtered'].values
            n_chains, n_draws, N_actual, T_actual, K = af.shape
            N = min(N, N_actual)
            T = min(T, T_actual)
            
            Z_pred = np.zeros((N, T), dtype=int)
            for i in range(N):
                for t in range(T):
                    states, counts = np.unique(af[:, :, i, t].flatten(), return_counts=True)
                    Z_pred[i, t] = states[np.argmax(counts)]
        else:
            if 'viterbi' not in idata.posterior:
                return None
            vit = idata.posterior['viterbi'].values
            n_chains, n_draws, N_actual, T_actual = vit.shape
            N = min(N, N_actual)
            T = min(T, T_actual)
            
            Z_pred = np.zeros((N, T), dtype=int)
            for i in range(N):
                for t in range(T):
                    states, counts = np.unique(vit[:, :, i, t].flatten(), return_counts=True)
                    Z_pred[i, t] = states[np.argmax(counts)]
        
        transitions = np.sum(np.abs(np.diff(Z_pred, axis=1)) > 0, axis=1)
        total_transitions = int(np.sum(transitions))
        static_customers = int(np.sum(transitions == 0))
        
        return {
            'pkl_path': str(pkl_path),
            'model_type': 'BEMMAOR' if 'BEMMAOR' in fname else 'Hurdle',
            'n_customers': N,
            'n_periods': T,
            'total_transitions': total_transitions,
            'transitions_per_customer_mean': float(np.mean(transitions)),
            'transitions_per_customer_std': float(np.std(transitions)),
            'static_customers': static_customers,
            'static_pct': 100.0 * static_customers / N,
            'overall_accuracy': np.nan,
            'log_evidence': res.get('log_evidence', res.get('log_ev', np.nan))
        }
    except Exception as e:
        return {'pkl_path': str(pkl_path), 'error': str(e)}


def process_folder(folder_path):
    """Process all PKLs in a folder with progress tracking."""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    dgp_files = list(folder.glob('dgp_*.pkl'))
    dgp_path = dgp_files[0] if dgp_files else None
    model_pkls = list(folder.glob('smc_*.pkl'))
    
    results = []
    for pkl_path in model_pkls:
        if dgp_path:
            ari_result = extract_ari_for_pkl(pkl_path, dgp_path)
            if ari_result and 'error' not in ari_result:
                results.append(('ari', ari_result))
        
        hess_result = extract_hessian_for_pkl(pkl_path)
        if hess_result and 'error' not in hess_result:
            results.append(('hessian', hess_result))
        
        eigen_result = extract_eigen_for_pkl(pkl_path)
        if eigen_result and 'error' not in eigen_result:
            results.append(('eigen', eigen_result))
        
        vit_result = extract_viterbi_for_pkl(pkl_path)
        if vit_result and 'error' not in vit_result:
            results.append(('viterbi', vit_result))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Unified post-processing for PKL folders')
    parser.add_argument('--folders', required=True, help='File with list of folder paths')
    parser.add_argument('--output', default='metrics_consolidated.csv', help='Output CSV')
    parser.add_argument('--n_jobs', type=int, default=4, help='Parallel workers')
    args = parser.parse_args()
    
    with open(args.folders) as f:
        folders = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(folders)} folders with {args.n_jobs} workers...")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Process with progress bar
    all_results = []
    with mp.Pool(args.n_jobs) as pool:
        for results in tqdm(pool.imap_unordered(process_folder, folders), 
                          total=len(folders), 
                          desc="Folders",
                          unit="folder"):
            all_results.extend(results)
    
    # Separate by type
    ari_rows = [r[1] for r in all_results if r[0] == 'ari']
    hess_rows = [r[1] for r in all_results if r[0] == 'hessian']
    eigen_rows = [r[1] for r in all_results if r[0] == 'eigen']
    vit_rows = [r[1] for r in all_results if r[0] == 'viterbi']
    
    print(f"\nExtracted: ARI={len(ari_rows)}, Hessian={len(hess_rows)}, Eigen={len(eigen_rows)}, Viterbi={len(vit_rows)}")
    
    # Build DataFrames
    df_ari = pd.DataFrame(ari_rows) if ari_rows else pd.DataFrame()
    df_hess = pd.DataFrame(hess_rows) if hess_rows else pd.DataFrame()
    df_eigen = pd.DataFrame(eigen_rows) if eigen_rows else pd.DataFrame()
    df_vit = pd.DataFrame(vit_rows) if vit_rows else pd.DataFrame()
    
    # Parse conditions
    for df in [df_ari, df_hess, df_eigen, df_vit]:
        if len(df) > 0:
            parsed = df['pkl_path'].apply(parse_condition_from_path).apply(pd.Series)
            for col in ['pi0', 'psi', 'rho', 'T', 'N']:
                if col in parsed.columns:
                    df[col] = parsed[col]
    
    # Merge
    df_master = df_ari.copy() if len(df_ari) > 0 else pd.DataFrame()
    
    if len(df_hess) > 0:
        hess_cols = [c for c in df_hess.columns if c not in df_master.columns and c != 'pkl_path']
        df_master = df_master.merge(df_hess[['pkl_path'] + hess_cols], on='pkl_path', how='outer')
    
    if len(df_eigen) > 0:
        eigen_cols = [c for c in df_eigen.columns if c not in df_master.columns and c != 'pkl_path']
        df_master = df_master.merge(df_eigen[['pkl_path'] + eigen_cols], on='pkl_path', how='outer')
    
    if len(df_vit) > 0:
        vit_cols = [c for c in df_vit.columns if c not in df_master.columns and c != 'pkl_path']
        df_master = df_master.merge(df_vit[['pkl_path'] + vit_cols], on='pkl_path', how='outer')
    
    if len(df_master) > 0:
        df_master.drop_duplicates(subset=['pkl_path'], keep='last', inplace=True)
    
    df_master.to_csv(args.output, index=False)
    print(f"\n{'='*60}")
    print(f"SAVED: {args.output}")
    print(f"  Rows: {len(df_master)}")
    print(f"  Cols: {len(df_master.columns)}")
    
    if len(df_master) > 0 and 'model_type' in df_master.columns:
        print(f"\nBy model: {df_master['model_type'].value_counts().to_dict()}")
        if 'ari' in df_master.columns:
            print(f"With ARI: {df_master['ari'].notna().sum()}")
        if 'transitions_per_customer_mean' in df_master.columns:
            print(f"With Viterbi: {df_master['transitions_per_customer_mean'].notna().sum()}")


if __name__ == '__main__':
    main()
