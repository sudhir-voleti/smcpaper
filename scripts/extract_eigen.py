#!/usr/bin/env python3
import pickle
import glob
import numpy as np
import pandas as pd
import re
import argparse
from pathlib import Path


def extract_covariance(idata, var_names=['theta', 'gamma_h', 'gamma_m', 'Gamma']):
    """Extract posterior covariance from ArviZ InferenceData."""
    try:
        posterior = idata.posterior
        samples_list = []
        param_info = []
        
        for var in var_names:
            if var not in posterior.data_vars:
                continue
            arr = np.squeeze(posterior[var].values)
            
            if arr.ndim == 1:
                flat = arr.flatten()
                samples_list.append(flat)
                param_info.append((var, len(flat)))
            elif arr.ndim == 2:
                n_samples = arr.shape[0] * arr.shape[1]
                flat = arr.reshape(-1)
                samples_list.append(flat)
                param_info.append((var, len(flat)))
            elif arr.ndim == 3:
                n_samples = arr.shape[0] * arr.shape[1]
                flat = arr.reshape(n_samples, -1)
                for i in range(flat.shape[1]):
                    samples_list.append(flat[:, i])
                    param_info.append((f"{var}_{i}", n_samples))
            elif arr.ndim == 4:
                n_samples = arr.shape[0] * arr.shape[1]
                flat = arr.reshape(n_samples, -1)
                for i in range(flat.shape[1]):
                    samples_list.append(flat[:, i])
                    param_info.append((f"{var}_{i}", n_samples))
        
        if not samples_list:
            return None, None, None
            
        min_len = min(len(s) for s in samples_list)
        samples_aligned = [s[:min_len] for s in samples_list]
        data_matrix = np.column_stack(samples_aligned)
        cov = np.cov(data_matrix, rowvar=False)
        return cov, param_info, data_matrix
        
    except Exception as e:
        return None, None, None


def process_pkl(pkl_path, model_type='BEMMAOR'):
    """Extract eigenvalues from idata posterior covariance."""
    try:
        with open(pkl_path, 'rb') as f:
            res = pickle.load(f)
        if 'idata' not in res or res['idata'] is None:
            return None
            
        idata = res['idata']
        cov, param_info, data = extract_covariance(idata)
        
        if cov is None or cov.ndim != 2:
            return None
            
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals_sorted = np.sort(eigvals)[::-1]
        threshold = 1e-10
        positive_eig = eigvals_sorted[eigvals_sorted > threshold]
        
        if len(positive_eig) == 0:
            return None
        
        theta_idx = next((i for i, (name, _) in enumerate(param_info) 
                         if name == 'theta' or name.startswith('theta_')), None)
        lambda_theta = positive_eig[theta_idx] if theta_idx is not None else np.nan
        
        return {
            'pkl_path': str(pkl_path),
            'model_type': model_type,
            'n_params': len(param_info),
            'n_positive_eig': len(positive_eig),
            'lambda_max': positive_eig[0],
            'lambda_theta': lambda_theta,
            'lambda_min': positive_eig[-1],
            'condition_number': positive_eig[0] / positive_eig[-1] 
                if len(positive_eig) > 1 and positive_eig[-1] > 0 else np.nan,
        }
    except Exception as e:
        return None


def parse_condition(pkl_path):
    """Extract pi0, rho from path."""
    path_str = str(pkl_path)
    pi0_match = re.search(r'pi0[_]?(\d+\.\d+|\d+)', path_str)
    pi0 = float(pi0_match.group(1)) if pi0_match else np.nan
    rho_match = re.search(r'rho[_]?(\d+\.\d+|\d+)', path_str)
    rho = float(rho_match.group(1)) if rho_match else np.nan
    return pi0, rho


def process_directory(base_dir, all_results):
    """Process all PKLs in a directory."""
    path = Path(base_dir)
    if not path.exists():
        print(f"Directory not found: {base_dir}, skipping...")
        return
        
    print(f"\nScanning {base_dir}...")
    
    # BEMMAOR
    bem_pkls = list(path.rglob('*BEMMAOR*.pkl'))
    print(f"  BEMMAOR: {len(bem_pkls)} PKLs")
    for i, pkl in enumerate(bem_pkls):
        if i % 10 == 0 and i > 0:
            print(f"    Processed {i}/{len(bem_pkls)}")
        res = process_pkl(pkl, 'BEMMAOR')
        if res:
            pi0, rho = parse_condition(pkl)
            res['pi0'] = pi0
            res['rho'] = rho
            all_results.append(res)
    
    # Hurdle
    hur_pkls = [p for p in path.rglob('*GAM*.pkl') if 'dgp' not in p.name.lower()]
    print(f"  Hurdle: {len(hur_pkls)} PKLs")
    for i, pkl in enumerate(hur_pkls):
        if i % 10 == 0 and i > 0:
            print(f"    Processed {i}/{len(hur_pkls)}")
        res = process_pkl(pkl, 'Hurdle')
        if res:
            pi0, rho = parse_condition(pkl)
            res['pi0'] = pi0
            res['rho'] = rho
            all_results.append(res)


def main():
    parser = argparse.ArgumentParser(description='Extract eigenvalues from PKL idata')
    parser.add_argument('--base_dir', nargs='+', required=True,
                        help='Base directories to scan (specify one or more)')
    parser.add_argument('--output', type=str, default='eigenvalues.csv',
                        help='Output CSV filename')
    args = parser.parse_args()
    
    all_results = []
    for base_dir in args.base_dir:
        process_directory(base_dir, all_results)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        print(f"\n{'='*60}")
        print(f"Saved {len(df)} records to {args.output}")
        print(f"{'='*60}")
        
        # Summary
        print("\n=== lambda_theta by model and pi0 ===")
        summary = df.groupby(['model_type', 'pi0']).agg({
            'lambda_theta': ['mean', 'std', 'count'],
            'lambda_min': ['mean', 'std'],
            'condition_number': ['mean', 'max']
        }).round(2)
        print(summary)
    else:
        print("No results extracted!")


if __name__ == '__main__':
    main()
