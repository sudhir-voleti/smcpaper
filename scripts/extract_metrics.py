#!/usr/bin/env python3
"""
extract_metrics.py
==================
Unified metrics extraction for JRSS-C/JASA simulation study.
Extracts ARI, raw accuracy, static_pct, and other diagnostics from
BEMMAOR and Hurdle PKL files, using DGP files for ground-truth states.

Usage:
    python extract_metrics.py \\
        --input_csv merged_24april.csv \\
        --output_csv metrics_complete.csv \\
        --base_dir /Users/sudhirvoleti/jrssc_april

Or as a module:
    from extract_metrics import extract_all_metrics
    df = extract_all_metrics("merged_24april.csv", "/Users/sudhirvoleti/jrssc_april")
    df.to_csv("out.csv", index=False)

Author: Sudhir Voleti
Date: 2026-04-24
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from scipy import stats


# =============================================================================
# CORE EXTRACTION FUNCTIONS
# =============================================================================

def extract_predicted_states(pkl_path):
    """
    Extract predicted state sequences from a model PKL file.
    
    Supports:
      - BEMMAOR: alpha_filtered (N, T, K) -> argmax -> (N, T)
      - Hurdle:  idata.posterior.viterbi -> mode over chains/draws -> (N, T)
    
    Parameters
    ----------
    pkl_path : str or Path
        Path to the model PKL file.
    
    Returns
    -------
    np.ndarray or None
        Predicted states array of shape (N, T), or None if extraction fails.
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot load PKL {pkl_path}: {e}")
        return None

    # --- Strategy 1: BEMMAOR via alpha_filtered ---
    if 'alpha_filtered' in data and data['alpha_filtered'] is not None:
        alpha = data['alpha_filtered']
        if hasattr(alpha, 'shape') and len(alpha.shape) == 3:
            return alpha.argmax(axis=-1)

    # --- Strategy 2: Hurdle via idata.posterior.viterbi ---
    if 'idata' in data and data['idata'] is not None:
        idata = data['idata']
        if hasattr(idata, 'posterior'):
            post = idata.posterior
            if 'viterbi' in post:
                v = post['viterbi'].values
                if len(v.shape) >= 3:
                    mode_res = stats.mode(v, axis=(0, 1), keepdims=False)
                    mode_arr = mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0]
                    if hasattr(mode_arr, 'shape') and len(mode_arr.shape) > 2:
                        mode_arr = np.squeeze(mode_arr)
                    return mode_arr

    # --- Strategy 3: Fallback — any integer array in posterior ---
    if 'idata' in data and data['idata'] is not None:
        idata = data['idata']
        if hasattr(idata, 'posterior'):
            for var_name in list(idata.posterior.data_vars):
                arr = idata.posterior[var_name].values
                if 'int' in str(arr.dtype) and len(arr.shape) >= 3:
                    mode_res = stats.mode(arr, axis=(0, 1), keepdims=False)
                    mode_arr = mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0]
                    if hasattr(mode_arr, 'shape') and len(mode_arr.shape) == 2:
                        return mode_arr

    print(f"  [WARN] No predicted states found in {Path(pkl_path).name}")
    return None


def load_true_states(pkl_path):
    """
    Load ground-truth latent states (Z) from the corresponding DGP PKL file.
    
    Parameters
    ----------
    pkl_path : str or Path
        Path to the model PKL file.
    
    Returns
    -------
    np.ndarray or None
        True states array of shape (N, T), or None if not found.
    """
    path = Path(pkl_path)
    rep_folder = path.parent

    dgp_files = list(rep_folder.glob("dgp_*.pkl"))
    if dgp_files:
        try:
            with open(dgp_files[0], 'rb') as f:
                dgp = pickle.load(f)
            if 'Z' in dgp:
                return dgp['Z']
            if 'true_states' in dgp:
                return dgp['true_states']
        except Exception as e:
            print(f"  [WARN] Cannot load DGP {dgp_files[0]}: {e}")

    parent_dgp = list(rep_folder.parent.glob("dgp_*.pkl")) if rep_folder.parent != rep_folder else []
    if parent_dgp:
        try:
            with open(parent_dgp[0], 'rb') as f:
                dgp = pickle.load(f)
            return dgp.get('Z', dgp.get('true_states', None))
        except Exception as e:
            print(f"  [WARN] Cannot load parent DGP {parent_dgp[0]}: {e}")

    return None


def compute_state_metrics(pred_states, true_states):
    """
    Compute accuracy, static_pct, and ARI from predicted and true states.
    
    Parameters
    ----------
    pred_states : np.ndarray, shape (N, T)
    true_states : np.ndarray, shape (N, T)
    
    Returns
    -------
    dict
        {'raw_accuracy': float, 'static_pct': float, 'ari': float, 'N': int}
    """
    min_N = min(pred_states.shape[0], true_states.shape[0])
    pred = pred_states[:min_N]
    true = true_states[:min_N]

    raw_accuracy = float(np.mean(pred == true))

    if pred.shape[1] > 1:
        n_switches = np.sum(np.diff(pred, axis=1) != 0, axis=1)
        static_pct = float(np.mean(n_switches == 0)) * 100
    else:
        static_pct = 100.0

    ari = float(adjusted_rand_score(true.flatten(), pred.flatten()))

    return {
        'raw_accuracy': raw_accuracy,
        'static_pct': static_pct,
        'ari': ari,
        'N': min_N
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def extract_all_metrics(input_csv, base_dir, pkl_path_col='pkl_path',
                        missing_only=True, verbose=True):
    """
    Extract or re-extract state metrics for all rows in a CSV.
    
    Parameters
    ----------
    input_csv : str
        Path to CSV containing at least a pkl_path column.
    base_dir : str
        Base directory to prepend to relative pkl_paths.
    pkl_path_col : str, default 'pkl_path'
        Name of the column holding PKL paths.
    missing_only : bool, default True
        If True, only compute for rows where 'ari' or 'overall_accuracy' is NaN.
        If False, recompute for ALL rows.
    verbose : bool, default True
        Print progress every 50 rows.
    
    Returns
    -------
    pd.DataFrame
        Copy of input with 'ari', 'overall_accuracy', 'static_pct' filled in.
    """
    df = pd.read_csv(input_csv)
    df[pkl_path_col] = df[pkl_path_col].astype(str).str.strip()

    if missing_only and 'ari' in df.columns:
        mask = df['ari'].isna()
        if 'overall_accuracy' in df.columns:
            mask = mask | df['overall_accuracy'].isna()
    else:
        mask = pd.Series([True] * len(df))

    to_process = df[mask].index
    n_total = len(to_process)
    print(f"Processing {n_total}/{len(df)} rows...")

    for i, idx in enumerate(to_process):
        row = df.loc[idx]
        rel_path = row[pkl_path_col]
        full_path = str(Path(base_dir) / rel_path)

        if verbose and i % 50 == 0:
            print(f"  [{i+1}/{n_total}] {rel_path[:60]}...")

        pred = extract_predicted_states(full_path)
        true = load_true_states(full_path)

        if pred is not None and true is not None:
            metrics = compute_state_metrics(pred, true)
            df.at[idx, 'overall_accuracy'] = metrics['raw_accuracy']
            df.at[idx, 'static_pct'] = metrics['static_pct']
            df.at[idx, 'ari'] = metrics['ari']
        else:
            if verbose:
                print(f"  [SKIP] pred={pred is not None}, true={true is not None}")

    print(f"\nFinal coverage:")
    for col in ['ari', 'overall_accuracy', 'static_pct']:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col}: {n}/{len(df)} ({100*n/len(df):.1f}%)")

    return df


def parse_path_features(pkl_path):
    """
    Extract metadata (N, T, pi0, psi, rho, model_type) from PKL path string.
    
    Parameters
    ----------
    pkl_path : str
        Relative or absolute path string.
    
    Returns
    -------
    dict
        Dictionary of parsed features.
    """
    import re
    features = {}
    path_str = str(pkl_path)

    patterns = {
        'N': r'N(\d+)',
        'T': r'T(\d+)',
        'pi0': r'pi0[_]?([0-9.]+)',
        'psi': r'psi[_]?(\d+)',
        'rho': r'rho[_]?([0-9.]+)',
        'rep': r'rep[_]?(\d+)',
        'K': r'K(\d+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, path_str, re.IGNORECASE)
        if m:
            val = m.group(1)
            features[key] = float(val) if '.' in val else int(val)

    features['model_type'] = (
        'BEMMAOR' if 'BEMMAOR' in path_str or 'bemmaor' in path_str
        else 'Hurdle'
    )

    return features


def enrich_with_path_features(df, pkl_path_col='pkl_path'):
    """
    Add N, T, pi0, psi, rho, model_type columns parsed from pkl_path.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pkl_path column.
    pkl_path_col : str, default 'pkl_path'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns added.
    """
    features_list = df[pkl_path_col].apply(parse_path_features).tolist()
    features_df = pd.DataFrame(features_list)

    for col in features_df.columns:
        if col not in df.columns:
            df[col] = features_df[col].values
        else:
            mask = df[col].isna()
            if mask.sum() > 0:
                df.loc[mask, col] = features_df.loc[mask, col].values

    return df


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract state metrics (ARI, accuracy, static_pct) from simulation PKLs"
    )
    parser.add_argument('--input_csv', required=True,
                        help='Input CSV with pkl_path column')
    parser.add_argument('--output_csv', required=True,
                        help='Output CSV path')
    parser.add_argument('--base_dir', required=True,
                        help='Base directory to prepend to relative pkl_paths')
    parser.add_argument('--recompute_all', action='store_true',
                        help='Recompute ALL rows, not just missing ones')
    parser.add_argument('--no_enrich', action='store_true',
                        help='Skip path-feature enrichment (N, pi0, psi, etc.)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress prints')

    args = parser.parse_args()

    df = extract_all_metrics(
        input_csv=args.input_csv,
        base_dir=args.base_dir,
        missing_only=not args.recompute_all,
        verbose=not args.quiet
    )

    if not args.no_enrich:
        df = enrich_with_path_features(df)

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv} ({len(df)} rows)")


if __name__ == '__main__':
    main()
