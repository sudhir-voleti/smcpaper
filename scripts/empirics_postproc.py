#!/usr/bin/env python3
"""
empirics_postproc_v2.py - Enhanced post-processing with absolute CLV, CIs, 
median ratios, and state diagnostics. UCI priority. T=53 excluded.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - UCI PRIORITY
# =============================================================================

BASE_DIR = Path("/Users/sudhirvoleti/research related/SMC paper Feb2026")
OUTPUT_DIR = BASE_DIR / "empirics_results"

# UCI ONLY - skip CDNOW for now
EMPIRICS_DIRS = {
    'uci': BASE_DIR / "empirics_uci",
    # 'cdnow': BASE_DIR / "empirics_cdnow"  # Deferred - 1990s data
}

# Skip T=53 files (legacy runs)
EXCLUDE_PATTERNS = ['_T53_', '_T54_', '_T100_']  # Add others as needed

# =============================================================================
# ENHANCED METRICS EXTRACTION
# =============================================================================

def extract_clv_with_ci(idata, res, K):
    """
    Extract absolute CLV by state with 95% CIs.
    Look for clv_proxy in idata posterior or res dict.
    """
    clv_stats = {}

    # Try to get from idata posterior first (preferred)
    if idata is not None and 'clv_proxy' in idata.posterior:
        clv = idata.posterior['clv_proxy']  # (chain, draw, N, T) or similar

        # Average over time if needed
        if clv.ndim >= 4:
            clv = clv.mean(dim='time') if 'time' in clv.dims else clv.mean(axis=-1)

        # Now should be (chain, draw, N) or (chain, draw, N, K)
        if 'state' in clv.dims or clv.shape[-1] == K:
            # CLV by state
            for k in range(K):
                state_clv = clv.sel(state=k) if 'state' in clv.dims else clv[..., k]
                # Flatten chains/draws
                flat_clv = state_clv.values.flatten()
                flat_clv = flat_clv[~np.isnan(flat_clv)]

                if len(flat_clv) > 0:
                    clv_stats[f'clv_state{k}_mean'] = np.mean(flat_clv)
                    clv_stats[f'clv_state{k}_median'] = np.median(flat_clv)
                    clv_stats[f'clv_state{k}_ci_low'] = np.percentile(flat_clv, 2.5)
                    clv_stats[f'clv_state{k}_ci_high'] = np.percentile(flat_clv, 97.5)
                    clv_stats[f'clv_state{k}_std'] = np.std(flat_clv)
                else:
                    clv_stats[f'clv_state{k}_mean'] = np.nan
                    clv_stats[f'clv_state{k}_median'] = np.nan
                    clv_stats[f'clv_state{k}_ci_low'] = np.nan
                    clv_stats[f'clv_state{k}_ci_high'] = np.nan
                    clv_stats[f'clv_state{k}_std'] = np.nan

    # Fallback: try to reconstruct from res dict
    elif 'clv_by_state' in res and res['clv_by_state']:
        clv_list = res['clv_by_state']
        for k, val in enumerate(clv_list):
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                arr = np.array(val).flatten()
                arr = arr[~np.isnan(arr)]
                if len(arr) > 0:
                    clv_stats[f'clv_state{k}_mean'] = np.mean(arr)
                    clv_stats[f'clv_state{k}_median'] = np.median(arr)
                    clv_stats[f'clv_state{k}_ci_low'] = np.percentile(arr, 2.5)
                    clv_stats[f'clv_state{k}_ci_high'] = np.percentile(arr, 97.5)
                    clv_stats[f'clv_state{k}_std'] = np.std(arr)
            else:
                clv_stats[f'clv_state{k}_mean'] = float(val) if not np.isnan(val) else np.nan

    return clv_stats

def compute_median_clv_ratio(clv_stats, K):
    """
    Compute CLV ratio as max/median instead of max/min to avoid Black Hole distortion.
    """
    medians = []
    for k in range(K):
        key = f'clv_state{k}_median'
        if key in clv_stats and not np.isnan(clv_stats[key]):
            medians.append((k, clv_stats[key]))

    if len(medians) < 2:
        return np.nan

    # Sort by median CLV
    medians.sort(key=lambda x: x[1])

    # Ratio of highest to median (not lowest)
    max_val = medians[-1][1]

    # If K=2, ratio is max/min (only 2 states)
    # If K>=3, ratio is max/middle (median of all states)
    if len(medians) == 2:
        min_val = medians[0][1]
        ratio = max_val / (min_val + 1e-10)  # Avoid div by zero
    else:
        mid_idx = len(medians) // 2
        median_val = medians[mid_idx][1]
        ratio = max_val / (median_val + 1e-10)

    return ratio

def extract_state_occupancy(idata, K):
    """
    Extract state occupancy rates from alpha_filtered or Viterbi.
    """
    occupancy = {}

    if idata is None:
        return occupancy

    # Try alpha_filtered (smoothed state probabilities)
    if 'alpha_filtered' in idata.posterior:
        alpha = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values
        # alpha shape: (N, T, K) or (N, K)
        if alpha.ndim == 3:
            # Average over time
            state_probs = alpha.mean(axis=1)  # (N, K)
        else:
            state_probs = alpha

        # Population occupancy
        pop_occ = state_probs.mean(axis=0)  # (K,)
        for k in range(K):
            occupancy[f'state{k}_occupancy'] = pop_occ[k] if k < len(pop_occ) else 0.0

        # Check for state collapse (any state < 5% occupancy)
        occupancy['min_occupancy'] = pop_occ.min()
        occupancy['max_occupancy'] = pop_occ.max()
        occupancy['occupancy_ratio'] = pop_occ.max() / (pop_occ.min() + 1e-10)

    # Also try Viterbi if available
    if 'z_viterbi' in idata.posterior:
        viterbi = idata.posterior['z_viterbi'].values
        # Count state assignments
        unique, counts = np.unique(viterbi, return_counts=True)
        total = counts.sum()
        for k in range(K):
            idx = np.where(unique == k)[0]
            if len(idx) > 0:
                occupancy[f'state{k}_viterbi_pct'] = counts[idx[0]] / total
            else:
                occupancy[f'state{k}_viterbi_pct'] = 0.0

    return occupancy

def extract_gamma_diagnostics(idata, K):
    """
    Extract Gamma transition matrix diagnostics.
    """
    gamma_diag = {}

    if idata is None or 'Gamma' not in idata.posterior:
        return gamma_diag

    Gamma = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values

    if Gamma.shape != (K, K):
        return gamma_diag

    # Diagonal elements (persistence)
    for k in range(K):
        gamma_diag[f'gamma_{k}{k}_persistence'] = Gamma[k, k]

    # Off-diagonal (switching)
    for i in range(K):
        for j in range(K):
            if i != j:
                gamma_diag[f'gamma_{i}{j}_switch'] = Gamma[i, j]

    # Implied dwell times
    for k in range(K):
        p_stay = Gamma[k, k]
        dwell = 1 / (1 - p_stay + 1e-10)
        gamma_diag[f'state{k}_dwell_time'] = dwell

    # Stationary distribution
    try:
        eigvals, eigvecs = np.linalg.eig(Gamma.T)
        stationary = np.real(eigvecs[:, np.isclose(eigvals, 1, atol=1e-8)])
        if stationary.size > 0:
            stationary = stationary[:, 0] / stationary[:, 0].sum()
            for k in range(K):
                gamma_diag[f'state{k}_stationary'] = stationary[k]
    except:
        pass

    return gamma_diag

def extract_oos_metrics(res):
    """
    Extract OOS metrics if available.
    """
    oos = {}

    # Standard OOS keys
    oos_keys = ['oos_rmse', 'oos_mae', 'oos_mape', 'oos_r2']
    for key in oos_keys:
        if key in res:
            oos[key] = res[key]

    # Check for nested OOS dict
    if 'oos' in res and isinstance(res['oos'], dict):
        for key, val in res['oos'].items():
            oos[f'oos_{key}'] = val

    return oos

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_single_pkl(pkl_path):
    """Process one PKL file with enhanced diagnostics."""

    # Skip T=53 and other legacy files
    for pattern in EXCLUDE_PATTERNS:
        if pattern in pkl_path.name:
            return {'skipped': True, 'reason': f'Legacy pattern: {pattern}'}

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        res = data.get('res', {})
        idata = data.get('idata', None)

        # Parse filename for K
        K = 2  # default
        if '_K2_' in pkl_path.name or 'K2' in pkl_path.name:
            K = 2
        elif '_K3_' in pkl_path.name or 'K3' in pkl_path.name:
            K = 3
        elif '_K4_' in pkl_path.name or 'K4' in pkl_path.name:
            K = 4

        record = {
            'pkl_path': str(pkl_path),
            'pkl_name': pkl_path.name,
            'K': K,
            'loaded': True
        }

        # Extract model type
        if 'bemmaor' in pkl_path.name.lower():
            record['model'] = 'BEMMAOR'
        elif 'hurdle' in pkl_path.name.lower() or '_GAM_' in pkl_path.name or '_GLM_' in pkl_path.name:
            record['model'] = 'Hurdle'
        elif 'tweedie' in pkl_path.name.lower() or 'uci' in pkl_path.name.lower():
            record['model'] = 'Tweedie'
        else:
            record['model'] = 'Unknown'

        # Basic metrics
        record['log_evidence'] = res.get('log_evidence', np.nan)
        record['elapsed_min'] = res.get('elapsed_min', np.nan)
        record['ess_min'] = res.get('ess_min', np.nan)

        # Enhanced CLV with CIs
        clv_stats = extract_clv_with_ci(idata, res, K)
        record.update(clv_stats)

        # Median-based CLV ratio (not min-based)
        record['clv_ratio_median'] = compute_median_clv_ratio(clv_stats, K)

        # State occupancy
        occupancy = extract_state_occupancy(idata, K)
        record.update(occupancy)

        # Gamma diagnostics
        gamma_diag = extract_gamma_diagnostics(idata, K)
        record.update(gamma_diag)

        # OOS metrics
        oos = extract_oos_metrics(res)
        record.update(oos)

        # Whale metrics (if available)
        record['whale_f1'] = res.get('whale_f1', np.nan)
        record['whale_precision'] = res.get('whale_precision', np.nan)
        record['whale_recall'] = res.get('whale_recall', np.nan)

        # PPC metrics
        record['ppc_pass'] = res.get('ppc_pass', False)
        record['ppc_zero_obs'] = res.get('ppc_zero_obs', np.nan)
        record['ppc_zero_sim'] = res.get('ppc_zero_sim_mean', np.nan)

        return record

    except Exception as e:
        return {
            'pkl_path': str(pkl_path),
            'pkl_name': pkl_path.name,
            'error': str(e),
            'loaded': False
        }

def main():
    print("=" * 80)
    print("EMPIRICS POST-PROCESSING V2 - UCI PRIORITY")
    print("Enhanced: Absolute CLV + CIs, Median Ratios, State Occupancy")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Excluding legacy patterns: {EXCLUDE_PATTERNS}")
    print()

    all_results = []

    # Process UCI only
    dataset = 'uci'
    print(f"\n{'='*80}")
    print(f"PROCESSING {dataset.upper()} (PRIORITY)")
    print(f"{'='*80}")

    base_path = EMPIRICS_DIRS[dataset]
    if not base_path.exists():
        print(f"ERROR: {base_path} does not exist")
        return

    pkls = list(base_path.rglob("*.pkl"))
    # Filter out N=4338 (full sample) and legacy files
    pkls = [p for p in pkls if "N4338" not in p.name]
    pkls = [p for p in pkls if not any(pat in p.name for pat in EXCLUDE_PATTERNS)]
    pkls = sorted(pkls)

    print(f"Found {len(pkls)} valid PKL files (N=500, T=42 only)")

    for i, pkl_path in enumerate(pkls, 1):
        print(f"\n[{i}/{len(pkls)}] {pkl_path.name}")
        result = process_single_pkl(pkl_path)
        result['dataset'] = dataset
        all_results.append(result)

        if result.get('skipped'):
            print(f"  SKIPPED: {result['reason']}")
            continue

        if result.get('error'):
            print(f"  ERROR: {result['error'][:60]}")
            continue

        if not result.get('loaded'):
            print(f"  FAILED TO LOAD")
            continue

        # Print summary
        print(f"  Model: {result.get('model', '?')}, K={result.get('K', '?')}")
        print(f"  Log-Ev: {result.get('log_evidence', 'N/A'):.2f}" if not np.isnan(result.get('log_evidence', np.nan)) else "  Log-Ev: N/A")

        # CLV summary
        clv_medians = [result.get(f'clv_state{k}_median', np.nan) for k in range(result.get('K', 2))]
        print(f"  CLV by state (median): {[f'{c:.2f}' if not np.isnan(c) else 'N/A' for c in clv_medians]}")
        print(f"  CLV Ratio (max/median): {result.get('clv_ratio_median', 'N/A'):.2f}" if not np.isnan(result.get('clv_ratio_median', np.nan)) else "  CLV Ratio: N/A")

        # State occupancy
        occs = [result.get(f'state{k}_occupancy', np.nan) for k in range(result.get('K', 2))]
        if not all(np.isnan(o) for o in occs):
            print(f"  State occupancy: {[f'{o:.1%}' if not np.isnan(o) else 'N/A' for o in occs]}")

        # Whale F1
        if not np.isnan(result.get('whale_f1', np.nan)):
            print(f"  Whale F1: {result['whale_f1']:.3f}")

        # OOS
        if not np.isnan(result.get('oos_rmse', np.nan)):
            print(f"  OOS RMSE: {result['oos_rmse']:.2f}")

    # Build DataFrame
    print(f"\n{'='*80}")
    print("BUILDING ENHANCED COMPARISON TABLE")
    print(f"{'='*80}")

    df = pd.DataFrame(all_results)

    # Filter out skipped/error rows for display
    valid_df = df[df['loaded'] == True].copy()

    print(f"Valid results: {len(valid_df)} / {len(df)}")

    # Select key columns for display
    display_cols = [
        'pkl_name', 'model', 'K', 'log_evidence', 
        'clv_state0_median', 'clv_state1_median', 'clv_ratio_median',
        'state0_occupancy', 'state1_occupancy',
        'whale_f1', 'oos_rmse', 'elapsed_min'
    ]

    # Only include columns that exist
    available_cols = [c for c in display_cols if c in valid_df.columns]
    display_df = valid_df[available_cols].copy()

    # Format
    numeric_cols = ['log_evidence', 'clv_state0_median', 'clv_state1_median', 
                   'clv_ratio_median', 'whale_f1', 'oos_rmse', 'elapsed_min']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

    # Sort
    display_df = display_df.sort_values(['model', 'K', 'log_evidence'], ascending=[True, True, False])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full results
    full_path = OUTPUT_DIR / "empirics_v2_full_results.csv"
    df.to_csv(full_path, index=False)
    print(f"\nSaved full results: {full_path}")

    # Display table
    display_path = OUTPUT_DIR / "empirics_v2_comparison.csv"
    display_df.to_csv(display_path, index=False)
    print(f"Saved comparison table: {display_path}")

    # Print to console
    print(f"\n{'='*80}")
    print("ENHANCED COMPARISON TABLE (UCI Only)")
    print(f"{'='*80}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(display_df.to_string(index=False))

    # Summary by model
    print(f"\n{'='*80}")
    print("SUMMARY BY MODEL")
    print(f"{'='*80}")

    for model in valid_df['model'].unique():
        if pd.isna(model):
            continue
        model_df = valid_df[valid_df['model'] == model]
        print(f"\n{model}:")
        print(f"  Runs: {len(model_df)}")
        print(f"  K values: {sorted(model_df['K'].unique())}")

        if 'log_evidence' in model_df.columns:
            best = model_df.loc[model_df['log_evidence'].idxmax()]
            print(f"  Best log-ev: K={best['K']} ({best['log_evidence']:.2f})")

        if 'clv_ratio_median' in model_df.columns:
            ratios = model_df['clv_ratio_median'].dropna()
            if len(ratios) > 0:
                print(f"  CLV ratio range: {ratios.min():.2f}x - {ratios.max():.2f}x")

        if 'whale_f1' in model_df.columns:
            f1s = model_df['whale_f1'].dropna()
            if len(f1s) > 0:
                print(f"  Whale F1 range: {f1s.min():.3f} - {f1s.max():.3f}")

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
