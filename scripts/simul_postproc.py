#!/usr/bin/env python3
"""
simul_postproc_v2.py - Two-table analysis with 95% CIs and ARI computation

Table 1: Consolidated (3 Models x 4 Worlds x K=2,3,4)
Table 2: Ablation (TWEEDIE/HURDLE variants vs baselines)

Usage:
    python simul_postproc_v2.py --root_dir "./march03_simul_full" --out_dir "./results" --true_states_dir "./march02_finalsim/data"
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import warnings
import re
from scipy.stats import bootstrap
from sklearn.metrics import adjusted_rand_score
import sys
warnings.filterwarnings('ignore')

# Configuration
MODELS = ['BEMMAOR', 'TWEEDIE', 'HURDLE']
WORLDS = ['Breeze', 'Cliff', 'Fog', 'Harbor']
K_VALUES = [2, 3, 4]
TRUE_STATES_N = 1000  # Original N in .npy files
TRUE_STATES_T = 52    # Original T in .npy files

# Metric groups for 95% CIs
FIT_METRICS = ['log_evidence']
PRED_METRICS = ['oos_rmse', 'oos_mae', 'clv_ratio']
STATE_METRICS = ['ari']  # Computed from idata + true states


def boot_ci(data, stat='median', n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap confidence intervals."""
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return {'pt': np.nan, 'lo': np.nan, 'hi': np.nan, 'n': n}
    if np.allclose(data, data[0]):
        return {'pt': data[0], 'lo': data[0], 'hi': data[0], 'n': n}

    func = np.median if stat == 'median' else np.mean
    pt = func(data)
    method = 'percentile' if n < 5 else 'BCa'

    try:
        res = bootstrap((data,), func, n_resamples=n_boot,
                       confidence_level=1-alpha, method=method, random_state=seed)
        return {'pt': pt, 'lo': res.confidence_interval.low, 
                'hi': res.confidence_interval.high, 'n': n}
    except:
        rng = np.random.default_rng(seed)
        bs = [func(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)]
        return {'pt': pt, 'lo': np.percentile(bs, alpha/2*100),
                'hi': np.percentile(bs, (1-alpha/2)*100), 'n': n}


def parse_fname(fname):
    """Parse PKL filename."""
    m = re.search(r'smc_K(\d+)_(.+?)_N(\d+)_T(\d+)_D(\d+)\.pkl', fname)
    if not m:
        return None
    K, model_variant, N, T, D = m.groups()

    mv_upper = model_variant.upper()
    if 'BEMMAOR' in mv_upper:
        model, variant = 'BEMMAOR', 'baseline'
    elif 'TWEEDIE' in mv_upper:
        model = 'TWEEDIE'
        if 'GAM' in mv_upper and 'STATEP' in mv_upper:
            variant = 'GAM_statep'
        elif 'GLM' in mv_upper and 'STATEP' in mv_upper:
            variant = 'GLM_statep'
        elif 'GAM' in mv_upper:
            variant = 'GAM_fixedp'
        else:
            variant = 'GLM_fixedp'
    elif 'GAM' in mv_upper:
        model, variant = 'HURDLE', 'GAM'
    elif 'GLM' in mv_upper:
        model, variant = 'HURDLE', 'GLM'
    else:
        model, variant = 'UNKNOWN', 'unknown'

    return {
        'K': int(K), 'model': model, 'variant': variant,
        'N': int(N), 'T': int(T), 'D': int(D), 'filename': fname
    }


def scan_pkls(root):
    """Scan for PKL files."""
    files = []
    for p in Path(root).rglob('*.pkl'):
        if p.name.startswith('.') or p.name.startswith('~'):
            continue
        md = parse_fname(p.name)
        if md:
            md['full_path'] = str(p)
            for part in p.parts:
                if part.lower() in [w.lower() for w in WORLDS]:
                    md['world'] = part.capitalize()
                    break
            files.append(md)
    return pd.DataFrame(files)


def load_true_states(world, target_N, target_T, true_states_dir):
    """
    Load true states from N=1000, T=52 and subsample to target_N, target_T.
    Uses fixed seed for reproducibility.
    """
    npy_path = Path(true_states_dir) / f"true_states_{world}_N{TRUE_STATES_N}_T{TRUE_STATES_T}.npy"

    if not npy_path.exists():
        print(f"    Warning: True states not found at {npy_path}")
        return None

    try:
        true_states_full = np.load(npy_path)
        # true_states_full shape: (1000, 52)

        # Subsample customers (N)
        rng = np.random.default_rng(42)
        customer_idx = rng.choice(TRUE_STATES_N, target_N, replace=False)
        true_states = true_states_full[customer_idx, :]

        # Truncate time periods (T)
        true_states = true_states[:, :target_T]

        return true_states

    except Exception as e:
        print(f"    Warning: Could not load true states for {world}: {e}")
        return None


def compute_ari(idata, true_states, K):
    """Compute ARI from estimated states vs true states."""
    try:
        if 'alpha_filtered' not in idata.posterior:
            return np.nan

        # Get estimated states (mean across chains/draws)
        # alpha_filtered shape: (chain, draw, customer, time, K)
        alpha = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values
        est_states = np.argmax(alpha, axis=-1)  # (customer, time)

        # Flatten both
        true_flat = true_states.flatten()
        est_flat = est_states.flatten()

        # Ensure same length
        min_len = min(len(true_flat), len(est_flat))
        true_flat = true_flat[:min_len]
        est_flat = est_flat[:min_len]

        # Remove invalid entries
        mask = (true_flat >= 0) & (true_flat < K)
        if mask.sum() == 0:
            return np.nan

        true_flat = true_flat[mask]
        est_flat = est_flat[mask]

        # Compute ARI
        ari = adjusted_rand_score(true_flat, est_flat)
        return ari

    except Exception as e:
        return np.nan


def extract_metrics(pkl_path, world, true_states_dir=None):
    """Extract all metrics including computed ARI."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        res = data.get('res', {}) if isinstance(data, dict) else {}
        idata = data.get('idata') if isinstance(data, dict) else None

        metrics = {
            'log_evidence': res.get('log_evidence', np.nan),
            'oos_rmse': res.get('oos_rmse', np.nan),
            'oos_mae': res.get('oos_mae', np.nan),
            'clv_ratio': res.get('clv_ratio', np.nan),
            'clv_total': res.get('clv_total', np.nan),
            'time_min': res.get('time_min', np.nan),
            'ari': np.nan,
            'ok': True
        }

        # Compute ARI if true states available
        if true_states_dir and idata is not None:
            N = res.get('N', 500)
            T = res.get('T', 41)
            K = res.get('K', 2)

            true_states = load_true_states(world, N, T, true_states_dir)
            if true_states is not None:
                metrics['ari'] = compute_ari(idata, true_states, K)

        return metrics
    except Exception as e:
        return {k: np.nan for k in ['log_evidence', 'oos_rmse', 'oos_mae', 'clv_ratio', 'clv_total', 'time_min', 'ari']}


def format_ci(pt, lo, hi, fmt):
    """Format value with CI."""
    if np.isnan(pt):
        return 'N/A'
    return f"{pt:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def create_table1_consolidated(df):
    """Table 1: 3 Models x 4 Worlds x K=2,3,4 with 95% CIs."""
    print("\n" + "="*80)
    print("TABLE 1: CONSOLIDATED (3 Models x 4 Worlds x K=2,3,4)")
    print("="*80)

    # Filter to baseline variants only
    baseline_df = df[
        ((df['model'] == 'BEMMAOR') & (df['variant'] == 'baseline')) |
        ((df['model'] == 'HURDLE') & (df['variant'] == 'GLM')) |
        ((df['model'] == 'TWEEDIE') & (df['variant'] == 'GLM_fixedp'))
    ].copy()

    results = []
    for model in MODELS:
        for world in WORLDS:
            for K in K_VALUES:
                subset = baseline_df[
                    (baseline_df['model'] == model) & 
                    (baseline_df['world'] == world) & 
                    (baseline_df['K'] == K)
                ]

                if len(subset) == 0:
                    continue

                row = {
                    'Model': model,
                    'World': world,
                    'K': K,
                    'N': len(subset)
                }

                # Fit metrics (LogEv)
                for metric in FIT_METRICS:
                    vals = subset[metric].dropna()
                    if len(vals) > 0:
                        b = boot_ci(vals.values, stat='median')
                        row[metric] = format_ci(b['pt'], b['lo'], b['hi'], '.1f')
                    else:
                        row[metric] = 'N/A'

                # Pred metrics (median for skewed)
                for metric in PRED_METRICS:
                    vals = subset[metric].dropna()
                    if len(vals) > 0:
                        b = boot_ci(vals.values, stat='median')
                        fmt = '.2f' if metric != 'clv_ratio' else '.1f'
                        row[metric] = format_ci(b['pt'], b['lo'], b['hi'], fmt)
                    else:
                        row[metric] = 'N/A'

                # State metrics (mean for bounded)
                for metric in STATE_METRICS:
                    vals = subset[metric].dropna()
                    if len(vals) > 0:
                        b = boot_ci(vals.values, stat='mean')
                        row[metric] = format_ci(b['pt'], b['lo'], b['hi'], '.3f')
                    else:
                        row[metric] = 'N/A'

                results.append(row)

    table = pd.DataFrame(results)

    # Sort
    table['Model_ord'] = table['Model'].map({m: i for i, m in enumerate(MODELS)})
    table['World_ord'] = table['World'].map({w: i for i, w in enumerate(WORLDS)})
    table = table.sort_values(['Model_ord', 'World_ord', 'K']).drop(columns=['Model_ord', 'World_ord'])

    return table


def create_table2_ablation(df):
    """Table 2: Ablation (variants vs baselines)."""
    print("\n" + "="*80)
    print("TABLE 2: ABLATION (Variants vs Baselines)")
    print("="*80)

    # All non-baseline variants
    ablation_df = df[
 <response clipped><NOTE>Result is longer than **10000 characters**, will be **truncated**.</NOTE>
