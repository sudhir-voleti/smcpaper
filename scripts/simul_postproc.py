#!/usr/bin/env python3
"""
simul_postproc.py
=================
Consolidated post-processing for HMM simulation results.
Handles Hurdle, Bemmaor, and Tweedie models with full metrics and 95% CIs.

Usage:
    python simul_postproc.py --root_dir "/path/to/march03_simul_full" --out_dir "./results"

Output:
    - results/master_comparison.csv (main table with CIs)
    - results/ablation_comparison.csv (TWEEDIE variants)
    - results/master_comparison.tex (LaTeX table)
    - results/pkl_inventory.csv (all PKLs found)
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
import re
from scipy.stats import bootstrap

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

METRICS_CONFIG = {
    'ari': {'stat': 'mean', 'format': '.3f', 'label': 'ARI'},
    'clv_ratio': {'stat': 'median', 'format': '.1f', 'label': 'CLV Ratio'},
    'whale_f1': {'stat': 'mean', 'format': '.3f', 'label': 'Whale F1'},
    'log_evidence': {'stat': 'median', 'format': '.1f', 'label': 'Log-Evidence'},
    'oos_rmse': {'stat': 'median', 'format': '.3f', 'label': 'OOS RMSE'},
    'state_accuracy': {'stat': 'mean', 'format': '.3f', 'label': 'State Accuracy'}
}

MODEL_ORDER = ['BEMMAOR', 'TWEEDIE', 'HURDLE']
WORLD_ORDER = ['Breeze', 'Cliff', 'Fog', 'Harbor']

# =============================================================================
# BOOTSTRAP UTILITIES
# =============================================================================

def bootstrap_ci(data, statistic='median', n_iterations=10000, alpha=0.05, random_state=42):
    """
    Calculate bootstrap confidence intervals with robust defaults.

    Features:
    - Uses median for skewed data (CLV ratio)
    - Handles constant data gracefully
    - Falls back to percentile when BCa unstable (n<5)
    """
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 2:
        return {'point': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'se': np.nan, 'n': n, 'method': 'insufficient'}

    # Constant data (e.g., HURDLE ARI = 0)
    if np.allclose(data, data[0], rtol=1e-10):
        return {'point': data[0], 'ci_low': data[0], 'ci_high': data[0], 'se': 0.0, 'n': n, 'method': 'constant'}

    stat_func = np.median if statistic == 'median' else np.mean
    point = stat_func(data)

    # Use percentile for small n (unstable BCa)
    method = 'percentile' if n < 5 else 'BCa'

    try:
        res = bootstrap((data,), stat_func, n_resamples=n_iterations,
                       confidence_level=1-alpha, method=method, random_state=random_state)
        return {
            'point': point,
            'ci_low': res.confidence_interval.low,
            'ci_high': res.confidence_interval.high,
            'se': res.standard_error,
            'n': n,
            'method': method
        }
    except:
        # Manual fallback
        rng = np.random.default_rng(random_state)
        boot_stats = [stat_func(rng.choice(data, size=n, replace=True)) for _ in range(n_iterations)]
        return {
            'point': point,
            'ci_low': np.percentile(boot_stats, alpha/2 * 100),
            'ci_high': np.percentile(boot_stats, (1-alpha/2) * 100),
            'se': np.std(boot_stats),
            'n': n,
            'method': 'percentile_fallback'
        }

# =============================================================================
# PKL PARSING
# =============================================================================

def parse_pkl_filename(filename):
    """Extract metadata from PKL filename."""
    pattern = r'smc_K(\d+)_(.+?)_N(\d+)_T(\d+)_D(\d+)\.pkl'
    match = re.search(pattern, filename)

    if not match:
        return None

    K, model_variant, N, T, D = match.groups()

    # Determine model type
    if 'BEMMAOR' in model_variant.upper():
        model = 'BEMMAOR'
        variant = 'baseline'
    elif 'TWEEDIE' in model_variant.upper():
        model = 'TWEEDIE'
        if 'GAM' in model_variant.upper() and 'statep' in model_variant.lower():
            variant = 'GAM_statep'
        elif 'GLM' in model_variant.upper() and 'statep' in model_variant.lower():
            variant = 'GLM_statep'
        elif 'GAM' in model_variant.upper():
            variant = 'GAM_fixedp'
        else:
            variant = 'GLM_fixedp'
    elif 'GAM' in model_variant.upper():
        model = 'HURDLE'
        variant = 'GAM'
    elif 'GLM' in model_variant.upper():
        model = 'HURDLE'
        variant = 'GLM'
    else:
        model = 'UNKNOWN'
        variant = 'unknown'

    return {
        'K': int(K),
        'model': model,
        'variant': variant,
        'N': int(N),
        'T': int(T),
        'D': int(D),
        'filename': filename
    }


def scan_pkl_directory(root_dir):
    """Scan directory for all PKL files and categorize."""
    root_path = Path(root_dir)
    pkl_files = []

    for pkl_file in root_path.rglob('*.pkl'):
        if pkl_file.name.startswith('.') or pkl_file.name.startswith('~'):
            continue

        metadata = parse_pkl_filename(pkl_file.name)
        if metadata:
            metadata['full_path'] = str(pkl_file)
            # Extract world from path
            for part in pkl_file.parts:
                if part.lower() in ['breeze', 'cliff', 'fog', 'harbor']:
                    metadata['world'] = part.capitalize()
                    break
            pkl_files.append(metadata)

    return pd.DataFrame(pkl_files)

# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def extract_metrics_from_pkl(pkl_path):
    """Extract key metrics from a PKL file."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Handle different save formats
        if isinstance(data, dict):
            idata = data.get('idata', data)
            log_ev = data.get('log_evidence', np.nan)
            res = data.get('res', {})
        else:
            idata = data
            log_ev = getattr(data, 'log_evidence', np.nan)
            res = {}

        metrics = {
            'log_evidence': log_ev,
            'ari': res.get('ari', np.nan),
            'state_accuracy': res.get('state_accuracy', np.nan),
            'clv_ratio': res.get('clv_ratio', np.nan),
            'whale_f1': res.get('whale_f1', np.nan),
            'whale_precision': res.get('whale_precision', np.nan),
            'whale_recall': res.get('whale_recall', np.nan),
            'oos_rmse': res.get('oos_rmse', np.nan),
            'oos_mae': res.get('oos_mae', np.nan),
            'success': True,
            'error': None
        }

        # Try to extract from idata if not in res
        try:
            if 'alpha_filtered' in idata.posterior:
                alpha = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values
                # Could compute ARI here if true states available
        except:
            pass

        return metrics

    except Exception as e:
        return {
            'log_evidence': np.nan, 'ari': np.nan, 'state_accuracy': np.nan,
            'clv_ratio': np.nan, 'whale_f1': np.nan, 'whale_precision': np.nan,
            'whale_recall': np.nan, 'oos_rmse': np.nan, 'oos_mae': np.nan,
            'success': False, 'error': str(e)
        }

# =============================================================================
# TABLE GENERATION
# =============================================================================

def create_master_comparison_table(df, metrics_config=METRICS_CONFIG):
    """
    Create master comparison table with bootstrap CIs.
    Aggregates to world-level first to address pseudoreplication.
    """
    results = []

    for model in MODEL_ORDER:
        model_data = df[df['model'] == model]
        if len(model_data) == 0:
            continue

        row = {
            'Model': model,
            'N_Worlds': model_data['world'].nunique(),
            'N_Runs': len(model_data)
        }

        for metric_name, config in metrics_config.items():
            if metric_name not in model_data.columns:
                row[config['label']] = "N/A"
                continue

            # Aggregate to world level (addresses pseudoreplication)
            world_agg = model_data.groupby('world')[metric_name].agg(config['stat'])

            # Bootstrap CI
            boot = bootstrap_ci(world_agg.values, statistic=config['stat'])

            # Format string
            fmt = config['format']
            if np.isnan(boot['point']):
                row[config['label']] = "N/A"
            else:
                row[config['label']] = (
                    f"{boot['point']:{fmt}} "
                    f"[{boot['ci_low']:{fmt}}, {boot['ci_high']:{fmt}}]"
                )

            # Also store raw values for sorting
            row[f"{metric_name}_point"] = boot['point']
            row[f"{metric_name}_ci_low"] = boot['ci_low']
            row[f"{metric_name}_ci_high"] = boot['ci_high']

        results.append(row)

    return pd.DataFrame(results)


def create_ablation_table(df):
    """Create ablation comparison table for TWEEDIE variants."""
    tweedie_df = df[df['model'] == 'TWEEDIE'].copy()

    if len(tweedie_df) == 0:
        return pd.DataFrame()

    results = []
    for variant in sorted(tweedie_df['variant'].unique()):
        variant_data = tweedie_df[tweedie_df['variant'] == variant]

        # World-level aggregation
        world_logev = variant_data.groupby('world')['log_evidence'].median()
        boot = bootstrap_ci(world_logev.values, statistic='median')

        results.append({
            'Variant': variant,
            <response clipped><NOTE>Result is longer than **10000 characters**, will be **truncated**.</NOTE>
