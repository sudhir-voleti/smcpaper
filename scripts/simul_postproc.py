#!/usr/bin/env python3
"""
simul_postproc.py
=================
Modern post-processing for HMM simulation results.
Handles Hurdle, Bemmaor, and Tweedie models with full metrics and 95% CIs.
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import arviz as az
from sklearn.metrics import adjusted_rand_score, confusion_matrix, accuracy_score
from itertools import permutations
import re


# =============================================================================
# 1. MODEL METADATA EXTRACTORS
# =============================================================================

def extract_model_info(pkl_path: Path) -> Dict:
    """Extract model type, K, world, etc. from filename and contents."""
    pkl_path = Path(pkl_path)
    name = pkl_path.name
    
    # Initialize
    info = {
        'file': name,
        'path': str(pkl_path),
        'model_type': 'unknown',
        'K': None,
        'N': None,
        'T': None,
        'D': None,
        'world': None,
        'glm_gam': None
    }
    
    # Detect model type from filename patterns
    if 'BEMMAOR' in name.upper():
        info['model_type'] = 'BEMMAOR'
    elif 'TWEEDIE' in name.upper():
        info['model_type'] = 'TWEEDIE'
    elif 'HURDLE' in name.upper():
        info['model_type'] = 'HURDLE'
    else:
        # Hurdle files don't have HURDLE in name - check by absence of others
        info['model_type'] = 'HURDLE'
    
    # Extract K, N, T, D from filename
    patterns = {
        'K': r'[Kk](\d+)',
        'N': r'[Nn](\d+)',
        'T': r'[Tt](\d+)',
        'D': r'[Dd](\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            info[key] = int(match.group(1))
    
    # Detect world from path
    world_candidates = ['harbor', 'breeze', 'fog', 'cliff']
    path_lower = str(pkl_path).lower()
    for world in world_candidates:
        if world in path_lower:
            info['world'] = world.capitalize()
            break
    
    # Detect GLM/GAM
    if 'GAM' in name.upper():
        info['glm_gam'] = 'GAM'
    elif 'GLM' in name.upper():
        info['glm_gam'] = 'GLM'
    
    return info


# =============================================================================
# 2. SAFE EXTRACTION UTILITIES
# =============================================================================

def safe_extract(idata, var_name: str, func='mean') -> Optional[np.ndarray]:
    """Safely extract posterior statistic."""
    if var_name not in idata.posterior:
        return None
    
    post = idata.posterior[var_name]
    
    if func == 'mean':
        return post.mean(dim=['chain', 'draw']).values
    elif func == 'std':
        return post.std(dim=['chain', 'draw']).values
    elif func == 'hdi_lower':
        try:
            hdi = az.hdi(post, hdi_prob=0.95)
            # Handle xarray structure
            if var_name in hdi:
                return hdi[var_name].sel(hdi='lower').values
            else:
                return list(hdi.data_vars.values())[0].sel(hdi='lower').values
        except:
            return None
    elif func == 'hdi_upper':
        try:
            hdi = az.hdi(post, hdi_prob=0.95)
            if var_name in hdi:
                return hdi[var_name].sel(hdi='higher').values
            else:
                return list(hdi.data_vars.values())[0].sel(hdi='higher').values
        except:
            return None
    
    return None


def safe_scalar(val) -> float:
    """Convert array/scalar to float."""
    if val is None:
        return np.nan
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        return float(np.mean(val))
    return float(val)


# =============================================================================
# 3. IN-SAMPLE FIT METRICS
# =============================================================================

def extract_fit_metrics(idata, res: Dict) -> Dict:
    """Extract log-evidence, WAIC, LOO with CIs where possible."""
    metrics = {}
    
    # Log-evidence (from results or extract)
    log_ev = res.get('log_evidence', res.get('log_ev', np.nan))
    if np.isnan(log_ev) and hasattr(idata, 'sample_stats'):
        try:
            lml = idata.sample_stats.log_marginal_likelihood.values
            if lml.dtype == object:
                finals = []
                for chain in lml.flatten():
                    clean = [float(v) for v in chain if np.isfinite(v)]
                    if clean:
                        finals.append(clean[-1])
                log_ev = np.mean(finals) if finals else np.nan
            else:
                log_ev = np.nanmean(lml[:, -1]) if lml.ndim > 1 else np.nanmean(lml)
        except:
            log_ev = np.nan
    
    metrics['log_evidence'] = log_ev
    
    # WAIC and LOO
    for ic_name, ic_func in [('waic', az.waic), ('loo', az.loo)]:
        try:
            ic_result = ic_func(idata)
            metrics[ic_name] = float(ic_result[ic_name])
            metrics[f'{ic_name}_se'] = float(ic_result[f'{ic_name}_se'])
        except:
            metrics[ic_name] = np.nan
            metrics[f'{ic_name}_se'] = np.nan
    
    # ESS and R-hat
    metrics['ess_min'] = res.get('ess_min', np.nan)
    metrics['rhat_max'] = res.get('rhat_max', np.nan)
    
    return metrics


# =============================================================================
# 4. PREDICTIVE METRICS (OOS)
# =============================================================================

def extract_predictive_metrics(res: Dict) -> Dict:
    """Extract OOS RMSE, MAE, and related metrics."""
    return {
        'oos_rmse': res.get('oos_rmse', np.nan),
        'oos_mae': res.get('oos_mae', np.nan),
        'oos_log_pred': res.get('oos_log_pred', res.get('log_pred_score', np.nan))
    }


# =============================================================================
# 5. STATE RECOVERY METRICS
# =============================================================================

def find_best_alignment(true_flat: np.ndarray, pred_flat: np.ndarray, K: int) -> Tuple[Dict, float, np.ndarray]:
    """Find optimal label permutation for state alignment."""
    if K > 6:
        return {i: i for i in range(K)}, np.nan, pred_flat
    
    best_acc = -1
    best_map = {i: i for i in range(K)}
    best_aligned = pred_flat
    
    for perm in permutations(range(K)):
        remap = {i: perm[i] for i in range(K)}
        aligned = np.vectorize(remap.get)(pred_flat)
        acc = accuracy_score(true_flat, aligned)
        
        if acc > best_acc:
            best_acc = acc
            best_map = remap
            best_aligned = aligned
    
    return best_map, best_acc, best_aligned


def extract_state_recovery(idata, data: Dict, world: str) -> Dict:
    """Extract ARI, accuracy, and confusion matrix against true states."""
    metrics = {
        'ari': np.nan,
        'state_accuracy': np.nan,
        'true_states_found': False
    }
    
    # Check for true states
    true_states = data.get('true_states')
    if true_states is None:
        # Try to load from file
        world_dir = Path(data.get('source_file', '.')).parent
        for pattern in [
            f"true_states_{world}_N{data.get('N')}_T{data.get('T')}.npy",
            f"true_states_{world}_N1000_T104.npy"
        ]:
            try:
                true_states = np.load(world_dir / pattern)
                break
            except:
                continue
    
    if true_states is None:
        return metrics
    
    # Decode estimated states
    if 'alpha_filtered' not in idata.posterior:
        return metrics
    
    try:
        alpha = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values
        est_states = np.argmax(alpha, axis=-1)
        
        true_flat = true_states.flatten()
        est_flat = est_states.flatten()
        
        # Mask valid entries
        mask = (true_flat >= 0) & (~np.isnan(true_flat))
        if mask.sum() == 0:
            return metrics
        
        true_flat = true_flat[mask]
        est_flat = est_flat[mask]
        
        K = len(np.unique(true_flat))
        
        # ARI (unadjusted)
        metrics['ari'] = adjusted_rand_score(true_flat, est_flat)
        
        # Best alignment accuracy
        label_map, acc, aligned = find_best_alignment(true_flat, est_flat, K)
        metrics['state_accuracy'] = acc
        metrics['label_map'] = str(label_map)
        metrics['true_states_found'] = True
        metrics['K_true'] = K
        
        # Confusion matrix
        cm = confusion_matrix(true_flat, aligned, labels=range(K))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        for i in range(min(K, 4)):  # Store up to 4 states
            metrics[f'cm_diag_{i}'] = cm_norm[i, i] if i < cm_norm.shape[0] else np.nan
        
    except Exception as e:
        print(f"    State recovery error: {e}")
    
    return metrics


# =============================================================================
# 6. CLV EXTRACTION WITH 95% CI
# =============================================================================

def extract_clv_with_ci(idata, res: Dict, model_type: str) -> Dict:
    """Extract CLV by state with 95% credible intervals."""
    metrics = {
        'clv_total': np.nan,
        'clv_ratio': np.nan,
        'clv_by_state': [],
        'clv_ci_low': [],
        'clv_ci_high': []
    }
    
    # Try different CLV variable names
    clv_var = None
    for var in ['clv_proxy', 'clv_by_state', 'clv']:
        if var in idata.posterior:
            clv_var = var
            break
    
    if clv_var is None:
        # Try from results
        clv_list = res.get('clv_by_state', [])
        if clv_list:
            metrics['clv_by_state'] = clv_list
            metrics['clv_total'] = sum(clv_list)
            if len(clv_list) >= 2:
                metrics['clv_ratio'] = max(clv_list) / (min(clv_list) + 1e-6)
        return metrics
    
    try:
        post = idata.posterior[clv_var]
        
        # Mean CLV by state
        clv_mean = post.mean(dim=['chain', 'draw']).values
        
        # Handle different shapes
        if clv_mean.ndim == 0:
            clv_mean = np.array([clv_mean.item()])
        elif clv_mean.ndim > 1:
            # Take mean across customers/time if needed
            clv_mean = clv_mean.mean(axis=tuple(range(clv_mean.ndim - 1)))
        
        K = len(clv_mean)
        metrics['clv_by_state'] = clv_mean.tolist()
        metrics['clv_total'] = float(np.sum(clv_mean))
        
        if K >= 2:
            metrics['clv_ratio'] = float(np.max(clv_mean) / (np.min(clv_mean) + 1e-6))
        
        # 95% HDI
        try:
            hdi = az.hdi(post, hdi_prob=0.95)
            if clv_var in hdi:
                hdi_low = hdi[clv_var].sel(hdi='lower').values
                hdi_high = hdi[clv_var].sel(hdi='higher').values
            else:
                hdi_var = list(hdi.data_vars.values())[0]
                hdi_low = hdi_var.sel(hdi='lower').values
                hdi_high = hdi_var.sel(hdi='higher').values
            
            # Align shapes
            if hdi_low.ndim > 1:
                hdi_low = hdi_low.mean(axis=tuple(range(hdi_low.ndim - 1)))
                hdi_high = hdi_high.mean(axis=tuple(range(hdi_high.ndim - 1)))
            
            metrics['clv_ci_low'] = hdi_low.tolist() if hasattr(hdi_low, 'tolist') else [float(hdi_low)]
            metrics['clv_ci_high'] = hdi_high.tolist() if hasattr(hdi_high, 'tolist') else [float(hdi_high)]
            
        except Exception as e:
            print(f"    HDI extraction failed: {e}")
            metrics['clv_ci_low'] = [np.nan] * K
            metrics['clv_ci_high'] = [np.nan] * K
            
    except Exception as e:
        print(f"    CLV extraction error: {e}")
    
    return metrics


# =============================================================================
# 7. MAIN PROCESSING PIPELINE
# =============================================================================

def process_single_pkl(pkl_path: Path) -> Optional[Dict]:
    """Full post-processing for a single PKL file."""
    pkl_path = Path(pkl_path)
    
    if not pkl_path.exists():
        print(f"  X File not found: {pkl_path}")
        return None
    
    # Load
    try:
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        
        idata = result['idata']
        res = result.get('res', {})
        data = result.get('data', {})
        
    except Exception as e:
        print(f"  X Load failed: {pkl_path.name}: {e}")
        return None
    
    # Basic info
    info = extract_model_info(pkl_path)
    info['time_min'] = res.get('time_min', np.nan)
    info['train_ratio'] = res.get('train_ratio', data.get('train_ratio', 1.0))
    
    # Extract all metric groups
    metrics = {}
    
    # 1. In-sample fit
    fit_metrics = extract_fit_metrics(idata, res)
    metrics.update(fit_metrics)
    
    # 2. Predictive
    pred_metrics = extract_predictive_metrics(res)
    metrics.update(pred_metrics)
    
    # 3. State recovery
    recovery_metrics = extract_state_recovery(idata, data, info['world'])
    metrics.update(recovery_metrics)
    
    # 4. CLV with CI
    clv_metrics = extract_clv_with_ci(idata, res, info['model_type'])
    metrics.update(clv_metrics)
    
    # Combine
    full_result = {**info, **metrics}
    
    return full_result


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Create formatted comparison table from all results."""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Define column order
    core_cols = ['model_type', 'world', 'K', 'glm_gam', 'N', 'T', 'D']
    fit_cols = ['log_evidence', 'waic', 'loo', 'ess_min', 'rhat_max']
    pred_cols = ['oos_rmse', 'oos_mae']
    recovery_cols = ['ari', 'state_accuracy', 'K_true']
    clv_cols = ['clv_total', 'clv_ratio']
    
    # Add CLV by state columns dynamically
    max_k = df['K'].max() if 'K' in df.columns else 0
    for k in range(int(max_k)):
        clv_cols.extend([f'clv_state{k}', f'clv_state{k}_ci_low', f'clv_state{k}_ci_high'])
    
    # Flatten CLV lists into columns
    for idx, row in df.iterrows():
        clv_list = row.get('clv_by_state', [])
        ci_low = row.get('clv_ci_low', [])
        ci_high = row.get('clv_ci_high', [])
        
        for k, (clv, low, high) in enumerate(zip(clv_list, ci_low, ci_high)):
            df.at[idx, f'clv_state{k}'] = clv
            df.at[idx, f'clv_state{k}_ci_low'] = low
            df.at[idx, f'clv_state{k}_ci_high'] = high
    
    # Select and order columns that exist
    all_cols = core_cols + fit_cols + pred_cols + recovery_cols + clv_cols + ['time_min', 'train_ratio']
    existing_cols = [c for c in all_cols if c in df.columns]
    
    df = df[existing_cols]
    
    # Sort by world, then log-evidence
    df = df.sort_values(['world', 'log_evidence'], ascending=[True, False], na_position='last')
    
    return df


def format_clv_string(row: pd.Series, k: int) -> str:
    """Format CLV with CI as string."""
    mean = row.get(f'clv_state{k}', np.nan)
    low = row.get(f'clv_state{k}_ci_low', np.nan)
    high = row.get(f'clv_state{k}_ci_high', np.nan)
    
    if np.isnan(mean):
        return "N/A"
    
    if not np.isnan(low) and not np.isnan(high):
        return f"${mean:.2f} [${low:.2f}, ${high:.2f}]"
    else:
        return f"${mean:.2f}"


def print_summary_table(df: pd.DataFrame):
    """Print formatted summary to console."""
    print("\n" + "="*120)
    print("SIMULATION RESULTS SUMMARY")
    print("="*120)
    
    # Group by world
    for world in sorted(df['world'].dropna().unique()):
        world_df = df[df['world'] == world]
        print(f"\n{'='*80}")
        print(f"WORLD: {world.upper()}")
        print(f"{'='*80}")
        
        # Display key columns
        display_cols = ['model_type', 'K', 'glm_gam', 'log_evidence', 'oos_rmse', 'ari', 'clv_ratio']
        display_cols = [c for c in display_cols if c in world_df.columns]
        
        for idx, row in world_df.iterrows():
            print(f"\n{row['model_type']} K={row['K']} ({row.get('glm_gam', 'N/A')}):")
            print(f"  Log-Ev: {row.get('log_evidence', np.nan):.2f} | "
                  f"WAIC: {row.get('waic', np.nan):.2f} | "
                  f"LOO: {row.get('loo', np.nan):.2f}")
            print(f"  OOS RMSE: {row.get('oos_rmse', np.nan):.4f} | "
                  f"MAE: {row.get('oos_mae', np.nan):.4f}")
            print(f"  ARI: {row.get('ari', np.nan):.3f} | "
                  f"State Acc: {row.get('state_accuracy', np.nan):.3f}")
            
            # CLV by state
            k = int(row.get('K', 0))
            print(f"  CLV Ratio: {row.get('clv_ratio', np.nan):.1f}x")
            for state in range(min(k, 4)):
                clv_str = format_clv_string(row, state)
                print(f"    State {state}: {clv_str}")
        
        # Best by log-ev
        best = world_df.loc[world_df['log_evidence'].idxmax()] if not world_df['log_evidence'].isna().all() else None
        if best is not None:
            print(f"\n  >>> BEST (Log-Ev): {best['model_type']} K={best['K']}")
    
    print("\n" + "="*120)


def export_results(df: pd.DataFrame, out_path: Path, formats: List[str] = ['csv', 'latex']):
    """Export to multiple formats."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV
    if 'csv' in formats:
        csv_path = out_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  Saved CSV: {csv_path}")
    
    # LaTeX
    if 'latex' in formats:
        latex_path = out_path.with_suffix('.tex')
        
        # Create simplified version for LaTeX
        latex_cols = ['model_type', 'world', 'K', 'log_evidence', 'oos_rmse', 'ari', 'clv_ratio']
        latex_cols = [c for c in latex_cols if c in df.columns]
        latex_df = df[latex_cols].copy().round(3)
        
        latex_str = latex_df.to_latex(
            index=False,
            float_format="%.3f",
            caption="HMM Model Comparison: Simulation Results",
            label="tab:simul_results",
            na_rep="NA"
        )
        
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        print(f"  Saved LaTeX: {latex_path}")
    
    # Markdown summary
    if 'md' in formats:
        md_path = out_path.with_suffix('.md')
        with open(md_path, 'w') as f:
            f.write("# Simulation Results\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
        print(f"  Saved Markdown: {md_path}")


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Post-process HMM simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PKL
  python simul_postproc.py --pkl /path/to/model.pkl
  
  # Process directory of PKLs
  python simul_postproc.py --dir /path/to/results/ --out summary
  
  # Process multiple specific PKLs
  python simul_postproc.py --pkls model1.pkl model2.pkl model3.pkl --out comparison
        """
    )
    
    parser.add_argument('--pkl', type=str, help='Single PKL file to process')
    parser.add_argument('--dir', type=str, help='Directory containing PKL files')
    parser.add_argument('--pkls', nargs='+', help='List of PKL files to process')
    parser.add_argument('--out', type=str, default='simul_comparison',
                       help='Output file base name (without extension)')
    parser.add_argument('--formats', nargs='+', default=['csv', 'latex'],
                       choices=['csv', 'latex', 'md'], help='Output formats')
    
    args = parser.parse_args()
    
    # Collect PKL files
    pkl_files = []
    
    if args.pkl:
        pkl_files = [Path(args.pkl)]
    elif args.dir:
        pkl_files = list(Path(args.dir).rglob("*.pkl"))
    elif args.pkls:
        pkl_files = [Path(p) for p in args.pkls]
    else:
        print("Error: Must specify --pkl, --dir, or --pkls")
        return
    
    print(f"Processing {len(pkl_files)} PKL files...")
    
    # Process all
    results = []
    for pkl_path in pkl_files:
        print(f"  Processing: {pkl_path.name}")
        result = process_single_pkl(pkl_path)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Print summary
    print_summary_table(df)
    
    # Export
    out_path = Path(args.out)
    export_results(df, out_path, args.formats)
    
    print(f"\n{'='*80}")
    print(f"Processed {len(results)} models successfully")
    print(f"Output base: {out_path.absolute()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
