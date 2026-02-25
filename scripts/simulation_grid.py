"""
Simulation: Compound Poisson-Gamma (Tweedie) DGP - Bulletproof Version
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


def make_serializable(obj):
    """Last resort JSON serializer"""
    try:
        # Test if JSON serializable
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # Convert numpy types
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        else:
            return str(obj)


def generate_one_dataset(lam, N, T, alpha, beta, seed, output_dir):
    """Generate single dataset - all native Python types"""
    rng = np.random.default_rng(int(seed))
    N, T, seed = int(N), int(T), int(seed)
    lam, alpha, beta = float(lam), float(alpha), float(beta)
    
    total_obs = N * T
    
    # Generate data
    m = rng.poisson(lam, size=total_obs)
    y = np.zeros(total_obs)
    pos_mask = m > 0
    if pos_mask.any():
        y[pos_mask] = rng.gamma(m[pos_mask] * alpha, 1.0 / beta)
    
    # Build DataFrame with Python lists, not numpy arrays
    customer_id = [int(i) for i in np.repeat(np.arange(N), T)]
    t = [int(i) for i in np.tile(np.arange(T), N)]
    y_list = [float(val) for val in y]
    m_list = [int(val) for val in m]
    is_zero_list = [int(val == 0) for val in m]
    
    df = pd.DataFrame({
        'customer_id': customer_id,
        't': t,
        'y': y_list,
        'purchase_count': m_list,
        'is_zero': is_zero_list,
        'lam_true': [float(lam)] * total_obs,
        'alpha_true': [float(alpha)] * total_obs,
        'beta_true': [float(beta)] * total_obs,
        'pi_0_true': [float(np.exp(-lam))] * total_obs
    })
    
    # Compute moments using Python floats only
    y_arr = np.array(y_list)
    y_pos = y_arr[y_arr > 0]
    
    moments = {
        'empirical_pi_0': float(np.mean(y_arr == 0)),
        'empirical_mean_purchases': float(np.mean(m_list)),
        'empirical_mean_spend': float(np.mean(y_pos)) if len(y_pos) > 0 else 0.0,
        'empirical_var_spend': float(np.var(y_pos, ddof=1)) if len(y_pos) > 1 else 0.0,
        'empirical_skew_spend': float(pd.Series(y_pos).skew()) if len(y_pos) > 2 else 0.0,
        'n_zeros': int(np.sum(y_arr == 0)),
        'n_positive': int(np.sum(y_arr > 0))
    }
    
    # Save
    filename = f"sim_lam{lam:.3f}_N{N}_T{T}_a{alpha:.1f}_b{beta:.1f}_seed{seed}.csv"
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    
    # Build metadata
    metadata = {
        'filename': str(filename),
        'config': {
            'lam': lam,
            'pi_0': float(np.exp(-lam)),
            'N': N,
            'T': T,
            'alpha': alpha,
            'beta': beta,
            'seed': seed
        },
        'true_params': {
            'lam': lam,
            'alpha': alpha,
            'beta': beta,
            'pi_0': float(np.exp(-lam)),
            'mean_spend': alpha / beta,
            'var_spend': alpha / (beta ** 2),
            'skewness': 2.0 / (alpha ** 0.5)
        },
        'empirical_moments': moments,
        'n_observations': total_obs,
        'n_customers': N,
        'n_periods': T
    }
    
    return make_serializable(metadata)


def main(output_dir):
    """Generate all 20 simulation datasets"""
    configs = []
    seed_base = 42
    
    # 16 core configs
    sparsity = [(0.50, 0.693), (0.70, 0.357), (0.85, 0.163), (0.95, 0.051)]
    Ns = [100, 200]
    alphas = [2.0, 0.5]
    
    cell_id = 0
    for pi0, lam in sparsity:
        for N in Ns:
            for alpha in alphas:
                configs.append((lam, N, 50, alpha, 0.5, seed_base + cell_id))
                cell_id += 1
    
    # 4 robustness
    configs.extend([
        (0.051, 100, 100, 0.5, 0.5, 200),
        (0.693, 200, 50, 2.0, 1.0, 201),
        (0.163, 100, 50, 2.0, 0.3, 202),
        (0.357, 500, 50, 1.0, 0.5, 203)
    ])
    
    print(f"Generating {len(configs)} datasets to {output_dir}")
    print("=" * 60)
    
    all_meta = []
    for i, (lam, N, T, alpha, beta, seed) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] lam={lam:.3f}, N={N}, alpha={alpha}")
        meta = generate_one_dataset(lam, N, T, alpha, beta, seed, output_dir)
        all_meta.append(meta)
        print(f"      pi_0={meta['empirical_moments']['empirical_pi_0']:.3f}")
    
    # Save metadata
    meta_path = Path(output_dir) / "simulation_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(all_meta, f, indent=2)
    
    print("=" * 60)
    print(f"Done. Saved to {meta_path}")


# RUN THIS
if __name__ == "__main__":
    OUTPUT_DIR = r"D:\Dropbox\research\SMC paper Feb2026\simul_data"
    main(OUTPUT_DIR)
