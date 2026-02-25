"""
Simulation: Compound Poisson-Gamma (Tweedie) DGP for SMC Estimation Paper
20-cell grid: sparsity (4 levels) x sample size (2 levels) x skewness (2 levels)
Plus 4 robustness checks = 24 total configurations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any
import json
from pathlib importPath


def to_python(obj: Any) -> Any:
    """Aggressive conversion of numpy/pandas types to native Python"""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


@dataclass
class SimulationConfig:
    """Configuration for a single simulation cell"""
    lam: float
    N: int
    T: int = 50
    alpha: float = 2.0
    beta: float = 0.5
    seed: int = 42
    
    def __post_init__(self):
        # Ensure native types
        self.lam = float(self.lam)
        self.N = int(self.N)
        self.T = int(self.T)
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.seed = int(self.seed)
    
    @property
    def pi_0(self) -> float:
        return float(np.exp(-self.lam))
    
    @property
    def true_params(self) -> Dict[str, float]:
        return {
            'lam': self.lam,
            'alpha': self.alpha,
            'beta': self.beta,
            'pi_0': self.pi_0,
            'mean_spend': float(self.alpha / self.beta),
            'var_spend': float(self.alpha / (self.beta ** 2)),
            'skewness': float(2 / np.sqrt(self.alpha))
        }


def generate_tweedie_data(config: SimulationConfig) -> pd.DataFrame:
    """Vectorized compound Poisson-Gamma (Tweedie) data generation"""
    rng = np.random.default_rng(config.seed)
    total_obs = config.N * config.T
    
    m = rng.poisson(config.lam, size=total_obs)
    y = np.zeros(total_obs, dtype=float)
    pos_mask = m > 0
    
    if pos_mask.any():
        shape_params = (m[pos_mask] * config.alpha).astype(float)
        y[pos_mask] = rng.gamma(shape_params, 1.0 / config.beta)
    
    df = pd.DataFrame({
        'customer_id': np.repeat(np.arange(config.N), config.T),
        't': np.tile(np.arange(config.T), config.N),
        'y': y,
        'purchase_count': m.astype(int),
        'is_zero': (m == 0).astype(int)
    })
    
    df['lam_true'] = config.lam
    df['alpha_true'] = config.alpha
    df['beta_true'] = config.beta
    df['pi_0_true'] = config.pi_0
    
    return df


def compute_moments(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute empirical moments - all native Python types"""
    y = df['y'].to_numpy()
    y_pos = y[y > 0]
    
    # Use .item() to force numpy scalar -> Python scalar
    empirical_pi_0 = ((y == 0).mean()).item()
    empirical_mean_purchases = (df['purchase_count'].to_numpy().mean()).item()
    
    if len(y_pos) > 0:
        empirical_mean_spend = (y_pos.mean()).item()
        empirical_var_spend = (y_pos.var(ddof=1)).item() if len(y_pos) > 1 else 0.0
        # pandas skew returns float, but force it anyway
        skew_val = pd.Series(y_pos).skew()
        empirical_skew_spend = float(skew_val) if pd.notna(skew_val) else 0.0
    else:
        empirical_mean_spend = 0.0
        empirical_var_spend = 0.0
        empirical_skew_spend = 0.0
    
    return {
        'empirical_pi_0': empirical_pi_0,
        'empirical_mean_purchases': empirical_mean_purchases,
        'empirical_mean_spend': empirical_mean_spend,
        'empirical_var_spend': empirical_var_spend,
        'empirical_skew_spend': empirical_skew_spend,
        'n_zeros': int((y == 0).sum()),
        'n_positive': int((y > 0).sum())
    }


def create_simulation_grid() -> List[SimulationConfig]:
    """Create 20-cell simulation grid"""
    configs = []
    seed_base = 42
    
    sparsity_params = [(0.50, 0.693), (0.70, 0.357), (0.85, 0.163), (0.95, 0.051)]
    N_levels = [100, 200]
    skew_params = [(2.0, 'low'), (0.5, 'high')]
    
    cell_id = 0
    for pi_0_target, lam in sparsity_params:
        for N in N_levels:
            for alpha, skew_label in skew_params:
                configs.append(SimulationConfig(
                    lam=float(lam), N=int(N), T=50,
                    alpha=float(alpha), beta=0.5, seed=seed_base + cell_id
                ))
                cell_id += 1
    
    # Robustness checks
    configs.extend([
        SimulationConfig(lam=0.051, N=100, T=100, alpha=0.5, beta=0.5, seed=200),
        SimulationConfig(lam=0.693, N=200, T=50, alpha=2.0, beta=1.0, seed=201),
        SimulationConfig(lam=0.163, N=100, T=50, alpha=2.0, beta=0.3, seed=202),
        SimulationConfig(lam=0.357, N=500, T=50, alpha=1.0, beta=0.5, seed=203)
    ])
    
    return configs


def run_simulation_cell(config: SimulationConfig, output_dir: str) -> Dict:
    """Run single simulation cell and save data"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df = generate_tweedie_data(config)
    moments = compute_moments(df)
    
    filename = f"sim_lam{config.lam:.3f}_N{config.N}_T{config.T}_a{config.alpha:.1f}_b{config.beta:.1f}_seed{config.seed}.csv"
    filepath = out_path / filename
    df.to_csv(filepath, index=False)
    
    # Build metadata with aggressive type conversion
    metadata = {
        'filename': str(filename),
        'filepath': str(filepath),
        'config': to_python({
            'lam': config.lam,
            'pi_0': config.pi_0,
            'N': config.N,
            'T': config.T,
            'alpha': config.alpha,
            'beta': config.beta,
            'seed': config.seed
        }),
        'true_params': to_python(config.true_params),
        'empirical_moments': to_python(moments),
        'n_observations': int(len(df)),
        'n_customers': int(config.N),
        'n_periods': int(config.T)
    }
    
    return metadata


def generate_all_simulations(output_dir: str = "data/simulation",
                             metadata_file: str = "data/simulation_metadata.json"):
    """Generate all 20 simulation cells"""
    configs = create_simulation_grid()
    all_metadata = []
    
    print(f"Generating {len(configs)} simulation datasets...")
    print("=" * 70)
    
    for i, config in enumerate(configs, 1):
        print(f"[{i:2d}/{len(configs)}] lam={config.lam:.3f} (pi_0={config.pi_0:.2f}), N={config.N}, alpha={config.alpha:.1f}")
        
        metadata = run_simulation_cell(config, output_dir)
        all_metadata.append(metadata)
        
        emp = metadata['empirical_moments']
        print(f"       Empirical pi_0: {emp['empirical_pi_0']:.3f} | Mean spend: {emp['empirical_mean_spend']:.2f}")
    
    # Save with default=str as final fallback
    meta_path = Path(metadata_file)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2, default=lambda x: str(x) if hasattr(x, '__class__') and 'numpy' in str(type(x)) else x)
    
    print("=" * 70)
    print(f"Complete. Metadata: {metadata_file}")
    return all_metadata


if __name__ == "__main__":
    # For Spyder: modify these paths
    import os
    OUTPUT_DIR = r"D:\Dropbox\research\SMC paper Feb2026\simul_data"
    
    metadata = generate_all_simulations(output_dir=OUTPUT_DIR, 
                                        metadata_file=os.path.join(OUTPUT_DIR, "simulation_metadata.json"))
    
    print("\nSummary:")
    for m in metadata[:5]:
        print(f"  {m['filename'][:50]}... pi_0={m['empirical_moments']['empirical_pi_0']:.3f}")
