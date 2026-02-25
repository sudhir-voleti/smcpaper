"""
Simulation: Compound Poisson-Gamma (Tweedie) DGP for SMC Estimation Paper
20-cell grid: sparsity (4 levels) x sample size (2 levels) x skewness (2 levels)
Plus 4 robustness checks = 24 total configurations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
import json
from pathlib import Path


@dataclass
class SimulationConfig:
    """Configuration for a single simulation cell"""
    lam: float         # Poisson rate for purchase count (controls sparsity)
    N: int             # sample size (100, 200)
    T: int = 50        # time periods (fixed)
    alpha: float = 2.0 # Gamma shape (controls skewness: low=2.0, high=0.5)
    beta: float = 0.5  # Gamma rate
    seed: int = 42     # reproducibility
    
    @property
    def pi_0(self) -> float:
        """Derived structural zero probability: P(m=0) = exp(-lambda)"""
        return np.exp(-self.lam)
    
    @property
    def true_params(self) -> Dict[str, float]:
        """Return true parameter dictionary"""
        return {
            'lam': self.lam,
            'alpha': self.alpha,
            'beta': self.beta,
            'pi_0': self.pi_0,
            'mean_spend': self.alpha / self.beta,
            'var_spend': self.alpha / (self.beta ** 2),
            'skewness': 2 / np.sqrt(self.alpha)
        }


def generate_tweedie_data(config: SimulationConfig) -> pd.DataFrame:
    """
    Vectorized compound Poisson-Gamma (Tweedie) data generation.
    
    DGP:
        m ~ Poisson(lam)              # purchase count
        y|m=0 = 0                      # structural zero
        y|m>0 ~ Gamma(m*alpha, beta)   # sum of m purchases
    
    Vectorized for speed using NumPy's Generator API.
    """
    rng = np.random.default_rng(config.seed)
    total_obs = config.N * config.T
    
    # 1. Generate purchase counts (Poisson)
    m = rng.poisson(config.lam, size=total_obs)
    
    # 2. Generate spend amounts (Gamma, vectorized)
    y = np.zeros(total_obs)
    pos_mask = m > 0
    
    # For m>0: sum of m i.i.d. Gamma(alpha, beta) = Gamma(m*alpha, beta)
    if pos_mask.any():
        y[pos_mask] = rng.gamma(
            m[pos_mask] * config.alpha,
            1 / config.beta,  # numpy uses scale=1/rate
            size=pos_mask.sum()
        )
    
    # 3. Construct DataFrame efficiently
    df = pd.DataFrame({
        'customer_id': np.repeat(np.arange(config.N), config.T),
        't': np.tile(np.arange(config.T), config.N),
        'y': y,
        'purchase_count': m,
        'is_zero': (m == 0).astype(int)
    })
    
    # Add metadata columns for recovery tracking
    df['lam_true'] = config.lam
    df['alpha_true'] = config.alpha
    df['beta_true'] = config.beta
    df['pi_0_true'] = config.pi_0
    
    return df


def compute_moments(df: pd.DataFrame) -> Dict[str, float]:
    """Compute empirical moments for validation"""
    y = df['y'].values
    y_pos = y[y > 0]
    
    return {
        'empirical_pi_0': (y == 0).mean(),
        'empirical_mean_purchases': df['purchase_count'].mean(),
        'empirical_mean_spend': y_pos.mean() if len(y_pos) > 0 else 0.0,
        'empirical_var_spend': y_pos.var() if len(y_pos) > 1 else 0.0,
        'empirical_skew_spend': pd.Series(y_pos).skew() if len(y_pos) > 2 else 0.0,
        'n_zeros': (y == 0).sum(),
        'n_positive': (y > 0).sum()
    }


def create_simulation_grid() -> List[SimulationConfig]:
    """
    Create the 20-cell simulation grid:
    - 4 sparsity levels (via lambda: 0.69, 0.36, 0.16, 0.05)
    - 2 sample sizes (100, 200)
    - 2 skewness levels (alpha: 2.0=low, 0.5=high)
    
    Total: 16 core cells + 4 robustness checks = 20 configurations
    """
    configs = []
    seed_base = 42
    
    # Sparsity levels: pi_0 = exp(-lambda)
    # pi_0=0.50 -> lam=0.693, pi_0=0.70 -> lam=0.357, etc.
    sparsity_params = [
        (0.50, 0.693),   # Moderate sparsity
        (0.70, 0.357),   # High sparsity  
        (0.85, 0.163),   # Very high sparsity
        (0.95, 0.051)    # Extreme sparsity
    ]
    
    # Sample sizes
    N_levels = [100, 200]
    
    # Skewness levels (via alpha)
    skew_params = [
        (2.0, 'low'),     # Skewness = 1.41
        (0.5, 'high')     # Skewness = 2.83
    ]
    
    # Generate 16 core configurations
    cell_id = 0
    for pi_0_target, lam in sparsity_params:
        for N in N_levels:
            for alpha, skew_label in skew_params:
                configs.append(SimulationConfig(
                    lam=lam,
                    N=N,
                    T=50,
                    alpha=alpha,
                    beta=0.5,
                    seed=seed_base + cell_id
                ))
                cell_id += 1
    
    # 4 robustness checks with different parameters
    robustness_configs = [
        # Extreme sparsity + high skew + longer panel
        SimulationConfig(lam=0.051, N=100, T=100, alpha=0.5, beta=0.5, seed=200),
        # Moderate sparsity + low skew + different beta
        SimulationConfig(lam=0.693, N=200, T=50, alpha=2.0, beta=1.0, seed=201),
        # High sparsity + low skew (unusual combination)
        SimulationConfig(lam=0.163, N=100, T=50, alpha=2.0, beta=0.3, seed=202),
        # Very high N for scalability test
        SimulationConfig(lam=0.357, N=500, T=50, alpha=1.0, beta=0.5, seed=203)
    ]
    
    configs.extend(robustness_configs)
    
    return configs


def run_simulation_cell(config: SimulationConfig, output_dir: str = "data/simulation") -> Dict:
    """
    Run single simulation cell and save data
    
    Returns metadata dictionary for tracking
    """
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_tweedie_data(config)
    
    # Compute moments
    moments = compute_moments(df)
    
    # Save data
    filename = (f"sim_lam{config.lam:.3f}_N{config.N}_T{config.T}_"
                f"a{config.alpha:.1f}_b{config.beta:.1f}_seed{config.seed}.csv")
    filepath = out_path / filename
    df.to_csv(filepath, index=False)
    
    # Return metadata
    metadata = {
        'filename': str(filename),
        'filepath': str(filepath),
        'config': {
            'lam': config.lam,
            'pi_0': config.pi_0,
            'N': config.N,
            'T': config.T,
            'alpha': config.alpha,
            'beta': config.beta,
            'seed': config.seed
        },
        'true_params': config.true_params,
        'empirical_moments': moments,
        'n_observations': len(df),
        'n_customers': config.N,
        'n_periods': config.T
    }
    
    return metadata


def generate_all_simulations(output_dir: str = "data/simulation",
                             metadata_file: str = "data/simulation_metadata.json"):
    """Generate all 20 simulation cells and save metadata"""
    
    configs = create_simulation_grid()
    all_metadata = []
    
    print(f"Generating {len(configs)} simulation datasets...")
    print("=" * 70)
    
    for i, config in enumerate(configs, 1):
        print(f"[{i:2d}/{len(configs)}] lam={config.lam:.3f} (pi_0={config.pi_0:.2f}), "
              f"N={config.N}, alpha={config.alpha:.1f}")
        
        metadata = run_simulation_cell(config, output_dir)
        all_metadata.append(metadata)
        
        # Print quick summary
        emp = metadata['empirical_moments']
        print(f"       Empirical pi_0: {emp['empirical_pi_0']:.3f} | "
              f"Mean spend: {emp['empirical_mean_spend']:.2f} | "
              f"Saved: {metadata['filename'][:50]}...")
    
    # Save metadata
    meta_path = Path(metadata_file)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print("=" * 70)
    print(f"All simulations complete. Metadata saved to {metadata_file}")
    
    return all_metadata


def load_simulation_metadata(metadata_file: str = "data/simulation_metadata.json") -> List[Dict]:
    """Load simulation metadata"""
    with open(metadata_file, 'r') as f:
        return json.load(f)


def get_simulation_summary(metadata_file: str = "data/simulation_metadata.json") -> pd.DataFrame:
    """Create summary table of all simulations"""
    metadata = load_simulation_metadata(metadata_file)
    
    rows = []
    for m in metadata:
        rows.append({
            'filename': m['filename'][:40],
            'lam': m['config']['lam'],
            'pi_0_true': m['config']['pi_0'],
            'N': m['config']['N'],
            'T': m['config']['T'],
            'alpha': m['config']['alpha'],
            'beta': m['config']['beta'],
            'empirical_pi_0': m['empirical_moments']['empirical_pi_0'],
            'empirical_skew': m['empirical_moments']['empirical_skew_spend'],
            'n_obs': m['n_observations']
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Generate all simulations
    metadata = generate_all_simulations()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    summary = get_simulation_summary()
    print(summary.to_string(index=False))
    
    # Print sparsity check
    print("\n" + "=" * 70)
    print("SPARSITY CHECK (True vs Empirical pi_0)")
    print("=" * 70)
    for m in metadata[:8]:  # First 8 only
        true_pi0 = m['config']['pi_0']
        emp_pi0 = m['empirical_moments']['empirical_pi_0']
        print(f"Target: {true_pi0:.3f} | Empirical: {emp_pi0:.3f} | "
              f"Diff: {abs(true_pi0 - emp_pi0):.4f}")
      
