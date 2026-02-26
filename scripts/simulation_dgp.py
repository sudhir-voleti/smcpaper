#!/usr/bin/env python3
"""
simulation_dgp.py - Local version with argparse
Generate HMM simulation data for any world
"""

import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WorldConfig:
    name: str
    sparsity_level: str
    tail_level: str
    pi0_base: np.ndarray
    mu_base: np.ndarray
    alpha_base: np.ndarray
    Gamma: np.ndarray
    sigma_re: float = 0.40
    N: int = 200
    T: int = 104
    seed: int = 42

def generate_hmm_data(cfg: WorldConfig):
    rng = np.random.default_rng(cfg.seed)
    K = 3
    
    # Hierarchical random effects
    re_pi0 = rng.normal(0, cfg.sigma_re, size=(cfg.N, K))
    re_mu = rng.normal(0, cfg.sigma_re, size=(cfg.N, K))
    
    logit_pi0 = np.log(cfg.pi0_base / (1 - cfg.pi0_base))
    pi0_ik = 1 / (1 + np.exp(-(logit_pi0 + re_pi0)))
    
    mu_ik = np.exp(np.log(cfg.mu_base) + re_mu)
    beta_ik = cfg.alpha_base / mu_ik
    
    # State evolution
    states = np.zeros((cfg.N, cfg.T), dtype=int)
    init_dist = [0.60, 0.30, 0.10]
    states[:, 0] = rng.choice(K, size=cfg.N, p=init_dist)
    
    for t in range(1, cfg.T):
        for i in range(cfg.N):
            states[i, t] = rng.choice(K, p=cfg.Gamma[states[i, t-1]])
    
    # Emissions
    cust_idx = np.repeat(np.arange(cfg.N), cfg.T)
    time_idx = np.tile(np.arange(cfg.T), cfg.N)
    flat_states = states.flatten()
    
    pi0_obs = pi0_ik[cust_idx, flat_states]
    alpha_obs = cfg.alpha_base[flat_states]
    beta_obs = beta_ik[cust_idx, flat_states]
    
    is_zero = rng.random(cfg.N * cfg.T) < pi0_obs
    y = rng.gamma(alpha_obs, 1.0 / beta_obs)
    y[is_zero] = 0.0
    
    df = pd.DataFrame({
        'customer_id': cust_idx,
        't': time_idx,
        'y': y,
        'true_state': flat_states
    })
    
    return df, states

def get_world_config(world_name, N, T, seed):
    """Get configuration for named world."""
    G = np.array([[0.92, 0.06, 0.02], 
                  [0.10, 0.80, 0.10], 
                  [0.05, 0.15, 0.80]])
    
    configs = {
        "Harbor": WorldConfig("Harbor", "Low", "Light",
                             np.array([0.90, 0.50, 0.15]),
                             np.array([5, 25, 120]),
                             np.array([4.0, 4.0, 4.0]), G, N=N, T=T, seed=seed),
        
        "Breeze": WorldConfig("Breeze", "Low", "Heavy",
                             np.array([0.90, 0.50, 0.15]),
                             np.array([5, 25, 120]),
                             np.array([0.8, 0.7, 0.5]), G, N=N, T=T, seed=seed),
        
        "Fog": WorldConfig("Fog", "High", "Light",
                          np.array([0.98, 0.85, 0.40]),
                          np.array([2, 10, 80]),
                          np.array([4.0, 4.0, 4.0]), G, N=N, T=T, seed=seed),
        
        "Cliff": WorldConfig("Cliff", "High", "Heavy",
                            np.array([0.98, 0.85, 0.40]),
                            np.array([2, 10, 80]),
                            np.array([0.8, 0.7, 0.5]), G, N=N, T=T, seed=seed),
    }
    
    return configs.get(world_name)

def main():
    parser = argparse.ArgumentParser(description='Generate HMM simulation data')
    parser.add_argument('--world', required=True, choices=['Harbor', 'Breeze', 'Fog', 'Cliff'])
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--T', type=int, default=104)
    parser.add_argument('--out_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Get config
    cfg = get_world_config(args.world, args.N, args.T, args.seed)
    if cfg is None:
        print(f"Error: Unknown world {args.world}")
        return
    
    print(f"Generating {args.world}: N={args.N}, T={args.T}, seed={args.seed}")
    
    # Generate
    df, states = generate_hmm_data(cfg)
    
    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / f"hmm_{args.world}_N{args.N}_T{args.T}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Also save states
    npy_path = out_dir / f"true_states_{args.world}_N{args.N}_T{args.T}.npy"
    np.save(npy_path, states)
    print(f"Saved: {npy_path}")

if __name__ == "__main__":
    main()
