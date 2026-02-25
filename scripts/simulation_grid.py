"""
simulation_dgp_hmm_final.py
Principled 4-World 3-State HMM with Hierarchical Heterogeneity
DGP for SMC Methodological Enablement Paper
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class WorldConfig:
    name: str
    sparsity_level: str  # "Low" or "High"
    tail_level: str      # "Light" or "Heavy"
    pi0_base: np.ndarray        # Base sparsity for [Dormant, Lukewarm, Whale]
    mu_base: np.ndarray         # Base mean spend for [Dormant, Lukewarm, Whale]
    alpha_base: np.ndarray      # Gamma shape for [Dormant, Lukewarm, Whale]
    Gamma: np.ndarray           # 3x3 Transition Matrix
    sigma_re: float = 0.40      # Strength of customer-level heterogeneity
    N: int = 200
    T: int = 52
    seed: int = 42

def generate_hmm_data(cfg: WorldConfig):
    rng = np.random.default_rng(cfg.seed)
    K = 3
    
    # 1. Hierarchical Customer Random Effects (Heterogeneity)
    # Draw shifts in logit(pi0) and log(mu) space for each customer-state pair
    re_pi0 = rng.normal(0, cfg.sigma_re, size=(cfg.N, K))
    re_mu  = rng.normal(0, cfg.sigma_re, size=(cfg.N, K))
    
    # Transform bases
    logit_pi0 = np.log(cfg.pi0_base / (1 - cfg.pi0_base))
    pi0_ik = 1 / (1 + np.exp(-(logit_pi0 + re_pi0)))
    
    mu_ik = np.exp(np.log(cfg.mu_base) + re_mu)
    beta_ik = cfg.alpha_base / mu_ik  # beta = shape / mean
    
    # 2. State Evolution (Markov Chain)
    states = np.zeros((cfg.N, cfg.T), dtype=int)
    init_dist = [0.60, 0.30, 0.10] # Your 60/30/10 split
    states[:, 0] = rng.choice(K, size=cfg.N, p=init_dist)
    
    for t in range(1, cfg.T):
        for i in range(cfg.N):
            states[i, t] = rng.choice(K, p=cfg.Gamma[states[i, t-1]])
            
    # 3. Fully Vectorized Emissions
    cust_idx = np.repeat(np.arange(cfg.N), cfg.T)
    time_idx = np.tile(np.arange(cfg.T), cfg.N)
    flat_states = states.flatten()
    
    # Broad-cast customer-state parameters to observation level
    pi0_obs = pi0_ik[cust_idx, flat_states]
    alpha_obs = cfg.alpha_base[flat_states]
    beta_obs = beta_ik[cust_idx, flat_states]
    
    # Generate data
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

# --- Define the 4 Worlds ---
# Persistence Matrix (Diagonals high)
G = np.array([[0.92, 0.06, 0.02], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]])

worlds = [
    # World 1: Harbor (Low Sparsity, Light Tails)
    WorldConfig("Harbor", "Low", "Light", 
                np.array([0.90, 0.50, 0.15]), np.array([5, 25, 120]), np.array([4.0, 4.0, 4.0]), G),
    
    # World 4: Cliff (High Sparsity, Heavy Tails)
    WorldConfig("Cliff", "High", "Heavy", 
                np.array([0.98, 0.85, 0.40]), np.array([2, 10, 80]), np.array([0.8, 0.7, 0.5]), G)
]

# Run simulation for 'The Cliff'
df_cliff, state_matrix = generate_hmm_data(worlds[1])
