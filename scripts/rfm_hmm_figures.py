"""
RFM-HMM Figure Generation for Marketing Science Submission
Author: Sudhir Voleti
Date: March 20, 2026
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_beta_trajectory(idata):
    beta_raw = idata.sample_stats['beta'].values
    trajectories = []
    for i in range(beta_raw.shape[0]):
        for j in range(beta_raw.shape[1]):
            beta_list = beta_raw[i, j]
            if isinstance(beta_list, list):
                trajectories.append(np.array(beta_list))
    return trajectories


def viterbi_decode(alpha_filtered):
    alpha_mean = alpha_filtered.mean(axis=(0, 1))
    return alpha_mean.argmax(axis=-1)


def compute_state_occupancy(z_viterbi, K):
    N, T = z_viterbi.shape
    occupancy = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            occupancy[t, k] = (z_viterbi[:, t] == k).mean()
    return occupancy


def plot_fig1_tempering(pkl_paths, labels, colors=None, save_path=None):
    if colors is None:
        colors = {'BEMMAOR': '#2E86AB', 'Hurdle': '#A23B72', 'Tweedie': '#F18F01'}
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    for model_id, pkl_path in pkl_paths.items():
        data = load_pkl(pkl_path)
        beta_traj = extract_beta_trajectory(data['idata'])
        model_type = model_id.split('_')[0]
        color = colors.get(model_type, 'gray')
        label = labels.get(model_id, model_id)
        
        for chain_idx, beta in enumerate(beta_traj):
            stages = np.arange(len(beta))
            alpha = 0.6 if len(beta_traj) > 1 else 1.0
            chain_label = label if chain_idx == 0 else None
            ax.plot(stages, beta, color=color, alpha=alpha, label=chain_label, linewidth=1.5)
    
    ax.set_xlabel('SMC Iteration', fontsize=11)
    ax.set_ylabel(r'Inverse Temperature $\beta$', fontsize=11)
    ax.set_title('Particle Tempering Schedule: Harbor World (K=3)', fontsize=12)
    ax.legend(loc='lower right', frameon=True, title='Model')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
        print(f"Saved Figure 1: {save_path}")
    return fig


def plot_fig2_ppc(pkl_paths, labels, colors=None, save_path=None):
    if colors is None:
        colors = {'BEMMAOR': '#2E86AB', 'Hurdle': '#A23B72'}
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    model_names = []
    
    # Collect data for all models first
    for model_id, pkl_path in pkl_paths.items():
        data = load_pkl(pkl_path)
        res = data['res']
        y_obs = data['data']['y']
        model_type = model_id.split('_')[0]
        color = colors.get(model_type, 'gray')
        label = labels.get(model_id, model_id)
        model_names.append(label)
        
        # Panel 1: Spend distribution (only for models with full PPC)
        if 'ppc_simulations' in res:
            ppc = res['ppc_simulations']
            y_obs_flat = y_obs.flatten()
            y_sim_flat = ppc.mean(axis=0).flatten()
            
            ax = axes[0]
            bins = np.logspace(0, np.log10(max(y_obs_flat.max(), y_sim_flat.max()) + 1), 50)
            ax.hist(y_obs_flat[y_obs_flat > 0], bins=bins, alpha=0.5, color='black', label='Observed', density=True)
            ax.hist(y_sim_flat[y_sim_flat > 0], bins=bins, alpha=0.5, color=color, label=label, density=True)
            ax.set_xscale('log')
            ax.set_xlabel('Spend ($)')
            ax.set_ylabel('Density')
            ax.set_title('Spend Distribution (Positive)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Panel 2 & 3: Bar charts for all models
    x = np.arange(len(model_names))
    width = 0.35
    
    # Zero proportion
    ax = axes[1]
    zero_obs_vals = []
    zero_sim_vals = []
    zero_sim_errs = []
    
    for model_id, pkl_path in pkl_paths.items():
        data = load_pkl(pkl_path)
        res = data['res']
        y_obs = data['data']['y']
        zero_obs_vals.append(res.get('ppc_zero_obs', (y_obs == 0).mean()) * 100)
        zero_sim_vals.append(res.get('ppc_zero_sim_mean', 0) * 100)
        zero_sim_errs.append(res.get('ppc_zero_sim_std', 0) * 100)
    
    ax.bar(x - width/2, zero_obs_vals, width, label='Observed', color='black', alpha=0.7)
    ax.bar(x + width/2, zero_sim_vals, width, yerr=zero_sim_errs, label='Simulated', 
           color=[colors.get(n.split('-')[0], 'gray') for n in model_names], alpha=0.7, capsize=3)
    ax.set_ylabel('Zero Spend %')
    ax.set_title('Zero-Inflation Check')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # P99 tail
    ax = axes[2]
    p99_obs_vals = []
    p99_sim_vals = []
    p99_sim_errs = []
    
    for model_id, pkl_path in pkl_paths.items():
        data = load_pkl(pkl_path)
        res = data['res']
        p99_obs_vals.append(res.get('ppc_p99_obs', 0))
        p99_sim_vals.append(res.get('ppc_p99_sim_mean', 0))
        p99_sim_errs.append(res.get('ppc_p99_sim_std', 0))
    
    ax.bar(x - width/2, p99_obs_vals, width, label='Observed', color='black', alpha=0.7)
    ax.bar(x + width/2, p99_sim_vals, width, yerr=p99_sim_errs, label='Simulated',
           color=[colors.get(n.split('-')[0], 'gray') for n in model_names], alpha=0.7, capsize=3)
    ax.set_ylabel('99th Percentile Spend ($)')
    ax.set_title('Tail Behavior (Whale Detection)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
        print(f"Saved Figure 2: {save_path}")
    return fig


def plot_fig3_occupancy(pkl_path, model_label, state_names=None, save_path=None):
    data = load_pkl(pkl_path)
    idata = data['idata']
    K = data['res']['K']
    T = data['res']['T']
    
    alpha_filtered = idata.posterior['alpha_filtered'].values
    z_viterbi = viterbi_decode(alpha_filtered)
    occupancy = compute_state_occupancy(z_viterbi, K)
    
    if state_names is None:
        state_names = [f'State {k}' for k in range(K)]
    colors = ['#d73027', '#fee08b', '#1a9850'][:K]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    weeks = np.arange(T)
    ax.stackplot(weeks, occupancy.T, labels=state_names, colors=colors, alpha=0.85)
    
    ax.set_xlabel('Week', fontsize=11)
    ax.set_ylabel('Population Proportion', fontsize=11)
    ax.set_title(f'State Occupancy Dynamics: {model_label}', fontsize=12)
    ax.legend(loc='upper left', frameon=True, bbox_to_anchor=(1.02, 1))
    ax.set_xlim(0, T-1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
        print(f"Saved Figure 3: {save_path}")
    return fig


if __name__ == "__main__":
    BASE_DIR = Path("/Users/sudhirvoleti/research related/SMC paper Feb2026")
    OUT_DIR = BASE_DIR / "figures"
    OUT_DIR.mkdir(exist_ok=True)
    
    print("Generating RFM-HMM Figures...")
    
    # Figure 1
    fig1_paths = {
        'BEMMAOR_Harbor': BASE_DIR / "simul_bemmaor_rerun/harbor/harbor/smc_K3_BEMMAOR_properCLV_N500_T41_D500.pkl",
        'Hurdle_Harbor': BASE_DIR / "simul_hurdle_relaxed/harbor/harbor/smc_K3_GAM_N500_T41_D500.pkl",
        'Tweedie_Harbor': BASE_DIR / "march10_simplest/harbor/smc_K3_TWEEDIE_GLM_p1.5_N500_T41_D1000.pkl"
    }
    fig1_labels = {
        'BEMMAOR_Harbor': 'BEMMAOR-SMC',
        'Hurdle_Harbor': 'Hurdle-GAM',
        'Tweedie_Harbor': 'Tweedie-GLM'
    }
    plot_fig1_tempering(fig1_paths, fig1_labels, save_path=OUT_DIR / "fig1_tempering.pdf")
    
    # Figure 2 (BEMMAOR and Hurdle only - they have full PPC)
    fig2_paths = {
        'BEMMAOR_UCI': BASE_DIR / "empirics_uci/bemmaor_k3_relaxed_terminal4/uci/smc_K3_BEMMAOR_properCLV_N500_T42_D500.pkl",
        'Hurdle_UCI': BASE_DIR / "empirics_uci/hurdle_rerun/uci/smc_K3_GLM_N500_T42_D1000.pkl"
    }
    fig2_labels = {
        'BEMMAOR_UCI': 'BEMMAOR-SMC',
        'Hurdle_UCI': 'Hurdle-GLM'
    }
    plot_fig2_ppc(fig2_paths, fig2_labels, save_path=OUT_DIR / "fig2_ppc.pdf")
    
    # Figure 3
    fig3_path = BASE_DIR / "empirics_uci/bemmaor_k3_relaxed_terminal4/uci/smc_K3_BEMMAOR_properCLV_N500_T42_D500.pkl"
    plot_fig3_occupancy(fig3_path, 'BEMMAOR-SMC (UCI)', 
                       state_names=['Dormant', 'Lukewarm', 'Engaged'],
                       save_path=OUT_DIR / "fig3_occupancy.pdf")
    
    print(f"All figures saved to: {OUT_DIR}")
