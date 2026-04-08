#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR: Flexible N, T, Chains, Draws
Usage: python master_orchestrator.py --N 750 --T 104 --terminal 1 --chains 2 --draws 1000
"""

import numpy as np
import pickle
import time
import json
import argparse
from pathlib import Path
import importlib.util

BEMMAOR_PATH = '/Users/sudhirvoleti/jrssc_april/smc_bemmaor_new.py'
HURDLE_PATH = '/Users/sudhirvoleti/jrssc_april/smc_hurdle_new.py'

def generate_dgp(N=250, T=52, K=2, pi0=0.9, psi=5, rho=0.4, seed=42):
    """Generate DGP with flexible N, T."""
    rng = np.random.default_rng(seed)
    
    stickiness = 0.85 + 0.1 * (1 / psi)
    Gamma = np.array([
        [stickiness, 1-stickiness],
        [(1-stickiness)*0.7, 1-(1-stickiness)*0.7]
    ])
    
    pi0_vec = np.array([pi0, 1-pi0])
    r_nb = np.array([1.0, 2.0])
    alpha_h = np.array([-1.0 - 0.5*(1-pi0), 0.5])
    alpha_gamma = np.array([2.0, 5.0])
    beta_m = np.array([1.0, 2.5])
    
    theta = rng.normal(0, 1, size=(N, 1))
    gamma_h = rho * 0.6
    gamma_m = rho * 1.0

    Z = np.zeros((N, T), dtype=int)
    for i in range(N):
        Z[i, 0] = rng.choice(K, p=pi0_vec)
        for t in range(1, T):
            Z[i, t] = rng.choice(K, p=Gamma[Z[i, t-1], :])

    Y = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            k = Z[i, t]
            lam = np.exp(alpha_h[k] + gamma_h * theta[i, 0])
            p_zero = (r_nb[k] / (r_nb[k] + lam)) ** r_nb[k]
            if rng.random() > p_zero:
                mu_spend = np.exp(beta_m[k] + gamma_m * theta[i, 0])
                beta_gamma = alpha_gamma[k] / mu_spend
                Y[i, t] = rng.gamma(alpha_gamma[k], 1/beta_gamma)

    sparsity = np.mean(Y == 0)
    return {
        'Y': Y, 'Z': Z, 'Gamma': Gamma, 'pi0': pi0_vec, 'theta': theta,
        'gamma_h': gamma_h, 'gamma_m': gamma_m, 'N': N, 'T': T, 'K': K,
        'seed': seed, 'params': {'pi0': pi0, 'psi': psi, 'rho': rho},
        'sparsity': sparsity, 'true_rho': gamma_h * gamma_m
    }

def compute_rfm_features(y, mask):
    """Compute RFM features."""
    N, T = y.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    
    for i in range(N):
        last_purchase, cum_freq, cum_spend = -1, 0, 0.0
        for t in range(T):
            if mask[i, t]:
                if y[i, t] > 0:
                    last_purchase, cum_freq = t, cum_freq + 1
                    cum_spend += y[i, t]
                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cum_freq
                    M[i, t] = cum_spend / cum_freq if cum_freq > 0 else 0.0
                else:
                    R[i, t], F[i, t], M[i, t] = t + 1, 0, 0.0
            else:
                R[i, t], F[i, t], M[i, t] = t + 1, 0, 0.0
    
    return R, F, M

def dgp_to_mksc_format(Y, Z_true=None, world="flexible"):
    """Convert DGP to MK SC format."""
    N, T_full = Y.shape
    y_train = Y.copy()
    mask_train = ~np.isnan(y_train) & (y_train >= 0)
    y_train = np.where(mask_train, y_train, 0.0).astype(np.float32)
    
    R_train, F_train, M_train = compute_rfm_features(y_train, mask_train)
    M_train_log = np.log1p(M_train)
    
    R_valid = R_train[mask_train]
    F_valid = F_train[mask_train]
    M_valid = M_train_log[mask_train]
    
    if len(R_valid) > 0 and R_valid.std() > 0:
        R_train = (R_train - R_valid.mean()) / (R_valid.std() + 1e-6)
    if len(F_valid) > 0 and F_valid.std() > 0:
        F_train = (F_train - F_valid.mean()) / (F_valid.std() + 1e-6)
    if len(M_valid) > 0 and M_valid.std() > 0:
        M_train_scaled = (M_train_log - M_valid.mean()) / (M_valid.std() + 1e-6)
    else:
        M_train_scaled = M_train_log
    
    return {
        'N': N, 'T': T_full,
        'y': y_train.astype(np.float32),
        'mask': mask_train.astype(bool),
        'R': R_train.astype(np.float32),
        'F': F_train.astype(np.float32),
        'M': M_train_scaled.astype(np.float32),
        'true_states': Z_true.astype(np.int32) if Z_true is not None else np.zeros((N, T_full), dtype=np.int32) - 1,
        'world': world,
        'M_raw': M_train.astype(np.float32),
        'T_total': T_full,
        'train_ratio': 1.0
    }

def run_model(model_type, data, K, draws, chains, out_dir, seed=42):
    """Run SMC model with flexible draws/chains."""
    model_path = BEMMAOR_PATH if model_type == "BEMMAOR" else HURDLE_PATH
    
    spec = importlib.util.spec_from_file_location(f"mksc_{model_type.lower()}", model_path)
    mksc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mksc)
    
    start = time.time()
    try:
        if hasattr(mksc, 'run_smc_bemmaor'):
            pkl_path, res, idata = mksc.run_smc_bemmaor(data, K, draws, chains, seed, out_dir)
        elif hasattr(mksc, 'run_smc_hurdle'):
            pkl_path, res, idata = mksc.run_smc_hurdle(data, K, draws, chains, seed, out_dir)
        else:
            pkl_path, res, idata = mksc.run_smc(data, K, draws, chains, seed, out_dir)
        
        elapsed = (time.time() - start) / 60
        return {
            'success': True,
            'pkl_path': str(pkl_path),
            'log_evidence': res.get('log_evidence', np.nan),
            'ess_min': res.get('ess_min', np.nan),
            'time_min': elapsed,
            'model_type': model_type
        }
    except Exception as e:
        elapsed = (time.time() - start) / 60
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'time_min': elapsed}

def run_single_condition(pi0, psi, rho, rep, N, T, base_dir, draws, chains, bem_draws=None, hur_draws=None):
    """Run single condition with flexible N, T, draws."""
    folder_name = f"pi0_{pi0:.2f}_psi_{psi}_rho_{rho:.1f}_T{T}_N{N}"
    rep_folder = f"rep_{rep:02d}"
    out_dir = Path(base_dir) / folder_name / rep_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seed = int(N*1000 + T*100 + pi0 * 10000 + psi * 100 + rho * 10 + rep)
    world_str = f"N{N}_T{T}_pi0{pi0:.2f}_psi{psi}_rho{rho:.1f}_rep{rep:02d}"

    print(f"\n[{folder_name}/{rep_folder}] π₀={pi0}, ψ={psi}, ρ={rho}, rep={rep}, N={N}, T={T}")

    dgp = generate_dgp(N=N, T=T, K=2, pi0=pi0, psi=psi, rho=rho, seed=seed)
    dgp_filename = f"dgp_N{N}_T{T}_pi0{pi0:.2f}_psi{psi}_rho{rho:.1f}_rep{rep:02d}_seed{seed}.pkl"
    dgp_path = out_dir / dgp_filename
    
    with open(dgp_path, 'wb') as f:
        pickle.dump(dgp, f)
    
    print(f" DGP: {dgp_filename} (sparsity={dgp['sparsity']:.1%}, N={N}, T={T})")

    data = dgp_to_mksc_format(dgp['Y'], dgp['Z'], world=world_str)

    results = {}
    
    # BEMMAOR with specified draws
    bem_d = bem_draws if bem_draws else draws
    print(f" BEMMAOR (D={bem_d})...")
    res_bem = run_model("BEMMAOR", data, 2, bem_d, chains, str(out_dir), seed)
    if res_bem['success']:
        results['BEMMAOR'] = res_bem
        print(f" ✓ Log-ev={res_bem['log_evidence']:.2f}, ESS={res_bem['ess_min']:.1f}, {res_bem['time_min']:.1f}min")
    else:
        print(f" ✗ Failed: {res_bem.get('error', 'Unknown')}")

    # Hurdle with specified draws (or default)
    hur_d = hur_draws if hur_draws else draws
    print(f" Hurdle (D={hur_d})...")
    res_hur = run_model("Hurdle", data, 2, hur_d, chains, str(out_dir), seed+1000)
    if res_hur['success']:
        results['Hurdle'] = res_hur
        print(f" ✓ Log-ev={res_hur['log_evidence']:.2f}, ESS={res_hur['ess_min']:.1f}, {res_hur['time_min']:.1f}min")
    else:
        print(f" ✗ Failed: {res_hur.get('error', 'Unknown')}")

    summary = {
        'params': {'pi0': pi0, 'psi': psi, 'rho': rho, 'rep': rep, 'seed': seed, 'N': N, 'T': T},
        'dgp': {'sparsity': dgp['sparsity'], 'true_rho': dgp['true_rho']},
        'results': results
    }
    summary_path = out_dir / f"summary_N{N}_T{T}_pi0{pi0:.2f}_psi{psi}_rho{rho:.1f}_rep{rep:02d}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return summary

def main():
    parser = argparse.ArgumentParser(description='Master Orchestrator: Flexible N, T, Chains, Draws')
    parser.add_argument('--N', type=int, required=True, help='Population size (250, 500, 750, 1000)')
    parser.add_argument('--T', type=int, required=True, help='Time periods (52, 104)')
    parser.add_argument('--pi0', type=float, required=True, help='Sparsity level (0.75, 0.90, 0.95)')
    parser.add_argument('--rho', type=float, required=True, help='Coupling (0.0, 0.4, 0.8)')
    parser.add_argument('--psi', type=int, default=5, help='Stickiness parameter')
    parser.add_argument('--terminal', type=int, default=1, help='Terminal ID for logging')
    parser.add_argument('--reps', type=int, default=5, help='Repetitions per condition')
    parser.add_argument('--chains', type=int, default=2, help='SMC chains')
    parser.add_argument('--draws', type=int, default=1000, help='Draws for both models')
    parser.add_argument('--bem-draws', type=int, default=None, help='BEMMAOR-specific draws (optional)')
    parser.add_argument('--hur-draws', type=int, default=None, help='Hurdle-specific draws (optional)')
    parser.add_argument('--base-dir', type=str, default='./results', help='Output directory')
    args = parser.parse_args()

    # Auto-create output folder based on N, T
    base_dir = f"{args.base_dir}_N{args.N}_T{args.T}"
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MASTER ORCHESTRATOR: N={args.N}, T={args.T}")
    print(f"Condition: π₀={args.pi0}, ρ={args.rho}, ψ={args.psi}")
    print(f"Reps: 1-{args.reps}, Chains: {args.chains}")
    print(f"BEMMAOR draws: {args.bem_draws or args.draws}, Hurdle draws: {args.hur_draws or args.draws}")
    print(f"Output: {base_dir}")
    print(f"Terminal: {args.terminal}")
    print("=" * 70)

    total_start = time.time()

    for rep in range(args.reps):
        print(f"\n{'='*70}")
        print(f"REP {rep+1}/{args.reps}")
        print(f"{'='*70}")
        
        run_single_condition(
            pi0=args.pi0, psi=args.psi, rho=args.rho, rep=rep,
            N=args.N, T=args.T, base_dir=base_dir,
            draws=args.draws, chains=args.chains,
            bem_draws=args.bem_draws, hur_draws=args.hur_draws
        )

    total_elapsed = (time.time() - total_start) / 60

    print("\n" + "=" * 70)
    print(f"TERMINAL {args.terminal} COMPLETE: {total_elapsed:.1f} minutes")
    print(f"N={args.N}, T={args.T}, π₀={args.pi0}, ρ={args.rho} finished")
    print("=" * 70)

if __name__ == "__main__":
    main()
EOFMASTER

echo "Master orchestrator saved."
