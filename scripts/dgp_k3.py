import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import argparse

def generate_dgp_k3(N=1000, T=104, pi0=0.9, psi=5, rho=0.4, seed=42):
    """Generate K=3 DGP with flexible N, T."""
    rng = np.random.default_rng(seed)
    K = 3
    
    # 3-state transition matrix: Dormant(0) -> Transactional(1) -> Champion(2)
    # Stickiness decreases as state value increases (Champions less sticky)
    stickiness = 0.85 + 0.1 * (1 / psi)
    Gamma = np.array([
        [stickiness, (1-stickiness)*0.7, (1-stickiness)*0.3],
        [(1-stickiness)*0.4, stickiness, (1-stickiness)*0.6],
        [(1-stickiness)*0.2, (1-stickiness)*0.3, 1-(1-stickiness)*0.5]
    ])
    # Normalize rows
    Gamma = Gamma / Gamma.sum(axis=1, keepdims=True)
    
    # Initial state distribution: more mass in lower states
    pi0_vec = np.array([pi0, (1-pi0)*0.6, (1-pi0)*0.4])
    pi0_vec = pi0_vec / pi0_vec.sum()
    
    # Emission parameters: increasing activity with state
    r_nb = np.array([1.0, 2.0, 3.0])
    alpha_h = np.array([-1.5 - 0.5*(1-pi0), -0.5, 0.5])
    alpha_gamma = np.array([2.0, 5.0, 8.0])
    beta_m = np.array([1.0, 2.5, 4.0])
    
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
        'sparsity': sparsity
    }

def dgp_to_csv(dgp, out_path):
    """Convert DGP to CSV format expected by model scripts."""
    N, T = dgp['N'], dgp['T']
    Y = dgp['Y']
    Z = dgp['Z']
    
    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({
                'customer_id': i,
                't': t,
                'y': Y[i, t],
                'true_state': Z[i, t]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--T', type=int, default=104)
    parser.add_argument('--pi0', type=float, default=0.9)
    parser.add_argument('--psi', type=int, default=5)
    parser.add_argument('--rho', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    dgp = generate_dgp_k3(args.N, args.T, args.pi0, args.psi, args.rho, args.seed)
    dgp_to_csv(dgp, args.output)
    
    # Also save DGP pickle
    pkl_path = args.output.replace('.csv', '_dgp.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dgp, f)
    
    print(f"K=3 DGP: N={dgp['N']}, T={dgp['T']}, pi0={dgp['params']['pi0']:.2f}, "
          f"psi={dgp['params']['psi']}, rho={dgp['params']['rho']:.1f}, "
          f"sparsity={dgp['sparsity']:.1%}")
    print(f"Saved: {args.output}")
    print(f"DGP saved: {pkl_path}")
