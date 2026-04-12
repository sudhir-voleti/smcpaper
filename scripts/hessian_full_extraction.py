import multiprocessing as mp
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

def parse_condition_from_path(pkl_path):
    path_str = str(pkl_path)
    n_match = re.search(r'N(\d+)', path_str)
    t_match = re.search(r'T(\d+)', path_str)
    N = int(n_match.group(1)) if n_match else 250
    T = int(t_match.group(1)) if t_match else 52
    
    pi0_match = re.search(r'pi0[_]?(\d+\.\d+)', path_str)
    rho_match = re.search(r'rho[_]?(\d+\.\d+)', path_str)
    pi0 = float(pi0_match.group(1)) if pi0_match else np.nan
    rho = float(rho_match.group(1)) if rho_match else np.nan
    
    rep_match = re.search(r'rep[_]?(\d+)', path_str)
    rep = int(rep_match.group(1)) if rep_match else -1
    
    return N, T, pi0, rho, rep

def process_single(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        
        idata = saved['idata']
        res = saved.get('res', {})
        
        N, T, pi0, rho, rep = parse_condition_from_path(pkl_path)
        fname = pkl_path.name
        model_type = 'BEMMAOR' if 'BEMMAOR' in fname else 'Hurdle'
        
        metrics = {
            'pkl_path': str(pkl_path),
            'model_type': model_type,
            'N': N,
            'T': T,
            'pi0': pi0,
            'rho': rho,
            'rep': rep,
            'log_evidence': res.get('log_evidence', np.nan),
        }
        
        if model_type == 'BEMMAOR':
            if 'gamma_h' in idata.posterior:
                gh = idata.posterior['gamma_h'].values
                metrics['gamma_h_mean'] = np.mean(gh)
                metrics['gamma_h_var'] = np.var(gh)
                metrics['gamma_h_precision'] = 1.0 / (np.var(gh) + 1e-6)
                metrics['info_silence_proxy'] = metrics['gamma_h_precision']
            
            if 'gamma_m' in idata.posterior:
                gm = idata.posterior['gamma_m'].values
                metrics['gamma_m_mean'] = np.mean(gm)
                metrics['gamma_m_var'] = np.var(gm)
            
            if 'theta' in idata.posterior:
                th = idata.posterior['theta'].values
                theta_var_per_customer = np.var(th, axis=(0, 1))
                metrics['theta_var_mean'] = np.mean(theta_var_per_customer)
                metrics['theta_var_min'] = np.min(theta_var_per_customer)
                metrics['theta_precision'] = 1.0 / (np.mean(theta_var_per_customer) + 1e-6)
        else:
            metrics['gamma_h_mean'] = 0.0
            metrics['gamma_h_var'] = 0.0
            metrics['gamma_h_precision'] = 0.0
            metrics['info_silence_proxy'] = 0.0
            
            if 'alpha0_h' in idata.posterior:
                a0 = idata.posterior['alpha0_h'].values
                metrics['alpha0_h_var'] = np.var(a0)
        
        return metrics
        
    except Exception as e:
        return {'pkl_path': str(pkl_path), 'error': str(e)}

if __name__ == '__main__':
    base_dirs = [
        Path('./results_T104_N250'),
        Path('./results_T52_N750'),
        Path('./results_T104_N500'),
        Path('./results_april05_N250_T52'),
        Path('./results_jasa_N500_T52'),
        Path('./results_jasa_pilot_N500_T52'),
    ]
    
    pkl_files = []
    for bd in base_dirs:
        if bd.exists():
            files = [p for p in bd.rglob('*.pkl') 
                    if ('BEMMAOR' in p.name or 'GAM' in p.name) 
                    and 'dgp' not in p.name]
            pkl_files.extend(files)
            print(f"{bd.name}: {len(files)} PKLs")
    
    print(f"\nTotal PKLs: {len(pkl_files)}")
    print(f"Using {mp.cpu_count()} cores...\n")
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(pool.imap_unordered(process_single, pkl_files, chunksize=10))
    
    results = [r for r in results if 'error' not in r]
    print(f"\nSuccessful: {len(results)}")
    
    df = pd.DataFrame(results)
    df.to_csv('hessian_all_models_complete.csv', index=False)
    print(f"\nSAVED: hessian_all_models_complete.csv")
    
    if len(df) > 0:
        print("\nSUMMARY: Information of Silence Proxy")
        summary = df.groupby(['model_type', 'N', 'T', 'pi0', 'rho'])['info_silence_proxy'].agg(['mean', 'count'])
        print(summary.round(2))
