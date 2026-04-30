import subprocess
import sys
from pathlib import Path
import time
import json

# Configurations to test
CONFIGS = [
    {'pi0': 0.90, 'rho': 0.4, 'psi': 5, 'reps': 5},
    {'pi0': 0.98, 'rho': 0.4, 'psi': 5, 'reps': 5},
    {'pi0': 0.98, 'rho': 0.3, 'psi': 5, 'reps': 5},
]

N, T = 1000, 104
BASE_DIR = Path("./results_k3_validation")
BASE_DIR.mkdir(exist_ok=True)

def run_single(config, rep, K_fit, model_script):
    """Run single K=3 DGP with K_fit model."""
    pi0, rho, psi = config['pi0'], config['rho'], config['psi']
    
    folder = BASE_DIR / f"pi0_{pi0:.2f}_rho_{rho:.1f}_psi_{psi}" / f"rep_{rep:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    
    seed = int(N*1000 + T*100 + pi0*10000 + psi*100 + rho*10 + rep + 5000)
    
    # Generate DGP
    dgp_path = folder / f"dgp_rep{rep:02d}.csv"
    if not dgp_path.exists():
        cmd = [
            "python", "dgp_k3.py",
            "--N", str(N), "--T", str(T),
            "--pi0", str(pi0), "--psi", str(psi), "--rho", str(rho),
            "--seed", str(seed),
            "--output", str(dgp_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  DGP failed: {result.stderr[:200]}")
            return None
    
    # Fit model
    out_dir = folder / f"K{K_fit}_{model_script.stem}"
    out_dir.mkdir(exist_ok=True)
    
    cmd = [
        "python", str(model_script),
        "--csv_path", str(dgp_path),
        "--K", str(K_fit),
        "--draws", "1000",
        "--chains", "2",
        "--out_dir", str(out_dir),
        "--seed", str(seed + 1000)
    ]
    
    print(f"  Running K={K_fit} {model_script.stem}...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start
    
    # Parse log-evidence from output
    log_ev = None
    for line in result.stdout.split('\n'):
        if 'Log-Evidence' in line or 'log_evidence' in line:
            try:
                log_ev = float(line.split(':')[1].strip())
            except:
                pass
    
    success = result.returncode == 0 and log_ev is not None
    
    print(f"  {'✓' if success else '✗'} K={K_fit} {model_script.stem}: "
          f"log_ev={log_ev:.1f if log_ev else 'N/A'}, {elapsed:.1f}s")
    
    return {
        'config': config,
        'rep': rep,
        'K_fit': K_fit,
        'model': model_script.stem,
        'log_evidence': log_ev,
        'success': success,
        'time': elapsed
    }

# Main
print("="*70)
print("K=3 VALIDATION RUN")
print("="*70)

results = []
for config in CONFIGS:
    print(f"\nConfig: π₀={config['pi0']:.2f}, ρ={config['rho']:.1f}, ψ={config['psi']}")
    
    for rep in range(config['reps']):
        print(f"\n  Rep {rep+1}/{config['reps']}")
        
        # K=3 BEMMAOR
        res = run_single(config, rep, 3, Path("smc_bemmaor_new.py"))
        if res: results.append(res)
        
        # K=2 BEMMAOR (underfit test)
        res = run_single(config, rep, 2, Path("smc_bemmaor_new.py"))
        if res: results.append(res)
        
        # K=3 Hurdle
        res = run_single(config, rep, 3, Path("smc_hurdle_new.py"))
        if res: results.append(res)
        
        # K=2 Hurdle
        res = run_single(config, rep, 2, Path("smc_hurdle_new.py"))
        if res: results.append(res)

# Save results
with open(BASE_DIR / "k3_validation_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"COMPLETE: {len(results)} runs")
print(f"Saved: {BASE_DIR / 'k3_validation_results.json'}")
print(f"{'='*70}")
