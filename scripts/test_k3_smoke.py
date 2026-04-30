import subprocess
import sys
from pathlib import Path

print("="*70)
print("K=3 PIPELINE SMOKE TEST (N=100, T=52, draws=100)")
print("="*70)

N, T = 100, 52
pi0, psi, rho = 0.90, 5, 0.4
seed = 42

test_dir = Path("./test_k3_pilot")
test_dir.mkdir(exist_ok=True)

# Step 1: Generate K=3 DGP
print("\n[1/4] Generating K=3 DGP...")
dgp_path = test_dir / "test_dgp.csv"
cmd = [
    "python", "dgp_k3.py",
    "--N", str(N), "--T", str(T),
    "--pi0", str(pi0), "--psi", str(psi), "--rho", str(rho),
    "--seed", str(seed),
    "--output", str(dgp_path)
]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"FAIL: {result.stderr}")
    sys.exit(1)
print("✓ DGP generated")

# Step 2: Fit K=3 BEMMAOR
print("\n[2/4] Fitting K=3 BEMMAOR (draws=100)...")
out_dir = test_dir / "k3_bem"
out_dir.mkdir(exist_ok=True)
cmd = [
    "python", "smc_bemmaor_new.py",
    "--csv_path", str(dgp_path),
    "--K", "3",
    "--draws", "100",
    "--chains", "2",
    "--out_dir", str(out_dir),
    "--seed", str(seed + 1000)
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print(f"FAIL: {result.stderr[:300]}")
    sys.exit(1)
print("✓ K=3 BEMMAOR completed")

# Step 3: Fit K=2 BEMMAOR
print("\n[3/4] Fitting K=2 BEMMAOR (draws=100)...")
out_dir = test_dir / "k2_bem"
out_dir.mkdir(exist_ok=True)
cmd = [
    "python", "smc_bemmaor_new.py",
    "--csv_path", str(dgp_path),
    "--K", "2",
    "--draws", "100",
    "--chains", "2",
    "--out_dir", str(out_dir),
    "--seed", str(seed + 2000)
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print(f"FAIL: {result.stderr[:300]}")
    sys.exit(1)
print("✓ K=2 BEMMAOR completed")

# Step 4: Fit K=3 Hurdle
print("\n[4/4] Fitting K=3 Hurdle (draws=100)...")
out_dir = test_dir / "k3_hur"
out_dir.mkdir(exist_ok=True)
cmd = [
    "python", "smc_hurdle_new.py",
    "--csv_path", str(dgp_path),
    "--K", "3",
    "--draws", "100",
    "--chains", "2",
    "--out_dir", str(out_dir),
    "--seed", str(seed + 3000)
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print(f"FAIL: {result.stderr[:300]}")
    sys.exit(1)
print("✓ K=3 Hurdle completed")

print("\n" + "="*70)
print("ALL SMOKE TESTS PASSED")
print("="*70)
