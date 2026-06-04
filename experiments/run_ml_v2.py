"""
Run ML-Enhanced Pipeline v2 experiments only.

Architecture changes vs v1:
  VAE  → C-VAE  (Transform-Type + Layer conditioned, N_COND=11)
  GCN  → ResGCN (ATF-weighted adjacency + residual skip connection)
  F=7  → F=8    (added path-distance-from-target feature)

Results saved to: results/ml_enhanced_v2_results.csv
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_ml_enhanced_experiments

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("FINRCA ML-Enhanced Pipeline v2")
print("  VAE  → C-VAE  (type+layer conditioned, N_COND=11)")
print("  GCN  → ResGCN (ATF-weighted edges + residual skip)")
print("  F=7  → F=8    (+ backward path-distance feature)")
print("=" * 60)

t0 = time.time()
run_ml_enhanced_experiments(output_dir)
print(f"\nDone in {(time.time()-t0)/60:.1f} min  →  results/ml_enhanced_v2_results.csv")
