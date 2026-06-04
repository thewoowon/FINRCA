"""
Score Blending Experiment: CVAE + z-score hybrid scoring.

Motivation
----------
v2 CVAE+APA-RCA+ResGCN improved Top-3 (+9.1pp) and Top-1 on A3/A4/A6,
but collapsed on A7 (coincidental: 76.7%→0%).  Root cause: the C-VAE's
source-type manifold (trained on GBM) misrepresents real market regime
changes (Fed 2022, AI rally 2023-24), underscoring source-node anomalies.

z-score handles A7 well (96.7% from APA-RCA-v2 baseline) precisely
because coincidental anomalies create large z-score deviations regardless
of distribution shift.

Hypothesis: blending CVAE and z-score preserves C-VAE gains on A3/A4/A6
while recovering A7 performance.

    blend_score(v) = α × score_CVAE(v) + (1-α) × score_z(v)

Protocol
--------
1. Train C-VAE on 60 clean pipelines (same as v2).
2. Collect 630 synthetic GNN training samples using CVAE scores.
3. Train final ResGCN on all 630 samples.
4. Run 210 RSHB trials; at each trial compute both scores,
   then evaluate α ∈ {0.0, 0.3, 0.5, 0.7, 1.0} in one pass.

Results saved to: results/ml_blend_results.csv
"""
import sys, os, time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synfrp import (RealDataConfig, RealMarketDataGenerator,
                    FinancialRiskPipeline, AnomalyInjector,
                    AnomalyType, get_injectable_nodes_for_type)
from frlg import FRLGBuilder, NodeAnomalyDetector
from rca import (APARCAEngine, VAEAnomalyScorer,
                 GNNReranker, build_gnn_sample, assign_cv_folds)
from baselines import select_target_node
from evaluation import ExperimentResults, compute_run_metrics
from experiments.runner import (
    PIPELINE_SIZES, ANOMALY_TYPES, N_TRIALS, RSHB_N_TRIALS,
    build_pipeline, _generate_clean_pipelines,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALPHAS     = [0.0, 0.3, 0.5, 0.7, 1.0]   # 0.0 = pure z-score, 1.0 = pure CVAE


def _blend(vae_scores: dict, z_scores: dict, alpha: float) -> dict:
    """α·CVAE + (1-α)·z-score per node."""
    all_nodes = set(vae_scores) | set(z_scores)
    return {
        nid: alpha * vae_scores.get(nid, 0.0) + (1 - alpha) * z_scores.get(nid, 0.0)
        for nid in all_nodes
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apa_engine = APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True)

    # ── Phase 1: Train C-VAE ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 1: Training C-VAE (type+layer conditioned)")
    print("="*60)
    clean_pipelines = _generate_clean_pipelines(n_per_size=20)
    vae_scorer = VAEAnomalyScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    vae_scorer.fit(clean_pipelines)

    # ── Phase 2: Collect GNN samples (630 synthetic trials) ──────────────────
    print("\n" + "="*60)
    print("Phase 2: Collecting GNN training samples (630 synthetic trials)")
    print("="*60)
    vae_samples = []
    total = len(ANOMALY_TYPES) * len(PIPELINE_SIZES) * N_TRIALS

    with tqdm(total=total, desc="GNN samples") as pbar:
        for size in PIPELINE_SIZES:
            for atype in ANOMALY_TYPES:
                for trial in range(N_TRIALS):
                    data_seed   = trial + 42
                    inject_seed = trial
                    pipeline    = build_pipeline(size, seed=data_seed)
                    rng         = np.random.default_rng(inject_seed)

                    cands = get_injectable_nodes_for_type(pipeline, atype)
                    if not cands:
                        pbar.update(1)
                        continue
                    target_inject = rng.choice(cands)
                    injector = AnomalyInjector(pipeline, rng)
                    injector, gt = injector.inject(
                        anomaly_type=atype, target_node_id=target_inject,
                        magnitude=8.0, seed=inject_seed,
                    )
                    pipeline.execute()

                    builder = FRLGBuilder(pipeline)
                    G       = builder.build()

                    try:
                        vae_scores = vae_scorer.score(pipeline)
                    except Exception:
                        pbar.update(1)
                        continue

                    builder.set_anomaly_scores(vae_scores)
                    obs = select_target_node(G, vae_scores, threshold=0.1)

                    try:
                        rca_r  = apa_engine.run(G, vae_scores, obs)
                        sample = build_gnn_sample(
                            G=G, scores=vae_scores, rca_result=rca_r,
                            true_rc=gt.target_node_id, top_k=10,
                            anomaly_type=atype.value,
                            pipeline_size=size, trial=trial,
                        )
                        if sample is not None:
                            vae_samples.append(sample)
                    except Exception:
                        pass
                    pbar.update(1)

    print(f"  Collected {len(vae_samples)} CVAE-GNN samples.")

    # ── Phase 3: Train final ResGCN on all synthetic samples ─────────────────
    print("\n" + "="*60)
    print("Phase 3: Training final ResGCN on all synthetic samples")
    print("="*60)
    vae_samples = assign_cv_folds(vae_samples, n_folds=5)
    gnn_final = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
    gnn_final.fit(vae_samples, verbose=False)
    print(f"  ResGCN trained on {len(vae_samples)} samples.")

    # ── Phase 4: RSHB blend evaluation ───────────────────────────────────────
    print("\n" + "="*60)
    print(f"Phase 4: RSHB blend evaluation  α={ALPHAS}")
    print("  α=0.0 → pure z-score, α=1.0 → pure CVAE")
    print("="*60)

    try:
        cfg      = RealDataConfig(size="medium", n_days=756, end_date="2024-12-31", seed=42)
        real_raw = RealMarketDataGenerator(cfg).generate_all()
        print("  Real market data loaded.")
    except Exception as e:
        print(f"  ERROR loading real data: {e}")
        return

    # One ExperimentResults per alpha
    all_results = {a: ExperimentResults() for a in ALPHAS}

    total_rshb = len(ANOMALY_TYPES) * RSHB_N_TRIALS
    with tqdm(total=total_rshb, desc="RSHB blend") as pbar:
        for atype in ANOMALY_TYPES:
            for trial in range(RSHB_N_TRIALS):
                inject_seed = trial
                rng_r       = np.random.default_rng(inject_seed)
                pipeline    = FinancialRiskPipeline(real_raw, size="medium")

                cands = get_injectable_nodes_for_type(pipeline, atype)
                if not cands:
                    pbar.update(1)
                    continue
                target_inject = rng_r.choice(cands)
                injector = AnomalyInjector(pipeline, rng_r)
                injector, gt = injector.inject(
                    anomaly_type=atype, target_node_id=target_inject,
                    magnitude=8.0, seed=inject_seed,
                )
                pipeline.execute()

                builder = FRLGBuilder(pipeline)
                G       = builder.build()

                # Compute both scores once per trial
                detector = NodeAnomalyDetector(threshold_zscore=2.0)
                z_scores = detector.detect_all(pipeline)
                try:
                    vae_scores = vae_scorer.score(pipeline)
                except Exception:
                    vae_scores = z_scores

                # Evaluate each alpha
                for alpha in ALPHAS:
                    blend_scores = _blend(vae_scores, z_scores, alpha)
                    builder.set_anomaly_scores(blend_scores)
                    obs = select_target_node(G, blend_scores, threshold=0.1)

                    try:
                        rca_r  = apa_engine.run(G, blend_scores, obs)
                        sample = build_gnn_sample(
                            G=G, scores=blend_scores, rca_result=rca_r,
                            true_rc=gt.target_node_id, top_k=10,
                            anomaly_type=atype.value,
                            pipeline_size="medium_real", trial=trial,
                        )
                        if sample is not None:
                            reranked = gnn_final.rerank(sample, rca_r)
                            mname = f"blend_a{int(alpha*10):02d}+APA+ResGCN"
                            m = compute_run_metrics(
                                result=reranked,
                                true_root_cause=gt.target_node_id,
                                anomaly_type=atype.value,
                                pipeline_size="medium_real",
                                trial=trial,
                                method_name=mname,
                            )
                            all_results[alpha].add(m)
                    except Exception:
                        pass

                pbar.update(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BLEND RESULTS — RSHB (210 trials each)")
    print("="*60)
    print(f"  {'α':>5}  {'Method':35s}  Top-1   Top-3   Top-5   MRR")
    print("  " + "-"*75)

    rows = []
    for alpha in ALPHAS:
        res = all_results[alpha]
        try:
            df = pd.DataFrame([vars(r) for r in res.records])
        except Exception:
            df = pd.DataFrame([r.__dict__ for r in res.records])

        if len(df) == 0:
            continue
        t1  = df["top1"].mean() * 100
        t3  = df["top3"].mean() * 100
        t5  = df["top5"].mean() * 100
        mrr = df["reciprocal_rank"].mean()
        mname = f"blend_a{int(alpha*10):02d}+APA+ResGCN"
        print(f"  {alpha:5.1f}  {mname:35s}  {t1:5.1f}%  {t3:5.1f}%  {t5:5.1f}%  {mrr:.3f}")
        rows.append({"alpha": alpha, "method": mname,
                     "top1": t1, "top3": t3, "top5": t5, "mrr": mrr,
                     "n": len(df)})

        # Per-type breakdown for this alpha
        for at in sorted(df["anomaly_type"].unique()):
            s = df[df["anomaly_type"] == at]
            rows[-1][at] = round(s["top1"].mean() * 100, 1)

    print()
    print("  Reference:")
    print("  v1 VAE+APA-RCA+GNN                     54.8%   65.7%   71.9%  0.620")
    print("  v2 CVAE+APA-RCA+ResGCN                 51.0%   74.8%   74.8%  0.626")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "ml_blend_results.csv")
    # Combine all into one df
    combined = pd.concat(
        [pd.DataFrame([vars(r) for r in res.records])
         for res in all_results.values()],
        ignore_index=True
    )
    combined.to_csv(out_path, index=False)
    print(f"\n  Full results → {out_path}")

    # Summary table
    summary_path = os.path.join(OUTPUT_DIR, "ml_blend_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"  Summary      → {summary_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
