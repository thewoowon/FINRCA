"""
v3 Formal Experiment: Hybrid TSD-CVAE + APA-RCA + ResGCN

Same protocol as v2 (run_ml_enhanced_experiments) but with Hybrid TSD-CVAE
replacing standard C-VAE as the anomaly scorer.

Phase 1: Train Hybrid TSD-CVAE on 60 clean pipelines
Phase 2: Collect 630 synthetic GNN samples using Hybrid TSD-CVAE scores
Phase 3: Train final ResGCN on all synthetic samples
Phase 4: RSHB 210 trials — report Top-1/3/5/MRR + per-type breakdown

Results saved to: results/v3_hybrid_results.csv
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
from rca import APARCAEngine, GNNReranker, build_gnn_sample, assign_cv_folds
from rca.arch_variants import HybridTSDCVAEScorer
from baselines import select_target_node
from evaluation import ExperimentResults, compute_run_metrics
from experiments.runner import (
    PIPELINE_SIZES, ANOMALY_TYPES, N_TRIALS, RSHB_N_TRIALS,
    build_pipeline, _generate_clean_pipelines,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apa_engine = APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True)

    # ── Phase 1: Train Hybrid TSD-CVAE ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1: Training Hybrid TSD-CVAE")
    print("=" * 60)
    clean_pipelines = _generate_clean_pipelines(n_per_size=20)
    scorer = HybridTSDCVAEScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    scorer.fit(clean_pipelines)

    # ── Phase 2: Collect GNN samples (630 synthetic) ────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Collecting ResGCN training samples (630 synthetic trials)")
    print("=" * 60)
    samples = []
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
                        scores = scorer.score(pipeline)
                    except Exception:
                        pbar.update(1)
                        continue

                    builder.set_anomaly_scores(scores)
                    obs = select_target_node(G, scores, threshold=0.1)

                    try:
                        rca_r  = apa_engine.run(G, scores, obs)
                        sample = build_gnn_sample(
                            G=G, scores=scores, rca_result=rca_r,
                            true_rc=gt.target_node_id, top_k=10,
                            anomaly_type=atype.value,
                            pipeline_size=size, trial=trial,
                        )
                        if sample is not None:
                            samples.append(sample)
                    except Exception:
                        pass
                    pbar.update(1)

    print(f"  Collected {len(samples)} samples.")

    # ── Phase 3: Train final ResGCN ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Training final ResGCN")
    print("=" * 60)
    samples = assign_cv_folds(samples, n_folds=5)
    reranker = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
    reranker.fit(samples, verbose=False)
    print(f"  ResGCN trained on {len(samples)} samples.")

    # ── Phase 4: RSHB evaluation ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 4: RSHB evaluation — Hybrid TSD-CVAE + APA-RCA + ResGCN (v3)")
    print("=" * 60)

    try:
        cfg      = RealDataConfig(size="medium", n_days=756, end_date="2024-12-31", seed=42)
        real_raw = RealMarketDataGenerator(cfg).generate_all()
        print("  Real market data loaded.")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    results     = ExperimentResults()
    results_apa = ExperimentResults()  # APA-RCA only (no GNN) for reference

    total_rshb = len(ANOMALY_TYPES) * RSHB_N_TRIALS
    with tqdm(total=total_rshb, desc="RSHB v3") as pbar:
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

                try:
                    scores = scorer.score(pipeline)
                except Exception:
                    pbar.update(1)
                    continue

                builder.set_anomaly_scores(scores)
                obs = select_target_node(G, scores, threshold=0.1)

                try:
                    rca_r = apa_engine.run(G, scores, obs)

                    # APA-RCA only (no GNN)
                    m_apa = compute_run_metrics(
                        result=rca_r,
                        true_root_cause=gt.target_node_id,
                        anomaly_type=atype.value,
                        pipeline_size="medium_real",
                        trial=trial,
                        method_name="HybridTSD+APA-RCA",
                    )
                    results_apa.add(m_apa)

                    # Full pipeline: + ResGCN
                    sample = build_gnn_sample(
                        G=G, scores=scores, rca_result=rca_r,
                        true_rc=gt.target_node_id, top_k=10,
                        anomaly_type=atype.value,
                        pipeline_size="medium_real", trial=trial,
                    )
                    if sample is not None:
                        reranked = reranker.rerank(sample, rca_r)
                        m = compute_run_metrics(
                            result=reranked,
                            true_root_cause=gt.target_node_id,
                            anomaly_type=atype.value,
                            pipeline_size="medium_real",
                            trial=trial,
                            method_name="HybridTSD+APA-RCA+ResGCN",
                        )
                        results.add(m)
                except Exception:
                    pass

                pbar.update(1)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("v3 FORMAL RESULTS — RSHB (210 trials)")
    print("=" * 60)

    for label, res in [("HybridTSD+APA-RCA (no GNN)", results_apa),
                       ("HybridTSD+APA-RCA+ResGCN (v3)", results)]:
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
        print(f"\n  {label}")
        print(f"    Overall:  Top-1={t1:.1f}%  Top-3={t3:.1f}%  Top-5={t5:.1f}%  MRR={mrr:.3f}  (N={len(df)})")
        print(f"    Per-type Top-1:")
        for at in sorted(df["anomaly_type"].unique()):
            s = df[df["anomaly_type"] == at]
            at_t1 = s["top1"].mean() * 100
            print(f"      {at:30s}  {at_t1:5.1f}%  (n={len(s)})")

    print("\n  Reference:")
    print("    v1 VAE+APA-RCA+GNN:         Top-1=54.8%  Top-3=65.7%  Top-5=71.9%  MRR=0.620")
    print("    v2 CVAE+APA-RCA+ResGCN:     Top-1=51.0%  Top-3=74.8%  Top-5=74.8%  MRR=0.626")

    # Save raw results
    all_records = []
    for r in results.records:
        d = vars(r) if hasattr(r, '__dict__') else r.__dict__
        all_records.append(d)
    if all_records:
        out_path = os.path.join(OUTPUT_DIR, "v3_hybrid_results.csv")
        pd.DataFrame(all_records).to_csv(out_path, index=False)
        print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
