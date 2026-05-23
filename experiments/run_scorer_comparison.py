"""
Anomaly Scorer Comparison: VAE vs IsolationForest vs OneClassSVM

Ablation experiment to answer: "왜 VAE인가, 다른 비지도 방법은 안 되는가?"

Protocol:
  1. Train all three scorers on the same 60 clean pipelines.
  2. For each scorer, run 630 synthetic trials → collect GNN samples.
  3. Train GNN for each scorer branch on synthetic samples.
  4. Run 210 RSHB trials → report Top-1 / Top-3 / Top-5 / MRR.

Results are printed to stdout and saved to:
    results/scorer_comparison.csv
"""

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synfrp import (MarketDataConfig, MarketDataGenerator,
                    RealDataConfig, RealMarketDataGenerator,
                    FinancialRiskPipeline, AnomalyInjector,
                    AnomalyType, get_injectable_nodes_for_type)
from frlg import FRLGBuilder, NodeAnomalyDetector
from rca import (APARCAEngine, VAEAnomalyScorer, IFAnomalyScorer,
                 OCSVMAnomalyScorer, GNNReranker, build_gnn_sample, assign_cv_folds)
from baselines import select_target_node
from evaluation import ExperimentResults, compute_run_metrics

# ── Re-use runner helpers ─────────────────────────────────────────────────────
from runner import (PIPELINE_SIZES, ANOMALY_TYPES, N_TRIALS,
                    build_pipeline, _generate_clean_pipelines)

RSHB_N_TRIALS = 30
OUTPUT_DIR    = "results"

# ── Scorer registry ───────────────────────────────────────────────────────────
SCORERS = {
    "VAE":   lambda: VAEAnomalyScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80),
    "IF":    lambda: IFAnomalyScorer(window_size=20, n_estimators=100, max_train_windows=50_000),
    "OCSVM": lambda: OCSVMAnomalyScorer(window_size=20, nu=0.1, n_components=100, max_train_windows=20_000),
}


def _collect_samples_for_scorer(scorer, scorer_name, apa_engine):
    """
    Run 630 synthetic trials with the given scorer and collect GNN samples.
    Returns list of GNNSample objects.
    """
    samples = []
    total   = len(ANOMALY_TYPES) * len(PIPELINE_SIZES) * N_TRIALS

    with tqdm(total=total, desc=f"  {scorer_name} samples") as pbar:
        for size in PIPELINE_SIZES:
            for atype in ANOMALY_TYPES:
                for trial in range(N_TRIALS):
                    data_seed   = trial + 42
                    inject_seed = trial

                    pipeline = build_pipeline(size, seed=data_seed)
                    rng      = np.random.default_rng(inject_seed)
                    cands    = get_injectable_nodes_for_type(pipeline, atype)
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
                        sc_scores = scorer.score(pipeline)
                    except Exception:
                        sc_scores = {n: 0.0 for n in G.nodes()}

                    builder.set_anomaly_scores(sc_scores)
                    obs = select_target_node(G, sc_scores, threshold=0.1)

                    try:
                        rca_r = apa_engine.run(G, sc_scores, obs)
                        sample = build_gnn_sample(
                            G=G, scores=sc_scores, rca_result=rca_r,
                            true_rc=gt.target_node_id, top_k=10,
                            anomaly_type=atype.value,
                            pipeline_size=size, trial=trial,
                        )
                        if sample is not None:
                            samples.append(sample)
                    except Exception as e:
                        pass
                    pbar.update(1)

    return samples


def _rshb_with_scorer(scorer, scorer_name, gnn, apa_engine, real_raw, results):
    """
    Run 210 RSHB trials using scorer + GNN. Adds metrics to results.
    """
    total = len(ANOMALY_TYPES) * RSHB_N_TRIALS
    method_name = f"{scorer_name}+APA-RCA+GNN"

    with tqdm(total=total, desc=f"  RSHB {scorer_name}") as pbar:
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
                    sc_scores = scorer.score(pipeline)
                except Exception:
                    sc_scores = {n: 0.0 for n in G.nodes()}

                builder.set_anomaly_scores(sc_scores)
                obs = select_target_node(G, sc_scores, threshold=0.1)

                try:
                    rca_r = apa_engine.run(G, sc_scores, obs)
                    sample = build_gnn_sample(
                        G=G, scores=sc_scores, rca_result=rca_r,
                        true_rc=gt.target_node_id, top_k=10,
                        anomaly_type=atype.value,
                        pipeline_size="medium_real", trial=trial,
                    )
                    if sample is not None:
                        reranked = gnn.rerank(sample, rca_r)
                        m = compute_run_metrics(
                            result=reranked, true_root_cause=gt.target_node_id,
                            anomaly_type=atype.value, pipeline_size="medium_real",
                            trial=trial, method_name=method_name,
                        )
                        results.add(m)
                except Exception as e:
                    pass
                pbar.update(1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apa_engine = APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True)
    results    = ExperimentResults()

    # ── Step 1: Generate 60 clean pipelines (shared across all scorers) ───────
    print("\n" + "="*60)
    print("Step 1: Generating 60 clean pipelines")
    print("="*60)
    clean_pipelines = _generate_clean_pipelines(n_per_size=20)

    # ── Step 2: Load real market data ─────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 2: Loading real market data (RSHB)")
    print("="*60)
    try:
        cfg      = RealDataConfig(size="medium", n_days=756, end_date="2024-12-31", seed=42)
        real_raw = RealMarketDataGenerator(cfg).generate_all()
        print("  Real market data loaded.")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # ── Step 3: For each scorer → train → collect samples → train GNN → RSHB ─
    all_rows = []

    for scorer_name, scorer_factory in SCORERS.items():
        print(f"\n{'='*60}")
        print(f"Scorer: {scorer_name}")
        print(f"{'='*60}")

        # Train scorer
        scorer = scorer_factory()
        scorer.fit(clean_pipelines)

        # Collect GNN samples (630 synthetic trials)
        print(f"\n  Collecting GNN samples ({len(ANOMALY_TYPES)*len(PIPELINE_SIZES)*N_TRIALS} trials)...")
        samples = _collect_samples_for_scorer(scorer, scorer_name, apa_engine)
        print(f"  Collected {len(samples)} GNN samples.")

        if len(samples) < 50:
            print(f"  WARNING: too few samples for {scorer_name}, skipping GNN.")
            continue

        # Train GNN on all synthetic samples
        samples = assign_cv_folds(samples, n_folds=5)
        gnn = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
        gnn.fit(samples, verbose=False)
        print(f"  GNN trained on {len(samples)} samples.")

        # RSHB evaluation
        print(f"\n  Running RSHB (210 trials)...")
        _rshb_with_scorer(scorer, scorer_name, gnn, apa_engine, real_raw, results)

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SCORER COMPARISON RESULTS (RSHB, 210 trials each)")
    print("="*60)

    df = results.to_dataframe() if hasattr(results, 'to_dataframe') else None
    if df is None:
        # fallback: use internal records
        try:
            df = pd.DataFrame([vars(r) for r in results._records])
        except Exception:
            df = None

    if df is not None:
        rshb_df = df[df["pipeline_size"] == "medium_real"].copy()
        mrr_col = "reciprocal_rank" if "reciprocal_rank" in df.columns else "mrr"
        summary_rows = []
        for method in rshb_df["method"].unique():
            sub = rshb_df[rshb_df["method"] == method]
            summary_rows.append({
                "Method": method,
                "Top-1": f"{sub['top1'].mean():.1%}",
                "Top-3": f"{sub['top3'].mean():.1%}",
                "Top-5": f"{sub['top5'].mean():.1%}",
                "MRR":   f"{sub[mrr_col].mean():.3f}",
                "N":     len(sub),
            })
        summary = pd.DataFrame(summary_rows)
        print(summary.to_string(index=False))

        # Add reference rows for context
        print("\nReference (from existing results):")
        print("  APA-RCA v2 (baseline)       Top-1=32.9%  Top-3=67.6%  Top-5=82.4%  MRR=0.503")
        print("  APA-RCA v2+GNN (z-score)    Top-1=31.4%  Top-3=49.5%  Top-5=68.6%  MRR=0.445")
        print("  VAE+APA-RCA v2+GNN          Top-1=54.8%  Top-3=65.7%  Top-5=71.9%  MRR=0.620")

        # Save full results
        out_path = os.path.join(OUTPUT_DIR, "scorer_comparison.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
    else:
        print("  No results to display.")


if __name__ == "__main__":
    main()
