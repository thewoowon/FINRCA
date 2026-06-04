"""
Architecture Variant Experiment: 2×2 comparison of C-VAE and GNN variants.

Combinations
------------
  (1) CVAE      + ResGCN    [current v2 — baseline for this experiment]
  (2) CVAE      + R-GCN     [GNN architecture change only]
  (3) TSD-CVAE  + ResGCN    [scorer architecture change only]
  (4) TSD-CVAE  + R-GCN     [both changed]

Protocol
--------
1. Train both scorers (CVAE, TSD-CVAE) on 60 clean pipelines.
2. Collect GNN/RGCN samples from 630 synthetic trials (for each scorer).
3. Train 4 rerankers (2 scorers × 2 GNN variants) on synthetic samples.
4. Evaluate all 4 on 210 RSHB trials.

Results saved to: results/arch_variant_results.csv
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
from rca import APARCAEngine, VAEAnomalyScorer, GNNReranker, build_gnn_sample, assign_cv_folds
from rca.arch_variants import (
    TSDCVAEScorer, RGCNReranker, build_rgcn_sample, RGCNSample,
)
from baselines import select_target_node
from evaluation import ExperimentResults, compute_run_metrics
from experiments.runner import (
    PIPELINE_SIZES, ANOMALY_TYPES, N_TRIALS, RSHB_N_TRIALS,
    build_pipeline, _generate_clean_pipelines,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 4 combinations
COMBOS = [
    ("CVAE+ResGCN",     "cvae",     "resgcn"),   # current v2
    ("CVAE+RGCN",       "cvae",     "rgcn"),
    ("TSD-CVAE+ResGCN", "tsd_cvae", "resgcn"),
    ("TSD-CVAE+RGCN",   "tsd_cvae", "rgcn"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apa_engine = APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True)

    # ── Phase 1: Train both scorers ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1: Training scorers (CVAE + TSD-CVAE)")
    print("=" * 60)
    clean_pipelines = _generate_clean_pipelines(n_per_size=20)

    cvae_scorer = VAEAnomalyScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    cvae_scorer.fit(clean_pipelines)

    tsd_scorer = TSDCVAEScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    tsd_scorer.fit(clean_pipelines)

    scorers = {"cvae": cvae_scorer, "tsd_cvae": tsd_scorer}

    # ── Phase 2: Collect GNN samples (630 synthetic, per scorer) ────────────
    print("\n" + "=" * 60)
    print("Phase 2: Collecting GNN training samples (630 trials × 2 scorers)")
    print("=" * 60)

    # For each scorer, collect both ResGCN samples and RGCN samples
    resgcn_samples = {"cvae": [], "tsd_cvae": []}
    rgcn_samples   = {"cvae": [], "tsd_cvae": []}

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

                    # Score with both scorers
                    for skey, scorer in scorers.items():
                        try:
                            vae_scores = scorer.score(pipeline)
                        except Exception:
                            continue

                        builder.set_anomaly_scores(vae_scores)
                        obs = select_target_node(G, vae_scores, threshold=0.1)

                        try:
                            rca_r = apa_engine.run(G, vae_scores, obs)

                            # ResGCN sample
                            s_res = build_gnn_sample(
                                G=G, scores=vae_scores, rca_result=rca_r,
                                true_rc=gt.target_node_id, top_k=10,
                                anomaly_type=atype.value,
                                pipeline_size=size, trial=trial,
                            )
                            if s_res is not None:
                                resgcn_samples[skey].append(s_res)

                            # R-GCN sample
                            s_rgcn = build_rgcn_sample(
                                G=G, scores=vae_scores, rca_result=rca_r,
                                true_rc=gt.target_node_id, top_k=10,
                                anomaly_type=atype.value,
                                pipeline_size=size, trial=trial,
                            )
                            if s_rgcn is not None:
                                rgcn_samples[skey].append(s_rgcn)
                        except Exception:
                            pass

                    pbar.update(1)

    for skey in scorers:
        print(f"  {skey}: {len(resgcn_samples[skey])} ResGCN, {len(rgcn_samples[skey])} RGCN samples")

    # ── Phase 3: Train 4 rerankers ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Training 4 rerankers")
    print("=" * 60)

    trained = {}
    for combo_name, skey, gkey in COMBOS:
        print(f"  Training: {combo_name}")
        if gkey == "resgcn":
            samples = assign_cv_folds(resgcn_samples[skey], n_folds=5)
            reranker = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
            reranker.fit(samples, verbose=False)
        else:  # rgcn
            samples = rgcn_samples[skey]
            reranker = RGCNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
            reranker.fit(samples, verbose=False)
        trained[combo_name] = (skey, gkey, reranker)
        print(f"    done ({len(samples)} samples)")

    # ── Phase 4: RSHB evaluation (210 trials) ──────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 4: RSHB evaluation — 4 combinations × 210 trials")
    print("=" * 60)

    try:
        cfg      = RealDataConfig(size="medium", n_days=756, end_date="2024-12-31", seed=42)
        real_raw = RealMarketDataGenerator(cfg).generate_all()
        print("  Real market data loaded.")
    except Exception as e:
        print(f"  ERROR loading real data: {e}")
        return

    all_results = {name: ExperimentResults() for name, _, _ in COMBOS}

    total_rshb = len(ANOMALY_TYPES) * RSHB_N_TRIALS
    with tqdm(total=total_rshb, desc="RSHB arch variants") as pbar:
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

                # Pre-compute scores for both scorers
                scorer_scores = {}
                for skey, scorer in scorers.items():
                    try:
                        scorer_scores[skey] = scorer.score(pipeline)
                    except Exception:
                        scorer_scores[skey] = None

                # Evaluate each combination
                for combo_name, skey, gkey, in COMBOS:
                    skey_actual, gkey_actual, reranker = trained[combo_name]
                    scores = scorer_scores.get(skey)
                    if scores is None:
                        continue

                    builder.set_anomaly_scores(scores)
                    obs = select_target_node(G, scores, threshold=0.1)

                    try:
                        rca_r = apa_engine.run(G, scores, obs)

                        if gkey == "resgcn":
                            sample = build_gnn_sample(
                                G=G, scores=scores, rca_result=rca_r,
                                true_rc=gt.target_node_id, top_k=10,
                                anomaly_type=atype.value,
                                pipeline_size="medium_real", trial=trial,
                            )
                        else:
                            sample = build_rgcn_sample(
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
                                method_name=combo_name,
                            )
                            all_results[combo_name].add(m)
                    except Exception:
                        pass

                pbar.update(1)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ARCHITECTURE VARIANT RESULTS — RSHB (210 trials each)")
    print("=" * 60)
    print(f"  {'Combination':25s}  Top-1   Top-3   Top-5   MRR     N")
    print("  " + "-" * 70)

    rows = []
    for combo_name, _, _ in COMBOS:
        res = all_results[combo_name]
        try:
            df = pd.DataFrame([vars(r) for r in res.records])
        except Exception:
            df = pd.DataFrame([r.__dict__ for r in res.records])

        if len(df) == 0:
            print(f"  {combo_name:25s}  (no results)")
            continue

        t1  = df["top1"].mean() * 100
        t3  = df["top3"].mean() * 100
        t5  = df["top5"].mean() * 100
        mrr = df["reciprocal_rank"].mean()
        n   = len(df)
        print(f"  {combo_name:25s}  {t1:5.1f}%  {t3:5.1f}%  {t5:5.1f}%  {mrr:.3f}   {n}")

        row = {"combination": combo_name, "top1": t1, "top3": t3, "top5": t5, "mrr": mrr, "n": n}
        # Per-type breakdown
        for at in sorted(df["anomaly_type"].unique()):
            s = df[df["anomaly_type"] == at]
            row[f"{at}_top1"] = round(s["top1"].mean() * 100, 1)
        rows.append(row)

    print()
    print("  Reference (current v2):")
    print("  CVAE+APA-RCA+ResGCN              51.0%   74.8%   74.8%  0.626")

    # Save
    if rows:
        summary_path = os.path.join(OUTPUT_DIR, "arch_variant_summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\n  Summary → {summary_path}")

    # Save all raw results combined
    all_records = []
    for combo_name, _, _ in COMBOS:
        res = all_results[combo_name]
        for r in res.records:
            d = vars(r) if hasattr(r, '__dict__') else r.__dict__
            d["combination"] = combo_name
            all_records.append(d)

    if all_records:
        out_path = os.path.join(OUTPUT_DIR, "arch_variant_results.csv")
        pd.DataFrame(all_records).to_csv(out_path, index=False)
        print(f"  Full results → {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
