"""
Hybrid TSD-CVAE quick comparison: CVAE vs TSD-CVAE vs Hybrid-TSD-CVAE
All three paired with ResGCN on RSHB (210 trials).
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
from rca.arch_variants import TSDCVAEScorer, HybridTSDCVAEScorer
from baselines import select_target_node
from evaluation import ExperimentResults, compute_run_metrics
from experiments.runner import (
    PIPELINE_SIZES, ANOMALY_TYPES, N_TRIALS, RSHB_N_TRIALS,
    build_pipeline, _generate_clean_pipelines,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SCORERS = [
    ("CVAE",        "cvae"),
    ("TSD-CVAE",    "tsd"),
    ("Hybrid-TSD",  "hybrid"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apa_engine = APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True)

    # ── Phase 1: Train 3 scorers ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1: Training 3 scorers")
    print("=" * 60)
    clean_pipelines = _generate_clean_pipelines(n_per_size=20)

    scorer_objs = {}
    scorer_objs["cvae"] = VAEAnomalyScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    scorer_objs["cvae"].fit(clean_pipelines)

    scorer_objs["tsd"] = TSDCVAEScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    scorer_objs["tsd"].fit(clean_pipelines)

    scorer_objs["hybrid"] = HybridTSDCVAEScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    scorer_objs["hybrid"].fit(clean_pipelines)

    # ── Phase 2: Collect GNN samples per scorer ─────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Collecting ResGCN samples (630 trials x 3 scorers)")
    print("=" * 60)

    all_samples = {k: [] for k in scorer_objs}
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

                    for skey, scorer in scorer_objs.items():
                        try:
                            scores = scorer.score(pipeline)
                        except Exception:
                            continue

                        builder.set_anomaly_scores(scores)
                        obs = select_target_node(G, scores, threshold=0.1)
                        try:
                            rca_r = apa_engine.run(G, scores, obs)
                            sample = build_gnn_sample(
                                G=G, scores=scores, rca_result=rca_r,
                                true_rc=gt.target_node_id, top_k=10,
                                anomaly_type=atype.value,
                                pipeline_size=size, trial=trial,
                            )
                            if sample is not None:
                                all_samples[skey].append(sample)
                        except Exception:
                            pass
                    pbar.update(1)

    for skey in scorer_objs:
        print(f"  {skey}: {len(all_samples[skey])} samples")

    # ── Phase 3: Train ResGCN per scorer ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Training 3 ResGCN rerankers")
    print("=" * 60)

    rerankers = {}
    for skey in scorer_objs:
        samples = assign_cv_folds(all_samples[skey], n_folds=5)
        reranker = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
        reranker.fit(samples, verbose=False)
        rerankers[skey] = reranker
        print(f"  {skey}: trained on {len(samples)} samples")

    # ── Phase 4: RSHB evaluation ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 4: RSHB evaluation — 3 scorers x 210 trials")
    print("=" * 60)

    try:
        cfg      = RealDataConfig(size="medium", n_days=756, end_date="2024-12-31", seed=42)
        real_raw = RealMarketDataGenerator(cfg).generate_all()
        print("  Real market data loaded.")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    results = {skey: ExperimentResults() for skey in scorer_objs}

    total_rshb = len(ANOMALY_TYPES) * RSHB_N_TRIALS
    with tqdm(total=total_rshb, desc="RSHB hybrid") as pbar:
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

                for skey, scorer in scorer_objs.items():
                    try:
                        scores = scorer.score(pipeline)
                    except Exception:
                        continue

                    builder.set_anomaly_scores(scores)
                    obs = select_target_node(G, scores, threshold=0.1)

                    try:
                        rca_r = apa_engine.run(G, scores, obs)
                        sample = build_gnn_sample(
                            G=G, scores=scores, rca_result=rca_r,
                            true_rc=gt.target_node_id, top_k=10,
                            anomaly_type=atype.value,
                            pipeline_size="medium_real", trial=trial,
                        )
                        if sample is not None:
                            reranked = rerankers[skey].rerank(sample, rca_r)
                            label = dict(SCORERS).get(skey, skey)  # won't exist, use below
                            for name, sk in SCORERS:
                                if sk == skey:
                                    label = name
                                    break
                            m = compute_run_metrics(
                                result=reranked,
                                true_root_cause=gt.target_node_id,
                                anomaly_type=atype.value,
                                pipeline_size="medium_real",
                                trial=trial,
                                method_name=f"{label}+ResGCN",
                            )
                            results[skey].add(m)
                    except Exception:
                        pass

                pbar.update(1)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HYBRID TSD-CVAE RESULTS — RSHB (210 trials)")
    print("=" * 60)
    print(f"  {'Scorer':20s}  Top-1   Top-3   Top-5   MRR     N")
    print("  " + "-" * 60)

    rows = []
    for name, skey in SCORERS:
        res = results[skey]
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
        n   = len(df)
        print(f"  {name+'+ResGCN':20s}  {t1:5.1f}%  {t3:5.1f}%  {t5:5.1f}%  {mrr:.3f}   {n}")

        row = {"scorer": name, "top1": t1, "top3": t3, "top5": t5, "mrr": mrr, "n": n}
        for at in sorted(df["anomaly_type"].unique()):
            s = df[df["anomaly_type"] == at]
            row[f"{at}_top1"] = round(s["top1"].mean() * 100, 1)
        rows.append(row)

    # Save
    if rows:
        out_path = os.path.join(OUTPUT_DIR, "hybrid_tsd_summary.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  Summary → {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
