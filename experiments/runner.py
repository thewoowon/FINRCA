"""
Experiment Runner: Executes all FINRCA experiments.

Main experiment: 4 anomaly types × 3 pipeline sizes × 10 trials = 120 runs
                 × 4 methods (BFS, VanillaRWR, AnomalyOnly, APA-RCA) = 480 evaluations

Ablation study: 4 APA-RCA variants × 120 runs = 480 evaluations
"""

import sys
import os
import copy
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synfrp import (MarketDataConfig, MarketDataGenerator,
                    RealDataConfig, RealMarketDataGenerator,
                    FinancialRiskPipeline, AnomalyInjector,
                    AnomalyType, get_injectable_nodes_for_type)
from frlg import FRLGBuilder, NodeAnomalyDetector
from rca import APARCAEngine, SchemaConstraintRCA, VAEAnomalyScorer, GNNReranker, build_gnn_sample, assign_cv_folds
from baselines import BFSBaseline, VanillaRWRBaseline, AnomalyScoreOnlyBaseline, StructuralRWRBaseline, PCRWRBaseline, select_target_node
from evaluation import ExperimentResults, compute_run_metrics


# ── Pipeline size configurations ─────────────────────────────────────────────
PIPELINE_SIZES = {
    "small":  {"n_stocks": 3, "n_rates": 2, "n_fx": 1, "n_positions": 3},
    "medium": {"n_stocks": 5, "n_rates": 3, "n_fx": 2, "n_positions": 5},
    "large":  {"n_stocks": 8, "n_rates": 3, "n_fx": 3, "n_positions": 8},
}

ANOMALY_TYPES = [
    AnomalyType.A1_SOURCE_CORRUPTION,
    AnomalyType.A2_ETL_LOGIC_ERROR,
    AnomalyType.A3_FEATURE_CALC_ERROR,
    AnomalyType.A4_AGGREGATION_ERROR,
    AnomalyType.A5_DOWNSTREAM_MASKING,
    AnomalyType.A6_MULTI_SOURCE,
    AnomalyType.A7_COINCIDENTAL,
]

N_TRIALS = 30


def build_pipeline(size: str, seed: int = 42) -> FinancialRiskPipeline:
    """Build a fresh pipeline for the given size."""
    size_cfg = PIPELINE_SIZES[size]
    cfg = MarketDataConfig(
        seed=seed,
        n_stocks=size_cfg["n_stocks"],
        n_rates=size_cfg["n_rates"],
        n_fx=size_cfg["n_fx"],
        n_positions=size_cfg["n_positions"],
    )
    raw = MarketDataGenerator(cfg).generate_all()
    return FinancialRiskPipeline(raw, size=size)


def run_single_experiment(
    pipeline_size: str,
    anomaly_type: AnomalyType,
    trial: int,
    methods: Dict,
    magnitude: float = 8.0,
    verbose: bool = False,
) -> List:
    """
    Execute one trial: inject anomaly, detect, run all methods, compute metrics.
    Returns list of SingleRunMetrics (one per method).
    """
    # Use trial directly as the data seed so each trial has genuinely different data
    data_seed  = trial + 42
    inject_seed = trial

    # 1. Build fresh pipeline
    pipeline = build_pipeline(pipeline_size, seed=data_seed)

    # 2. Pick target node for this anomaly type (deterministically by trial)
    rng = np.random.default_rng(inject_seed)
    candidates = get_injectable_nodes_for_type(pipeline, anomaly_type)
    if not candidates:
        return []
    target_inject = rng.choice(candidates)

    # 3. Inject anomaly
    injector = AnomalyInjector(pipeline, rng)
    injector, gt = injector.inject(
        anomaly_type=anomaly_type,
        target_node_id=target_inject,
        magnitude=magnitude,
        seed=inject_seed,
    )

    # 4. Execute pipeline
    pipeline.execute()

    # 5. Build FRLG + detect anomalies
    builder  = FRLGBuilder(pipeline)
    G        = builder.build()
    detector = NodeAnomalyDetector(threshold_zscore=2.0)
    scores   = detector.detect_all(pipeline)
    builder.set_anomaly_scores(scores)

    # 6. Select target node (most anomalous report node)
    observed_target = select_target_node(G, scores, threshold=0.1)

    if verbose:
        print(f"  Injected: {gt.target_node_id} | Observed: {observed_target} | "
              f"Score of RC: {scores.get(gt.target_node_id, 0):.3f}")

    # 7. Run each method
    run_metrics = []
    for method_name, method in methods.items():
        try:
            # PC+RWR needs pipeline attached for time series extraction
            if isinstance(method, PCRWRBaseline):
                method.set_pipeline(pipeline)
            result = method.run(G, scores, observed_target)
            metrics = compute_run_metrics(
                result=result,
                true_root_cause=gt.target_node_id,
                anomaly_type=anomaly_type.value,
                pipeline_size=pipeline_size,
                trial=trial,
                method_name=method_name,
            )
            run_metrics.append(metrics)
        except Exception as e:
            print(f"    ERROR in {method_name}: {e}")

    # 8. Schema-Constraint RCA (runs for all types; only fires on A4)
    try:
        sc_rca = SchemaConstraintRCA()
        sc_result = sc_rca.check_and_run(G, pipeline, scores, observed_target)
        if sc_result is None:
            # No violation found — build a zero-result (correctly abstains)
            from rca.apa_rca import RCAResult as _RCAResult
            sc_result = _RCAResult(
                ranked_nodes=[(n, 0.0) for n in G.nodes() if n != observed_target],
                convergence_iters=0, elapsed_ms=0.0,
                target_node=observed_target, method="Schema-Constraint",
            )
        metrics = compute_run_metrics(
            result=sc_result,
            true_root_cause=gt.target_node_id,
            anomaly_type=anomaly_type.value,
            pipeline_size=pipeline_size,
            trial=trial,
            method_name="Schema-Constraint",
        )
        run_metrics.append(metrics)
    except Exception as e:
        print(f"    ERROR in Schema-Constraint: {e}")

    return run_metrics


def run_main_experiments(output_dir: str = "results") -> ExperimentResults:
    """
    Main experiment: 4 types × 3 sizes × 10 trials × 4 methods.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "BFS":            BFSBaseline(),
        "Vanilla-RWR":    VanillaRWRBaseline(),
        "Structural-RWR": StructuralRWRBaseline(),
        "PC+RWR":         PCRWRBaseline(pc_alpha=0.05, max_cond_vars=2),
        "AnomalyOnly":    AnomalyScoreOnlyBaseline(),
        "APA-RCA":        APARCAEngine(gamma=0.15, alpha=0.7),
        "APA-RCA-v2":     APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True),
        "APA-RCA-v3":     APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True, use_atf_weights=True),
    }

    results = ExperimentResults()
    total = len(ANOMALY_TYPES) * len(PIPELINE_SIZES) * N_TRIALS

    print(f"\n{'='*60}")
    print(f"FINRCA Main Experiment: {total} runs × {len(methods)} methods")
    print(f"{'='*60}\n")

    with tqdm(total=total, desc="Experiments") as pbar:
        for size in PIPELINE_SIZES:
            for atype in ANOMALY_TYPES:
                for trial in range(N_TRIALS):
                    run_metrics = run_single_experiment(
                        pipeline_size=size,
                        anomaly_type=atype,
                        trial=trial,
                        methods=methods,
                    )
                    for m in run_metrics:
                        results.add(m)
                    pbar.update(1)
                    pbar.set_postfix({
                        "size": size,
                        "type": atype.value.split("_")[0],
                        "trial": trial
                    })

    results.save(os.path.join(output_dir, "main_results.csv"))

    print("\n=== MAIN RESULTS ===")
    print(results.main_results_table().to_string(index=False))

    return results


def run_ablation_experiments(output_dir: str = "results") -> ExperimentResults:
    """
    Ablation study: evaluate contribution of each APA-RCA component.
    4 variants: Full, w/o transform weight, w/o attenuation, w/o anomaly score.
    """
    os.makedirs(output_dir, exist_ok=True)

    ablation_configs = {
        # v1 ablation (baseline components)
        "APA-RCA (Full)":           APARCAEngine(use_anomaly_scores=True,  use_transform_weights=True,  use_attenuation=True),
        "w/o Transform Weight":     APARCAEngine(use_anomaly_scores=True,  use_transform_weights=False, use_attenuation=True),
        "w/o Attenuation":          APARCAEngine(use_anomaly_scores=True,  use_transform_weights=True,  use_attenuation=False),
        "w/o Anomaly Score":        APARCAEngine(use_anomaly_scores=False, use_transform_weights=True,  use_attenuation=True),
        # v2 ablation (adaptive components)
        "APA-RCA v2 (Full)":        APARCAEngine(adaptive_weighting=True,  use_score_inv_beta=True,  use_ancestor_constraint=True),
        "v2 w/o Adaptive Weight":   APARCAEngine(adaptive_weighting=False, use_score_inv_beta=True,  use_ancestor_constraint=True),
        "v2 w/o Score-inv Beta":    APARCAEngine(adaptive_weighting=True,  use_score_inv_beta=False, use_ancestor_constraint=True),
        "v2 w/o Ancestor Const":    APARCAEngine(adaptive_weighting=True,  use_score_inv_beta=True,  use_ancestor_constraint=False),
    }

    results = ExperimentResults()
    # Ablation on medium size only, all anomaly types, 10 trials
    total = len(ANOMALY_TYPES) * N_TRIALS

    print(f"\n{'='*60}")
    print(f"FINRCA Ablation Study: {total} runs × {len(ablation_configs)} configs")
    print(f"{'='*60}\n")

    with tqdm(total=total, desc="Ablation") as pbar:
        for atype in ANOMALY_TYPES:
            for trial in range(N_TRIALS):
                run_metrics = run_single_experiment(
                    pipeline_size="medium",
                    anomaly_type=atype,
                    trial=trial,
                    methods=ablation_configs,
                )
                for m in run_metrics:
                    results.add(m)
                pbar.update(1)

    results.save(os.path.join(output_dir, "ablation_results.csv"))

    print("\n=== ABLATION RESULTS ===")
    print(results.ablation_table().to_string(index=False))

    return results


def run_scalability_analysis(output_dir: str = "results") -> ExperimentResults:
    """
    Scalability: APA-RCA performance vs pipeline size.
    All 3 sizes, all anomaly types, 10 trials.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "APA-RCA": APARCAEngine(gamma=0.15, alpha=0.7),
    }

    results = ExperimentResults()
    total = len(ANOMALY_TYPES) * len(PIPELINE_SIZES) * N_TRIALS

    print(f"\n{'='*60}")
    print(f"FINRCA Scalability Analysis: {total} runs")
    print(f"{'='*60}\n")

    with tqdm(total=total, desc="Scalability") as pbar:
        for size in PIPELINE_SIZES:
            for atype in ANOMALY_TYPES:
                for trial in range(N_TRIALS):
                    run_metrics = run_single_experiment(
                        pipeline_size=size,
                        anomaly_type=atype,
                        trial=trial,
                        methods=methods,
                    )
                    for m in run_metrics:
                        results.add(m)
                    pbar.update(1)

    results.save(os.path.join(output_dir, "scalability_results.csv"))

    print("\n=== SCALABILITY RESULTS ===")
    print(results.pipeline_size_breakdown().to_string(index=False))

    return results


RSHB_N_TRIALS = 30   # trials per anomaly type in RSHB


def run_real_source_hybrid(output_dir: str = "results") -> ExperimentResults:
    """
    Real-Source Hybrid Benchmark (RSHB):
    Replace synthetic GBM/Vasicek/RW source data with real Yahoo Finance time series.
    ETL/Feature/Risk/Report layers and anomaly injection are identical to main experiment.
    Only medium pipeline size, all 7 anomaly types, 30 trials.

    Data coverage (medium config):
      - 10 US equities + Samsung (005930.KS) + SK Hynix (000660.KS) via KRX
      - 3 FX pairs (EUR/GBP/JPY vs USD)
      - US Treasury curve (^IRX/^FVX/^TYX)
      - 756 trading days (2022-01-03 – 2024-12-31): three distinct market regimes
        covering the 2022 Fed tightening cycle, 2023 banking stress, and 2023-24 rally
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "APA-RCA-v2":  APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True),
        "APA-RCA-v3":  APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True, use_atf_weights=True),
        "AnomalyOnly": AnomalyScoreOnlyBaseline(),
    }
    # Schema-Constraint is evaluated inline per trial (like main experiment)

    results = ExperimentResults()
    total = len(ANOMALY_TYPES) * RSHB_N_TRIALS

    print(f"\n{'='*60}")
    print(f"FINRCA Real-Source Hybrid Benchmark: {total} runs")
    print(f"  Tickers: 10 US equities + 2 KRX + 3 FX + 3 rates")
    print(f"  Period: 756 trading days (2022-01-03 – 2024-12-31)")
    print(f"{'='*60}\n")

    try:
        cfg = RealDataConfig(size="medium", n_days=756, end_date="2024-12-31", seed=42)
        real_raw = RealMarketDataGenerator(cfg).generate_all()
        print("  Real market data loaded from Yahoo Finance.")
        print(f"  Stocks: {len([c for c in real_raw['stock_prices'].columns])} series")
        print(f"  Rates:  {len(real_raw['interest_rates'].columns)} series")
        print(f"  FX:     {len(real_raw['fx_rates'].columns)} series")
    except Exception as e:
        print(f"  WARNING: Could not load real data ({e}). Skipping RSHB.")
        return results

    with tqdm(total=total, desc="Real-Source Hybrid") as pbar:
        for atype in ANOMALY_TYPES:
            for trial in range(RSHB_N_TRIALS):
                inject_seed = trial
                rng = np.random.default_rng(inject_seed)

                # Build pipeline with real source data
                pipeline = FinancialRiskPipeline(real_raw, size="medium")

                candidates = get_injectable_nodes_for_type(pipeline, atype)
                if not candidates:
                    pbar.update(1)
                    continue
                target_inject = rng.choice(candidates)

                injector = AnomalyInjector(pipeline, rng)
                injector, gt = injector.inject(
                    anomaly_type=atype,
                    target_node_id=target_inject,
                    magnitude=8.0,
                    seed=inject_seed,
                )

                pipeline.execute()

                builder  = FRLGBuilder(pipeline)
                G        = builder.build()
                detector = NodeAnomalyDetector(threshold_zscore=2.0)
                scores   = detector.detect_all(pipeline)
                builder.set_anomaly_scores(scores)

                observed_target = select_target_node(G, scores, threshold=0.1)

                for method_name, method in methods.items():
                    try:
                        result = method.run(G, scores, observed_target)
                        metrics = compute_run_metrics(
                            result=result,
                            true_root_cause=gt.target_node_id,
                            anomaly_type=atype.value,
                            pipeline_size="medium_real",
                            trial=trial,
                            method_name=method_name,
                        )
                        results.add(metrics)
                    except Exception as e:
                        print(f"    ERROR in {method_name}: {e}")

                # Schema-Constraint RCA (same as main experiment)
                try:
                    sc_rca = SchemaConstraintRCA()
                    sc_result = sc_rca.check_and_run(G, pipeline, scores, observed_target)
                    if sc_result is None:
                        from rca.apa_rca import RCAResult as _RCAResult
                        sc_result = _RCAResult(
                            ranked_nodes=[(n, 0.0) for n in G.nodes() if n != observed_target],
                            convergence_iters=0, elapsed_ms=0.0,
                            target_node=observed_target, method="Schema-Constraint",
                        )
                    metrics = compute_run_metrics(
                        result=sc_result,
                        true_root_cause=gt.target_node_id,
                        anomaly_type=atype.value,
                        pipeline_size="medium_real",
                        trial=trial,
                        method_name="Schema-Constraint",
                    )
                    results.add(metrics)
                except Exception as e:
                    print(f"    ERROR in Schema-Constraint: {e}")

                pbar.update(1)

    results.save(os.path.join(output_dir, "rshb_results.csv"))

    print("\n=== REAL-SOURCE HYBRID RESULTS ===")
    print(results.main_results_table().to_string(index=False))

    return results


def _generate_clean_pipelines(n_per_size: int = 20) -> list:
    """
    Generate anomaly-free executed pipelines for VAE training.
    Uses seeds 1000+ to avoid collision with main experiment seeds.
    """
    clean = []
    for size in PIPELINE_SIZES:
        for i in range(n_per_size):
            seed = 1000 + i
            pl   = build_pipeline(size, seed=seed)
            pl.execute()
            clean.append(pl)
    return clean


def _run_single_with_scores(
    pipeline_size: str,
    anomaly_type: AnomalyType,
    trial: int,
    apa_engine: APARCAEngine,
    top_k: int = 10,
    vae_scorer: Optional[VAEAnomalyScorer] = None,
) -> Optional[tuple]:
    """
    One trial with anomaly scoring + APA-RCA-v2.
    If vae_scorer is provided, uses VAE scores; otherwise uses z-score.
    Returns (zscore_dict, vae_scores_or_None, G, gt, rca_result, gnn_sample) or None.
    """
    data_seed   = trial + 42
    inject_seed = trial

    pipeline = build_pipeline(pipeline_size, seed=data_seed)
    rng      = np.random.default_rng(inject_seed)
    candidates_inj = get_injectable_nodes_for_type(pipeline, anomaly_type)
    if not candidates_inj:
        return None
    target_inject = rng.choice(candidates_inj)

    injector = AnomalyInjector(pipeline, rng)
    injector, gt = injector.inject(
        anomaly_type=anomaly_type, target_node_id=target_inject,
        magnitude=8.0, seed=inject_seed,
    )
    pipeline.execute()

    builder  = FRLGBuilder(pipeline)
    G        = builder.build()

    # Always compute z-scores (used for z-score+GNN baseline)
    detector  = NodeAnomalyDetector(threshold_zscore=2.0)
    z_scores  = detector.detect_all(pipeline)

    # Optionally compute VAE scores
    vae_scores = None
    if vae_scorer is not None:
        try:
            vae_scores = vae_scorer.score(pipeline)
        except Exception:
            vae_scores = z_scores  # fallback

    # ── z-score branch: APA-RCA-v2 with z-scores ────────────────────────────
    builder.set_anomaly_scores(z_scores)
    observed_target = select_target_node(G, z_scores, threshold=0.1)

    try:
        rca_z = apa_engine.run(G, z_scores, observed_target)
        rca_z.method = "APA-RCA-v2"
        metrics_z = compute_run_metrics(
            result=rca_z, true_root_cause=gt.target_node_id,
            anomaly_type=anomaly_type.value, pipeline_size=pipeline_size,
            trial=trial, method_name="APA-RCA-v2",
        )
        sample_z = build_gnn_sample(
            G=G, scores=z_scores, rca_result=rca_z,
            true_rc=gt.target_node_id, top_k=top_k,
            anomaly_type=anomaly_type.value,
            pipeline_size=pipeline_size, trial=trial,
        )
    except Exception as e:
        print(f"    ERROR APA-RCA-v2 [{pipeline_size}/{anomaly_type.value}/{trial}]: {e}")
        return None

    # ── VAE branch: APA-RCA-v2 with VAE scores (if VAE provided) ─────────────
    metrics_vae = None
    sample_vae  = None
    if vae_scorer is not None and vae_scores is not None:
        try:
            rca_vae = apa_engine.run(G, vae_scores, observed_target)
            rca_vae.method = "CVAE+APA-RCA"
            metrics_vae = compute_run_metrics(
                result=rca_vae, true_root_cause=gt.target_node_id,
                anomaly_type=anomaly_type.value, pipeline_size=pipeline_size,
                trial=trial, method_name="CVAE+APA-RCA",
            )
            sample_vae = build_gnn_sample(
                G=G, scores=vae_scores, rca_result=rca_vae,
                true_rc=gt.target_node_id, top_k=top_k,
                anomaly_type=anomaly_type.value,
                pipeline_size=pipeline_size, trial=trial,
            )
        except Exception:
            pass

    return metrics_z, sample_z, metrics_vae, sample_vae


def run_ml_enhanced_experiments(output_dir: str = "results") -> ExperimentResults:
    """
    ML-Enhanced Pipeline: two GNN re-ranking experiments run in parallel.

    Experiment A — APA-RCA-v2 + GNN  (z-score anomaly scorer + GNN re-ranker)
    Experiment B — VAE+APA-RCA + GNN  (VAE anomaly scorer  + GNN re-ranker)

    Phase 1 — Train VAE
        Generate 20 anomaly-free pipelines per size (60 total).
        Train one shared MLP-VAE on sliding windows.

    Phase 2 — Collect GNN samples (all 630 synthetic trials)
        For each trial:
          • z-score → APA-RCA-v2 → GNN sample   [z-branch]
          • VAE score → APA-RCA-v2 → GNN sample  [VAE branch]
        Record APA-RCA-v2 and VAE+APA-RCA metrics as intermediate baselines.

    Phase 3 — 5-fold CV (synthetic evaluation)
        Stratified CV by (anomaly_type, pipeline_size).
        Train separate GNNs for z-branch and VAE-branch.
        Report: APA-RCA-v2+GNN and VAE+APA-RCA+GNN.

    Phase 4 — RSHB evaluation (cross-domain transfer)
        Train GNN on all z_samples  → apply to RSHB with z-scores.
        Train GNN on all vae_samples → apply to RSHB with VAE scores.
        Report: APA-RCA-v2+GNN and VAE+APA-RCA+GNN on real data.
    """
    os.makedirs(output_dir, exist_ok=True)

    apa_engine = APARCAEngine(gamma=0.15, alpha=0.7, adaptive_weighting=True)
    results    = ExperimentResults()

    # ── Phase 1: Train VAE ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Training VAE anomaly scorer")
    print(f"{'='*60}")
    print("  Generating 20 clean pipelines per size (60 total)...")
    clean_pipelines = _generate_clean_pipelines(n_per_size=20)
    vae_scorer = VAEAnomalyScorer(window_size=20, latent_dim=8, hidden_dim=64, epochs=80)
    vae_scorer.fit(clean_pipelines)

    # ── Phase 2: Collect GNN samples for all 630 synthetic trials ────────────
    print(f"\n{'='*60}")
    print("Phase 2: Collecting GNN training samples (630 trials)")
    print(f"{'='*60}")
    z_samples   = []  # z-score branch
    vae_samples = []  # VAE branch
    total = len(ANOMALY_TYPES) * len(PIPELINE_SIZES) * N_TRIALS

    with tqdm(total=total, desc="Sample collection") as pbar:
        for size in PIPELINE_SIZES:
            for atype in ANOMALY_TYPES:
                for trial in range(N_TRIALS):
                    out = _run_single_with_scores(
                        size, atype, trial, apa_engine,
                        top_k=10, vae_scorer=vae_scorer,
                    )
                    if out is not None:
                        metrics_z, sample_z, metrics_vae, sample_vae = out
                        results.add(metrics_z)
                        if metrics_vae is not None:
                            results.add(metrics_vae)
                        if sample_z is not None:
                            z_samples.append(sample_z)
                        if sample_vae is not None:
                            vae_samples.append(sample_vae)
                    pbar.update(1)

    print(f"  Collected {len(z_samples)} z-score GNN samples, "
          f"{len(vae_samples)} VAE GNN samples.")

    # ── Phase 3: 5-fold CV on synthetic ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 3: 5-fold CV — APA-RCA-v2+GNN and VAE+APA-RCA+GNN (synthetic)")
    print(f"{'='*60}")
    N_FOLDS = 5
    z_samples   = assign_cv_folds(z_samples,   n_folds=N_FOLDS)
    vae_samples = assign_cv_folds(vae_samples, n_folds=N_FOLDS)

    from rca.apa_rca import RCAResult as _RC

    def _eval_fold(sample_list, reranker, method_name):
        for s in sample_list:
            mock_result = _RC(
                ranked_nodes=sorted(
                    [(nid, float(s.features[i, 0]))
                     for i, nid in enumerate(s.node_ids)
                     if nid in s.candidate_ids],
                    key=lambda x: x[1], reverse=True,
                ),
                convergence_iters=0, elapsed_ms=0.0,
                target_node="", method=method_name,
            )
            reranked = reranker.rerank(s, mock_result)
            m = compute_run_metrics(
                result=reranked, true_root_cause=s.true_rc,
                anomaly_type=s.anomaly_type, pipeline_size=s.pipeline_size,
                trial=s.trial, method_name=method_name,
            )
            results.add(m)

    for fold in range(N_FOLDS):
        # z-score GNN fold
        z_train = [s for s in z_samples   if s.fold != fold]
        z_test  = [s for s in z_samples   if s.fold == fold]
        z_rr = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=120)
        z_rr.fit(z_train, verbose=False)
        print(f"  Fold {fold+1}/{N_FOLDS}: z-GNN  trained={len(z_train)}, test={len(z_test)}")
        _eval_fold(z_test, z_rr, "APA-RCA-v2+ResGCN")

        # C-VAE + ResGCN fold
        vae_train = [s for s in vae_samples if s.fold != fold]
        vae_test  = [s for s in vae_samples if s.fold == fold]
        vae_rr = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=120)
        vae_rr.fit(vae_train, verbose=False)
        print(f"  Fold {fold+1}/{N_FOLDS}: CVAE-ResGCN trained={len(vae_train)}, test={len(vae_test)}")
        _eval_fold(vae_test, vae_rr, "CVAE+APA-RCA+ResGCN")

    # ── Phase 4: Train on all synthetic → test on RSHB ───────────────────────
    print(f"\n{'='*60}")
    print("Phase 4: RSHB evaluation — APA-RCA-v2+ResGCN and CVAE+APA-RCA+ResGCN")
    print(f"{'='*60}")

    z_final   = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
    z_final.fit(z_samples, verbose=False)
    print(f"  Final z-GNN trained on all {len(z_samples)} synthetic samples.")

    vae_final = GNNReranker(top_k=10, hidden_dim=32, dropout=0.3, epochs=150)
    vae_final.fit(vae_samples, verbose=False)
    print(f"  Final VAE-GNN trained on all {len(vae_samples)} synthetic samples.")

    try:
        cfg      = RealDataConfig(size="medium", n_days=756, end_date="2024-12-31", seed=42)
        real_raw = RealMarketDataGenerator(cfg).generate_all()
        print("  Real market data loaded.")
    except Exception as e:
        print(f"  WARNING: Could not load real data ({e}). Skipping RSHB phase.")
        real_raw = None

    if real_raw is not None:
        total_rshb = len(ANOMALY_TYPES) * RSHB_N_TRIALS
        with tqdm(total=total_rshb, desc="RSHB ML") as pbar:
            for atype in ANOMALY_TYPES:
                for trial in range(RSHB_N_TRIALS):
                    inject_seed = trial
                    rng_r       = np.random.default_rng(inject_seed)
                    pipeline    = FinancialRiskPipeline(real_raw, size="medium")

                    cands_r = get_injectable_nodes_for_type(pipeline, atype)
                    if not cands_r:
                        pbar.update(1)
                        continue
                    target_inject = rng_r.choice(cands_r)

                    injector = AnomalyInjector(pipeline, rng_r)
                    injector, gt = injector.inject(
                        anomaly_type=atype, target_node_id=target_inject,
                        magnitude=8.0, seed=inject_seed,
                    )
                    pipeline.execute()

                    builder  = FRLGBuilder(pipeline)
                    G        = builder.build()

                    # always compute z-scores
                    detector = NodeAnomalyDetector(threshold_zscore=2.0)
                    z_scores = detector.detect_all(pipeline)

                    # try VAE scores
                    try:
                        vae_scores = vae_scorer.score(pipeline)
                    except Exception:
                        vae_scores = z_scores

                    # ── z-score branch ───────────────────────────────────────
                    builder.set_anomaly_scores(z_scores)
                    obs_z = select_target_node(G, z_scores, threshold=0.1)
                    try:
                        rca_z = apa_engine.run(G, z_scores, obs_z)
                        rca_z.method = "APA-RCA-v2"
                        sample_z = build_gnn_sample(
                            G=G, scores=z_scores, rca_result=rca_z,
                            true_rc=gt.target_node_id, top_k=10,
                            anomaly_type=atype.value, pipeline_size="medium_real",
                            trial=trial,
                        )
                        if sample_z is not None:
                            reranked_z = z_final.rerank(sample_z, rca_z)
                            m = compute_run_metrics(
                                result=reranked_z, true_root_cause=gt.target_node_id,
                                anomaly_type=atype.value, pipeline_size="medium_real",
                                trial=trial, method_name="APA-RCA-v2+ResGCN",
                            )
                            results.add(m)
                    except Exception as e:
                        print(f"    ERROR z-RSHB [{atype.value}/{trial}]: {e}")

                    # ── VAE branch ───────────────────────────────────────────
                    builder.set_anomaly_scores(vae_scores)
                    obs_vae = select_target_node(G, vae_scores, threshold=0.1)
                    try:
                        rca_vae = apa_engine.run(G, vae_scores, obs_vae)
                        rca_vae.method = "CVAE+APA-RCA"
                        sample_vae = build_gnn_sample(
                            G=G, scores=vae_scores, rca_result=rca_vae,
                            true_rc=gt.target_node_id, top_k=10,
                            anomaly_type=atype.value, pipeline_size="medium_real",
                            trial=trial,
                        )
                        if sample_vae is not None:
                            reranked_vae = vae_final.rerank(sample_vae, rca_vae)
                            for mname, res in [("CVAE+APA-RCA", rca_vae),
                                               ("CVAE+APA-RCA+ResGCN", reranked_vae)]:
                                res.method = mname
                                m = compute_run_metrics(
                                    result=res, true_root_cause=gt.target_node_id,
                                    anomaly_type=atype.value, pipeline_size="medium_real",
                                    trial=trial, method_name=mname,
                                )
                                results.add(m)
                    except Exception as e:
                        print(f"    ERROR vae-RSHB [{atype.value}/{trial}]: {e}")

                    pbar.update(1)

    # ── Save & summarise ─────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "ml_enhanced_v2_results.csv")
    results.save(out_path)

    print(f"\n{'='*60}")
    print("ML-Enhanced Results Summary")
    print(f"{'='*60}")
    df = pd.DataFrame([vars(r) for r in results.records])
    if len(df) > 0:
        synth = df[~df["pipeline_size"].str.contains("real", na=False)]
        real  = df[df["pipeline_size"].str.contains("real", na=False)]
        for label, sub in [("Synthetic (CV)", synth), ("RSHB (real)", real)]:
            if len(sub) == 0:
                continue
            print(f"\n  [{label}]")
            grp = sub.groupby("method")[["top1", "top5", "reciprocal_rank"]].mean()
            grp.columns = ["Top-1", "Top-5", "MRR"]
            print(grp.round(3).to_string())

    return results


def run_bootstrap_ci_analysis(output_dir: str = "results") -> None:
    """
    Compute bootstrap confidence intervals from existing main_results.csv.
    Bootstraps over trial-level data (1000 resamples, 95% CI).
    Prints CI table and saves to bootstrap_ci.csv.
    """
    results_path = os.path.join(output_dir, "main_results.csv")
    if not os.path.exists(results_path):
        print(f"  No results found at {results_path}. Run main experiments first.")
        return

    results = ExperimentResults()
    results.load(results_path)

    print(f"\n{'='*60}")
    print(f"Bootstrap CI Analysis (n_bootstrap=1000, 95% CI)")
    print(f"  Source: {results_path}  ({len(results.records)} records)")
    print(f"{'='*60}\n")

    # Overall CI per method
    ci_df = results.summary_with_ci(group_by=["method"], n_bootstrap=1000, ci=0.95)

    # Pretty-print: show as "mean [lo, hi]"
    display = ci_df.copy()
    display["Top-1 (95% CI)"] = display.apply(
        lambda r: f"{r.top1_mean:.3f} [{r.top1_ci_lo:.3f}, {r.top1_ci_hi:.3f}]", axis=1)
    display["Top-5 (95% CI)"] = display.apply(
        lambda r: f"{r.top5_mean:.3f} [{r.top5_ci_lo:.3f}, {r.top5_ci_hi:.3f}]", axis=1)
    display["MRR  (95% CI)"] = display.apply(
        lambda r: f"{r.mrr_mean:.3f} [{r.mrr_ci_lo:.3f}, {r.mrr_ci_hi:.3f}]", axis=1)
    print(display[["method", "Top-1 (95% CI)", "Top-5 (95% CI)", "MRR  (95% CI)", "n_runs"]].to_string(index=False))

    # Per anomaly type CI
    print(f"\n--- Per Anomaly Type (APA-RCA-v2 only) ---\n")
    rshb_path = os.path.join(output_dir, "rshb_results.csv")
    if os.path.exists(rshb_path):
        rshb_results = ExperimentResults()
        rshb_results.load(rshb_path)
        rshb_ci = rshb_results.summary_with_ci(group_by=["method", "anomaly_type"], n_bootstrap=1000, ci=0.95)
        v2_ci = rshb_ci[rshb_ci["method"] == "APA-RCA-v2"].copy()
        v2_ci["Top-5 (95% CI)"] = v2_ci.apply(
            lambda r: f"{r.top5_mean:.3f} [{r.top5_ci_lo:.3f}, {r.top5_ci_hi:.3f}]", axis=1)
        print(v2_ci[["anomaly_type", "Top-5 (95% CI)", "n_runs"]].to_string(index=False))

    ci_df.to_csv(os.path.join(output_dir, "bootstrap_ci.csv"), index=False)
    print(f"\nSaved CI results to {os.path.join(output_dir, 'bootstrap_ci.csv')}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "results")

    print("Starting FINRCA Experiments...")
    t_start = time.time()

    main_results        = run_main_experiments(output_dir)
    ablation_results    = run_ablation_experiments(output_dir)
    scalability_results = run_scalability_analysis(output_dir)
    real_source_results = run_real_source_hybrid(output_dir)
    run_bootstrap_ci_analysis(output_dir)
    run_ml_enhanced_experiments(output_dir)

    print(f"\n{'='*60}")
    print(f"All experiments complete in {(time.time()-t_start)/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
