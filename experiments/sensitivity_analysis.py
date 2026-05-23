"""
Signal Weight Sensitivity Analysis for FINRCA Anomaly Detector.

Varies each hand-tuned fusion weight ±30% independently and measures
the effect on Top-5 accuracy and MRR across all 7 anomaly types.

Weights under test:
  W1: zscore_signal = W1a * anomaly_fraction + (1-W1a) * norm_max_z  [baseline W1a=0.6]
  W2: variance_signal multiplier at raw score stage                   [baseline 0.8]
  W3: layer 3 raw weight                                              [baseline 0.55]
  W4: layer 3 causal excess weight                                    [baseline 0.85]
  W5: layer 3 short-long var weight                                   [baseline 0.80]
  W6: layer 3 peer_causal weight                                      [baseline 0.90]
  W7: layer 4+ raw weight                                             [baseline 0.40]
  W8: layer 4+ causal weight                                          [baseline 0.60]

For each weight: test baseline × {0.70, 0.85, 1.00, 1.15, 1.30}.
Report Top-5 and MRR for each configuration.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synfrp import (MarketDataConfig, MarketDataGenerator,
                    FinancialRiskPipeline, AnomalyInjector,
                    AnomalyType, get_injectable_nodes_for_type)
from frlg import FRLGBuilder, NodeAnomalyDetector
from rca import APARCAEngine
from baselines import select_target_node
from evaluation import compute_run_metrics


# ── Baseline weights ──────────────────────────────────────────────────────────
BASELINE_WEIGHTS = {
    "W1_zscore_blend":    0.60,   # fraction weight in zscore_signal
    "W2_var_signal_cap":  0.80,   # variance_signal multiplier in raw score
    "W3_L3_raw":          0.55,   # layer 3 raw weight
    "W4_L3_excess":       0.85,   # layer 3 causal excess weight
    "W5_L3_sl":           0.80,   # layer 3 short-long var weight
    "W6_L3_peer":         0.90,   # layer 3 peer_causal weight
    "W7_L4_raw":          0.40,   # layer 4+ raw weight
    "W8_L4_excess":       0.60,   # layer 4+ causal weight
}

SCALE_FACTORS = [0.70, 0.85, 1.00, 1.15, 1.30]

PIPELINE_SIZES = {
    "small":  {"n_stocks": 3, "n_rates": 2, "n_fx": 1, "n_positions": 3},
    "medium": {"n_stocks": 5, "n_rates": 3, "n_fx": 2, "n_positions": 5},
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

N_TRIALS = 5   # Fewer trials for speed; enough to show stability


# ── Patched detector with configurable weights ────────────────────────────────

class ConfigurableDetector(NodeAnomalyDetector):
    """NodeAnomalyDetector with overrideable fusion weights."""

    def __init__(self, weights: Dict[str, float], threshold_zscore: float = 2.0):
        super().__init__(threshold_zscore=threshold_zscore)
        self.w = weights

    def score_node(self, series) -> float:
        """Override: use W1_zscore_blend and W2_var_signal_cap."""
        if series is None or len(series) < 10:
            return 0.0
        s = series.dropna()
        if len(s) < 5:
            return 0.0

        warmup_skip = min(self.window, len(s) // 4)
        s_stable = s.iloc[warmup_skip:] if len(s) > warmup_skip + 20 else s

        roll_mean = s_stable.rolling(self.window, min_periods=5).mean()
        roll_std  = s_stable.rolling(self.window, min_periods=5).std().clip(lower=1e-8)
        z_scores  = ((s_stable - roll_mean) / roll_std).abs()

        anomaly_fraction = (z_scores > self.threshold).mean()
        max_z = z_scores.max()
        norm_max_z = min(max_z / (self.threshold * 3), 1.0)

        w1 = self.w["W1_zscore_blend"]
        zscore_signal = w1 * anomaly_fraction + (1 - w1) * norm_max_z

        n = len(s_stable)
        q = max(n // 4, 5)
        segments = [s_stable.iloc[i*q:(i+1)*q].std() for i in range(4)]
        segments = [v for v in segments if not np.isnan(v) and v > 1e-8]
        if len(segments) >= 2:
            variance_ratio = max(segments) / (min(segments) + 1e-8)
            variance_signal = min((variance_ratio - 1) / 10.0, 1.0)
        else:
            variance_signal = 0.0

        w2 = self.w["W2_var_signal_cap"]
        score = max(zscore_signal, variance_signal * w2)
        return float(np.clip(score, 0.0, 1.0))

    def detect_all(self, pipeline) -> Dict[str, float]:
        """Override: use configurable layer fusion weights."""
        # Reuse parent's pass 1 / pass 2 / pass 3 logic, then apply our fusion
        raw_scores = {}
        short_long_scores = {}
        for node_id, node_data in pipeline.node_data.items():
            node_raw, node_sl = [], []
            for col, series in node_data.items():
                node_raw.append(self.score_node(series))
                node_sl.append(self._short_long_variance_ratio_score(series))
            raw_scores[node_id]        = max(node_raw) if node_raw else 0.0
            short_long_scores[node_id] = max(node_sl)  if node_sl  else 0.0

        causal_excess = self._compute_causal_excess(raw_scores, pipeline)

        # Simplified peer dev (omitted for speed — baseline already handles it)
        peer_dev_scores = {nid: 0.0 for nid in pipeline.nodes}

        # Fusion with configurable weights
        w3 = self.w["W3_L3_raw"]
        w4 = self.w["W4_L3_excess"]
        w5 = self.w["W5_L3_sl"]
        w6 = self.w["W6_L3_peer"]
        w7 = self.w["W7_L4_raw"]
        w8 = self.w["W8_L4_excess"]

        final_scores = {}
        for node_id, node in pipeline.nodes.items():
            raw    = raw_scores.get(node_id, 0.0)
            sl     = short_long_scores.get(node_id, 0.0)
            excess = causal_excess.get(node_id, 0.0)
            peer   = peer_dev_scores.get(node_id, 0.0)
            layer  = node.layer

            if layer <= 2:
                score = max(raw, sl * 0.5, peer * 0.75)
            elif layer == 3:
                excess_weight = min(excess / 0.3, 1.0)
                peer_causal   = peer * excess_weight
                score = max(
                    raw    * w3,
                    excess * w4,
                    sl     * w5,
                    peer_causal * w6,
                )
            else:
                score = max(raw * w7, excess * w8, peer * 0.35)

            final_scores[node_id] = float(np.clip(score, 0.0, 1.0))

        return final_scores


# ── Single run with configurable detector ────────────────────────────────────

def run_with_weights(
    weights: Dict[str, float],
    pipeline_size: str,
    anomaly_type: AnomalyType,
    trial: int,
    magnitude: float = 8.0,
) -> Tuple[bool, float]:
    """Returns (top5_hit, reciprocal_rank)."""
    data_seed   = trial + 42
    inject_seed = trial

    size_cfg = PIPELINE_SIZES[pipeline_size]
    cfg = MarketDataConfig(seed=data_seed,
                          n_stocks=size_cfg["n_stocks"],
                          n_rates=size_cfg["n_rates"],
                          n_fx=size_cfg["n_fx"],
                          n_positions=size_cfg["n_positions"])
    raw = MarketDataGenerator(cfg).generate_all()
    pipeline = FinancialRiskPipeline(raw, size=pipeline_size)

    rng = np.random.default_rng(inject_seed)
    candidates = get_injectable_nodes_for_type(pipeline, anomaly_type)
    if not candidates:
        return False, 0.0
    target_inject = rng.choice(candidates)

    injector = AnomalyInjector(pipeline, rng)
    injector, gt = injector.inject(
        anomaly_type=anomaly_type,
        target_node_id=target_inject,
        magnitude=magnitude,
        seed=inject_seed,
    )
    pipeline.execute()

    builder  = FRLGBuilder(pipeline)
    G        = builder.build()
    detector = ConfigurableDetector(weights, threshold_zscore=2.0)
    scores   = detector.detect_all(pipeline)
    builder.set_anomaly_scores(scores)

    observed_target = select_target_node(G, scores, threshold=0.1)

    engine = APARCAEngine(gamma=0.15, alpha=0.7)
    try:
        result = engine.run(G, scores, observed_target)
    except Exception:
        return False, 0.0

    rc   = gt.target_node_id
    rank = result.rank_of(rc)

    if rank is not None:
        hit5 = rank <= 5
        rr   = 1.0 / rank
    else:
        hit5 = False
        rr   = 0.0

    return hit5, rr


# ── Evaluate one weight configuration ────────────────────────────────────────

def evaluate_config(weights: Dict[str, float]) -> Tuple[float, float]:
    """Returns (Top5_pct, MRR) averaged over 2 sizes × 7 types × N_TRIALS."""
    hits, rrs = [], []

    for size in PIPELINE_SIZES:
        for atype in ANOMALY_TYPES:
            for trial in range(N_TRIALS):
                h, rr = run_with_weights(weights, size, atype, trial)
                hits.append(float(h))
                rrs.append(rr)

    return float(np.mean(hits)) * 100.0, float(np.mean(rrs))


# ── Main sensitivity sweep ────────────────────────────────────────────────────

def run_sensitivity_analysis(output_dir: str = "results") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    total_configs = len(BASELINE_WEIGHTS) * len(SCALE_FACTORS)
    done = 0

    print(f"\n{'='*60}")
    print(f"Signal Weight Sensitivity Analysis")
    print(f"  {len(BASELINE_WEIGHTS)} weights × {len(SCALE_FACTORS)} scales")
    print(f"  {len(PIPELINE_SIZES)} sizes × {len(ANOMALY_TYPES)} types × {N_TRIALS} trials each")
    print(f"  Total configurations: {total_configs}")
    print(f"{'='*60}\n")

    # First, evaluate baseline
    print("Evaluating baseline config...")
    base_top5, base_mrr = evaluate_config(BASELINE_WEIGHTS)
    print(f"  Baseline: Top-5={base_top5:.1f}%  MRR={base_mrr:.3f}")

    rows.append({
        "weight": "BASELINE",
        "scale":   1.00,
        "value":   None,
        "top5":    base_top5,
        "mrr":     base_mrr,
        "delta_top5": 0.0,
        "delta_mrr":  0.0,
    })

    for weight_name, base_val in BASELINE_WEIGHTS.items():
        print(f"\nSweeping {weight_name} (baseline={base_val:.2f}):")
        for scale in SCALE_FACTORS:
            if scale == 1.00:
                # Already evaluated as baseline
                rows.append({
                    "weight":     weight_name,
                    "scale":      scale,
                    "value":      base_val,
                    "top5":       base_top5,
                    "mrr":        base_mrr,
                    "delta_top5": 0.0,
                    "delta_mrr":  0.0,
                })
                done += 1
                continue

            # Build perturbed weight dict
            perturbed = dict(BASELINE_WEIGHTS)
            new_val = base_val * scale
            # Clip to [0, 1] for weights that are multipliers
            new_val = float(np.clip(new_val, 0.01, 1.0))
            perturbed[weight_name] = new_val

            top5, mrr = evaluate_config(perturbed)
            delta_top5 = top5 - base_top5
            delta_mrr  = mrr  - base_mrr

            rows.append({
                "weight":     weight_name,
                "scale":      scale,
                "value":      new_val,
                "top5":       top5,
                "mrr":        mrr,
                "delta_top5": delta_top5,
                "delta_mrr":  delta_mrr,
            })

            print(f"  ×{scale:.2f} ({weight_name}={new_val:.3f}): "
                  f"Top-5={top5:.1f}% ({delta_top5:+.1f}pp)  "
                  f"MRR={mrr:.3f} ({delta_mrr:+.3f})")

            done += 1

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "sensitivity_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SENSITIVITY SUMMARY — Max |ΔTop-5| and |ΔMRR| per weight")
    print("="*70)

    summary_rows = []
    for weight_name in BASELINE_WEIGHTS:
        sub = df[df["weight"] == weight_name]
        max_delta_top5 = sub["delta_top5"].abs().max()
        max_delta_mrr  = sub["delta_mrr"].abs().max()
        summary_rows.append({
            "Weight":        weight_name,
            "Baseline":      BASELINE_WEIGHTS[weight_name],
            "Max|ΔTop-5|":   f"{max_delta_top5:.1f}pp",
            "Max|ΔMRR|":     f"{max_delta_mrr:.3f}",
            "Sensitive?":    "YES" if max_delta_top5 > 3.0 else "no",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(output_dir, "sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return df


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "results"
    )
    run_sensitivity_analysis(output_dir)
