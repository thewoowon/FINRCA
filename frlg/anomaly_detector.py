"""
FRLG Anomaly Detector
Assigns anomaly scores to each pipeline node based on its output data.

Two-pass approach:
  Pass 1 (raw): z-score + variance-ratio detection per node (existing)
  Pass 2 (causal): subtract upstream signal — isolate anomaly *originated* at node
    - causal_excess = raw_score(node) - max(raw_score(upstream))
    - correlation_break: detect when input→output relationship changes (key for A3)
    - short_long_variance_ratio: high-frequency variance spike (A3 wrong window)

Final score = max(raw_signal, causal_excess_signal, correlation_break_signal)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats

from synfrp.pipeline import FinancialRiskPipeline


class NodeAnomalyDetector:
    """
    Detects anomalies in pipeline node outputs using statistical methods.
    Computes a scalar anomaly score for each node in [0, 1].
    """

    def __init__(self, threshold_zscore: float = 3.0, window: int = 60):
        self.threshold = threshold_zscore
        self.window    = window

    # ── Core signal: rolling z-score + variance ratio ──────────────────────

    def score_node(self, series: pd.Series) -> float:
        """
        Compute raw anomaly score for a single time series.
        Returns value in [0, 1].
        """
        if series is None or len(series) < 10:
            return 0.0

        s = series.dropna()
        if len(s) < 5:
            return 0.0

        # Skip rolling warmup period: first `window` rows may be filled constants
        # (from bfill/ffill), causing artificial variance inflation in segment 1.
        # Drop initial flat/constant prefix (where std of rolling 10-window is ~0).
        warmup_skip = min(self.window, len(s) // 4)
        s_stable = s.iloc[warmup_skip:] if len(s) > warmup_skip + 20 else s

        # Signal 1: Rolling z-score exceedance (use stable portion)
        roll_mean = s_stable.rolling(self.window, min_periods=5).mean()
        roll_std  = s_stable.rolling(self.window, min_periods=5).std().clip(lower=1e-8)
        z_scores  = ((s_stable - roll_mean) / roll_std).abs()

        anomaly_fraction = (z_scores > self.threshold).mean()
        max_z = z_scores.max()
        norm_max_z = min(max_z / (self.threshold * 3), 1.0)
        zscore_signal = 0.6 * anomaly_fraction + 0.4 * norm_max_z

        # Signal 2: Segment-based variance jump (4 quarters, stable portion only)
        n = len(s_stable)
        q = max(n // 4, 5)
        segments = [s_stable.iloc[i*q:(i+1)*q].std() for i in range(4)]
        segments = [v for v in segments if not np.isnan(v) and v > 1e-8]
        if len(segments) >= 2:
            max_seg = max(segments)
            min_seg = min(segments)
            variance_ratio = max_seg / (min_seg + 1e-8)
            variance_signal = min((variance_ratio - 1) / 10.0, 1.0)
        else:
            variance_signal = 0.0

        score = max(zscore_signal, variance_signal * 0.8)
        return float(np.clip(score, 0.0, 1.0))

    # ── New: short-vs-long variance ratio (A3 wrong rolling window) ─────────

    def _short_long_variance_ratio_score(self, series: pd.Series) -> float:
        """
        Detect A3-type anomaly: wrong short rolling window inflates
        high-frequency variance relative to long-window variance.

        Computes ratio of short-window (5-day) std vs long-window (30-day) std.
        Under a wrong window (e.g., 2 instead of 20), the output series has
        disproportionately high short-term variability.
        """
        s = series.dropna()
        if len(s) < 40:
            return 0.0

        short_w = 5
        long_w  = 30

        short_vol = s.rolling(short_w, min_periods=3).std()
        long_vol  = s.rolling(long_w,  min_periods=10).std()

        # Ratio of mean short-vol to mean long-vol (aligned on non-NaN)
        mask = short_vol.notna() & long_vol.notna() & (long_vol > 1e-8)
        if mask.sum() < 10:
            return 0.0

        ratio_series = short_vol[mask] / long_vol[mask]
        median_ratio = ratio_series.median()
        max_ratio    = ratio_series.quantile(0.95)

        # Healthy: short ~ long → ratio ≈ 1
        # Wrong window: short >> long → ratio >> 1
        # Threshold: ratio > 3 is suspicious
        score_median = min((median_ratio - 1.0) / 4.0, 1.0)
        score_max    = min((max_ratio - 1.0) / 6.0, 1.0)
        return float(np.clip(max(score_median, score_max) * 0.9, 0.0, 1.0))

    # ── New: peer z-score (cross-node comparison) ────────────────────────────

    def _peer_deviation_score(
        self,
        series: pd.Series,
        peer_series_list: list,
    ) -> float:
        """
        Detect when a node's output is a statistical outlier among its peers
        (nodes of the same type/layer computing the same quantity for different instruments).

        For A3: FEAT_VOL20_STOCK_5 should have similar mean and std as
        FEAT_VOL20_STOCK_1/2/3/4. If one is 8× larger, it stands out.

        Strategy: z-score the node's mean value against the distribution of
        peer means. High z-score → anomaly origin likely here.
        """
        if not peer_series_list:
            return 0.0

        s = series.dropna()
        if len(s) < 10:
            return 0.0

        # Use stable portion (skip warmup)
        warmup = min(60, len(s) // 4)
        s_stable = s.iloc[warmup:] if len(s) > warmup + 10 else s

        node_mean = float(s_stable.mean())
        node_std  = float(s_stable.std())

        peer_means = []
        peer_cvs   = []   # coefficient of variation = std/|mean| (scale-invariant)
        for ps in peer_series_list:
            ps2 = ps.dropna()
            pw  = min(60, len(ps2) // 4)
            ps_stable = ps2.iloc[pw:] if len(ps2) > pw + 10 else ps2
            if len(ps_stable) > 5:
                pm_ = float(ps_stable.mean())
                ps_ = float(ps_stable.std())
                peer_means.append(pm_)
                # CV: use abs(mean) to avoid division by near-zero
                cv = ps_ / (abs(pm_) + 1e-8)
                peer_cvs.append(cv)

        if len(peer_means) < 1:
            return 0.0

        # CV of this node (computed before branching)
        node_cv = node_std / (abs(node_mean) + 1e-8)

        # With a single peer, use direct ratio-based comparison instead of z-score
        if len(peer_means) == 1:
            peer_m = peer_means[0]
            peer_cv = peer_cvs[0] if peer_cvs else 1.0
            # Mean-level difference (directional: are we lower than peer?)
            mean_ratio = abs(node_mean - peer_m) / (abs(peer_m) + 1e-8)
            # CV difference
            cv_diff = abs(node_cv - peer_cv) / (peer_cv + 1e-8)
            score = min(max(mean_ratio, cv_diff) * 2.0, 1.0)
            return float(np.clip(score, 0.0, 1.0))

        # Z-score of this node's mean vs peer distribution (scale-normalized)
        pm = np.mean(peer_means)
        ps_std = np.std(peer_means) + 1e-10
        z_mean = abs(node_mean - pm) / ps_std

        # Z-score of this node's CV vs peer distribution
        pms_cv  = np.mean(peer_cvs)
        pss_cv  = np.std(peer_cvs) + 1e-10
        z_cv    = abs(node_cv - pms_cv) / pss_cv

        # Directional mean drop (A4: sector aggregation missing instrument → lower mean)
        signed_z_mean_drop = max(0.0, (pm - node_mean) / (ps_std + 1e-10))

        score = min(max(z_mean, z_cv, signed_z_mean_drop * 0.8) / 5.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    # ── New: φ_peer — temporal fraction of peer deviation (ATF-derived) ────
    #
    # Derived from the Anomaly Transfer Function (ATF) framework:
    # - Code bugs (A3) produce a function replacement f→f', affecting ALL time
    #   steps → φ_peer ≈ 1.0 (node deviates from peers at every point in time)
    # - Data spikes (A5) propagate through a rolling window of width w,
    #   affecting only w/T time steps → φ_peer ≈ w/T ≈ 0.08
    #
    # This is distribution-free: it depends only on the temporal structure of
    # the anomaly, not on the specific data distribution.

    def _phi_peer(
        self,
        series: pd.Series,
        peer_series_list: list,
        threshold_z: float = 2.0,
    ) -> float:
        """
        Temporal fraction: what fraction of time steps does this node
        deviate from its peer mean by more than threshold_z std deviations?

        A3 (wrong window): φ ≈ 0.7–0.9 (persistent deviation at all times)
        A5 (spike propagation): φ ≈ 0.05–0.37 (transient, window-sized)
        """
        s = series.dropna()
        if not peer_series_list or len(s) < 20:
            return 0.0

        peer_df = pd.concat(
            [p.reindex(s.index) for p in peer_series_list], axis=1
        )
        peer_mean = peer_df.mean(axis=1)
        peer_std = peer_df.std(axis=1).clip(lower=1e-8)
        z = ((s - peer_mean) / peer_std).abs()
        phi = float((z > threshold_z).mean())
        return phi

    # ── New: causal excess (isolate origin vs propagation) ──────────────────

    # ATF-derived transfer coefficients for edge-specific causal discount
    ATF_DISCOUNT = {
        "source":     1.0,
        "direct_map": 1.0,
        "filter":     0.8,
        "calculate":  1.0,
        "aggregate":  0.45,
        "report":     0.1,
    }

    def _compute_causal_excess(
        self,
        raw_scores: Dict[str, float],
        pipeline: FinancialRiskPipeline,
    ) -> Dict[str, float]:
        """
        For each node, subtract the max anomaly score of its upstream nodes,
        discounted by the ATF transfer coefficient ρ_τ of the node's transform.

        causal_excess(v) = raw(v) - ρ_τ(v) × max(raw(u) for u in upstream(v))

        ATF-derived: ρ_τ is the expected z-score transfer ratio through
        transform type τ. Higher ρ means more anomaly propagation from upstream,
        so a higher discount is applied (less credit for origin).
        """
        excess: Dict[str, float] = {}

        for node_id, node in pipeline.nodes.items():
            raw = raw_scores.get(node_id, 0.0)
            upstream_ids = node.upstream_nodes

            if not upstream_ids:
                # Source node: no upstream to blame → full score
                excess[node_id] = raw
                continue

            max_upstream = max(
                (raw_scores.get(u, 0.0) for u in upstream_ids),
                default=0.0,
            )
            # Edge-specific discount: use this node's transform type
            t_type = node.transform_type.value if hasattr(node.transform_type, 'value') else str(node.transform_type)
            discount = self.ATF_DISCOUNT.get(t_type, 0.7)
            upstream_contribution = discount * max_upstream
            causal_excess = raw - upstream_contribution
            excess[node_id] = float(np.clip(causal_excess, 0.0, 1.0))

        return excess

    # ── Main API ─────────────────────────────────────────────────────────────

    def detect_all(
        self,
        pipeline: FinancialRiskPipeline,
    ) -> Dict[str, float]:
        """
        Compute anomaly scores for all nodes in the pipeline.
        Returns {node_id: score}.

        Multi-signal fusion:
          - raw_score: z-score + variance ratio (existing)
          - short_long_var_score: high-frequency variance spike (A3)
          - causal_excess: isolate origin from propagation (A3/A4)
          - correlation_break: input-output relationship change (A3)
        """
        # ── Pass 1: raw scores ───────────────────────────────────────────────
        raw_scores: Dict[str, float] = {}
        short_long_scores: Dict[str, float] = {}

        for node_id, node_data in pipeline.node_data.items():
            node_raw = []
            node_sl  = []
            for col, series in node_data.items():
                node_raw.append(self.score_node(series))
                node_sl.append(self._short_long_variance_ratio_score(series))
            raw_scores[node_id]        = max(node_raw) if node_raw else 0.0
            short_long_scores[node_id] = max(node_sl)  if node_sl  else 0.0

        # ── Pass 2: causal excess ────────────────────────────────────────────
        causal_excess = self._compute_causal_excess(raw_scores, pipeline)

        # ── Pass 3: peer deviation (cross-node comparison for same-type nodes) ──
        # Group nodes by (layer, transform_type prefix) — e.g., all FEAT_VOL20_*
        # nodes are peers and should produce similar-scale outputs.
        from collections import defaultdict

        # Build peer groups: nodes computing the same metric for different instruments.
        # Key = (layer, metric_type, instrument_family)
        # instrument_family distinguishes STOCK vs SECTOR vs RATE vs FX so that
        # SECTOR nodes don't get grouped with individual STOCK return nodes.
        def _instrument_family(col: str) -> str:
            """Extract the instrument family from a column name."""
            col_upper = col.upper()
            if "SECTOR" in col_upper:
                return "SECTOR"
            if "STOCK" in col_upper or "PRICE" in col_upper:
                return "STOCK"
            if "RATE" in col_upper:
                return "RATE"
            if any(fx in col_upper for fx in ["USD", "EUR", "JPY", "GBP", "CNY", "KRW", "FX"]):
                return "FX"
            if "POSITION" in col_upper or "SIZE" in col_upper:
                return "POSITION"
            if "VAR" in col_upper or "ES" in col_upper or "STRESS" in col_upper or "CONCENTRATION" in col_upper:
                return "RISK"
            if "TOTAL" in col_upper or "VALUE" in col_upper:
                return "PORTFOLIO"
            return "OTHER"

        def _peer_key(node):
            out_cols = node.output_columns
            if not out_cols:
                return None
            col = out_cols[0]
            parts = col.split("_")
            # Metric type = last token (VOL20, VOL60, RETURN, SIZE, etc.)
            metric = parts[-1] if len(parts) >= 2 else col
            family = _instrument_family(col)
            return (node.layer, metric, family)

        peer_groups: dict = defaultdict(list)
        for node_id, node in pipeline.nodes.items():
            key = _peer_key(node)
            if key:
                peer_groups[key].append(node_id)

        # For each node, compute peer deviation score and φ_peer (temporal fraction)
        peer_dev_scores: Dict[str, float] = {}
        phi_peer_scores: Dict[str, float] = {}
        for node_id, node in pipeline.nodes.items():
            key = _peer_key(node)
            if not key:
                peer_dev_scores[node_id] = 0.0
                phi_peer_scores[node_id] = 0.0
                continue

            peers = [p for p in peer_groups[key] if p != node_id]
            if len(peers) < 1:  # need at least 1 peer
                peer_dev_scores[node_id] = 0.0
                phi_peer_scores[node_id] = 0.0
                continue

            node_data = pipeline.node_data.get(node_id, {})
            out_col = node.output_columns[0] if node.output_columns else None
            if not out_col or out_col not in node_data:
                peer_dev_scores[node_id] = 0.0
                phi_peer_scores[node_id] = 0.0
                continue

            out_series = node_data[out_col]
            peer_series = []
            for peer_id in peers:
                peer_data = pipeline.node_data.get(peer_id, {})
                peer_node = pipeline.nodes[peer_id]
                peer_out_col = peer_node.output_columns[0] if peer_node.output_columns else None
                if peer_out_col and peer_out_col in peer_data:
                    peer_series.append(peer_data[peer_out_col])

            peer_dev_scores[node_id] = self._peer_deviation_score(out_series, peer_series)
            phi_peer_scores[node_id] = self._phi_peer(out_series, peer_series)

        # ── Fusion ───────────────────────────────────────────────────────────
        # For feature-layer nodes (layer 3), weight causal signals more heavily
        # since A3/A4 occur there and propagation masking is the main problem.
        # For source/ETL nodes (layer 1-2), raw signal is already reliable.
        final_scores: Dict[str, float] = {}
        for node_id, node in pipeline.nodes.items():
            raw    = raw_scores.get(node_id, 0.0)
            sl     = short_long_scores.get(node_id, 0.0)
            excess = causal_excess.get(node_id, 0.0)
            peer   = peer_dev_scores.get(node_id, 0.0)
            layer  = node.layer

            if layer <= 2:
                # Source/ETL: trust raw signal (A1, A2 are well-detected)
                # Also use peer deviation to boost source that stands out from peers
                score = max(raw, sl * 0.5, peer * 0.75)
            elif layer == 3:
                # Feature: use causal + peer signals to isolate origin
                # Key insight: peer_deviation alone can be high for propagated anomalies
                # (e.g., FEAT_RETURN_STOCK_5 is a peer outlier because its upstream SRC is bad).
                # Normally, multiply peer by excess-based weight so only origin nodes are boosted.
                #
                # Gate peer by causal excess: only boost nodes where anomaly
                # is NOT fully explained by upstream (excess ∈ [0,1])
                excess_weight = min(excess / 0.3, 1.0)
                peer_causal = peer * excess_weight

                # peer_origin signal (A3), derived from ATF temporal analysis:
                # φ_peer measures what fraction of time steps this node deviates
                # from its peers. Code bugs (A3) affect ALL time steps (φ≈0.7-0.9),
                # while data spike propagation (A5) affects only w/T steps (φ≈0.05-0.3).
                # This is distribution-free — it works on both synthetic and real data.
                out_cols = node.output_columns
                is_vol_feature = any("VOL" in c for c in out_cols)
                phi = phi_peer_scores.get(node_id, 0.0)
                upstream_peer_max = max(
                    (peer_dev_scores.get(u, 0.0) for u in node.upstream_nodes),
                    default=0.0,
                )
                if is_vol_feature and phi > 0.50 and peer > 0.70 and upstream_peer_max < 0.50:
                    peer_origin = peer * 0.85
                else:
                    peer_origin = 0.0

                score = max(
                    raw * 0.55,          # raw (downweighted)
                    excess * 0.85,       # causal excess (origin signal)
                    sl * 0.80,           # short-long var ratio (A3 window bug)
                    peer_causal * 0.90,  # peer deviation, gated by excess
                    peer_origin,         # A3 wrong-window: stable elevated VOL outlier
                )
            else:
                # Risk/Report: primarily propagation — downweight
                score = max(
                    raw * 0.40,
                    excess * 0.60,
                    peer * 0.35,
                )

            final_scores[node_id] = float(np.clip(score, 0.0, 1.0))

        return final_scores

    def detect_all_with_change_point(
        self,
        pipeline: FinancialRiskPipeline,
    ) -> Dict[str, float]:
        """
        Alias for detect_all (change-point logic is now integrated).
        Kept for backwards compatibility.
        """
        scores = self.detect_all(pipeline)

        # Additional half-split variance jump pass (kept for compatibility)
        for node_id, node_data in pipeline.node_data.items():
            for col, series in node_data.items():
                s = series.dropna()
                if len(s) < 20:
                    continue
                mid = len(s) // 2
                s1_std = s[:mid].std()
                s2_std = s[mid:].std()
                if s1_std > 1e-8:
                    variance_jump = abs(s2_std - s1_std) / s1_std
                    jump_score = min(variance_jump / 5.0, 1.0)
                    scores[node_id] = max(scores[node_id], jump_score * 0.7)

        return scores


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from synfrp import (MarketDataConfig, MarketDataGenerator,
                        FinancialRiskPipeline, AnomalyInjector, AnomalyType)

    for atype in [AnomalyType.A1_SOURCE_CORRUPTION, AnomalyType.A2_ETL_LOGIC_ERROR,
                  AnomalyType.A3_FEATURE_CALC_ERROR, AnomalyType.A4_AGGREGATION_ERROR]:
        cfg = MarketDataConfig(seed=42)
        raw = MarketDataGenerator(cfg).generate_all()
        pipeline = FinancialRiskPipeline(raw)

        injector = AnomalyInjector(pipeline)
        injector, gt = injector.inject(atype, seed=0, magnitude=8.0)
        pipeline.execute()

        detector = NodeAnomalyDetector(threshold_zscore=2.0)
        scores = detector.detect_all(pipeline)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rc_rank = next((i+1 for i, (nid, _) in enumerate(sorted_scores)
                        if nid == gt.target_node_id), None)

        print(f"\n=== {atype.value} | RC: {gt.target_node_id} | Rank: {rc_rank} ===")
        for nid, sc in sorted_scores[:8]:
            marker = " ← RC" if nid == gt.target_node_id else ""
            print(f"  {nid}: {sc:.4f}{marker}")
