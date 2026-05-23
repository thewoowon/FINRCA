"""
SynFRP: Anomaly Injector
Injects controlled anomalies into a single pipeline node.
7 anomaly types matching realistic financial pipeline failures:
  A1: Source data corruption (spike in raw data)
  A2: ETL logic error (wrong conversion factor)
  A3: Feature calculation error (wrong rolling window)
  A4: Aggregation error (partial sum — missing instrument)
  A5: Downstream masking (source spike absorbed by ETL clipping)
  A6: Multi-source ambiguity (multiple weak sources → strong aggregation anomaly)
  A7: Coincidental anomaly (true root cause + unrelated high-score noise node)
"""

import numpy as np
import pandas as pd
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class AnomalyType(str, Enum):
    A1_SOURCE_CORRUPTION   = "A1_source_corruption"
    A2_ETL_LOGIC_ERROR     = "A2_etl_logic_error"
    A3_FEATURE_CALC_ERROR  = "A3_feature_calc_error"
    A4_AGGREGATION_ERROR   = "A4_aggregation_error"
    A5_DOWNSTREAM_MASKING  = "A5_downstream_masking"
    A6_MULTI_SOURCE        = "A6_multi_source"
    A7_COINCIDENTAL        = "A7_coincidental"


@dataclass
class InjectionResult:
    """Records ground truth of an anomaly injection."""
    anomaly_type: AnomalyType
    target_node_id: str          # node where anomaly was injected
    affected_columns: List[str]  # output columns of that node
    injection_magnitude: float   # severity parameter
    seed: int


class AnomalyInjector:
    """
    Injects anomalies into the financial risk pipeline.
    Returns a modified pipeline that produces anomalous outputs,
    along with ground truth of what was changed.
    """

    def __init__(self, pipeline, rng: Optional[np.random.Generator] = None):
        self.pipeline = pipeline
        self.rng = rng or np.random.default_rng(0)

    # ── Public API ──────────────────────────────────────────────────────────

    def inject(
        self,
        anomaly_type: AnomalyType,
        target_node_id: Optional[str] = None,
        magnitude: float = 5.0,
        seed: int = 0
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        Inject an anomaly into the pipeline.
        Returns a new injector with modified pipeline + ground truth.
        """
        self.rng = np.random.default_rng(seed)
        injector = AnomalyInjector(self.pipeline, self.rng)

        if anomaly_type == AnomalyType.A1_SOURCE_CORRUPTION:
            return injector._inject_source_corruption(target_node_id, magnitude, seed)
        elif anomaly_type == AnomalyType.A2_ETL_LOGIC_ERROR:
            return injector._inject_etl_logic_error(target_node_id, magnitude, seed)
        elif anomaly_type == AnomalyType.A3_FEATURE_CALC_ERROR:
            return injector._inject_feature_calc_error(target_node_id, magnitude, seed)
        elif anomaly_type == AnomalyType.A4_AGGREGATION_ERROR:
            return injector._inject_aggregation_error(target_node_id, magnitude, seed)
        elif anomaly_type == AnomalyType.A5_DOWNSTREAM_MASKING:
            return injector._inject_downstream_masking(target_node_id, magnitude, seed)
        elif anomaly_type == AnomalyType.A6_MULTI_SOURCE:
            return injector._inject_multi_source(target_node_id, magnitude, seed)
        elif anomaly_type == AnomalyType.A7_COINCIDENTAL:
            return injector._inject_coincidental(target_node_id, magnitude, seed)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    # ── Injection Methods ────────────────────────────────────────────────────

    def _inject_source_corruption(
        self, target_node_id: Optional[str], magnitude: float, seed: int
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        A1: Insert a price spike in a source node's data.
        Simulates incorrect feed data (e.g., a data vendor error).
        """
        # Pick a source node if not specified
        source_nodes = self.pipeline.get_source_nodes()
        # Filter to stock price sources only (most realistic)
        stock_sources = [n for n in source_nodes if "STOCK" in n and "PRICE" in n]
        if target_node_id is None or target_node_id not in stock_sources:
            target_node_id = self.rng.choice(stock_sources)

        node = self.pipeline.nodes[target_node_id]
        col  = node.output_columns[0]

        # Inject spike: randomly choose 5 consecutive days, multiply by magnitude
        n = self.pipeline.raw_data["stock_prices"].shape[0]
        spike_start = int(self.rng.integers(60, n - 10))
        spike_len   = int(self.rng.integers(1, 6))

        # Override compute_fn to inject corrupted data
        original_fn = node.compute_fn
        spike_s, spike_e = spike_start, spike_start + spike_len

        def corrupted_fn(d, fn=original_fn, c=col, s=spike_s, e=spike_e, m=magnitude):
            result = fn(d)
            series = result[c].copy()
            series.iloc[s:e] = series.iloc[s:e] * m
            return {c: series}

        node.compute_fn = corrupted_fn

        result = InjectionResult(
            anomaly_type=AnomalyType.A1_SOURCE_CORRUPTION,
            target_node_id=target_node_id,
            affected_columns=[col],
            injection_magnitude=magnitude,
            seed=seed
        )
        return self, result

    def _inject_etl_logic_error(
        self, target_node_id: Optional[str], magnitude: float, seed: int
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        A2: Wrong conversion factor in an ETL node.
        Simulates a misconfigured FX rate or scale factor.
        Adds both a scale error AND a spike to make detection easier.
        """
        etl_nodes = [nid for nid, n in self.pipeline.nodes.items()
                     if n.layer == 2 and "ETL_FX" in nid]
        if target_node_id is None or target_node_id not in etl_nodes:
            target_node_id = self.rng.choice(etl_nodes)

        node   = self.pipeline.nodes[target_node_id]
        col    = node.output_columns[0]
        orig_fn = node.compute_fn
        # Wrong factor: off by magnitude AND add systematic bias shift
        wrong_factor = magnitude
        rng = self.rng

        def wrong_etl_fn(d, fn=orig_fn, c=col, wf=wrong_factor, r=rng):
            result = fn(d)
            series = result[c]
            # Scale error + spike injection
            corrupted = series * wf
            # Additional spike in middle portion
            n = len(corrupted)
            spike_start = n // 3
            corrupted.iloc[spike_start:spike_start + 10] *= 3.0
            return {c: corrupted}

        node.compute_fn = wrong_etl_fn

        result = InjectionResult(
            anomaly_type=AnomalyType.A2_ETL_LOGIC_ERROR,
            target_node_id=target_node_id,
            affected_columns=[col],
            injection_magnitude=magnitude,
            seed=seed
        )
        return self, result

    def _inject_feature_calc_error(
        self, target_node_id: Optional[str], magnitude: float, seed: int
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        A3: Wrong rolling window in volatility calculation.
        Simulates a code bug where window parameter is wrong.
        """
        feat_nodes = [nid for nid, n in self.pipeline.nodes.items()
                      if n.layer == 3 and "VOL20" in nid]
        if target_node_id is None or target_node_id not in feat_nodes:
            target_node_id = self.rng.choice(feat_nodes)

        node    = self.pipeline.nodes[target_node_id]
        in_col  = node.input_columns[0]
        out_col = node.output_columns[0]
        # Wrong window: use 2 instead of 20 + add noise to make it detectable
        wrong_window = max(2, int(20 / magnitude))

        def wrong_vol_fn(d, ic=in_col, oc=out_col, w=wrong_window):
            # Wrong window + scale by magnitude to amplify the discrepancy
            vol = d[ic].rolling(w).std().fillna(0)
            return {oc: vol * magnitude}

        node.compute_fn = wrong_vol_fn

        result = InjectionResult(
            anomaly_type=AnomalyType.A3_FEATURE_CALC_ERROR,
            target_node_id=target_node_id,
            affected_columns=[out_col],
            injection_magnitude=magnitude,
            seed=seed
        )
        return self, result

    def _inject_aggregation_error(
        self, target_node_id: Optional[str], magnitude: float, seed: int
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        A4: Partial aggregation — one instrument dropped from the sum.
        Simulates a sector aggregation missing one stock.
        """
        agg_nodes = [nid for nid, n in self.pipeline.nodes.items()
                     if n.layer == 3 and "SECTOR" in nid]
        if target_node_id is None or target_node_id not in agg_nodes:
            target_node_id = self.rng.choice(agg_nodes)

        node    = self.pipeline.nodes[target_node_id]
        in_cols = node.input_columns.copy()
        out_col = node.output_columns[0]

        # Drop one random input column from the aggregation
        drop_idx = int(self.rng.integers(0, len(in_cols)))
        reduced  = [c for i, c in enumerate(in_cols) if i != drop_idx]
        if not reduced:   # fallback: keep all but scale down
            reduced = in_cols

        def partial_agg_fn(d, cols=reduced, oc=out_col):
            return {oc: sum(d[c] for c in cols) / len(cols)}

        node.compute_fn = partial_agg_fn
        # Update input_columns to reflect the dropped instrument
        # This enables Schema-Constraint RCA to detect the violation
        # (actual input count < expected_input_count)
        node.input_columns = reduced

        result = InjectionResult(
            anomaly_type=AnomalyType.A4_AGGREGATION_ERROR,
            target_node_id=target_node_id,
            affected_columns=[out_col],
            injection_magnitude=magnitude,
            seed=seed
        )
        return self, result


    def _inject_downstream_masking(
        self, target_node_id: Optional[str], magnitude: float, seed: int
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        A5: Source spike is largely absorbed by ETL clipping.
        Source node anomaly score drops to 0.3-0.5 range.
        Downstream nodes (feature/risk) show residual elevated scores (0.4-0.7).
        Score-only picks ETL/feature node; graph should trace back to source.

        Mechanism: inject spike of moderate size (not extreme), so that the
        4-sigma clipping removes ~70% of the spike energy. The source still
        has a detectable but non-dominant anomaly signal, while mid-pipeline
        nodes carry the attenuated residual.
        """
        stock_sources = [n for n in self.pipeline.get_source_nodes()
                         if "STOCK" in n and "PRICE" in n]
        if target_node_id is None or target_node_id not in stock_sources:
            target_node_id = self.rng.choice(stock_sources)

        node = self.pipeline.nodes[target_node_id]
        col  = node.output_columns[0]
        original_fn = node.compute_fn

        n = self.pipeline.raw_data["stock_prices"].shape[0]
        spike_start = int(self.rng.integers(60, n - 20))
        # Moderate magnitude: ETL clipping (4-sigma) will absorb most of it.
        # Use magnitude of 3.5–4.5x std range — just at/above clip threshold.
        # This leaves source score ~0.35 after partial clipping rather than 1.0.
        clip_factor = 3.8  # just above the 4-sigma clip boundary

        def masked_source_fn(d, fn=original_fn, c=col, s=spike_start,
                             cf=clip_factor):
            result = fn(d)
            series = result[c].copy()
            mu = float(series.mean())
            sigma = float(series.std())
            # Spike sits just above the 4-sigma clip boundary → partially clipped
            series.iloc[s:s + 8] = mu + cf * sigma * magnitude / 8.0
            return {c: series}

        node.compute_fn = masked_source_fn

        result = InjectionResult(
            anomaly_type=AnomalyType.A5_DOWNSTREAM_MASKING,
            target_node_id=target_node_id,
            affected_columns=[col],
            injection_magnitude=magnitude,
            seed=seed
        )
        return self, result

    def _inject_multi_source(
        self, target_node_id: Optional[str], magnitude: float, seed: int
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        A6: Multiple source nodes simultaneously exhibit moderate anomalies.
        These combine through aggregation so the aggregation node has the
        highest anomaly score, but each source contributed independently.
        Ground truth: the *first* (largest-contributing) source is root cause.

        Score-only: picks the sector aggregation node (highest score).
        Graph: traces back through multiple source paths; APA-RCA distributes
        walk mass across all contributing ancestors.
        """
        stock_sources = [n for n in self.pipeline.get_source_nodes()
                         if "STOCK" in n and "PRICE" in n]

        # Pick 2-3 sources to corrupt simultaneously
        n_corrupt = min(3, len(stock_sources))
        chosen = list(self.rng.choice(stock_sources, size=n_corrupt, replace=False))
        # Ground truth: the first chosen source (largest magnitude)
        primary_target = chosen[0]

        data_len = self.pipeline.raw_data["stock_prices"].shape[0]
        spike_start = int(self.rng.integers(60, data_len - 15))

        for i, src_id in enumerate(chosen):
            node = self.pipeline.nodes[src_id]
            col  = node.output_columns[0]
            orig_fn = node.compute_fn
            # Decaying magnitude: primary=1.0, secondary=0.6, tertiary=0.35
            factor = [1.0, 0.6, 0.35][i]
            spike_mag = magnitude * factor * 0.5  # moderate: score ~0.4-0.6

            def multi_corrupt_fn(d, fn=orig_fn, c=col, s=spike_start,
                                 m=spike_mag):
                result = fn(d)
                series = result[c].copy()
                series.iloc[s:s + 5] = series.iloc[s:s + 5] * m
                return {c: series}

            node.compute_fn = multi_corrupt_fn

        result = InjectionResult(
            anomaly_type=AnomalyType.A6_MULTI_SOURCE,
            target_node_id=primary_target,
            affected_columns=[self.pipeline.nodes[primary_target].output_columns[0]],
            injection_magnitude=magnitude,
            seed=seed
        )
        return self, result

    def _inject_coincidental(
        self, target_node_id: Optional[str], magnitude: float, seed: int
    ) -> Tuple["AnomalyInjector", InjectionResult]:
        """
        A7: True anomaly at a source (A1-style) + a coincidentally high-scoring
        node in a completely different branch (market-driven noise amplification).

        Score-only: may pick the coincidental high-score node as root cause.
        Graph (ancestor constraint): the coincidental node is NOT an ancestor
        of the target report node → gets 0.05× penalty → correctly deprioritised.
        """
        stock_sources = [n for n in self.pipeline.get_source_nodes()
                         if "STOCK" in n and "PRICE" in n]
        if len(stock_sources) < 2:
            # Fallback to A1 if only one source
            return self._inject_source_corruption(target_node_id, magnitude, seed)

        # Pick the true root cause source
        if target_node_id is None or target_node_id not in stock_sources:
            target_node_id = stock_sources[0]

        # Inject A1-style spike at true root cause
        node = self.pipeline.nodes[target_node_id]
        col  = node.output_columns[0]
        orig_fn = node.compute_fn
        data_len = self.pipeline.raw_data["stock_prices"].shape[0]
        spike_start = int(self.rng.integers(60, data_len - 10))
        spike_len   = int(self.rng.integers(2, 6))

        def true_corrupt_fn(d, fn=orig_fn, c=col, s=spike_start,
                            e=spike_start + spike_len, m=magnitude):
            result = fn(d)
            series = result[c].copy()
            series.iloc[s:e] = series.iloc[s:e] * m
            return {c: series}

        node.compute_fn = true_corrupt_fn

        # Now pick a DIFFERENT branch source and give it elevated variance
        # (simulating market-driven volatility in an unrelated instrument).
        # This node is NOT an ancestor of the same report target → ancestor
        # constraint will filter it.
        other_sources = [s for s in stock_sources if s != target_node_id]
        noise_src_id  = self.rng.choice(other_sources)
        noise_node    = self.pipeline.nodes[noise_src_id]
        noise_col     = noise_node.output_columns[0]
        noise_orig_fn = noise_node.compute_fn
        # Add high-variance noise (score will be ~0.7-0.9, higher than some
        # true-RC propagation nodes)
        noise_mag = magnitude * 0.7

        def noise_fn(d, fn=noise_orig_fn, c=noise_col, m=noise_mag,
                     rng=np.random.default_rng(seed + 999)):
            result = fn(d)
            series = result[c].copy()
            sigma  = float(series.std())
            noise  = rng.normal(0, sigma * m, size=len(series))
            series = series + pd.Series(noise, index=series.index)
            return {c: series}

        noise_node.compute_fn = noise_fn

        result = InjectionResult(
            anomaly_type=AnomalyType.A7_COINCIDENTAL,
            target_node_id=target_node_id,
            affected_columns=[col],
            injection_magnitude=magnitude,
            seed=seed
        )
        return self, result


def get_injectable_nodes_for_type(pipeline, anomaly_type: AnomalyType) -> List[str]:
    """Return list of eligible target nodes for a given anomaly type."""
    if anomaly_type == AnomalyType.A1_SOURCE_CORRUPTION:
        return [n for n in pipeline.get_source_nodes() if "STOCK" in n]
    elif anomaly_type == AnomalyType.A2_ETL_LOGIC_ERROR:
        return [nid for nid, n in pipeline.nodes.items()
                if n.layer == 2 and "ETL_FX" in nid]
    elif anomaly_type == AnomalyType.A3_FEATURE_CALC_ERROR:
        return [nid for nid, n in pipeline.nodes.items()
                if n.layer == 3 and "VOL20" in nid]
    elif anomaly_type == AnomalyType.A4_AGGREGATION_ERROR:
        return [nid for nid, n in pipeline.nodes.items()
                if n.layer == 3 and "SECTOR" in nid]
    elif anomaly_type == AnomalyType.A5_DOWNSTREAM_MASKING:
        return [n for n in pipeline.get_source_nodes() if "STOCK" in n and "PRICE" in n]
    elif anomaly_type == AnomalyType.A6_MULTI_SOURCE:
        # Ground truth is the primary (first) corrupted source; return all sources
        return [n for n in pipeline.get_source_nodes() if "STOCK" in n and "PRICE" in n]
    elif anomaly_type == AnomalyType.A7_COINCIDENTAL:
        return [n for n in pipeline.get_source_nodes() if "STOCK" in n and "PRICE" in n]
    return []
