"""
Evaluation Framework: Metrics for FINRCA experiments.

Accuracy metrics:
  - Top-K Accuracy (K=1,3,5): fraction of runs where RC is in Top-K
  - MRR (Mean Reciprocal Rank)
  - False Root Cause Rate @ K

Efficiency metrics:
  - MTTD (Mean Time to Detect): time from injection to RCA output
  - Query Latency

Coverage metric:
  - Trace Completeness: fraction of true propagation path captured in FRLG
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from rca.apa_rca import RCAResult


@dataclass
class SingleRunMetrics:
    """Metrics for a single experiment run."""
    method: str
    anomaly_type: str
    pipeline_size: str
    trial: int
    true_root_cause: str
    target_node: str

    rank: Optional[int]          # rank of true RC (None if not found)
    top1: int                    # 1 if RC is rank-1, else 0
    top3: int
    top5: int
    reciprocal_rank: float       # 1/rank if found, else 0
    false_rc_rate_k5: float      # fraction of top-5 that are NOT true RC

    elapsed_ms: float            # total time for this run
    convergence_iters: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "anomaly_type": self.anomaly_type,
            "pipeline_size": self.pipeline_size,
            "trial": self.trial,
            "true_root_cause": self.true_root_cause,
            "target_node": self.target_node,
            "rank": self.rank,
            "top1": self.top1,
            "top3": self.top3,
            "top5": self.top5,
            "reciprocal_rank": self.reciprocal_rank,
            "false_rc_rate_k5": self.false_rc_rate_k5,
            "elapsed_ms": self.elapsed_ms,
            "convergence_iters": self.convergence_iters,
        }


def compute_run_metrics(
    result: RCAResult,
    true_root_cause: str,
    anomaly_type: str,
    pipeline_size: str,
    trial: int,
    method_name: Optional[str] = None,
) -> SingleRunMetrics:
    """Compute metrics for a single RCA result."""
    rank = result.rank_of(true_root_cause)

    top1 = int(rank == 1)
    top3 = int(rank is not None and rank <= 3)
    top5 = int(rank is not None and rank <= 5)
    rr   = (1.0 / rank) if rank is not None else 0.0

    # False RC rate @ K=5: fraction of top-5 that are NOT the true RC
    top5_nodes = result.top_k(5)
    false_at_5 = sum(1 for n in top5_nodes if n != true_root_cause) / max(len(top5_nodes), 1)

    return SingleRunMetrics(
        method=method_name if method_name is not None else result.method,
        anomaly_type=anomaly_type,
        pipeline_size=pipeline_size,
        trial=trial,
        true_root_cause=true_root_cause,
        target_node=result.target_node,
        rank=rank,
        top1=top1,
        top3=top3,
        top5=top5,
        reciprocal_rank=rr,
        false_rc_rate_k5=false_at_5,
        elapsed_ms=result.elapsed_ms,
        convergence_iters=result.convergence_iters,
    )


class ExperimentResults:
    """
    Aggregates results across all experiment runs.
    Supports per-method, per-anomaly-type, per-pipeline-size breakdowns.
    """

    def __init__(self):
        self.records: List[SingleRunMetrics] = []

    def add(self, metrics: SingleRunMetrics):
        self.records.append(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.records])

    def summary(self, group_by: List[str] = ["method"]) -> pd.DataFrame:
        """Aggregate metrics grouped by specified columns."""
        df = self.to_dataframe()
        if df.empty:
            return df

        agg = df.groupby(group_by).agg(
            top1_acc=("top1", "mean"),
            top3_acc=("top3", "mean"),
            top5_acc=("top5", "mean"),
            mrr=("reciprocal_rank", "mean"),
            false_rc_rate=("false_rc_rate_k5", "mean"),
            avg_elapsed_ms=("elapsed_ms", "mean"),
            n_runs=("top1", "count"),
        ).reset_index()

        # Round for display
        for col in ["top1_acc", "top3_acc", "top5_acc", "mrr", "false_rc_rate"]:
            agg[col] = agg[col].round(4)
        agg["avg_elapsed_ms"] = agg["avg_elapsed_ms"].round(2)

        return agg

    def main_results_table(self) -> pd.DataFrame:
        """Primary results table: method × metric (Table 6.1 in thesis)."""
        return self.summary(["method"])

    def anomaly_type_breakdown(self) -> pd.DataFrame:
        """Results per anomaly type (Table 6.2 in thesis)."""
        return self.summary(["method", "anomaly_type"])

    def pipeline_size_breakdown(self) -> pd.DataFrame:
        """Scalability analysis (Table 6.4 in thesis)."""
        return self.summary(["method", "pipeline_size"])

    def ablation_table(self) -> pd.DataFrame:
        """For ablation study runs (Table 6.3 in thesis)."""
        return self.summary(["method"])

    def summary_with_ci(
        self,
        group_by: List[str] = ["method"],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """
        Bootstrap confidence intervals over trial-level results.
        Resamples existing trials (with replacement) n_bootstrap times.
        Returns mean ± CI for Top-1, Top-5, MRR per group.
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        alpha_lo = (1 - ci) / 2 * 100
        alpha_hi = (1 + ci) / 2 * 100
        rng = np.random.default_rng(42)

        rows = []
        for keys, group in df.groupby(group_by):
            top1_vals = group["top1"].values
            top5_vals = group["top5"].values
            mrr_vals  = group["reciprocal_rank"].values
            n = len(top1_vals)

            # Bootstrap resample
            idx = rng.integers(0, n, size=(n_bootstrap, n))
            bs_top1 = top1_vals[idx].mean(axis=1)
            bs_top5 = top5_vals[idx].mean(axis=1)
            bs_mrr  = mrr_vals[idx].mean(axis=1)

            key_dict = dict(zip(group_by, keys if isinstance(keys, tuple) else [keys]))
            rows.append({
                **key_dict,
                "top1_mean":    round(float(np.mean(top1_vals)), 4),
                "top1_ci_lo":   round(float(np.percentile(bs_top1, alpha_lo)), 4),
                "top1_ci_hi":   round(float(np.percentile(bs_top1, alpha_hi)), 4),
                "top5_mean":    round(float(np.mean(top5_vals)), 4),
                "top5_ci_lo":   round(float(np.percentile(bs_top5, alpha_lo)), 4),
                "top5_ci_hi":   round(float(np.percentile(bs_top5, alpha_hi)), 4),
                "mrr_mean":     round(float(np.mean(mrr_vals)), 4),
                "mrr_ci_lo":    round(float(np.percentile(bs_mrr, alpha_lo)), 4),
                "mrr_ci_hi":    round(float(np.percentile(bs_mrr, alpha_hi)), 4),
                "n_runs":       n,
            })

        return pd.DataFrame(rows)

    def save(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} records to {path}")

    def load(self, path: str):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            self.records.append(SingleRunMetrics(**row.to_dict()))


def trace_completeness(
    frlg_builder,
    true_root_cause: str,
    target_node: str,
) -> float:
    """
    Compute trace completeness:
    fraction of true propagation path (RC → target) captured in FRLG.
    Returns 1.0 if any path exists, 0.0 if no path found.
    (For synthetic pipeline, should always be 1.0.)
    """
    paths = frlg_builder.get_propagation_paths(true_root_cause, target_node)
    return 1.0 if len(paths) > 0 else 0.0
