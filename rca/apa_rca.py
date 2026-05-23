"""
APA-RCA: Anomaly-Propagation-Aware Root Cause Analysis
Core algorithm: Random Walk with Restart on transposed FRLG,
with anomaly-score-weighted transitions and transform-type-weighted edges.

Algorithm (from thesis specification):
  1. Transpose FRLG: G^T (reverse edge directions for backward propagation)
  2. Compute edge weights: w(u,v) = a(u) × β(edge_type) × α^{hop}
  3. Normalize to transition matrix P
  4. RWR from target node t: r^(k+1) = (1-γ)·P·r^(k) + γ·e_t
  5. Converge and rank all nodes by r
  6. Return Top-K root cause candidates
"""

import numpy as np
import networkx as nx
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# Transform type weight priors (β) — hand-tuned (v1/v2 legacy)
TRANSFORM_WEIGHTS = {
    "aggregate":  1.5,
    "calculate":  1.2,
    "report":     1.3,
    "direct_map": 1.0,
    "filter":     0.8,
    "source":     1.0,
}

# ATF-derived transfer coefficients (ρ_τ) — principled β (v3)
#
# Each ρ_τ is the z-score transfer ratio: how much of the input anomaly
# z-score survives the transform. Derived from closed-form analysis of
# each SynFRP transform (see paper §4, ATF Framework).
#
#   source:     identity → ρ = 1.0
#   direct_map: identity / scalar multiply → ρ = 1.0
#   filter:     clip at c std → ρ = min(1, c/z) ≈ 0.8 (avg for 4σ clip)
#   calculate:  log-return / multiplication → ρ ≈ 1.0
#   aggregate:  mean of k inputs → ρ = 1/√k ≈ 0.45 (k=5 medium pipeline)
#   report:     binary threshold → ρ ≈ 0 (discontinuous, no z-score info)
ATF_TRANSFER_COEFFICIENTS = {
    "source":     1.0,
    "direct_map": 1.0,
    "filter":     0.8,
    "calculate":  1.0,
    "aggregate":  0.45,
    "report":     0.1,   # near-zero but non-zero to avoid dead edges
}


@dataclass
class RCAResult:
    """Output of the APA-RCA algorithm."""
    ranked_nodes: List[Tuple[str, float]]   # [(node_id, score), ...] sorted desc
    convergence_iters: int
    elapsed_ms: float
    target_node: str
    method: str = "APA-RCA"

    def top_k(self, k: int = 5) -> List[str]:
        return [nid for nid, _ in self.ranked_nodes[:k]]

    def rank_of(self, node_id: str) -> Optional[int]:
        """Return 1-based rank of a node (None if not in list)."""
        for i, (nid, _) in enumerate(self.ranked_nodes):
            if nid == node_id:
                return i + 1
        return None


class APARCAEngine:
    """
    Anomaly-Propagation-Aware RCA Engine.

    Parameters
    ----------
    gamma : float
        Restart probability (default 0.15).
    alpha : float
        Anomaly attenuation factor per hop (default 0.7).
    max_iter : int
        Maximum RWR iterations.
    tol : float
        Convergence tolerance.
    use_anomaly_scores : bool
        If False, uniform scores (ablation: remove anomaly component).
    use_transform_weights : bool
        If False, uniform edge weights (ablation: remove β).
    use_attenuation : bool
        If False, no hop-distance decay (ablation: remove α).
    adaptive_weighting : bool
        If True (v2), use hop-distance-adaptive graph/score weighting and
        score-inversed β. If False (v1), use fixed 0.4/0.6 split.
    use_score_inv_beta : bool
        If True (v2 default), apply score-inversed type bonus. Ablation flag.
    use_ancestor_constraint : bool
        If True (default), non-ancestor nodes get 0.05× penalty. Ablation flag.
    use_atf_weights : bool
        If True (v3), use ATF-derived ρ_τ as β instead of hand-tuned TRANSFORM_WEIGHTS.
    """

    def __init__(
        self,
        gamma: float = 0.15,
        alpha: float = 0.7,
        max_iter: int = 1000,
        tol: float = 1e-8,
        use_anomaly_scores: bool = True,
        use_transform_weights: bool = True,
        use_attenuation: bool = True,
        adaptive_weighting: bool = False,
        use_score_inv_beta: bool = True,
        use_ancestor_constraint: bool = True,
        use_atf_weights: bool = False,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.use_anomaly_scores      = use_anomaly_scores
        self.use_transform_weights   = use_transform_weights
        self.use_attenuation         = use_attenuation
        self.adaptive_weighting      = adaptive_weighting
        self.use_score_inv_beta      = use_score_inv_beta
        self.use_ancestor_constraint = use_ancestor_constraint
        self.use_atf_weights         = use_atf_weights

    def run(
        self,
        graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        target_node: str,
    ) -> RCAResult:
        """
        Run APA-RCA from target_node backward through the FRLG.

        Parameters
        ----------
        graph : nx.DiGraph
            The FRLG (directed: upstream → downstream).
        anomaly_scores : Dict[str, float]
            Per-node anomaly scores in [0, 1].
        target_node : str
            The node where anomaly was observed (starting point for backward walk).

        Returns
        -------
        RCAResult with ranked root cause candidates.
        """
        t0 = time.time()

        # Step 1: Transpose graph (reverse edges for backward propagation)
        G_T = graph.reverse(copy=True)

        nodes = list(G_T.nodes())
        n     = len(nodes)
        idx   = {nid: i for i, nid in enumerate(nodes)}

        if target_node not in idx:
            raise ValueError(f"Target node '{target_node}' not in graph.")

        # Step 2: Compute hop distances from target in transposed graph.
        # G_T edge: downstream→upstream, so hop_distances[v] = upstream distance from target.
        #
        # Multi-seed expansion: if highly anomalous nodes (score > 0.4) in L4-L5 are NOT
        # reachable from target_node in G_T, add them as auxiliary seeds. This handles cases
        # where the observed target is a report node whose lineage path does not pass through
        # the actual root cause (e.g., A3: REPORT_LIMIT does not use VOL20 features directly).
        # The hop distance for auxiliary seeds is offset by +1 so target_node remains primary.
        hop_distances: Dict[str, int] = {}
        try:
            lengths = nx.single_source_shortest_path_length(G_T, target_node)
            hop_distances = dict(lengths)
        except Exception:
            hop_distances = {}

        # Also compute maximum reachable hop for normalization
        max_hop = max(hop_distances.values(), default=1)

        # Step 3: Build anomaly-weighted transition matrix
        # w(u, v) = a(u) × β(edge_type(u,v)) × α^hop(u)
        W = np.zeros((n, n))

        for u, v, data in G_T.edges(data=True):
            # In G_T: u is downstream (original), v is upstream (original)
            # We want to walk from target toward root causes (upstream)
            ui = idx[u]
            vi = idx[v]

            # Anomaly score of destination node v (the upstream candidate)
            # Higher score = more likely to be a root cause → stronger pull
            a_v = anomaly_scores.get(v, 0.0) if self.use_anomaly_scores else 1.0
            a_v = max(a_v, 1e-4)   # avoid zero-weight dead ends

            # Transform weight β for transition matrix
            # v3 (use_atf_weights): ATF-derived ρ_τ as β in transitions
            # v2 (adaptive_weighting): fixed β removed from transitions; βeff applied at scoring
            # v1: hand-tuned β used in transitions
            edge_type = data.get("edge_type", "direct_map")
            if self.use_atf_weights:
                beta = ATF_TRANSFER_COEFFICIENTS.get(edge_type, 1.0) if self.use_transform_weights else 1.0
            elif self.adaptive_weighting:
                beta = 1.0  # no fixed prior in v2 transitions; score-inv βeff applied at scoring
            else:
                beta = TRANSFORM_WEIGHTS.get(edge_type, 1.0) if self.use_transform_weights else 1.0

            # Attenuation: mild decay per hop so distant nodes can still receive score
            # We use a soft attenuation so deep source nodes aren't completely zeroed
            hop = hop_distances.get(v, 0)  # hop of the destination (upstream)
            if self.use_attenuation:
                att = self.alpha ** (hop / max(max_hop, 1))  # normalized: never below alpha^1
            else:
                att = 1.0

            W[vi, ui] = a_v * beta * att   # row=destination(upstream), col=source(downstream)

        # Step 4: Column-normalize W → transition matrix P
        col_sums = W.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        P = W / col_sums

        # Step 5: RWR — r^(k+1) = (1-γ)·P·r^(k) + γ·e_t
        # e_t: restart distribution. Primary weight at target_node.
        # Secondary restart: share 0.3 weight proportionally among highly anomalous
        # non-target nodes (score > 0.3) — helps guide walk toward actual root causes.
        e_t = np.zeros(n)
        e_t[idx[target_node]] = 0.7   # primary restart at target

        if self.use_anomaly_scores:
            other_anomalous = {nid: sc for nid, sc in anomaly_scores.items()
                               if nid != target_node and nid in idx and sc > 0.3}
            if other_anomalous:
                total_other = sum(other_anomalous.values())
                for nid, sc in other_anomalous.items():
                    e_t[idx[nid]] += 0.3 * (sc / total_other)

        e_t = e_t / (e_t.sum() + 1e-12)  # normalize

        r = np.ones(n) / n   # uniform init
        converged = False
        iters     = 0

        for iters in range(1, self.max_iter + 1):
            r_new = (1 - self.gamma) * P @ r + self.gamma * e_t
            delta = np.linalg.norm(r_new - r, ord=1)
            r = r_new
            if delta < self.tol:
                converged = True
                break

        elapsed_ms = (time.time() - t0) * 1000

        # Step 6: Rank all nodes
        # APA-RCA composite score combines:
        #   1. RWR stationary score (graph propagation signal)
        #   2. Anomaly score (direct detection evidence)
        #   3. Ancestor constraint (only ancestors of target can be root causes)
        #   4. Upstream preference (source-layer nodes preferred as root cause)
        #
        # This is the key novelty vs AnomalyOnly: when anomaly scores are tied
        # (common in propagation scenarios), graph structure breaks ties by
        # preferring nodes that are (a) ancestors of target and (b) more upstream.
        ancestors = nx.ancestors(graph, target_node)

        scored = []
        for i in range(n):
            nid = nodes[i]
            if nid == target_node:
                continue
            rwr_score = float(r[i])

            # Direct anomaly score
            a_direct = anomaly_scores.get(nid, 0.0) if self.use_anomaly_scores else 0.5
            a_direct = max(a_direct, 0.01)

            # Ancestor constraint: non-ancestors get a heavy penalty
            # (they cannot be root causes of anomalies observed at target)
            if not self.use_ancestor_constraint:
                ancestor_factor = 1.0
            elif nid in ancestors:
                ancestor_factor = 1.0
            else:
                ancestor_factor = 0.05   # strong penalty: non-ancestors unlikely RC

            # Layer-based upstream preference
            node_layer = graph.nodes[nid].get("layer", 3)
            if self.use_attenuation:
                # Layer 1 (source) = highest upstream bonus
                layer_bonus = 1.0 + (5 - node_layer) * 0.15
            else:
                layer_bonus = 1.0

            # Transform-type weight: sources/ETL more likely as root cause origin
            # Note: ATF ρ_τ is used in transitions only (forward transfer).
            # Scoring type_bonus uses hand-tuned weights regardless of use_atf_weights,
            # because ρ_τ models "how much anomaly passes through" (forward),
            # not "how likely this node is a root cause" (backward).
            t_type = graph.nodes[nid].get("transform_type", "direct_map")
            type_bonus = TRANSFORM_WEIGHTS.get(t_type, 1.0) if self.use_transform_weights else 1.0

            # Final score
            # v2: adaptive graph/score weighting + score-inversed type_bonus
            #   - Nodes close to target: trust graph structure more (RWR already placed mass)
            #   - Nodes far from target: score signal is more reliable (distant propagation
            #     attenuates RWR mass, but high anomaly score = direct evidence)
            #   w_graph(v) = 0.3 + 0.4 × (1 - hop/max_hop)  — closer → trust graph more
            #   β_eff(v)   = β_base × (1 - 0.5×a(v))        — score-inversed type bonus
            # v1: fixed 0.4 graph / 0.6 score, unmodified type_bonus
            if self.adaptive_weighting:
                hop_v   = hop_distances.get(nid, 0)
                w_graph = 0.3 + 0.4 * (1.0 - hop_v / max(max_hop, 1))
                w_score = 1.0 - w_graph
                if self.use_score_inv_beta:
                    type_bonus_eff = max(type_bonus * (1.0 - 0.5 * a_direct), 0.1)
                else:
                    type_bonus_eff = type_bonus
                final = (w_graph * rwr_score + w_score * a_direct) * ancestor_factor * layer_bonus * type_bonus_eff
            else:
                final = (0.4 * rwr_score + 0.6 * a_direct) * ancestor_factor * layer_bonus * type_bonus
            scored.append((nid, final))

        scored.sort(key=lambda x: x[1], reverse=True)

        return RCAResult(
            ranked_nodes=scored,
            convergence_iters=iters,
            elapsed_ms=elapsed_ms,
            target_node=target_node,
            method="APA-RCA-v3" if self.use_atf_weights else ("APA-RCA-v2" if self.adaptive_weighting else "APA-RCA"),
        )

    def run_from_reports(
        self,
        graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        threshold: float = 0.3,
    ) -> Optional[RCAResult]:
        """
        Auto-select target node: report node with highest anomaly score.
        Used when no specific target is given.
        """
        report_nodes = [
            n for n in graph.nodes
            if graph.nodes[n].get("layer") == 5
        ]
        if not report_nodes:
            return None

        target = max(report_nodes, key=lambda n: anomaly_scores.get(n, 0.0))
        if anomaly_scores.get(target, 0.0) < threshold:
            # No clear anomaly in reports — use highest-scored node overall
            target = max(anomaly_scores, key=anomaly_scores.get)

        return self.run(graph, anomaly_scores, target)



if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from synfrp import (MarketDataConfig, MarketDataGenerator,
                        FinancialRiskPipeline, AnomalyInjector, AnomalyType)
    from frlg import FRLGBuilder, NodeAnomalyDetector

    cfg = MarketDataConfig(seed=42)
    raw = MarketDataGenerator(cfg).generate_all()
    pipeline = FinancialRiskPipeline(raw)

    injector = AnomalyInjector(pipeline)
    injector, gt = injector.inject(AnomalyType.A1_SOURCE_CORRUPTION, seed=0, magnitude=8.0)
    print(f"Injected anomaly at: {gt.target_node_id} (type: {gt.anomaly_type})")

    pipeline.execute()

    builder  = FRLGBuilder(pipeline)
    G        = builder.build()
    detector = NodeAnomalyDetector()
    scores   = detector.detect_all(pipeline)
    builder.set_anomaly_scores(scores)

    engine = APARCAEngine(gamma=0.15, alpha=0.7)
    # Use the most anomalous report node as target
    result = engine.run_from_reports(G, scores)

    print(f"\nTarget node: {result.target_node}")
    print(f"Converged in {result.convergence_iters} iterations ({result.elapsed_ms:.2f} ms)")
    print(f"\nTop-10 Root Cause Candidates:")
    for rank, (nid, score) in enumerate(result.ranked_nodes[:10], 1):
        marker = " ← TRUE ROOT CAUSE" if nid == gt.target_node_id else ""
        print(f"  #{rank}: {nid} (score={score:.6f}){marker}")

    rank = result.rank_of(gt.target_node_id)
    print(f"\nTrue root cause rank: {rank}")
