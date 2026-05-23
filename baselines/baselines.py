"""
Baseline RCA methods for comparison with APA-RCA.

B1: Manual BFS  — backward BFS from target, ranked by hop distance
B2: Vanilla RWR — graph structure only, no anomaly scores
B3: Anomaly Score Only — ignore graph, rank by anomaly score

All baselines share the same RCAResult interface.
"""

import numpy as np
import networkx as nx
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from rca.apa_rca import RCAResult, APARCAEngine


class BFSBaseline:
    """
    B1: Manual BFS (backward traversal from target node).
    Nodes are ranked by BFS hop distance (closer = more suspicious).
    Ties broken by anomaly score.
    """

    def run(
        self,
        graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        target_node: str,
    ) -> RCAResult:
        t0 = time.time()

        G_T = graph.reverse(copy=True)  # transpose for backward traversal
        nodes = list(graph.nodes())

        # BFS from target on transposed graph
        visited: Dict[str, int] = {}  # node -> hop distance
        queue = deque([(target_node, 0)])
        visited[target_node] = 0

        while queue:
            node, dist = queue.popleft()
            for neighbor in G_T.successors(node):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        # Nodes not reached get max hop + 1
        max_hop = max(visited.values(), default=0) + 1
        all_hops = {n: visited.get(n, max_hop) for n in nodes}

        # Rank: lower hop = more suspicious, ties broken by anomaly score (desc)
        # Exclude target node itself (it's the symptom, not the root cause)
        scored = [
            (n, (-all_hops[n], anomaly_scores.get(n, 0.0)))
            for n in nodes if n != target_node
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.time() - t0) * 1000

        # Convert to RCAResult format (use negative hop as pseudo-score)
        max_h = max(all_hops.values(), default=1)
        ranked = [
            (n, 1.0 - all_hops[n] / (max_h + 1))
            for n, _ in scored
            if n != target_node
        ]

        return RCAResult(
            ranked_nodes=ranked,
            convergence_iters=max_hop,
            elapsed_ms=elapsed_ms,
            target_node=target_node,
            method="BFS",
        )


class VanillaRWRBaseline:
    """
    B2: Vanilla Random Walk with Restart.
    Uses graph structure only — no anomaly scores, no transform weights, no attenuation.
    """

    def __init__(self, gamma: float = 0.15, max_iter: int = 1000, tol: float = 1e-8):
        self.engine = APARCAEngine(
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
            use_anomaly_scores=False,     # uniform scores
            use_transform_weights=False,  # uniform edge weights
            use_attenuation=False,        # no hop decay
        )

    def run(
        self,
        graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        target_node: str,
    ) -> RCAResult:
        result = self.engine.run(graph, anomaly_scores, target_node)
        result.method = "Vanilla-RWR"
        return result


class AnomalyScoreOnlyBaseline:
    """
    B3: Anomaly Score Only.
    Ignores graph structure; ranks nodes purely by their anomaly score.
    Demonstrates what happens without lineage information.
    """

    def run(
        self,
        graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        target_node: str,
    ) -> RCAResult:
        t0 = time.time()

        nodes = list(graph.nodes())
        # Exclude target node (symptom, not root cause)
        scored = [(n, anomaly_scores.get(n, 0.0)) for n in nodes if n != target_node]
        scored.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.time() - t0) * 1000

        return RCAResult(
            ranked_nodes=scored,
            convergence_iters=1,
            elapsed_ms=elapsed_ms,
            target_node=target_node,
            method="AnomalyScore-Only",
        )


class StructuralRWRBaseline:
    """
    Literature-style RWR (MicroRCA/CloudRanger style):
    - Transition matrix: uniform column-normalized (no anomaly weighting)
    - Restart vector: proportional to anomaly scores (anomaly-biased restart)
    - No hop attenuation, no transform-type weights

    Isolates the contribution of anomaly-weighted TRANSITIONS (APA-RCA novelty)
    vs. anomaly-biased RESTART (prior literature approach).
    """

    def __init__(self, gamma: float = 0.15, max_iter: int = 1000, tol: float = 1e-8):
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def run(
        self,
        graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        target_node: str,
    ) -> RCAResult:
        t0 = time.time()

        nodes = list(graph.nodes())
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}

        # Backward walk: transpose graph
        G_T = graph.reverse(copy=True)

        # Transition matrix: uniform column-normalized adjacency
        W = np.zeros((n, n))
        for v in nodes:
            neighbors = list(G_T.successors(v))
            for u in neighbors:
                W[idx[v], idx[u]] = 1.0
        col_sums = W.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        P = W / col_sums

        # Anomaly-biased restart vector (proportional to anomaly scores)
        scores_arr = np.array([max(anomaly_scores.get(v, 0.0), 0.0) for v in nodes])
        total = scores_arr.sum()
        e = scores_arr / total if total > 0 else np.ones(n) / n

        # RWR: r^(k+1) = (1-γ)·P·r^(k) + γ·e
        r = e.copy()
        iters = 0
        for iters in range(self.max_iter):
            r_new = (1 - self.gamma) * (P @ r) + self.gamma * e
            if np.linalg.norm(r_new - r, 1) < self.tol:
                break
            r = r_new

        elapsed_ms = (time.time() - t0) * 1000

        ranked = sorted(
            [(nodes[i], float(r[i])) for i in range(n) if nodes[i] != target_node],
            key=lambda x: x[1], reverse=True,
        )

        return RCAResult(
            ranked_nodes=ranked,
            convergence_iters=iters,
            elapsed_ms=elapsed_ms,
            target_node=target_node,
            method="Structural-RWR",
        )


class PCRWRBaseline:
    """
    CloudRanger-style: PC algorithm causal discovery + RWR.
    - PC algorithm on node time series to discover causal graph
    - Backward RWR on discovered graph with anomaly-biased restart

    Represents approaches that automatically infer causal structure from data
    rather than using domain-knowledge FRLG.
    The pipeline object must be set via set_pipeline() before calling run().
    """

    def __init__(self, gamma: float = 0.15, pc_alpha: float = 0.05,
                 max_cond_vars: int = 3, max_iter: int = 1000, tol: float = 1e-8):
        self.gamma = gamma
        self.pc_alpha = pc_alpha
        self.max_cond_vars = max_cond_vars
        self.max_iter = max_iter
        self.tol = tol
        self._pipeline = None

    def set_pipeline(self, pipeline) -> "PCRWRBaseline":
        """Attach pipeline for time series extraction. Returns self for chaining."""
        self._pipeline = pipeline
        return self

    def _build_timeseries_matrix(self, nodes: List[str]) -> np.ndarray:
        """
        Build T×N matrix from pipeline node data.
        Each column = log-diff of one node's primary time series.
        Log-diff improves stationarity for Fisher-Z independence tests.
        """
        nd = self._pipeline.node_data
        series = []
        for node in nodes:
            if node in nd and nd[node]:
                raw = list(nd[node].values())[0]
                arr = np.array(raw, dtype=float)
                # log-diff for stationarity; clip to avoid log(0)
                arr = np.clip(np.abs(arr), 1e-8, None)
                arr = np.diff(np.log(arr))
                arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            else:
                T = 251
                arr = np.zeros(T)
            series.append(arr)

        # Align lengths (take minimum)
        T = min(len(s) for s in series)
        return np.column_stack([s[-T:] for s in series])  # T × N

    def _pc_graph_to_nx(self, cg_graph, nodes: List[str], ref_graph: nx.DiGraph) -> nx.DiGraph:
        """
        Convert causal-learn CPDAG to NetworkX DiGraph.
        Encoding: graph[i][j] == -1 and graph[j][i] == 1  →  i → j
                  graph[i][j] == -1 and graph[j][i] == -1 →  i -- j (undirected)
        Undirected edges oriented by layer order (upstream → downstream).
        """
        adj = cg_graph.graph  # (N, N) numpy array
        n = len(nodes)
        G_pc = nx.DiGraph()
        G_pc.add_nodes_from(nodes)

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == 0 and adj[j][i] == 0:
                    continue
                # Directed i → j
                if adj[i][j] == -1 and adj[j][i] == 1:
                    G_pc.add_edge(nodes[i], nodes[j])
                # Directed j → i
                elif adj[i][j] == 1 and adj[j][i] == -1:
                    G_pc.add_edge(nodes[j], nodes[i])
                # Undirected i -- j: orient by layer
                else:
                    li = ref_graph.nodes[nodes[i]].get("layer", 0)
                    lj = ref_graph.nodes[nodes[j]].get("layer", 0)
                    if li < lj:
                        G_pc.add_edge(nodes[i], nodes[j])
                    elif lj < li:
                        G_pc.add_edge(nodes[j], nodes[i])
                    # same layer → skip

        return G_pc

    def run(
        self,
        graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        target_node: str,
    ) -> RCAResult:
        t0 = time.time()

        nodes = list(graph.nodes())
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}

        # Step 1: Extract and run PC (skip if no pipeline attached)
        G_pc = graph  # fallback to known graph
        if self._pipeline is not None:
            try:
                from causallearn.search.ConstraintBased.PC import pc as run_pc
                X = self._build_timeseries_matrix(nodes)
                cg = run_pc(
                    X,
                    alpha=self.pc_alpha,
                    indep_test="fisherz",
                    depth=self.max_cond_vars,
                    show_progress=False,
                    verbose=False,
                )
                G_pc = self._pc_graph_to_nx(cg.G, nodes, graph)
                if G_pc.number_of_edges() == 0:
                    G_pc = graph  # fallback: empty graph → use FRLG
            except Exception:
                G_pc = graph  # fallback on any error

        # Step 2: RWR on PC-discovered graph (backward walk)
        G_T = G_pc.reverse(copy=True)

        W = np.zeros((n, n))
        for v in nodes:
            for u in G_T.successors(v):
                if u in idx:
                    W[idx[v], idx[u]] = 1.0
        col_sums = W.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        P = W / col_sums

        # Anomaly-biased restart
        scores_arr = np.array([max(anomaly_scores.get(v, 0.0), 0.0) for v in nodes])
        total = scores_arr.sum()
        e = scores_arr / total if total > 0 else np.ones(n) / n

        r = e.copy()
        iters = 0
        for iters in range(self.max_iter):
            r_new = (1 - self.gamma) * (P @ r) + self.gamma * e
            if np.linalg.norm(r_new - r, 1) < self.tol:
                break
            r = r_new

        elapsed_ms = (time.time() - t0) * 1000

        ranked = sorted(
            [(nodes[i], float(r[i])) for i in range(n) if nodes[i] != target_node],
            key=lambda x: x[1], reverse=True,
        )

        return RCAResult(
            ranked_nodes=ranked,
            convergence_iters=iters,
            elapsed_ms=elapsed_ms,
            target_node=target_node,
            method="PC+RWR",
        )


def select_target_node(
    graph: nx.DiGraph,
    anomaly_scores: Dict[str, float],
    threshold: float = 0.1,
) -> str:
    """
    Select the target (observed anomaly) node for RCA.

    Strategy (v2): Find the highest-layer (report > risk > feature) node that
    is reachable downstream from the most anomalous upstream nodes.
    This ensures the target is downstream of the true root cause regardless
    of whether the target itself has a high anomaly score (e.g., A3 where
    the feature node anomaly may not propagate strongly to all report nodes).

    Concretely:
      1. Identify top-k anomalous nodes (k=5, score >= threshold).
      2. For each candidate, find the highest-layer descendant.
      3. Pick the descendant that covers the most anomalous ancestors.
    Fallback: original report-score-based selection.
    """

    all_nodes = list(graph.nodes())

    # Step 1: top-k anomalous nodes above threshold.
    # Include both overall top-5 AND top-5 non-source anomalous nodes so that
    # feature/ETL-layer root causes (A2, A3) can contribute coverage even when
    # noisy source nodes dominate the overall top-5.
    anomalous_all = sorted(
        [(n, anomaly_scores.get(n, 0.0)) for n in all_nodes
         if anomaly_scores.get(n, 0.0) >= threshold],
        key=lambda x: x[1], reverse=True,
    )[:5]
    anomalous_nonsrc = sorted(
        [(n, anomaly_scores.get(n, 0.0)) for n in all_nodes
         if anomaly_scores.get(n, 0.0) >= threshold
         and graph.nodes[n].get("layer", 1) > 1],
        key=lambda x: x[1], reverse=True,
    )[:5]
    # Union: non-source entries may add new seeds; keep highest score per node
    seed_dict: Dict[str, float] = {n: s for n, s in anomalous_all}
    for n, s in anomalous_nonsrc:
        if s > seed_dict.get(n, 0.0):
            seed_dict[n] = s
    anomalous = list(seed_dict.items())

    # Step 2 & 3: for each anomalous source node, collect its downstream
    # report/risk nodes and accumulate their anomaly-weighted coverage.
    # Key insight: a target is "good" if it is downstream of many high-scoring
    # anomalous nodes — meaning those nodes can actually explain the target's anomaly.
    # We track per-target the SUM of anomaly scores of its TOP-k anomalous ancestors
    # that are themselves in the anomalous set (i.e., we only credit ancestors
    # that are actually anomalous, not all ancestors of target).
    coverage: Dict[str, float] = {}   # target -> sum of anomalous ancestor scores
    layer_of: Dict[str, int]   = {}

    for src, src_score in anomalous:
        desc = nx.descendants(graph, src)
        for dn in desc:
            layer = graph.nodes[dn].get("layer", 0)
            if layer < 4:
                continue
            coverage[dn] = coverage.get(dn, 0.0) + src_score
            layer_of[dn] = layer

    if coverage:
        # Score: strong preference for report layer (5 > 4), then highest coverage
        best = max(
            coverage,
            key=lambda dn: layer_of.get(dn, 0) * 10.0 + coverage[dn],
        )
        return best

    # Fallback: report node with highest (own_score + anomalous_ancestors)
    report_nodes = [n for n in all_nodes if graph.nodes[n].get("layer") == 5]
    best_target = None
    best_composite = -1.0
    for rn in report_nodes:
        r_score = anomaly_scores.get(rn, 0.0)
        ancestors = nx.ancestors(graph, rn)
        n_anc = sum(1 for a in ancestors if anomaly_scores.get(a, 0.0) >= threshold)
        composite = r_score + 0.1 * n_anc
        if composite > best_composite:
            best_composite = composite
            best_target = rn

    if best_target is not None and anomaly_scores.get(best_target, 0.0) >= threshold:
        return best_target

    # Fallback: highest-scored non-source node
    non_source = [n for n in all_nodes if graph.nodes[n].get("layer", 1) > 1]
    if non_source:
        return max(non_source, key=lambda n: anomaly_scores.get(n, 0.0))

    return max(anomaly_scores, key=anomaly_scores.get, default=all_nodes[0])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from synfrp import (MarketDataConfig, MarketDataGenerator,
                        FinancialRiskPipeline, AnomalyInjector, AnomalyType)
    from frlg import FRLGBuilder, NodeAnomalyDetector
    from rca import APARCAEngine

    cfg = MarketDataConfig(seed=42)
    raw = MarketDataGenerator(cfg).generate_all()
    pipeline = FinancialRiskPipeline(raw)

    injector = AnomalyInjector(pipeline)
    injector, gt = injector.inject(AnomalyType.A1_SOURCE_CORRUPTION, seed=0, magnitude=8.0)
    pipeline.execute()

    builder  = FRLGBuilder(pipeline)
    G        = builder.build()
    detector = NodeAnomalyDetector()
    scores   = detector.detect_all(pipeline)

    target = select_target_node(G, scores)
    print(f"Target: {target}, True RC: {gt.target_node_id}")

    methods = {
        "BFS":            BFSBaseline(),
        "Vanilla-RWR":    VanillaRWRBaseline(),
        "AnomalyOnly":    AnomalyScoreOnlyBaseline(),
        "APA-RCA":        APARCAEngine(),
    }

    print(f"\n{'Method':<20} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Rank':>6}")
    print("-" * 50)
    for name, method in methods.items():
        result = method.run(G, scores, target)
        rank = result.rank_of(gt.target_node_id)
        top1 = int(rank == 1) if rank else 0
        top3 = int(rank is not None and rank <= 3)
        top5 = int(rank is not None and rank <= 5)
        print(f"{name:<20} {top1:>6} {top3:>6} {top5:>6} {str(rank):>6}")
