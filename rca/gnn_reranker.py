"""
ATF-Informed Residual GNN Re-ranker for APA-RCA top-K candidates.

After APA-RCA produces a ranked list, this module builds a local subgraph
around the top-K candidates and applies a 2-layer Residual GCN to produce
refined re-ranking scores that incorporate neighbourhood structure.

Architecture improvements over baseline GCN
-------------------------------------------
1. ATF-Weighted Adjacency:
   The standard GCN uses a binary adjacency (A[v,u] = 1 for all edges).
   This module encodes ATF transfer coefficients ρ_τ as edge weights:

       A[v, u] = ρ_τ(edge_type of u→v)

   where ρ_τ ∈ {1.0 (DirectMap/Source/Calculate), 0.8 (Filter),
                 0.45 (Aggregate), 0.1 (Report)}.

   Rationale: when doing backward traversal to find the root cause,
   the weight of the path from v back through u should reflect how
   much of u's anomaly would survive the transformation at v.
   A DirectMap edge (ρ=1.0) preserves the signal fully → high backward
   weight. An Aggregate edge (ρ=0.45) dilutes the signal across k inputs
   → lower backward weight. This directly encodes the ATF framework into
   the GNN's message-passing kernel.

2. Residual Skip Connection:
   Standard GCN layers apply graph diffusion A·W(x), which can over-smooth
   node features over multiple hops and erase the individual anomaly signal
   that makes root-cause nodes identifiable. We add a learned skip connection:

       h = ReLU(LayerNorm(A·W_conv(x) + W_skip(x)))

   W_skip(x) preserves each node's own feature vector without neighbourhood
   aggregation. This is critical because the true root-cause node has a
   distinctly high VAE anomaly score and is_ancestor flag; over-smoothing
   from pure GCN can dilute these signals.

   LayerNorm is applied after the residual sum for training stability
   (variable graph sizes make BatchNorm inappropriate).

Node features (F=8)
-------------------
    0  rwr_score_norm   APA-RCA RWR score / max(scores in subgraph)
    1  anomaly_score    VAE reconstruction-error anomaly score ∈ (0,1)
    2  layer_norm       layer index / 5.0
    3  is_ancestor      1 if node is ancestor of target in FRLG
    4  in_degree_norm   in-degree / max_in_degree
    5  out_degree_norm  out-degree / max_out_degree
    6  is_candidate     1 if node is in top-K candidate list
    7  path_dist_norm   backward hop-distance from target / (max_dist + 1)
                        (0 = target itself; higher = farther upstream.
                         Non-ancestors get distance max_dist + 1 → value = 1.0)

    Rationale for feature 7: root causes in FRLG are typically far upstream
    from the observed anomaly (high path_dist), while downstream symptoms
    are close.  Combined with is_ancestor (feature 3), this disambiguates
    true root causes (far + ancestor) from downstream noise (close + ancestor)
    and unrelated nodes (non-ancestor → penalised to 1.0).

Training
--------
    - Loss: weighted BCE (positive class = true root cause)
    - Optimizer: Adam + weight_decay
    - Gradient clipping for stability

Evaluation strategy
-------------------
    Synthetic:  5-fold cross-validation
    RSHB:       train on all 630 synthetic, test on 210 real-source trials
                (no fine-tuning on real data — cross-domain transfer)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ATF transfer coefficients — shared with APA-RCA engine
# ρ_τ: expected z-score transfer ratio through each transform type
# (see Chapter 4, ATF Framework)
ATF_EDGE_WEIGHTS: Dict[str, float] = {
    "source":     1.0,
    "direct_map": 1.0,
    "calculate":  1.0,
    "filter":     0.8,
    "aggregate":  0.45,
    "report":     0.1,
}

FEATURE_DIM = 8


# ── ATF-Informed Residual GCN ─────────────────────────────────────────────────

class _ResGCN(nn.Module):
    """
    2-layer Residual GCN with ATF-weighted adjacency support.

    Layer 1: h = ReLU(LayerNorm(A · W_conv(x) + W_skip(x)))
    Layer 2: out = Sigmoid(A · W_out(h))

    The skip connection W_skip(x) passes each node's own feature vector
    directly to the next layer without graph diffusion, preventing
    over-smoothing and preserving node-level anomaly signals.
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        # Layer 1: graph conv + skip
        self.W_conv  = nn.Linear(in_dim, hidden, bias=True)
        self.W_skip  = nn.Linear(in_dim, hidden, bias=True)
        self.norm1   = nn.LayerNorm(hidden)
        # Layer 2: output projection
        self.W_out   = nn.Linear(hidden, 1, bias=True)
        self.drop    = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x : (N, F)   node features
        A : (N, N)   ATF-weighted, symmetrically-normalised adjacency
        → (N, 1) scores in (0, 1)
        """
        # Layer 1: graph conv + residual skip + LayerNorm
        h_conv = self.drop(A @ self.W_conv(x))    # neighbourhood aggregation
        h_skip = self.W_skip(x)                    # node-level skip (no diffusion)
        h = self.relu(self.norm1(h_conv + h_skip)) # (N, hidden)

        # Layer 2: output
        return torch.sigmoid(A @ self.W_out(h))    # (N, 1)


# ── Sample dataclass ──────────────────────────────────────────────────────────

@dataclass
class GNNSample:
    """One (subgraph, label) pair collected from a single experiment trial."""
    features:      np.ndarray       # (N, F) float32
    adj:           np.ndarray       # (N, N) ATF-weighted, sym-norm, float32
    node_ids:      List[str]        # subgraph node order
    candidate_ids: List[str]        # top-K node ids from APA-RCA
    true_rc:       str              # ground-truth root cause
    # bookkeeping for CV & metrics
    anomaly_type:  str = ""
    pipeline_size: str = ""
    trial:         int = -1
    fold:          int = -1         # assigned during CV split


# ── ATF-weighted subgraph builder ─────────────────────────────────────────────

def build_gnn_sample(
    G:        nx.DiGraph,
    scores:   Dict[str, float],
    rca_result,                        # RCAResult from APA-RCA
    true_rc:  str,
    top_k:    int = 10,
    anomaly_type:  str = "",
    pipeline_size: str = "",
    trial:    int = -1,
) -> Optional[GNNSample]:
    """
    Build a GNNSample from one experiment trial.

    Key change vs baseline: the adjacency matrix A uses ATF transfer
    coefficients ρ_τ as edge weights rather than binary 1/0.  This
    encodes domain knowledge about anomaly propagation directly into
    the GNN's message-passing kernel.
    """

    # ── Candidate set ──────────────────────────────────────────────────────────
    candidates = [nid for nid, _ in rca_result.ranked_nodes[:top_k]]
    rwr_dict   = {nid: sc for nid, sc in rca_result.ranked_nodes[:top_k]}
    if not candidates:
        return None

    # ── 1-hop subgraph ─────────────────────────────────────────────────────────
    sub_nodes = set(candidates)
    for nid in candidates:
        if nid in G:
            sub_nodes.update(G.predecessors(nid))
            sub_nodes.update(G.successors(nid))
    sub_nodes = sorted(sub_nodes)          # deterministic order
    nidx = {nid: i for i, nid in enumerate(sub_nodes)}
    N    = len(sub_nodes)

    # ── Ancestor set + backward hop distances from target ──────────────────────
    #
    # We reverse G (giving G^T) and run single-source BFS from the target node.
    # This yields, for each ancestor of target, its hop-distance in G^T —
    # equivalently, the number of backward steps needed to reach it from target
    # in the original FRLG.  Root causes are typically far upstream (high dist).
    #
    target         = rca_result.target_node
    ancestors: set = set()
    dist_from_target: Dict[str, int] = {}

    if target and target in G:
        try:
            Gt = G.reverse(copy=False)
            dist_from_target = dict(
                nx.single_source_shortest_path_length(Gt, target)
            )
            ancestors = set(dist_from_target.keys()) - {target}
        except Exception:
            pass

    max_dist = max(dist_from_target.values(), default=1)
    # Non-ancestors receive distance = max_dist + 1 (penalised to norm value 1.0)
    penalty_dist = max_dist + 1

    # ── Global degree normalisation ────────────────────────────────────────────
    max_in  = max((d for _, d in G.in_degree()),  default=1) + 1
    max_out = max((d for _, d in G.out_degree()), default=1) + 1
    max_rwr = max(rwr_dict.values(), default=1e-8)

    # ── Feature matrix (F=8) ───────────────────────────────────────────────────
    feats = np.zeros((N, FEATURE_DIM), dtype=np.float32)
    for i, nid in enumerate(sub_nodes):
        nd   = G.nodes.get(nid, {})
        dist = dist_from_target.get(nid, penalty_dist)
        feats[i, 0] = rwr_dict.get(nid, 0.0) / max_rwr
        feats[i, 1] = float(scores.get(nid, 0.0))
        feats[i, 2] = nd.get("layer", 3) / 5.0
        feats[i, 3] = float(nid in ancestors)
        feats[i, 4] = (G.in_degree(nid)  if nid in G else 0) / max_in
        feats[i, 5] = (G.out_degree(nid) if nid in G else 0) / max_out
        feats[i, 6] = float(nid in set(candidates))
        feats[i, 7] = dist / penalty_dist   # 0 = target, 1 = max upstream / non-ancestor

    # ── ATF-weighted adjacency (transposed FRLG + self-loops) ─────────────────
    #
    # For each original FRLG edge u → v (upstream to downstream):
    #   - In the transposed graph used for backward traversal, this becomes v → u
    #   - We set A[v_idx, u_idx] = ρ_τ(edge_type), where edge_type is the
    #     transform type of node v (the target node of the original edge).
    #
    # Interpretation: when the GNN aggregates neighbourhood information at v,
    # the signal coming from upstream u is weighted by how well the transform
    # at v preserves anomaly signal (ρ_τ).  High ρ (DirectMap=1.0) → strong
    # backward attribution; low ρ (Aggregate=0.45) → weak attribution.
    #
    A = np.zeros((N, N), dtype=np.float32)
    for u, v in G.edges():
        if u in nidx and v in nidx:
            edge_attrs = G.edges[u, v]
            edge_type  = edge_attrs.get("edge_type", "direct_map")
            rho        = ATF_EDGE_WEIGHTS.get(edge_type, 1.0)
            A[nidx[v], nidx[u]] = rho   # transposed: v ← u with weight ρ_τ

    # Self-loops (weight 1.0 — node always attends to itself fully)
    np.fill_diagonal(A, 1.0)

    # Symmetric normalisation: D^{-1/2} · A · D^{-1/2}
    # Degree is now ρ-weighted sum of incoming edges (+ 1 self-loop)
    deg        = A.sum(axis=1, keepdims=True).clip(min=1.0)
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    A_norm     = D_inv_sqrt * A * D_inv_sqrt.T   # (N, N)

    return GNNSample(
        features=feats, adj=A_norm,
        node_ids=sub_nodes, candidate_ids=candidates,
        true_rc=true_rc,
        anomaly_type=anomaly_type, pipeline_size=pipeline_size, trial=trial,
    )


# ── Re-ranker ─────────────────────────────────────────────────────────────────

class GNNReranker:
    """
    Trains a 2-layer Residual GCN with ATF-weighted adjacency to re-rank
    APA-RCA top-K candidates.

    Architectural improvements over baseline GCN
    --------------------------------------------
    1. ATF-Weighted Adjacency: edge weights encode ρ_τ transfer coefficients
       derived from the Anomaly Transfer Function framework (Chapter 4).
       The GNN's message-passing kernel now reflects domain knowledge about
       how anomaly signals propagate through each transform type.

    2. Residual Skip Connection: each GCN layer adds a learned projection
       of the original node features (W_skip · x) to the graph-diffused
       output (A · W_conv · x), preventing over-smoothing and preserving
       node-level anomaly signals (VAE score, is_ancestor) that are
       diagnostically critical for root-cause identification.

    Usage
    -----
    reranker = GNNReranker()
    reranker.fit(training_samples)
    new_result = reranker.rerank(sample, rca_result)
    """

    def __init__(
        self,
        top_k:            int   = 10,
        hidden_dim:       int   = 32,
        dropout:          float = 0.3,
        epochs:           int   = 150,
        lr:               float = 5e-3,
        weight_decay:     float = 1e-4,
        pos_weight_scale: float = 3.0,
    ):
        self.top_k             = top_k
        self.hidden_dim        = hidden_dim
        self.dropout           = dropout
        self.epochs            = epochs
        self.lr                = lr
        self.weight_decay      = weight_decay
        self.pos_weight_scale  = pos_weight_scale
        self.device            = torch.device("cpu")
        self.model: Optional[_ResGCN] = None

    # ── training ───────────────────────────────────────────────────────────────

    def fit(self, samples: List[GNNSample], verbose: bool = False) -> "GNNReranker":
        """
        Train the Residual GCN on collected (subgraph, label) samples.

        Only samples where true_rc appears in the subgraph contribute a
        positive supervision signal; the rest still contribute negatives.
        """
        valid = [s for s in samples if s.true_rc in s.node_ids]
        if not valid:
            if verbose:
                print("  [ResGCN] No valid training samples — model not updated.")
            return self

        self.model = _ResGCN(FEATURE_DIM, self.hidden_dim, self.dropout).to(self.device)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in range(self.epochs):
            rng      = np.random.default_rng(epoch)
            shuffled = [valid[i] for i in rng.permutation(len(valid))]
            total_loss = 0.0

            for s in shuffled:
                x = torch.tensor(s.features, dtype=torch.float32, device=self.device)
                A = torch.tensor(s.adj,      dtype=torch.float32, device=self.device)
                y = torch.tensor(
                    [1.0 if nid == s.true_rc else 0.0 for nid in s.node_ids],
                    dtype=torch.float32, device=self.device,
                )

                opt.zero_grad()
                pred = self.model(x, A).squeeze(-1)        # (N,)

                # Weighted BCE: up-weight the single positive (true root cause)
                w    = 1.0 + y * (self.pos_weight_scale - 1.0)
                bce  = nn.functional.binary_cross_entropy(pred, y, weight=w)
                bce.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += bce.item()

            if verbose and (epoch + 1) % 30 == 0:
                print(f"    epoch {epoch+1}/{self.epochs}  loss={total_loss/len(shuffled):.4f}")

        return self

    # ── inference ──────────────────────────────────────────────────────────────

    def rerank(self, sample: GNNSample, rca_result) -> object:
        """
        Apply trained ResGCN to re-rank candidates in rca_result.
        Returns a new RCAResult with GNN-adjusted ranking.
        """
        if self.model is None:
            return rca_result

        self.model.eval()
        with torch.no_grad():
            x   = torch.tensor(sample.features, dtype=torch.float32, device=self.device)
            A   = torch.tensor(sample.adj,      dtype=torch.float32, device=self.device)
            gnn = self.model(x, A).squeeze(-1).cpu().numpy()   # (N,)

        gnn_dict = {sample.node_ids[i]: float(gnn[i]) for i in range(len(sample.node_ids))}

        # Re-rank candidates by GNN score (descending)
        new_ranked = [
            (nid, gnn_dict.get(nid, 0.0))
            for nid, _ in rca_result.ranked_nodes
        ]
        new_ranked.sort(key=lambda x: x[1], reverse=True)

        from rca.apa_rca import RCAResult
        result              = copy.copy(rca_result)
        result.ranked_nodes = new_ranked
        result.method       = "ATF-CVAE+APA-RCA+ResGCN"
        return result


# ── 5-fold CV helper ──────────────────────────────────────────────────────────

def assign_cv_folds(samples: List[GNNSample], n_folds: int = 5) -> List[GNNSample]:
    """
    Assign CV fold index to each sample, stratified by (anomaly_type, pipeline_size).
    Within each stratum, trials are distributed round-robin across folds.
    """
    strata: Dict[Tuple[str, str], List[int]] = {}
    for i, s in enumerate(samples):
        key = (s.anomaly_type, s.pipeline_size)
        strata.setdefault(key, []).append(i)

    for indices in strata.values():
        for rank, idx in enumerate(sorted(indices, key=lambda i: samples[i].trial)):
            samples[idx].fold = rank % n_folds

    return samples
