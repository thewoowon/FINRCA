"""
GNN Re-ranker for APA-RCA top-K candidates.

After APA-RCA produces a ranked list, this module builds a local subgraph
around the top-K candidates and applies a 2-layer GCN to produce refined
re-ranking scores that incorporate neighbourhood structure.

Why GCN works here
------------------
APA-RCA walks the *entire* transposed FRLG.  The GNN focuses only on the
local subgraph around candidates and can learn structural patterns that
distinguish true root causes from high-scoring downstream symptoms:
  - true root cause  → low layer, low in-degree, high is_ancestor
  - downstream noise → high layer, many successors, not always ancestor

Architecture (pure PyTorch, no torch_geometric)
-----------------------------------------------
    GCN-1 : (N, F) → (N, 32)  [D^{-1/2} A D^{-1/2} · X · W1, ReLU, Dropout]
    GCN-2 : (N, 32) → (N, 1)  [D^{-1/2} A D^{-1/2} · H · W2, Sigmoid]

Node features (F=7)
-------------------
    0  rwr_score_norm   APA-RCA RWR score / max(scores in subgraph)
    1  anomaly_score    raw anomaly score from detector / VAE
    2  layer_norm       layer index / 5.0
    3  is_ancestor      1 if node is ancestor of target in FRLG
    4  in_degree_norm   in-degree / max_in_degree
    5  out_degree_norm  out-degree / max_out_degree
    6  is_candidate     1 if node is in top-K candidate list

Training
--------
    - Loss: weighted BCE (positive class = true root cause)
    - Optimizer: Adam + weight_decay
    - Gradient clipping for stability

Evaluation strategy
-------------------
    Synthetic:  5-fold cross-validation (126 train / 504 test per fold reversed)
    RSHB:       train on all 630 synthetic, test on 210 real-source trials
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


FEATURE_DIM = 7


# ── GCN Model ────────────────────────────────────────────────────────────────

class _GCN(nn.Module):
    """2-layer GCN with symmetric-normalised adjacency."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.W1      = nn.Linear(in_dim, hidden, bias=True)
        self.W2      = nn.Linear(hidden,       1, bias=True)
        self.drop    = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x : (N, F)   node features
        A : (N, N)   symmetric-normalised adjacency with self-loops
        → (N, 1) scores in (0, 1)
        """
        h = self.relu(self.drop(A @ self.W1(x)))   # (N, hidden)
        return torch.sigmoid(A @ self.W2(h))        # (N, 1)


# ── Sample dataclass ─────────────────────────────────────────────────────────

@dataclass
class GNNSample:
    """One (subgraph, label) pair collected from a single experiment trial."""
    features:    np.ndarray          # (N, F) float32
    adj:         np.ndarray          # (N, N) symmetric-normalised float32
    node_ids:    List[str]           # subgraph node order
    candidate_ids: List[str]         # top-K node ids from APA-RCA
    true_rc:     str                 # ground-truth root cause
    # bookkeeping for CV & metrics
    anomaly_type:  str = ""
    pipeline_size: str = ""
    trial:         int = -1
    fold:          int = -1          # assigned during CV split


# ── Subgraph builder ─────────────────────────────────────────────────────────

def build_gnn_sample(
    G:        nx.DiGraph,
    scores:   Dict[str, float],
    rca_result,                       # RCAResult from APA-RCA
    true_rc:  str,
    top_k:    int = 10,
    anomaly_type:  str = "",
    pipeline_size: str = "",
    trial:    int = -1,
) -> Optional[GNNSample]:
    """Build a GNNSample from one experiment trial."""

    # ── Candidate set ─────────────────────────────────────────────────────────
    candidates = [nid for nid, _ in rca_result.ranked_nodes[:top_k]]
    rwr_dict   = {nid: sc for nid, sc in rca_result.ranked_nodes[:top_k]}
    if not candidates:
        return None

    # ── 1-hop subgraph ───────────────────────────────────────────────────────
    sub_nodes = set(candidates)
    for nid in candidates:
        if nid in G:
            sub_nodes.update(G.predecessors(nid))
            sub_nodes.update(G.successors(nid))
    sub_nodes = sorted(sub_nodes)          # deterministic order
    nidx = {nid: i for i, nid in enumerate(sub_nodes)}
    N    = len(sub_nodes)

    # ── Ancestor set from target node ────────────────────────────────────────
    target = rca_result.target_node
    ancestors: set = set()
    if target and target in G:
        try:
            Gt = G.reverse(copy=False)
            ancestors = nx.ancestors(Gt, target)
        except Exception:
            pass

    # ── Global degree normalisation ──────────────────────────────────────────
    max_in  = max((d for _, d in G.in_degree()),  default=1) + 1
    max_out = max((d for _, d in G.out_degree()), default=1) + 1
    max_rwr = max(rwr_dict.values(), default=1e-8)

    # ── Feature matrix ───────────────────────────────────────────────────────
    feats = np.zeros((N, FEATURE_DIM), dtype=np.float32)
    for i, nid in enumerate(sub_nodes):
        nd = G.nodes.get(nid, {})
        feats[i, 0] = rwr_dict.get(nid, 0.0) / max_rwr
        feats[i, 1] = float(scores.get(nid, 0.0))
        feats[i, 2] = nd.get("layer", 3) / 5.0
        feats[i, 3] = float(nid in ancestors)
        feats[i, 4] = (G.in_degree(nid)  if nid in G else 0) / max_in
        feats[i, 5] = (G.out_degree(nid) if nid in G else 0) / max_out
        feats[i, 6] = float(nid in set(candidates))

    # ── Symmetrically-normalised adjacency (transposed FRLG + self-loops) ───
    A = np.zeros((N, N), dtype=np.float32)
    for u, v in G.edges():                # u → v in FRLG  ⇒  v ← u (transposed)
        if u in nidx and v in nidx:
            A[nidx[v], nidx[u]] = 1.0
    np.fill_diagonal(A, 1.0)
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
    Trains a 2-layer GCN to re-rank APA-RCA top-K candidates.

    Usage
    -----
    reranker = GNNReranker()
    reranker.fit(training_samples)
    new_result = reranker.rerank(G, scores, rca_result)
    """

    def __init__(
        self,
        top_k:      int   = 10,
        hidden_dim: int   = 32,
        dropout:    float = 0.3,
        epochs:     int   = 120,
        lr:         float = 5e-3,
        weight_decay: float = 1e-4,
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
        self.model: Optional[_GCN] = None

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, samples: List[GNNSample], verbose: bool = False) -> "GNNReranker":
        """
        Train GCN on collected (subgraph, label) samples.

        Only samples where true_rc appears in the top-K candidates contribute
        a positive supervision signal; the rest are still used as negatives.
        """
        # Filter: keep only samples where true_rc is in the subgraph
        valid = [s for s in samples if s.true_rc in s.node_ids]
        if not valid:
            if verbose:
                print("  [GNN] No valid training samples — model not updated.")
            return self

        self.model = _GCN(FEATURE_DIM, self.hidden_dim, self.dropout).to(self.device)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in range(self.epochs):
            rng      = np.random.default_rng(epoch)
            shuffled = [valid[i] for i in rng.permutation(len(valid))]
            total_loss = 0.0

            for s in shuffled:
                x   = torch.tensor(s.features, dtype=torch.float32, device=self.device)
                A   = torch.tensor(s.adj,      dtype=torch.float32, device=self.device)
                y   = torch.tensor(
                    [1.0 if nid == s.true_rc else 0.0 for nid in s.node_ids],
                    dtype=torch.float32, device=self.device,
                )

                opt.zero_grad()
                pred  = self.model(x, A).squeeze(-1)        # (N,)
                # Weighted BCE: up-weight the single positive node
                w     = 1.0 + y * (self.pos_weight_scale - 1.0)
                bce   = nn.functional.binary_cross_entropy(pred, y, weight=w)
                bce.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += bce.item()

            if verbose and (epoch + 1) % 30 == 0:
                print(f"    epoch {epoch+1}/{self.epochs}  loss={total_loss/len(shuffled):.4f}")

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def rerank(self, sample: GNNSample, rca_result):
        """
        Apply trained GNN to re-rank candidates in rca_result.
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

        # Re-rank: candidates sorted by GNN score
        new_ranked = [
            (nid, gnn_dict.get(nid, 0.0))
            for nid, _ in rca_result.ranked_nodes
        ]
        new_ranked.sort(key=lambda x: x[1], reverse=True)

        from rca.apa_rca import RCAResult
        result              = copy.copy(rca_result)
        result.ranked_nodes = new_ranked
        result.method       = "VAE+APA-RCA+GNN"
        return result


# ── 5-fold CV helper ─────────────────────────────────────────────────────────

def assign_cv_folds(samples: List[GNNSample], n_folds: int = 5) -> List[GNNSample]:
    """
    Assign CV fold index to each sample, stratified by (anomaly_type, pipeline_size).
    Within each stratum, trials are distributed round-robin across folds.
    """
    from itertools import cycle

    strata: Dict[Tuple[str, str], List[int]] = {}
    for i, s in enumerate(samples):
        key = (s.anomaly_type, s.pipeline_size)
        strata.setdefault(key, []).append(i)

    for indices in strata.values():
        for rank, idx in enumerate(sorted(indices, key=lambda i: samples[i].trial)):
            samples[idx].fold = rank % n_folds

    return samples
