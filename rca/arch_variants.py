"""
Architecture Variants for ablation study.

Variant A: TypeSpecificDecoderCVAE — shared encoder + 6 type-specific decoders
Variant B: RGCN — edge-type-specific message passing (one W_r per relation)

These can be combined with existing models for a 2×2 comparison:
  (1) Standard CVAE  + Standard ResGCN    [current v2]
  (2) Standard CVAE  + RGCN               [variant B only]
  (3) TSD-CVAE       + Standard ResGCN    [variant A only]
  (4) TSD-CVAE       + RGCN               [both variants]
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .vae_scorer import (
    TYPE_ORDER, N_TYPES, N_LAYERS, N_COND,
    TYPE_TO_IDX, _cond_vec, _elbo,
    VAEAnomalyScorer,
)
from .gnn_reranker import (
    ATF_EDGE_WEIGHTS, FEATURE_DIM,
    _ResGCN, GNNSample, GNNReranker,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Variant A: Type-Specific Decoder CVAE (TSD-CVAE)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Motivation: Standard CVAE concatenates conditioning vector to decoder input,
# so all types share the same decoder weights.  TSD-CVAE routes latent z to
# a type-specific decoder sub-network, giving each transform type its own
# reconstruction pathway — a stronger structural prior than conditioning.
#
# Architecture:
#   Encoder (shared): [x, t] → h → (μ, log σ²)     [same as CVAE]
#   Decoder (per-type): z → x̂_τ                     [6 separate decoders]
#
#   During forward, type index selects the decoder: x̂ = Dec_τ(z)
#   No conditioning concat needed in decoder — type is implicit in routing.

class _TSDDecoder(nn.Module):
    """Single type-specific decoder: z → x̂."""
    def __init__(self, latent_dim: int, hidden_dim: int, window_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim,      hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),      nn.ReLU(),
            nn.Linear(hidden_dim,      window_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _TSDDecoderHybrid(nn.Module):
    """Single type-specific decoder with conditioning: [z, t] → x̂."""
    def __init__(self, latent_dim: int, n_cond: int, hidden_dim: int, window_size: int):
        super().__init__()
        dec_in = latent_dim + n_cond
        self.net = nn.Sequential(
            nn.Linear(dec_in,          hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),      nn.ReLU(),
            nn.Linear(hidden_dim,      window_size),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, t], dim=-1))


class _TSDCVAE(nn.Module):
    """
    Type-Specific Decoder CVAE (pure routing, no conditioning in decoder).
    """
    def __init__(self, window_size: int, n_cond: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        enc_in = window_size + n_cond

        self.enc = nn.Sequential(
            nn.Linear(enc_in,     hidden_dim),      nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoders = nn.ModuleList([
            _TSDDecoder(latent_dim, hidden_dim, window_size)
            for _ in range(N_TYPES)
        ])

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(torch.cat([x, t], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, t)
        z = self.reparametrize(mu, logvar)

        type_idx = t[:, :N_TYPES].argmax(dim=1)
        x_hat = torch.zeros_like(x)
        for tidx in range(N_TYPES):
            mask = (type_idx == tidx)
            if mask.any():
                x_hat[mask] = self.decoders[tidx](z[mask])

        return x_hat, mu, logvar


class _TSDCVAEHybrid(nn.Module):
    """
    Hybrid TSD-CVAE: type-specific decoders + conditioning in both encoder & decoder.

    Encoder: [x, t] → (mu, logvar)          [shared, conditioned]
    Decoder: Dec_tau([z, t]) → x_hat         [type-routed + conditioned]

    Combines structural separation (per-type decoder) with information
    preservation (conditioning vector carries layer info to decoder).
    """
    def __init__(self, window_size: int, n_cond: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        enc_in = window_size + n_cond

        self.enc = nn.Sequential(
            nn.Linear(enc_in,     hidden_dim),      nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Type-specific decoders that ALSO receive conditioning
        self.decoders = nn.ModuleList([
            _TSDDecoderHybrid(latent_dim, n_cond, hidden_dim, window_size)
            for _ in range(N_TYPES)
        ])

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(torch.cat([x, t], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, t)
        z = self.reparametrize(mu, logvar)

        type_idx = t[:, :N_TYPES].argmax(dim=1)
        x_hat = torch.zeros_like(x)
        for tidx in range(N_TYPES):
            mask = (type_idx == tidx)
            if mask.any():
                x_hat[mask] = self.decoders[tidx](z[mask], t[mask])

        return x_hat, mu, logvar


class TSDCVAEScorer(VAEAnomalyScorer):
    """
    Drop-in replacement for VAEAnomalyScorer using Type-Specific Decoder CVAE.

    Inherits all data handling (window extraction, scoring, normalization)
    from VAEAnomalyScorer — only replaces the model architecture.
    """

    def fit(self, clean_pipelines: list) -> "TSDCVAEScorer":
        """Train TSD-CVAE instead of standard CVAE."""
        import pandas as pd
        from torch.utils.data import DataLoader, TensorDataset

        all_windows: List[np.ndarray] = []
        all_types:   List[np.ndarray] = []

        for pl in clean_pipelines:
            series = self._series_from_pipeline(pl)
            W, C   = self._make_windows_with_cond(series, pl)
            if len(W) > 0:
                all_windows.append(W)
                all_types.append(C)

        if not all_windows:
            raise ValueError("No valid windows extracted from clean pipelines.")

        X = np.concatenate(all_windows, axis=0)
        T = np.concatenate(all_types,   axis=0)

        rng  = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        X, T = X[perm], T[perm]

        type_counts = T[:, :N_TYPES].sum(axis=0).astype(int)
        type_summary = ", ".join(
            f"{TYPE_ORDER[i]}={type_counts[i]:,}"
            for i in range(N_TYPES) if type_counts[i] > 0
        )
        print(f"  [TSD-CVAE] Training on {len(X):,} windows  "
              f"(window_size={self.W}, types: {type_summary})")

        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(T, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t, T_t),
            batch_size=self.batch_size, shuffle=True,
        )

        # Use TSD-CVAE instead of standard CVAE
        self.cvae = _TSDCVAE(self.W, N_COND, self.H, self.L).to(self.device)
        opt = torch.optim.Adam(self.cvae.parameters(), lr=self.lr)

        self.cvae.train()
        for epoch in range(self.epochs):
            for (batch_x, batch_t) in loader:
                batch_x = batch_x.to(self.device)
                batch_t = batch_t.to(self.device)
                opt.zero_grad()
                x_hat, mu, lv = self.cvae(batch_x, batch_t)
                loss = _elbo(batch_x, x_hat, mu, lv, self.beta)
                loss.backward()
                opt.step()

        # Calibrate normalization
        self.cvae.eval()
        with torch.no_grad():
            x_hat_all, _, _ = self.cvae(
                X_t.to(self.device), T_t.to(self.device)
            )
            errs = ((X_t.to(self.device) - x_hat_all) ** 2).mean(dim=1).cpu().numpy()

        self._err_mu  = float(errs.mean())
        self._err_sig = float(errs.std() + 1e-8)
        print(f"  [TSD-CVAE] Training complete. "
              f"err_mu={self._err_mu:.4f}, err_sig={self._err_sig:.4f}")
        return self


class HybridTSDCVAEScorer(VAEAnomalyScorer):
    """
    Hybrid TSD-CVAE scorer: type-specific decoders + conditioning in decoder.
    Combines structural separation with layer-info preservation.
    """

    def fit(self, clean_pipelines: list) -> "HybridTSDCVAEScorer":
        from torch.utils.data import DataLoader, TensorDataset

        all_windows: List[np.ndarray] = []
        all_types:   List[np.ndarray] = []

        for pl in clean_pipelines:
            series = self._series_from_pipeline(pl)
            W, C   = self._make_windows_with_cond(series, pl)
            if len(W) > 0:
                all_windows.append(W)
                all_types.append(C)

        if not all_windows:
            raise ValueError("No valid windows extracted from clean pipelines.")

        X = np.concatenate(all_windows, axis=0)
        T = np.concatenate(all_types,   axis=0)

        rng  = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        X, T = X[perm], T[perm]

        type_counts = T[:, :N_TYPES].sum(axis=0).astype(int)
        type_summary = ", ".join(
            f"{TYPE_ORDER[i]}={type_counts[i]:,}"
            for i in range(N_TYPES) if type_counts[i] > 0
        )
        print(f"  [Hybrid-TSD-CVAE] Training on {len(X):,} windows  "
              f"(window_size={self.W}, types: {type_summary})")

        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(T, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t, T_t),
            batch_size=self.batch_size, shuffle=True,
        )

        self.cvae = _TSDCVAEHybrid(self.W, N_COND, self.H, self.L).to(self.device)
        opt = torch.optim.Adam(self.cvae.parameters(), lr=self.lr)

        self.cvae.train()
        for epoch in range(self.epochs):
            for (batch_x, batch_t) in loader:
                batch_x = batch_x.to(self.device)
                batch_t = batch_t.to(self.device)
                opt.zero_grad()
                x_hat, mu, lv = self.cvae(batch_x, batch_t)
                loss = _elbo(batch_x, x_hat, mu, lv, self.beta)
                loss.backward()
                opt.step()

        self.cvae.eval()
        with torch.no_grad():
            x_hat_all, _, _ = self.cvae(
                X_t.to(self.device), T_t.to(self.device)
            )
            errs = ((X_t.to(self.device) - x_hat_all) ** 2).mean(dim=1).cpu().numpy()

        self._err_mu  = float(errs.mean())
        self._err_sig = float(errs.std() + 1e-8)
        print(f"  [Hybrid-TSD-CVAE] Training complete. "
              f"err_mu={self._err_mu:.4f}, err_sig={self._err_sig:.4f}")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Variant B: Relational GCN (R-GCN)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Motivation: Standard ResGCN uses scalar ρ_τ as edge weights in a single
# adjacency matrix.  R-GCN gives each edge type its own learnable weight
# matrix W_r, allowing the model to learn type-specific message transformations
# rather than just scalar scaling.
#
# Architecture:
#   h = ReLU(LayerNorm( Σ_r A_r · W_r(x)  +  W_skip(x) ))
#   out = Sigmoid(A_full · W_out(h))
#
#   where r ∈ {source, direct_map, calculate, filter, aggregate, report}
#   A_r = binary adjacency for edges of type r (no ρ_τ scalar — learned instead)
#   A_full = union adjacency with self-loops (for output layer)

EDGE_TYPES = list(ATF_EDGE_WEIGHTS.keys())
N_EDGE_TYPES = len(EDGE_TYPES)
EDGE_TYPE_TO_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}


class _RGCN(nn.Module):
    """
    Relational GCN with per-edge-type message passing + residual skip.

    Layer 1: h = ReLU(LayerNorm( Σ_r A_r · W_r(x) + W_skip(x) ))
    Layer 2: out = Sigmoid(A · W_out(h))
    """
    def __init__(self, in_dim: int, hidden: int, dropout: float, n_relations: int):
        super().__init__()
        # Per-relation weight matrices
        self.W_rel = nn.ModuleList([
            nn.Linear(in_dim, hidden, bias=False) for _ in range(n_relations)
        ])
        # Skip connection (same as ResGCN)
        self.W_skip = nn.Linear(in_dim, hidden, bias=True)
        self.norm1  = nn.LayerNorm(hidden)
        # Output layer
        self.W_out  = nn.Linear(hidden, 1, bias=True)
        self.drop   = nn.Dropout(dropout)
        self.relu   = nn.ReLU()

    def forward(
        self, x: torch.Tensor, A_rels: List[torch.Tensor], A_full: torch.Tensor
    ) -> torch.Tensor:
        """
        x      : (N, F)              node features
        A_rels : list of (N, N)       per-relation adjacency matrices (sym-norm)
        A_full : (N, N)               full adjacency with self-loops (for layer 2)
        → (N, 1)
        """
        # Layer 1: per-relation message passing + skip
        h_rel = sum(
            self.drop(A_r @ W_r(x))
            for A_r, W_r in zip(A_rels, self.W_rel)
        )
        h_skip = self.W_skip(x)
        h = self.relu(self.norm1(h_rel + h_skip))

        # Layer 2: output
        return torch.sigmoid(A_full @ self.W_out(h))


def build_rgcn_sample(
    G,
    scores:     Dict[str, float],
    rca_result,
    true_rc:    str,
    top_k:      int = 10,
    anomaly_type:  str = "",
    pipeline_size: str = "",
    trial:      int = -1,
) -> Optional["RGCNSample"]:
    """
    Build an RGCNSample with per-relation adjacency matrices.

    Same subgraph and features as build_gnn_sample, but adjacency is split
    into N_EDGE_TYPES separate binary matrices (one per relation type).
    """
    import networkx as nx

    candidates = [nid for nid, _ in rca_result.ranked_nodes[:top_k]]
    rwr_dict   = {nid: sc for nid, sc in rca_result.ranked_nodes[:top_k]}
    if not candidates:
        return None

    sub_nodes = set(candidates)
    for nid in candidates:
        if nid in G:
            sub_nodes.update(G.predecessors(nid))
            sub_nodes.update(G.successors(nid))
    sub_nodes = sorted(sub_nodes)
    nidx = {nid: i for i, nid in enumerate(sub_nodes)}
    N = len(sub_nodes)

    target = rca_result.target_node
    ancestors: set = set()
    dist_from_target: Dict[str, int] = {}
    if target and target in G:
        try:
            Gt = G.reverse(copy=False)
            dist_from_target = dict(nx.single_source_shortest_path_length(Gt, target))
            ancestors = set(dist_from_target.keys()) - {target}
        except Exception:
            pass

    max_dist     = max(dist_from_target.values(), default=1)
    penalty_dist = max_dist + 1
    max_in  = max((d for _, d in G.in_degree()),  default=1) + 1
    max_out = max((d for _, d in G.out_degree()), default=1) + 1
    max_rwr = max(rwr_dict.values(), default=1e-8)

    # Features (same F=8 as ResGCN)
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
        feats[i, 7] = dist / penalty_dist

    # Per-relation adjacency matrices (transposed, binary)
    A_rels = [np.zeros((N, N), dtype=np.float32) for _ in range(N_EDGE_TYPES)]
    A_full = np.zeros((N, N), dtype=np.float32)

    for u, v in G.edges():
        if u in nidx and v in nidx:
            edge_type = G.edges[u, v].get("edge_type", "direct_map")
            ridx = EDGE_TYPE_TO_IDX.get(edge_type, 1)  # default direct_map
            A_rels[ridx][nidx[v], nidx[u]] = 1.0  # transposed
            A_full[nidx[v], nidx[u]] = 1.0

    # Self-loops on full adjacency
    np.fill_diagonal(A_full, 1.0)

    # Symmetric normalization for each relation matrix
    A_rels_norm = []
    for A_r in A_rels:
        np.fill_diagonal(A_r, 0.0)  # no self-loops in per-relation
        deg = A_r.sum(axis=1, keepdims=True).clip(min=1.0)
        D_inv_sqrt = 1.0 / np.sqrt(deg)
        A_rels_norm.append(D_inv_sqrt * A_r * D_inv_sqrt.T)

    # Normalize full adjacency
    deg_full = A_full.sum(axis=1, keepdims=True).clip(min=1.0)
    D_inv_sqrt_full = 1.0 / np.sqrt(deg_full)
    A_full_norm = D_inv_sqrt_full * A_full * D_inv_sqrt_full.T

    return RGCNSample(
        features=feats,
        adj_rels=A_rels_norm,
        adj_full=A_full_norm,
        node_ids=sub_nodes,
        candidate_ids=candidates,
        true_rc=true_rc,
        anomaly_type=anomaly_type,
        pipeline_size=pipeline_size,
        trial=trial,
    )


from dataclasses import dataclass, field

@dataclass
class RGCNSample:
    """One (subgraph, label) pair with per-relation adjacency."""
    features:      np.ndarray
    adj_rels:      List[np.ndarray]   # list of (N, N) per-relation
    adj_full:      np.ndarray         # (N, N) full adjacency
    node_ids:      List[str]
    candidate_ids: List[str]
    true_rc:       str
    anomaly_type:  str = ""
    pipeline_size: str = ""
    trial:         int = -1
    fold:          int = -1


class RGCNReranker:
    """
    R-GCN reranker with per-edge-type message passing.
    Drop-in replacement for GNNReranker with different model architecture.
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
        self.model: Optional[_RGCN] = None

    def fit(self, samples: List[RGCNSample], verbose: bool = False) -> "RGCNReranker":
        valid = [s for s in samples if s.true_rc in s.node_ids]
        if not valid:
            if verbose:
                print("  [R-GCN] No valid training samples.")
            return self

        self.model = _RGCN(
            FEATURE_DIM, self.hidden_dim, self.dropout, N_EDGE_TYPES
        ).to(self.device)
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
                A_rels = [
                    torch.tensor(a, dtype=torch.float32, device=self.device)
                    for a in s.adj_rels
                ]
                A_full = torch.tensor(s.adj_full, dtype=torch.float32, device=self.device)
                y = torch.tensor(
                    [1.0 if nid == s.true_rc else 0.0 for nid in s.node_ids],
                    dtype=torch.float32, device=self.device,
                )

                opt.zero_grad()
                pred = self.model(x, A_rels, A_full).squeeze(-1)

                w   = 1.0 + y * (self.pos_weight_scale - 1.0)
                bce = nn.functional.binary_cross_entropy(pred, y, weight=w)
                bce.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += bce.item()

            if verbose and (epoch + 1) % 30 == 0:
                print(f"    epoch {epoch+1}/{self.epochs}  loss={total_loss/len(shuffled):.4f}")

        return self

    def rerank(self, sample: RGCNSample, rca_result) -> object:
        if self.model is None:
            return rca_result

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(sample.features, dtype=torch.float32, device=self.device)
            A_rels = [
                torch.tensor(a, dtype=torch.float32, device=self.device)
                for a in sample.adj_rels
            ]
            A_full = torch.tensor(sample.adj_full, dtype=torch.float32, device=self.device)
            gnn = self.model(x, A_rels, A_full).squeeze(-1).cpu().numpy()

        gnn_dict = {sample.node_ids[i]: float(gnn[i]) for i in range(len(sample.node_ids))}

        new_ranked = [
            (nid, gnn_dict.get(nid, 0.0))
            for nid, _ in rca_result.ranked_nodes
        ]
        new_ranked.sort(key=lambda x: x[1], reverse=True)

        from .apa_rca import RCAResult
        result              = copy.copy(rca_result)
        result.ranked_nodes = new_ranked
        result.method       = "R-GCN reranked"
        return result
