"""
Anomaly Scorers for FRLG nodes.

Three unsupervised scorers share the same interface (fit / score):
  - VAEAnomalyScorer  : Transform-Type Conditioned VAE (C-VAE) — primary scorer
  - IFAnomalyScorer   : IsolationForest scorer (ablation baseline)
  - OCSVMAnomalyScorer: OneClassSVM scorer (ablation baseline)

All scorers:
  1. Extract per-node time-series windows (size W, z-normalized)
  2. Train on anomaly-free pipeline windows
  3. Return Dict[node_id -> float ∈ (0,1)] via score()
     where higher = more anomalous

─────────────────────────────────────────────────────────────────────────────
C-VAE Architecture (Transform-Type Conditioned VAE)
─────────────────────────────────────────────────────────────────────────────
Motivation
----------
The ATF framework (Chapter 4) shows that different transform types have
structurally different input-output characteristics:

  Source / DirectMap / Calculate  →  ρ = 1.0  (signal preserved)
  Filter (4σ clip)                →  ρ ≈ 0.8  (signal attenuated)
  Aggregate (mean of k inputs)    →  ρ ≈ 0.45 (signal diluted)
  Report (binary threshold)       →  ρ ≈ 0    (signal destroyed)

This implies that the *normal* time-series patterns of nodes differ
systematically by transform type:
  - Source nodes: GBM-like dynamics, fat tails, high autocorrelation
  - Filter nodes: clipped series, bounded variance, attenuated peaks
  - Aggregate nodes: smoothed series, reduced variance (averaging effect)
  - Report nodes: discrete/binary patterns, no continuous anomaly signal

A standard (unconditional) VAE trains a single "normal" manifold pooled
across all node types.  Reconstruction error for a Filter node is then
measured against the average of Source + Aggregate + Report patterns —
an imprecise reference that inflates false-positive rates.

The C-VAE conditions both encoder and decoder on the node's transform type
via a one-hot vector, learning a *separate* normal manifold per type.
Reconstruction error is then type-specific: "is this Filter node behaving
anomalously *for a Filter node*?"

This design principle directly materialises the ATF framework in the model
architecture: the same theoretical insight (type-specific propagation
characteristics) that drives APA-RCA's transition weights now also drives
the anomaly scorer's reference distribution.

Architecture
------------
    type_onehot : R^{N_TYPES}   (6-dim one-hot for transform type)

    Encoder:
        input  = [x(W), type_onehot(N_TYPES)]     ← concat → (W + N_TYPES)
        Linear(W + N_TYPES, H)  → ReLU
        Linear(H, H//2)         → ReLU
        → μ(L),  log σ²(L)

    Decoder:
        input  = [z(L), type_onehot(N_TYPES)]     ← concat → (L + N_TYPES)
        Linear(L + N_TYPES, H//2) → ReLU
        Linear(H//2, H)           → ReLU
        Linear(H, W)              → x̂(W)

    where W = window_size, H = hidden_dim, L = latent_dim, N_TYPES = 6

Loss
----
    ELBO = MSE(x, x̂) + β · KL(q(z|x,t) ‖ p(z))

    KL = -0.5 · mean(1 + log σ² - μ² - exp(log σ²))
    β  = 0.5  (down-weights KL relative to reconstruction, suitable for
               anomaly detection where reconstruction accuracy is primary)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple


# ── Transform-type constants ───────────────────────────────────────────────────
#
# Aligned with synfrp.pipeline.TransformType enum values (str enum).
# Order determines the one-hot index; must be consistent between fit() and score().

TYPE_ORDER: List[str] = [
    "source",       # index 0
    "direct_map",   # index 1
    "filter",       # index 2
    "calculate",    # index 3
    "aggregate",    # index 4
    "report",       # index 5
]
N_TYPES: int = len(TYPE_ORDER)
TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(TYPE_ORDER)}

# Pipeline layers: 1=source, 2=ETL/filter, 3=feature calc, 4=risk agg, 5=report
N_LAYERS: int = 5
MIN_LAYER: int = 1

# Combined conditioning dimension: transform type + pipeline layer
N_COND: int = N_TYPES + N_LAYERS   # 6 + 5 = 11


def _type_onehot(type_str: str) -> np.ndarray:
    """Return N_TYPES-dimensional one-hot vector for a transform type string."""
    vec = np.zeros(N_TYPES, dtype=np.float32)
    idx = TYPE_TO_IDX.get(type_str, 0)   # fallback to "source" index
    vec[idx] = 1.0
    return vec


def _layer_onehot(layer: int) -> np.ndarray:
    """Return N_LAYERS-dimensional one-hot vector for a pipeline layer (1–5)."""
    vec = np.zeros(N_LAYERS, dtype=np.float32)
    idx = max(0, min(N_LAYERS - 1, layer - MIN_LAYER))   # clamp to [0, 4]
    vec[idx] = 1.0
    return vec


def _cond_vec(type_str: str, layer: int) -> np.ndarray:
    """
    Build the N_COND-dimensional conditioning vector.

    Concatenates transform-type one-hot (6-dim) with pipeline-layer
    one-hot (5-dim) → 11-dim total.

    Rationale: the normal time-series pattern of a node depends on BOTH
    its transform type (how it processes inputs) AND its position in the
    pipeline hierarchy (what data has already been processed upstream).
    A Filter node at layer 2 (ETL stage) sees raw GBM prices; a Filter
    node at layer 3 (feature stage) sees log-return derived features —
    structurally different normal windows.
    """
    return np.concatenate([_type_onehot(type_str), _layer_onehot(layer)]).astype(np.float32)


# ── C-VAE model ───────────────────────────────────────────────────────────────

class _CVAE(nn.Module):
    """
    Transform-Type + Pipeline-Layer Conditioned Variational Autoencoder.

    Both encoder and decoder receive an 11-dim conditioning vector that
    encodes the node's transform type (6-dim one-hot) AND its pipeline
    layer (5-dim one-hot).

    Type conditioning:  separate normal manifold per transform type.
    Layer conditioning: further refines the manifold by pipeline depth —
                        a Filter node at layer 2 (raw price data) has
                        different normal patterns than a Filter node at
                        layer 3 (log-return features).

    Parameters
    ----------
    window_size : int    W     — time-series window length
    n_cond      : int    N_COND — conditioning dimension (N_TYPES + N_LAYERS = 11)
    hidden_dim  : int    H     — hidden layer width
    latent_dim  : int    L     — latent space dimension
    """

    def __init__(
        self,
        window_size: int,
        n_cond:      int,
        hidden_dim:  int,
        latent_dim:  int,
    ):
        super().__init__()
        enc_in  = window_size + n_cond   # W + N_COND
        dec_in  = latent_dim  + n_cond   # L + N_COND

        # Encoder: [x, t] → h → (μ, log σ²)
        self.enc = nn.Sequential(
            nn.Linear(enc_in,        hidden_dim),      nn.ReLU(),
            nn.Linear(hidden_dim,    hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: [z, t] → x̂
        self.dec = nn.Sequential(
            nn.Linear(dec_in,           hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2,  hidden_dim),      nn.ReLU(),
            nn.Linear(hidden_dim,       window_size),
        )

    def encode(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode [x, t] → (μ, log σ²)."""
        h = self.enc(torch.cat([x, t], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparametrisation trick: z = μ + σ · ε,  ε ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, W)         time-series windows, z-normalised
        t : (B, N_TYPES)   transform-type one-hot vectors

        Returns
        -------
        x_hat   : (B, W)   reconstructed windows
        mu      : (B, L)   posterior mean
        logvar  : (B, L)   posterior log-variance
        """
        mu, logvar = self.encode(x, t)
        z     = self.reparametrize(mu, logvar)
        x_hat = self.dec(torch.cat([z, t], dim=-1))
        return x_hat, mu, logvar


def _elbo(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Evidence Lower BOund (ELBO) loss.

    L = MSE(x, x̂)  +  β · KL(q(z|x,t) ‖ p(z))

    β < 1 prioritises reconstruction accuracy over posterior regularisation,
    which is appropriate for anomaly detection (we want minimal reconstruction
    error for normal inputs, not necessarily a well-disentangled latent space).
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kld   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld


# ── Public API: VAEAnomalyScorer (C-VAE) ─────────────────────────────────────

class VAEAnomalyScorer:
    """
    Transform-Type Conditioned VAE anomaly scorer.

    Scores nodes by reconstruction error under the C-VAE, where error is
    measured against the type-appropriate normal manifold (not a pooled
    average over all node types).

    Design rationale
    ----------------
    Standard (unconditional) VAE:  one shared normal manifold
    C-VAE:                         one manifold per transform type

    The C-VAE directly encodes the ATF framework's finding that transform
    types have structurally different normal time-series patterns.  This
    produces more precise anomaly signals and, crucially, a more stable
    feature space for cross-domain GNN generalisation: type-specific
    manifold structure is a structural property of the FRLG that holds
    across both synthetic (GBM) and real (RSHB) data, unlike the absolute
    distributional properties (variance level, autocorrelation magnitude)
    that differ across domains.

    Usage
    -----
    scorer = VAEAnomalyScorer()
    scorer.fit(clean_pipelines)        # list of anomaly-free executed pipelines
    scores = scorer.score(pipeline)    # Dict[node_id -> float in (0,1)]
    """

    def __init__(
        self,
        window_size: int   = 20,
        latent_dim:  int   = 8,
        hidden_dim:  int   = 64,
        epochs:      int   = 80,
        batch_size:  int   = 512,
        lr:          float = 1e-3,
        beta:        float = 0.5,
    ):
        self.W          = window_size
        self.L          = latent_dim
        self.H          = hidden_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.beta       = beta
        self.device     = torch.device("cpu")
        self.cvae: Optional[_CVAE] = None
        self._err_mu:  float = 0.0
        self._err_sig: float = 1.0

    # ── helpers ────────────────────────────────────────────────────────────────

    def _series_from_pipeline(self, pipeline) -> Dict[str, np.ndarray]:
        """
        Extract a scalar float32 time series for every node.
        Multi-column nodes are averaged across columns.
        Returns {node_id: np.ndarray of length ≥ W}.
        """
        out: Dict[str, np.ndarray] = {}
        for node_id, col_dict in pipeline.node_data.items():
            if not col_dict:
                continue
            series_list = [
                s.values for s in col_dict.values()
                if isinstance(s, pd.Series) and len(s) >= self.W
            ]
            if not series_list:
                continue
            vals = np.stack(series_list, axis=1).mean(axis=1).astype(np.float32)
            if np.isfinite(vals).any():
                out[node_id] = vals
        return out

    def _get_node_type(self, pipeline, node_id: str) -> str:
        """Retrieve transform type string; falls back to 'source'."""
        node = getattr(pipeline, "nodes", {}).get(node_id)
        if node is None:
            return "source"
        transform_type = getattr(node, "transform_type", None)
        if transform_type is None:
            return "source"
        return str(transform_type.value) if hasattr(transform_type, "value") else str(transform_type)

    def _get_node_layer(self, pipeline, node_id: str) -> int:
        """Retrieve pipeline layer (1–5); falls back to 3 (middle)."""
        node = getattr(pipeline, "nodes", {}).get(node_id)
        if node is None:
            return 3
        layer = getattr(node, "layer", 3)
        return int(layer) if layer is not None else 3

    def _make_windows_with_cond(
        self,
        series_dict: Dict[str, np.ndarray],
        pipeline,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide W-length windows over all node series.

        Returns
        -------
        windows : (N_windows, W)        float32, z-normalised
        conds   : (N_windows, N_COND)   float32, [type_onehot | layer_onehot]
        """
        wins: List[np.ndarray] = []
        conds: List[np.ndarray] = []

        for nid, vals in series_dict.items():
            mu, sig = vals.mean(), vals.std()
            if sig < 1e-8:
                continue
            normed   = (vals - mu) / sig
            cond     = _cond_vec(
                self._get_node_type(pipeline, nid),
                self._get_node_layer(pipeline, nid),
            )
            for s in range(len(normed) - self.W + 1):
                wins.append(normed[s: s + self.W])
                conds.append(cond)

        if not wins:
            return (
                np.zeros((0, self.W),    dtype=np.float32),
                np.zeros((0, N_COND),    dtype=np.float32),
            )
        return (
            np.stack(wins,  axis=0).astype(np.float32),
            np.stack(conds, axis=0).astype(np.float32),
        )

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(self, clean_pipelines: list) -> "VAEAnomalyScorer":
        """
        Train the C-VAE on windows from anomaly-free executed pipelines.

        Each window is paired with its node's transform-type one-hot vector,
        so the model learns separate normal manifolds per type.

        Parameters
        ----------
        clean_pipelines : list[FinancialRiskPipeline]
            Pipelines executed without any injected anomaly.
        """
        all_windows: List[np.ndarray] = []
        all_types:   List[np.ndarray] = []

        for pl in clean_pipelines:
            series   = self._series_from_pipeline(pl)
            W, C     = self._make_windows_with_cond(series, pl)
            if len(W) > 0:
                all_windows.append(W)
                all_types.append(C)

        if not all_windows:
            raise ValueError("No valid windows extracted from clean pipelines.")

        X = np.concatenate(all_windows, axis=0)   # (N_total, W)
        T = np.concatenate(all_types,   axis=0)   # (N_total, N_COND)

        # Shuffle jointly
        rng  = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        X, T = X[perm], T[perm]

        # Log type distribution for transparency
        type_counts = T[:, :N_TYPES].sum(axis=0).astype(int)
        type_summary = ", ".join(
            f"{TYPE_ORDER[i]}={type_counts[i]:,}"
            for i in range(N_TYPES) if type_counts[i] > 0
        )
        print(f"  [C-VAE] Training on {len(X):,} windows  "
              f"(window_size={self.W}, cond_dim={N_COND}, types: {type_summary})")

        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(T, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t, T_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.cvae = _CVAE(self.W, N_COND, self.H, self.L).to(self.device)
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

        # Calibrate per-type normalisation on training data
        # (reconstruction error distribution used for sigmoid scoring)
        self.cvae.eval()
        with torch.no_grad():
            x_hat_all, _, _ = self.cvae(
                X_t.to(self.device), T_t.to(self.device)
            )
            errs = ((X_t.to(self.device) - x_hat_all) ** 2).mean(dim=1).cpu().numpy()

        self._err_mu  = float(errs.mean())
        self._err_sig = float(errs.std() + 1e-8)
        print(f"  [C-VAE] Training complete. "
              f"err_mu={self._err_mu:.4f}, err_sig={self._err_sig:.4f}")
        return self

    # ── score ──────────────────────────────────────────────────────────────────

    def score(self, pipeline) -> Dict[str, float]:
        """
        Compute C-VAE reconstruction-error anomaly score per node.

        Each node's windows are scored against the manifold for that node's
        transform type.  The score is sigmoid-normalized:

            score(v) = sigmoid((err_v - μ_err) / σ_err)

        where μ_err and σ_err are the mean and std of reconstruction error
        across all training windows (calibrated in fit()).

        Returns
        -------
        Dict[node_id -> float ∈ (0, 1)]   higher = more anomalous
        """
        if self.cvae is None:
            raise RuntimeError("Call fit() before score().")

        series = self._series_from_pipeline(pipeline)
        self.cvae.eval()
        node_scores: Dict[str, float] = {}

        with torch.no_grad():
            for nid, vals in series.items():
                mu_s, sig_s = vals.mean(), vals.std()
                if sig_s < 1e-8:
                    node_scores[nid] = 0.0
                    continue

                normed = (vals - mu_s) / sig_s
                wins   = np.stack(
                    [normed[s: s + self.W] for s in range(len(normed) - self.W + 1)]
                ).astype(np.float32)

                # Build conditioning tensor: [type_onehot | layer_onehot]
                cond   = _cond_vec(
                    self._get_node_type(pipeline, nid),
                    self._get_node_layer(pipeline, nid),
                )
                n_wins = len(wins)

                x_t = torch.tensor(wins, dtype=torch.float32).to(self.device)
                t_t = torch.tensor(
                    np.tile(cond, (n_wins, 1)), dtype=torch.float32
                ).to(self.device)

                x_hat, _, _ = self.cvae(x_t, t_t)
                err = ((x_t - x_hat) ** 2).mean(dim=1).cpu().numpy().mean()

                # Sigmoid-normalize
                z = (err - self._err_mu) / self._err_sig
                node_scores[nid] = float(1.0 / (1.0 + np.exp(-z)))

        return node_scores


# ── Shared helpers for sklearn-based scorers ───────────────────────────────────

class _SklearnScorerBase:
    """
    Base class for sklearn unsupervised anomaly scorers.
    Provides shared pipeline/window extraction and sigmoid calibration.
    """

    def __init__(self, window_size: int = 20):
        self.W          = window_size
        self._score_mu:  float = 0.0
        self._score_sig: float = 1.0
        self.model = None

    def _series_from_pipeline(self, pipeline) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for node_id, col_dict in pipeline.node_data.items():
            if not col_dict:
                continue
            series_list = [
                s.values for s in col_dict.values()
                if isinstance(s, pd.Series) and len(s) >= self.W
            ]
            if not series_list:
                continue
            vals = np.stack(series_list, axis=1).mean(axis=1).astype(np.float32)
            if np.isfinite(vals).any():
                out[node_id] = vals
        return out

    def _make_windows(self, series_dict: Dict[str, np.ndarray]) -> np.ndarray:
        wins = []
        for vals in series_dict.values():
            mu, sig = vals.mean(), vals.std()
            if sig < 1e-8:
                continue
            normed = (vals - mu) / sig
            for s in range(len(normed) - self.W + 1):
                wins.append(normed[s: s + self.W])
        return (
            np.stack(wins, axis=0).astype(np.float32)
            if wins else np.zeros((0, self.W), dtype=np.float32)
        )

    def _collect_all_windows(self, clean_pipelines: list) -> np.ndarray:
        parts = []
        for pl in clean_pipelines:
            W = self._make_windows(self._series_from_pipeline(pl))
            if len(W) > 0:
                parts.append(W)
        if not parts:
            raise ValueError("No valid windows from clean pipelines.")
        return np.concatenate(parts, axis=0)

    def _calibrate(self, anomaly_scores: np.ndarray):
        self._score_mu  = float(anomaly_scores.mean())
        self._score_sig = float(anomaly_scores.std() + 1e-8)

    def _sigmoid_normalize(self, raw: float) -> float:
        z = (raw - self._score_mu) / self._score_sig
        return float(1.0 / (1.0 + np.exp(-z)))

    def _score_windows(self, windows: np.ndarray) -> float:
        raise NotImplementedError

    def score(self, pipeline) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Call fit() before score().")
        series = self._series_from_pipeline(pipeline)
        node_scores: Dict[str, float] = {}
        for nid, vals in series.items():
            mu_s, sig_s = vals.mean(), vals.std()
            if sig_s < 1e-8:
                node_scores[nid] = 0.0
                continue
            normed = (vals - mu_s) / sig_s
            wins   = np.stack([normed[s: s + self.W]
                               for s in range(len(normed) - self.W + 1)])
            node_scores[nid] = self._sigmoid_normalize(self._score_windows(wins))
        return node_scores


# ── IsolationForest Scorer (ablation baseline) ────────────────────────────────

class IFAnomalyScorer(_SklearnScorerBase):
    """
    IsolationForest anomaly scorer. Ablation baseline for C-VAE comparison.

    Rationale: IF is a boundary-based method — it partitions the feature space
    of time-series windows and flags windows far from the training distribution
    boundary.  Unlike C-VAE, it has no type conditioning and no temporal
    reconstruction objective.  This ablation tests whether the C-VAE's
    type-conditioned manifold is necessary for cross-domain GNN generalisation.

    Usage
    -----
    scorer = IFAnomalyScorer()
    scorer.fit(clean_pipelines)
    scores = scorer.score(pipeline)    # Dict[node_id -> float ∈ (0,1)]
    """

    def __init__(
        self,
        window_size:       int = 20,
        n_estimators:      int = 100,
        max_train_windows: int = 50_000,
    ):
        super().__init__(window_size)
        self.n_estimators      = n_estimators
        self.max_train_windows = max_train_windows

    def fit(self, clean_pipelines: list) -> "IFAnomalyScorer":
        from sklearn.ensemble import IsolationForest

        X = self._collect_all_windows(clean_pipelines)
        if len(X) > self.max_train_windows:
            rng = np.random.default_rng(42)
            X   = X[rng.choice(len(X), self.max_train_windows, replace=False)]
        print(f"  [IF] Training on {len(X):,} windows (window_size={self.W})")

        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=42, n_jobs=-1
        )
        self.model.fit(X)

        raw_anomaly = -self.model.score_samples(X)
        self._calibrate(raw_anomaly)
        print(f"  [IF] Training complete. "
              f"score_mu={self._score_mu:.4f}, score_sig={self._score_sig:.4f}")
        return self

    def _score_windows(self, windows: np.ndarray) -> float:
        return float(-self.model.score_samples(windows).mean())


# ── OneClass-SVM Scorer (ablation baseline) ───────────────────────────────────

class OCSVMAnomalyScorer(_SklearnScorerBase):
    """
    OneClass-SVM anomaly scorer via SGD. Ablation baseline for C-VAE comparison.

    Uses SGDOneClassSVM (linear approximation via Nystroem kernel map) to avoid
    the O(n²) cost of kernel SVM on large window sets.

    Rationale: OCSVM is a classical one-class boundary method — it learns a
    hyperplane in the kernel feature space separating normal from anomalous
    windows.  Like IF, it has no type conditioning and no temporal reconstruction
    objective.  This ablation tests whether *any* boundary-based method can
    substitute for the C-VAE's manifold-based representation under distribution
    shift (GBM → real market data).

    Usage
    -----
    scorer = OCSVMAnomalyScorer()
    scorer.fit(clean_pipelines)
    scores = scorer.score(pipeline)    # Dict[node_id -> float ∈ (0,1)]
    """

    def __init__(
        self,
        window_size:       int   = 20,
        nu:                float = 0.1,
        n_components:      int   = 100,
        max_train_windows: int   = 20_000,
    ):
        super().__init__(window_size)
        self.nu                = nu
        self.n_components      = n_components
        self.max_train_windows = max_train_windows
        self.scaler     = None
        self.kernel_map = None

    def fit(self, clean_pipelines: list) -> "OCSVMAnomalyScorer":
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.kernel_approximation import Nystroem
        from sklearn.preprocessing import StandardScaler

        X = self._collect_all_windows(clean_pipelines)
        if len(X) > self.max_train_windows:
            rng = np.random.default_rng(42)
            X   = X[rng.choice(len(X), self.max_train_windows, replace=False)]
        print(f"  [OCSVM] Training on {len(X):,} windows (window_size={self.W})")

        self.scaler     = StandardScaler().fit(X)
        X_scaled        = self.scaler.transform(X)
        self.kernel_map = Nystroem(
            kernel="rbf", n_components=self.n_components, random_state=42
        ).fit(X_scaled)
        X_mapped = self.kernel_map.transform(X_scaled)

        self.model = SGDOneClassSVM(nu=self.nu, random_state=42)
        self.model.fit(X_mapped)

        raw_anomaly = -self.model.decision_function(X_mapped)
        self._calibrate(raw_anomaly)
        print(f"  [OCSVM] Training complete. "
              f"score_mu={self._score_mu:.4f}, score_sig={self._score_sig:.4f}")
        return self

    def _score_windows(self, windows: np.ndarray) -> float:
        X_s = self.scaler.transform(windows)
        X_m = self.kernel_map.transform(X_s)
        return float(-self.model.decision_function(X_m).mean())
