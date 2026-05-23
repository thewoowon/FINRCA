"""
Anomaly Scorers for FRLG nodes.

Three unsupervised scorers share the same interface (fit / score):
  - VAEAnomalyScorer  : MLP-VAE reconstruction-error scorer (primary)
  - IFAnomalyScorer   : IsolationForest scorer (ablation baseline)
  - OCSVMAnomalyScorer: OneClassSVM scorer via SGD (ablation baseline)

All scorers:
  1. Extract per-node time-series windows (size W, z-normalized)
  2. Train on anomaly-free pipeline windows
  3. Return Dict[node_id -> float ∈ (0,1)] via score()
     where higher = more anomalous

VAE Architecture (MLP-VAE):
    Encoder: Linear(W, H) → ReLU → Linear(H, H//2) → (μ: L, log σ²: L)
    Decoder: Linear(L, H//2) → ReLU → Linear(H//2, H) → ReLU → Linear(H, W)
    where W = window_size, H = hidden_dim, L = latent_dim
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional


# ── MLP-VAE ──────────────────────────────────────────────────────────────────

class _MLPVAE(nn.Module):
    def __init__(self, window_size: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(window_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, window_size),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z   = self.reparametrize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar


def _elbo(x, x_hat, mu, logvar, beta: float = 0.5):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kld   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld


# ── Public API ────────────────────────────────────────────────────────────────

class VAEAnomalyScorer:
    """
    Shared MLP-VAE trained on time-series windows from anomaly-free pipelines.
    One scorer instance is shared across all pipeline sizes because each series
    is z-normalized before windowing — the VAE learns window *shape*, not scale.

    Usage
    -----
    scorer = VAEAnomalyScorer()
    scorer.fit(clean_pipelines)          # list of executed pipelines, no anomaly
    scores = scorer.score(pipeline)      # Dict[node_id -> float in (0,1)]
    """

    def __init__(
        self,
        window_size: int  = 20,
        latent_dim:  int  = 8,
        hidden_dim:  int  = 64,
        epochs:      int  = 80,
        batch_size:  int  = 512,
        lr:          float = 1e-3,
        beta:        float = 0.5,
    ):
        self.W  = window_size
        self.L  = latent_dim
        self.H  = hidden_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr   = lr
        self.beta = beta
        self.device = torch.device("cpu")
        self.vae: Optional[_MLPVAE] = None
        self._err_mu:  float = 0.0
        self._err_sig: float = 1.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _series_from_pipeline(self, pipeline) -> Dict[str, np.ndarray]:
        """
        Extract a scalar float32 time series for every node.
        Uses pipeline.node_data (Dict[node_id -> Dict[col -> pd.Series]]).
        Multi-column nodes are averaged across columns.
        """
        out = {}
        for node_id, col_dict in pipeline.node_data.items():
            if not col_dict:
                continue
            series_list = [s.values for s in col_dict.values()
                           if isinstance(s, pd.Series) and len(s) >= self.W]
            if not series_list:
                continue
            vals = np.stack(series_list, axis=1).mean(axis=1).astype(np.float32)
            if np.isfinite(vals).any():
                out[node_id] = vals
        return out

    def _make_windows(self, series_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Slide W-length windows over all series. Returns (N_windows, W)."""
        wins = []
        for vals in series_dict.values():
            mu, sig = vals.mean(), vals.std()
            if sig < 1e-8:
                continue
            normed = (vals - mu) / sig
            for s in range(len(normed) - self.W + 1):
                wins.append(normed[s: s + self.W])
        return np.stack(wins, axis=0).astype(np.float32) if wins else np.zeros((0, self.W), dtype=np.float32)

    def _node_score_from_windows(self, windows: np.ndarray) -> float:
        """Return sigmoid-normalized mean reconstruction error."""
        x_t   = torch.tensor(windows, dtype=torch.float32).to(self.device)
        x_hat, _, _ = self.vae(x_t)
        err   = ((x_t - x_hat) ** 2).mean(dim=1).cpu().numpy().mean()
        z     = (err - self._err_mu) / self._err_sig
        return float(1.0 / (1.0 + np.exp(-z)))

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, clean_pipelines: list) -> "VAEAnomalyScorer":
        """
        Train VAE on windows from anomaly-free executed pipelines.

        Parameters
        ----------
        clean_pipelines : list[FinancialRiskPipeline]
            Pipelines that have been executed with NO anomaly injected.
        """
        all_windows: List[np.ndarray] = []
        for pl in clean_pipelines:
            series = self._series_from_pipeline(pl)
            W = self._make_windows(series)
            if len(W) > 0:
                all_windows.append(W)

        if not all_windows:
            raise ValueError("No valid windows from clean pipelines.")

        X = np.concatenate(all_windows, axis=0)
        rng = np.random.default_rng(42)
        X   = X[rng.permutation(len(X))]
        print(f"  [VAE] Training on {len(X):,} windows  (window_size={self.W})")

        X_t    = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=True)

        self.vae = _MLPVAE(self.W, self.H, self.L).to(self.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)

        self.vae.train()
        for epoch in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                x_hat, mu, lv = self.vae(batch)
                loss = _elbo(batch, x_hat, mu, lv, self.beta)
                loss.backward()
                opt.step()

        # Calibrate normalization on training data
        self.vae.eval()
        with torch.no_grad():
            x_hat, _, _ = self.vae(X_t.to(self.device))
            errs = ((X_t.to(self.device) - x_hat) ** 2).mean(dim=1).cpu().numpy()
        self._err_mu  = float(errs.mean())
        self._err_sig = float(errs.std() + 1e-8)
        print(f"  [VAE] Training complete. err_mu={self._err_mu:.4f}, err_sig={self._err_sig:.4f}")
        return self

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, pipeline) -> Dict[str, float]:
        """
        Compute VAE reconstruction-error anomaly score per node.
        Returns Dict[node_id -> float ∈ (0,1)].
        Nodes with insufficient data get score 0.0.
        """
        if self.vae is None:
            raise RuntimeError("Call fit() before score().")

        series = self._series_from_pipeline(pipeline)
        self.vae.eval()
        node_scores: Dict[str, float] = {}
        with torch.no_grad():
            for nid, vals in series.items():
                mu_s, sig_s = vals.mean(), vals.std()
                if sig_s < 1e-8:
                    node_scores[nid] = 0.0
                    continue
                normed = (vals - mu_s) / sig_s
                wins   = np.stack([normed[s: s + self.W] for s in range(len(normed) - self.W + 1)])
                node_scores[nid] = self._node_score_from_windows(wins)
        return node_scores


# ── Shared helpers for sklearn-based scorers ──────────────────────────────────

class _SklearnScorerBase:
    """
    Base class for sklearn unsupervised anomaly scorers.
    Provides shared pipeline/window extraction and sigmoid calibration.
    """

    def __init__(self, window_size: int = 20):
        self.W = window_size
        self._score_mu:  float = 0.0
        self._score_sig: float = 1.0
        self.model = None

    def _series_from_pipeline(self, pipeline) -> Dict[str, np.ndarray]:
        out = {}
        for node_id, col_dict in pipeline.node_data.items():
            if not col_dict:
                continue
            series_list = [s.values for s in col_dict.values()
                           if isinstance(s, pd.Series) and len(s) >= self.W]
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
        return np.stack(wins, axis=0).astype(np.float32) if wins else np.zeros((0, self.W), dtype=np.float32)

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
        """Store (μ, σ) of raw anomaly scores for sigmoid normalization."""
        self._score_mu  = float(anomaly_scores.mean())
        self._score_sig = float(anomaly_scores.std() + 1e-8)

    def _sigmoid_normalize(self, raw: float) -> float:
        z = (raw - self._score_mu) / self._score_sig
        return float(1.0 / (1.0 + np.exp(-z)))

    def _score_windows(self, windows: np.ndarray) -> float:
        """Override in subclass. Returns mean raw anomaly score for a window set."""
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
            wins = np.stack([normed[s: s + self.W]
                             for s in range(len(normed) - self.W + 1)])
            raw = self._score_windows(wins)
            node_scores[nid] = self._sigmoid_normalize(raw)
        return node_scores


# ── IsolationForest Scorer ────────────────────────────────────────────────────

class IFAnomalyScorer(_SklearnScorerBase):
    """
    IsolationForest-based anomaly scorer. Same fit/score interface as VAEAnomalyScorer.

    Rationale for ablation:
        IF produces a continuous anomaly score from an ensemble of random trees.
        Unlike VAE, it has no temporal reconstruction objective — it treats each
        window as an i.i.d. point in R^W. This ablation tests whether the VAE's
        reconstruction-based representation is necessary for cross-domain GNN
        generalization.

    Usage
    -----
    scorer = IFAnomalyScorer()
    scorer.fit(clean_pipelines)
    scores = scorer.score(pipeline)    # Dict[node_id -> float ∈ (0,1)]
    """

    def __init__(self, window_size: int = 20, n_estimators: int = 100,
                 max_train_windows: int = 50_000):
        super().__init__(window_size)
        self.n_estimators = n_estimators
        self.max_train_windows = max_train_windows

    def fit(self, clean_pipelines: list) -> "IFAnomalyScorer":
        from sklearn.ensemble import IsolationForest

        X = self._collect_all_windows(clean_pipelines)
        if len(X) > self.max_train_windows:
            rng = np.random.default_rng(42)
            X = X[rng.choice(len(X), self.max_train_windows, replace=False)]
        print(f"  [IF] Training on {len(X):,} windows (window_size={self.W})")

        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=42, n_jobs=-1
        )
        self.model.fit(X)

        # score_samples: higher = more normal → negate for anomaly score
        raw_anomaly = -self.model.score_samples(X)
        self._calibrate(raw_anomaly)
        print(f"  [IF] Training complete. score_mu={self._score_mu:.4f}, "
              f"score_sig={self._score_sig:.4f}")
        return self

    def _score_windows(self, windows: np.ndarray) -> float:
        # negate: higher value = more anomalous
        return float(-self.model.score_samples(windows).mean())


# ── OneClass-SVM Scorer (SGD) ─────────────────────────────────────────────────

class OCSVMAnomalyScorer(_SklearnScorerBase):
    """
    OneClass-SVM anomaly scorer using SGD optimization (sklearn SGDOneClassSVM).
    Same fit/score interface as VAEAnomalyScorer.

    Uses SGDOneClassSVM (linear approximation via Nystroem kernel map) to avoid
    the O(n²) cost of kernel SVM on large window sets.

    Rationale for ablation:
        OCSVM is a classical one-class boundary method. It learns a decision
        boundary in feature space rather than reconstructing temporal patterns.
        This tests whether any generative/reconstruction objective (VAE) is
        necessary for the GNN feature to generalize across domains.

    Usage
    -----
    scorer = OCSVMAnomalyScorer()
    scorer.fit(clean_pipelines)
    scores = scorer.score(pipeline)    # Dict[node_id -> float ∈ (0,1)]
    """

    def __init__(self, window_size: int = 20, nu: float = 0.1,
                 n_components: int = 100, max_train_windows: int = 20_000):
        super().__init__(window_size)
        self.nu = nu
        self.n_components = n_components
        self.max_train_windows = max_train_windows
        self.scaler = None
        self.kernel_map = None

    def fit(self, clean_pipelines: list) -> "OCSVMAnomalyScorer":
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.kernel_approximation import Nystroem
        from sklearn.preprocessing import StandardScaler

        X = self._collect_all_windows(clean_pipelines)
        if len(X) > self.max_train_windows:
            rng = np.random.default_rng(42)
            X = X[rng.choice(len(X), self.max_train_windows, replace=False)]
        print(f"  [OCSVM] Training on {len(X):,} windows (window_size={self.W})")

        self.scaler     = StandardScaler().fit(X)
        X_scaled        = self.scaler.transform(X)
        self.kernel_map = Nystroem(kernel="rbf", n_components=self.n_components,
                                   random_state=42).fit(X_scaled)
        X_mapped        = self.kernel_map.transform(X_scaled)

        self.model = SGDOneClassSVM(nu=self.nu, random_state=42)
        self.model.fit(X_mapped)

        # decision_function: positive = normal, negative = anomaly → negate
        raw_anomaly = -self.model.decision_function(X_mapped)
        self._calibrate(raw_anomaly)
        print(f"  [OCSVM] Training complete. score_mu={self._score_mu:.4f}, "
              f"score_sig={self._score_sig:.4f}")
        return self

    def _score_windows(self, windows: np.ndarray) -> float:
        X_s = self.scaler.transform(windows)
        X_m = self.kernel_map.transform(X_s)
        return float(-self.model.decision_function(X_m).mean())
