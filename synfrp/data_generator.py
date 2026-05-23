"""
SynFRP: Synthetic Financial Risk Pipeline - Data Generator
Generates synthetic market data using GBM, Vasicek, and random walk models.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MarketDataConfig:
    """Configuration for synthetic market data generation."""
    n_days: int = 252          # trading days
    n_stocks: int = 5
    n_rates: int = 3
    n_fx: int = 2
    n_positions: int = 5
    seed: int = 42

    # GBM parameters for stock prices
    stock_mu: float = 0.08     # annual drift
    stock_sigma: float = 0.20  # annual volatility
    stock_S0: float = 100.0    # initial price

    # Vasicek parameters for interest rates
    rate_kappa: float = 0.5    # mean reversion speed
    rate_theta: float = 0.03   # long-term mean
    rate_sigma: float = 0.01   # volatility
    rate_r0: float = 0.02      # initial rate

    # FX parameters
    fx_drift: float = 0.0
    fx_sigma: float = 0.05
    fx_S0: float = 1.0

    # Position sizes
    position_min: float = 1e5
    position_max: float = 1e6


class MarketDataGenerator:
    """
    Generates synthetic financial market data for the SynFRP benchmark.
    Produces: stock prices (GBM), interest rates (Vasicek), FX rates (RW).
    """

    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate_stock_prices(self) -> pd.DataFrame:
        """Geometric Brownian Motion for stock prices."""
        cfg = self.config
        dt = 1 / 252
        n = cfg.n_days
        m = cfg.n_stocks

        # drift and diffusion
        drift = (cfg.stock_mu - 0.5 * cfg.stock_sigma ** 2) * dt
        diff = cfg.stock_sigma * np.sqrt(dt)

        log_returns = drift + diff * self.rng.standard_normal((n, m))
        log_prices = np.cumsum(log_returns, axis=0)
        prices = cfg.stock_S0 * np.exp(log_prices)

        cols = [f"STOCK_{i+1}_PRICE" for i in range(m)]
        dates = pd.bdate_range("2023-01-02", periods=n)
        return pd.DataFrame(prices, index=dates, columns=cols)

    def generate_interest_rates(self) -> pd.DataFrame:
        """Vasicek model for interest rates."""
        cfg = self.config
        dt = 1 / 252
        n = cfg.n_days
        k = cfg.n_rates

        rates = np.zeros((n, k))
        # slightly different theta per tenor
        thetas = [cfg.rate_theta * (1 + 0.5 * i) for i in range(k)]

        for j in range(k):
            r = cfg.rate_r0
            for i in range(n):
                dW = self.rng.standard_normal() * np.sqrt(dt)
                dr = cfg.rate_kappa * (thetas[j] - r) * dt + cfg.rate_sigma * dW
                r = max(r + dr, 0.0)  # floor at 0
                rates[i, j] = r

        cols = [f"RATE_{['SHORT','MID','LONG'][j]}" for j in range(k)]
        dates = pd.bdate_range("2023-01-02", periods=n)
        return pd.DataFrame(rates, index=dates, columns=cols)

    def generate_fx_rates(self) -> pd.DataFrame:
        """Random walk with drift for FX rates."""
        cfg = self.config
        dt = 1 / 252
        n = cfg.n_days
        k = cfg.n_fx

        drift = cfg.fx_drift * dt
        diff = cfg.fx_sigma * np.sqrt(dt)
        log_returns = drift + diff * self.rng.standard_normal((n, k))
        log_fx = np.cumsum(log_returns, axis=0)
        fx = cfg.fx_S0 * np.exp(log_fx)

        base_pairs = ["USD_KRW", "EUR_KRW", "JPY_KRW", "GBP_KRW", "CNY_KRW"]
        pairs = base_pairs[:k]
        dates = pd.bdate_range("2023-01-02", periods=n)
        return pd.DataFrame(fx, index=dates, columns=pairs)

    def generate_positions(self) -> pd.DataFrame:
        """Static position sizes with slow variation."""
        cfg = self.config
        n = cfg.n_days
        m = cfg.n_positions

        base = self.rng.uniform(cfg.position_min, cfg.position_max, m)
        # slow drift: positions change ~1% per day
        drift = self.rng.standard_normal((n, m)) * 0.01
        positions = base * np.cumprod(1 + drift, axis=0)

        cols = [f"POSITION_{i+1}_SIZE" for i in range(m)]
        dates = pd.bdate_range("2023-01-02", periods=n)
        return pd.DataFrame(positions, index=dates, columns=cols)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all raw market data sources."""
        return {
            "stock_prices": self.generate_stock_prices(),
            "interest_rates": self.generate_interest_rates(),
            "fx_rates": self.generate_fx_rates(),
            "positions": self.generate_positions(),
        }


if __name__ == "__main__":
    cfg = MarketDataConfig(seed=42)
    gen = MarketDataGenerator(cfg)
    data = gen.generate_all()
    for name, df in data.items():
        print(f"{name}: shape={df.shape}, cols={list(df.columns)}")
        print(df.head(3))
        print()
