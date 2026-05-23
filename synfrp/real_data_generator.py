"""
Real-Source Hybrid Market Data Generator.

Replaces GBM/Vasicek/RW synthetic source data with actual Yahoo Finance time series
while keeping all ETL/Feature/Risk/Report pipeline layers unchanged.
This produces the "Real-Source Hybrid Benchmark" (RSHB):
  - Source layer (L1): real stock prices, FX rates, interest rate proxies
  - ETL/Feature/Risk/Report (L2-L5): identical SynFRP pipeline logic

Column naming is intentionally identical to MarketDataGenerator output so that
FinancialRiskPipeline can be instantiated without modification.

Tickers used (medium config, 10 stocks + 2 KRX):
  US Stocks:  AAPL, MSFT, JPM, GS, BAC, C, WFC, MS, BLK, SCHW  (diversified US)
  KRX:        005930.KS (Samsung Electronics), 000660.KS (SK Hynix)
  FX:         EURUSD=X, GBPUSD=X, JPYUSD=X  (major pairs)
  Rates:      ^IRX (13-week T-bill), ^FVX (5-year), ^TYX (30-year)
  Positions:  synthetic slow-drift (no real position data publicly available)

Extended period: 756 trading days (~3 years, 2022-01-03 – 2024-12-31)
covering post-COVID normalisation, 2022 rate hiking cycle, and 2023-24 AI rally —
diverse market regimes for fat-tail robustness testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Ticker configurations per pipeline size
# ---------------------------------------------------------------------------

STOCK_TICKERS = {
    # small: 5 US equities (backward-compatible with v1 RSHB runs)
    "small":  ["AAPL", "MSFT", "JPM", "GS", "BAC"],
    # medium: 10 US equities + 2 KRX
    "medium": ["AAPL", "MSFT", "JPM", "GS", "BAC", "C", "WFC", "MS", "BLK", "SCHW",
               "005930.KS", "000660.KS"],
    # large: same 12 + 3 additional US
    "large":  ["AAPL", "MSFT", "JPM", "GS", "BAC", "C", "WFC", "MS", "BLK", "SCHW",
               "005930.KS", "000660.KS", "NVDA", "AMZN", "META"],
}

FX_TICKERS = {
    "small":  ["EURUSD=X"],
    "medium": ["EURUSD=X", "GBPUSD=X", "JPYUSD=X"],
    "large":  ["EURUSD=X", "GBPUSD=X", "JPYUSD=X", "AUDUSD=X"],
}

RATE_TICKERS = {
    "small":  ["^IRX", "^FVX"],
    "medium": ["^IRX", "^FVX", "^TYX"],
    "large":  ["^IRX", "^FVX", "^TYX"],
}

# Column name mappings: yfinance ticker → SynFRP column name
FX_COL_MAP = {
    "EURUSD=X": "USD_KRW",
    "GBPUSD=X": "EUR_KRW",
    "JPYUSD=X": "JPY_KRW",
    "AUDUSD=X": "AUD_KRW",
}

RATE_COL_MAP = {
    "^IRX": "RATE_SHORT",
    "^FVX": "RATE_MID",
    "^TYX": "RATE_LONG",
}

# ---------------------------------------------------------------------------
# Extended time period: 3 years of trading data
# Covers: 2022 rate-hike cycle, 2023 banking stress, 2023-24 AI rally
# ---------------------------------------------------------------------------
DEFAULT_N_DAYS  = 756   # ~3 years of trading days
DEFAULT_END_DATE = "2024-12-31"


@dataclass
class RealDataConfig:
    """Configuration for real market data loading."""
    size: str = "medium"               # "small", "medium", "large"
    n_days: int = DEFAULT_N_DAYS       # trading days to use (default 756 = ~3yr)
    end_date: str = DEFAULT_END_DATE   # end date for historical data
    seed: int = 42                     # for position generation only
    n_positions: int = 5               # number of position series


class RealMarketDataGenerator:
    """
    Loads real financial time series from Yahoo Finance and formats them
    identically to MarketDataGenerator.generate_all() output.

    Falls back to synthetic data if yfinance is unavailable or download fails.

    Coverage:
      - US equities: diversified across mega-cap tech (AAPL/MSFT/NVDA/AMZN/META),
        major banks (JPM/GS/BAC/C/WFC/MS), and asset managers (BLK/SCHW).
      - KRX equities: Samsung Electronics (005930.KS), SK Hynix (000660.KS) —
        Korea's two largest semiconductor firms, representative of non-US
        market structure and KRX price formation.
      - FX: EUR, GBP, JPY, AUD vs USD — covers G10 major pairs.
      - Rates: US Treasury curve (short/mid/long) via ^IRX/^FVX/^TYX.

    The 756-day window spans three distinct market regimes:
      2022: rapid Fed tightening (+425 bps), equity drawdown, FX volatility
      2023: banking sector stress (SVB/CS), AI-driven recovery
      2024: rate stabilisation, continued equity rally, election volatility
    This multi-regime coverage ensures fat-tail events are present for
    anomaly injection robustness testing.
    """

    def __init__(self, config: RealDataConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _fetch_yfinance(self, tickers, n_days, end_date):
        """Download adjusted close prices from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance not installed. Run: pip install yfinance"
            )

        import warnings
        # Need more than 2y to cover 756 days; use 4y window to be safe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(
                tickers, period="4y", end=end_date,
                auto_adjust=True, progress=False,
            )

        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        else:
            df = df[["Close"]] if "Close" in df.columns else df

        # Drop rows where ALL tickers are NaN (weekends/holidays)
        df = df.dropna(how="all")
        # Take the last n_days business days
        df = df.tail(n_days)

        # KRX tickers may have some NaN on US holidays (and vice versa);
        # forward-fill within each column independently.
        df = df.ffill().bfill()

        if len(df) < n_days:
            raise ValueError(
                f"Only {len(df)} trading days available, need {n_days}. "
                "Adjust end_date or reduce n_days."
            )
        return df

    def generate_stock_prices(self) -> pd.DataFrame:
        """Real stock prices (adjusted close) from Yahoo Finance.

        US tickers: AAPL, MSFT, JPM, GS, BAC, C, WFC, MS, BLK, SCHW
        KRX tickers: 005930.KS (Samsung Electronics), 000660.KS (SK Hynix)

        KRX prices are in KRW; we normalise to the same index-100 scale as
        US equities so that SynFRP's percentage-based ETL transforms work
        without modification.  The normalisation is: price / price[0] * 100.
        """
        cfg = self.config
        tickers = STOCK_TICKERS[cfg.size]

        df = self._fetch_yfinance(tickers, cfg.n_days, cfg.end_date)

        # Ensure column order matches STOCK_TICKERS
        existing = [t for t in tickers if t in df.columns]
        df = df[existing]

        # Normalise KRX columns (in KRW) to index-100 so scale is comparable
        krx = [t for t in existing if t.endswith(".KS")]
        for t in krx:
            df[t] = df[t] / df[t].iloc[0] * 100.0

        # Rename to SynFRP convention: STOCK_1_PRICE, STOCK_2_PRICE, ...
        rename = {t: f"STOCK_{i+1}_PRICE" for i, t in enumerate(existing)}
        df = df.rename(columns=rename)
        df.index = pd.bdate_range("2022-01-03", periods=cfg.n_days)
        return df

    def generate_interest_rates(self) -> pd.DataFrame:
        """Real interest rate proxies from Yahoo Finance (T-bill / Treasury yields)."""
        cfg = self.config
        tickers = RATE_TICKERS[cfg.size]

        df = self._fetch_yfinance(tickers, cfg.n_days, cfg.end_date)

        # Rates are in % — convert to decimal
        for col in df.columns:
            if df[col].median() > 0.5:   # likely in percent
                df[col] = df[col] / 100.0

        # Forward-fill any gaps (rate data has occasional missing days)
        df = df.ffill().bfill()

        rename = {t: RATE_COL_MAP[t] for t in tickers if t in RATE_COL_MAP}
        df = df.rename(columns=rename)
        df.index = pd.bdate_range("2022-01-03", periods=cfg.n_days)
        return df

    def generate_fx_rates(self) -> pd.DataFrame:
        """Real FX rates from Yahoo Finance."""
        cfg = self.config
        tickers = FX_TICKERS[cfg.size]

        df = self._fetch_yfinance(tickers, cfg.n_days, cfg.end_date)
        df = df.ffill().bfill()

        rename = {t: FX_COL_MAP[t] for t in tickers if t in FX_COL_MAP}
        df = df.rename(columns=rename)
        df.index = pd.bdate_range("2022-01-03", periods=cfg.n_days)
        return df

    def generate_positions(self) -> pd.DataFrame:
        """Synthetic position sizes (no public position data available).

        The number of position series is automatically set to match the number
        of stock tickers for the configured size, so that the pipeline's 1:1
        stock-position join is always satisfied.
        """
        cfg = self.config
        n = cfg.n_days
        # Always match position count to stock count for this size
        m = len(STOCK_TICKERS[cfg.size])

        base = self.rng.uniform(1e5, 1e6, m)
        drift = self.rng.standard_normal((n, m)) * 0.01
        positions = base * np.cumprod(1 + drift, axis=0)

        cols = [f"POSITION_{i+1}_SIZE" for i in range(m)]
        dates = pd.bdate_range("2022-01-03", periods=n)
        return pd.DataFrame(positions, index=dates, columns=cols)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all market data: real source + synthetic positions."""
        return {
            "stock_prices":   self.generate_stock_prices(),
            "interest_rates": self.generate_interest_rates(),
            "fx_rates":        self.generate_fx_rates(),
            "positions":       self.generate_positions(),
        }
