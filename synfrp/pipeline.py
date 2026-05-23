"""
SynFRP: Synthetic Financial Risk Pipeline
Defines the 5-layer risk calculation pipeline with explicit data lineage metadata.
Each node records: inputs, outputs, transform_type, and computation logic.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum


class TransformType(str, Enum):
    SOURCE      = "source"
    DIRECT_MAP  = "direct_map"
    FILTER      = "filter"
    CALCULATE   = "calculate"
    AGGREGATE   = "aggregate"
    REPORT      = "report"


@dataclass
class PipelineNode:
    """
    Represents a single node in the financial risk pipeline.
    Captures lineage metadata + computation function.
    """
    node_id: str
    name: str
    layer: int                        # 1=source, 2=ETL, 3=feature, 4=risk, 5=report
    transform_type: TransformType
    input_columns: List[str]          # upstream column references
    output_columns: List[str]         # columns this node produces
    upstream_nodes: List[str]         # node_ids this node depends on
    compute_fn: Optional[Callable]    = field(default=None, repr=False)
    description: str                  = ""
    expected_input_count: Optional[int] = None  # schema constraint: expected # of upstream inputs

    def compute(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Execute this node's computation given upstream data."""
        if self.compute_fn is None:
            raise NotImplementedError(f"No compute_fn for node {self.node_id}")
        return self.compute_fn(data)


class FinancialRiskPipeline:
    """
    5-layer synthetic financial risk pipeline.
    Layer 1: Raw Data Sources
    Layer 2: ETL (cleaning, normalization, joining)
    Layer 3: Feature Engineering (returns, vol, correlation)
    Layer 4: Risk Calculation (VaR, ES, stress)
    Layer 5: Reporting
    """

    def __init__(self, raw_data: Dict[str, pd.DataFrame], size: str = "medium"):
        """
        size: 'small' (~50 nodes), 'medium' (~100 nodes), 'large' (~200 nodes)
        """
        self.raw_data = raw_data
        self.size = size
        self.nodes: Dict[str, PipelineNode] = {}
        self.node_data: Dict[str, Dict[str, pd.Series]] = {}  # node_id -> col -> series
        self._build_pipeline()

    def _register(self, node: PipelineNode):
        self.nodes[node.node_id] = node

    def _build_pipeline(self):
        stocks   = self.raw_data["stock_prices"]
        rates    = self.raw_data["interest_rates"]
        fx       = self.raw_data["fx_rates"]
        positions = self.raw_data["positions"]

        stock_cols    = list(stocks.columns)
        rate_cols     = list(rates.columns)
        fx_cols       = list(fx.columns)
        pos_cols      = list(positions.columns)

        # ── Layer 1: Source Nodes ───────────────────────────────────────────
        for col in stock_cols:
            node_id = f"SRC_{col}"
            series = stocks[col]
            self._register(PipelineNode(
                node_id=node_id, name=f"Source: {col}", layer=1,
                transform_type=TransformType.SOURCE,
                input_columns=[], output_columns=[col],
                upstream_nodes=[],
                compute_fn=lambda d, s=series, c=col: {c: s},
                description=f"Raw stock price series for {col}"
            ))

        for col in rate_cols:
            node_id = f"SRC_{col}"
            series = rates[col]
            self._register(PipelineNode(
                node_id=node_id, name=f"Source: {col}", layer=1,
                transform_type=TransformType.SOURCE,
                input_columns=[], output_columns=[col],
                upstream_nodes=[],
                compute_fn=lambda d, s=series, c=col: {c: s},
                description=f"Raw interest rate series for {col}"
            ))

        for col in fx_cols:
            node_id = f"SRC_{col}"
            series = fx[col]
            self._register(PipelineNode(
                node_id=node_id, name=f"Source: {col}", layer=1,
                transform_type=TransformType.SOURCE,
                input_columns=[], output_columns=[col],
                upstream_nodes=[],
                compute_fn=lambda d, s=series, c=col: {c: s},
                description=f"Raw FX rate series for {col}"
            ))

        for col in pos_cols:
            node_id = f"SRC_{col}"
            series = positions[col]
            self._register(PipelineNode(
                node_id=node_id, name=f"Source: {col}", layer=1,
                transform_type=TransformType.SOURCE,
                input_columns=[], output_columns=[col],
                upstream_nodes=[],
                compute_fn=lambda d, s=series, c=col: {c: s},
                description=f"Raw position size for {col}"
            ))

        # ── Layer 2: ETL ────────────────────────────────────────────────────
        # 2a. Missing value imputation (forward fill) for each stock
        for col in stock_cols:
            out_col = f"{col}_CLEAN"
            src_id  = f"SRC_{col}"
            self._register(PipelineNode(
                node_id=f"ETL_IMPUTE_{col}", name=f"Impute: {col}", layer=2,
                transform_type=TransformType.FILTER,
                input_columns=[col], output_columns=[out_col],
                upstream_nodes=[src_id],
                compute_fn=lambda d, c=col, oc=out_col: {oc: d[c].ffill()},
                description="Forward-fill missing values"
            ))

        # 2b. Outlier clipping (z-score > 4 → clip)
        # Uses historical (first-60-day) mean/std as clip boundaries so that
        # a spike injected later is genuinely absorbed — not just included in
        # the boundary calculation.  This enables A5 (Downstream Masking).
        for col in stock_cols:
            in_col  = f"{col}_CLEAN"
            out_col = f"{col}_FILTERED"
            up_id   = f"ETL_IMPUTE_{col}"
            def _clip(d, c=in_col, oc=out_col, warmup=60):
                s = d[c]
                # Estimate distribution from historical warmup window only
                ref = s.iloc[:warmup]
                mu, sigma = ref.mean(), ref.std()
                if sigma < 1e-8:
                    sigma = 1e-8
                return {oc: s.clip(mu - 4*sigma, mu + 4*sigma)}
            self._register(PipelineNode(
                node_id=f"ETL_CLIP_{col}", name=f"Clip outliers: {col}", layer=2,
                transform_type=TransformType.FILTER,
                input_columns=[in_col], output_columns=[out_col],
                upstream_nodes=[up_id],
                compute_fn=_clip,
                description="Clip outliers beyond 4 std deviations (historical boundary)"
            ))

        # 2c. FX conversion: convert stock prices to KRW using USD_KRW
        fx_col = "USD_KRW"
        for col in stock_cols:
            in_col  = f"{col}_FILTERED"
            out_col = f"{col}_KRW"
            up_ids  = [f"ETL_CLIP_{col}", "SRC_USD_KRW"]
            def _fx(d, sc=in_col, oc=out_col, fc=fx_col):
                return {oc: d[sc] * d[fc]}
            self._register(PipelineNode(
                node_id=f"ETL_FX_{col}", name=f"FX convert: {col}", layer=2,
                transform_type=TransformType.CALCULATE,
                input_columns=[in_col, fx_col], output_columns=[out_col],
                upstream_nodes=up_ids,
                compute_fn=_fx,
                description="Convert USD stock price to KRW"
            ))

        # 2d. Join price + position
        for i, (scol, pcol) in enumerate(zip(stock_cols, pos_cols)):
            price_col = f"{scol}_KRW"
            out_col   = f"PORTFOLIO_{i+1}_VALUE"
            up_ids    = [f"ETL_FX_{scol}", f"SRC_{pcol}"]
            def _join(d, pc=price_col, pos=pcol, oc=out_col):
                return {oc: d[pc] * d[pos]}
            self._register(PipelineNode(
                node_id=f"ETL_JOIN_{i+1}", name=f"Join portfolio {i+1}", layer=2,
                transform_type=TransformType.CALCULATE,
                input_columns=[price_col, pcol], output_columns=[out_col],
                upstream_nodes=up_ids,
                compute_fn=_join,
                description="Compute portfolio position value (price × size)"
            ))

        # ── Layer 3: Feature Engineering ────────────────────────────────────
        # 3a. Daily log returns for each stock
        for col in stock_cols:
            in_col  = f"{col}_FILTERED"
            out_col = f"{col}_RETURN"
            up_id   = f"ETL_CLIP_{col}"
            def _ret(d, c=in_col, oc=out_col):
                return {oc: np.log(d[c]).diff().fillna(0)}
            self._register(PipelineNode(
                node_id=f"FEAT_RETURN_{col}", name=f"Returns: {col}", layer=3,
                transform_type=TransformType.CALCULATE,
                input_columns=[in_col], output_columns=[out_col],
                upstream_nodes=[up_id],
                compute_fn=_ret,
                description="Daily log returns"
            ))

        # 3b. Rolling 20d volatility
        for col in stock_cols:
            in_col  = f"{col}_RETURN"
            out_col = f"{col}_VOL20"
            up_id   = f"FEAT_RETURN_{col}"
            def _vol(d, c=in_col, oc=out_col):
                s = d[c].rolling(20).std()
                return {oc: s.bfill().ffill().fillna(0)}
            self._register(PipelineNode(
                node_id=f"FEAT_VOL20_{col}", name=f"Vol20: {col}", layer=3,
                transform_type=TransformType.AGGREGATE,
                input_columns=[in_col], output_columns=[out_col],
                upstream_nodes=[up_id],
                compute_fn=_vol,
                description="20-day rolling volatility"
            ))

        # 3c. Rolling 60d volatility
        for col in stock_cols:
            in_col  = f"{col}_RETURN"
            out_col = f"{col}_VOL60"
            up_id   = f"FEAT_RETURN_{col}"
            def _vol60(d, c=in_col, oc=out_col):
                s = d[c].rolling(60).std()
                return {oc: s.bfill().ffill().fillna(0)}
            self._register(PipelineNode(
                node_id=f"FEAT_VOL60_{col}", name=f"Vol60: {col}", layer=3,
                transform_type=TransformType.AGGREGATE,
                input_columns=[in_col], output_columns=[out_col],
                upstream_nodes=[up_id],
                compute_fn=_vol60,
                description="60-day rolling volatility"
            ))

        # 3d. Portfolio total value aggregation
        pv_cols = [f"PORTFOLIO_{i+1}_VALUE" for i in range(len(stock_cols))]
        up_ids  = [f"ETL_JOIN_{i+1}" for i in range(len(stock_cols))]
        def _total(d, cols=pv_cols):
            s = sum(d[c] for c in cols)
            return {"PORTFOLIO_TOTAL_VALUE": s}
        self._register(PipelineNode(
            node_id="FEAT_TOTAL_VALUE", name="Total portfolio value", layer=3,
            transform_type=TransformType.AGGREGATE,
            input_columns=pv_cols, output_columns=["PORTFOLIO_TOTAL_VALUE"],
            upstream_nodes=up_ids,
            compute_fn=_total,
            description="Sum of all portfolio position values"
        ))

        # 3e. Sector-level aggregation
        # Sector A = first half of stocks, Sector B = second half (may be empty for small)
        half       = max(1, len(stock_cols) // 2)
        sectorA    = [f"{c}_RETURN" for c in stock_cols[:half]]
        sectorB    = [f"{c}_RETURN" for c in stock_cols[half:]] or sectorA  # fallback
        upA = [f"FEAT_RETURN_{c}" for c in stock_cols[:half]]
        upB = [f"FEAT_RETURN_{c}" for c in stock_cols[half:]] or upA

        def _sector(d, cols=sectorA, oc="SECTOR_A_RETURN"):
            return {oc: sum(d[c] for c in cols) / len(cols)}
        self._register(PipelineNode(
            node_id="FEAT_SECTOR_A", name="Sector A avg return", layer=3,
            transform_type=TransformType.AGGREGATE,
            input_columns=sectorA, output_columns=["SECTOR_A_RETURN"],
            upstream_nodes=upA,
            compute_fn=_sector,
            description="Sector A average return (first half of stocks)",
            expected_input_count=len(sectorA),
        ))

        def _sectorB(d, cols=sectorB, oc="SECTOR_B_RETURN"):
            return {oc: sum(d[c] for c in cols) / len(cols)}
        self._register(PipelineNode(
            node_id="FEAT_SECTOR_B", name="Sector B avg return", layer=3,
            transform_type=TransformType.AGGREGATE,
            input_columns=sectorB, output_columns=["SECTOR_B_RETURN"],
            upstream_nodes=upB,
            compute_fn=_sectorB,
            description="Sector B average return (second half of stocks)",
            expected_input_count=len(sectorB),
        ))

        # ── Layer 4: Risk Calculation ────────────────────────────────────────
        ret_cols = [f"{c}_RETURN" for c in stock_cols]
        vol_cols = [f"{c}_VOL20" for c in stock_cols]
        ret_ups  = [f"FEAT_RETURN_{c}" for c in stock_cols]
        vol_ups  = [f"FEAT_VOL20_{c}" for c in stock_cols]

        # 4a. Historical VaR 95%
        def _var95(d, cols=ret_cols):
            portfolio_ret = sum(d[c] for c in cols) / len(cols)
            var95 = portfolio_ret.rolling(60).quantile(0.05)
            return {"VAR_95": var95.fillna(0)}
        self._register(PipelineNode(
            node_id="RISK_VAR95", name="Historical VaR 95%", layer=4,
            transform_type=TransformType.AGGREGATE,
            input_columns=ret_cols, output_columns=["VAR_95"],
            upstream_nodes=ret_ups,
            compute_fn=_var95,
            description="60-day rolling historical VaR at 95% confidence"
        ))

        # 4b. Historical VaR 99%
        def _var99(d, cols=ret_cols):
            portfolio_ret = sum(d[c] for c in cols) / len(cols)
            var99 = portfolio_ret.rolling(60).quantile(0.01)
            return {"VAR_99": var99.fillna(0)}
        self._register(PipelineNode(
            node_id="RISK_VAR99", name="Historical VaR 99%", layer=4,
            transform_type=TransformType.AGGREGATE,
            input_columns=ret_cols, output_columns=["VAR_99"],
            upstream_nodes=ret_ups,
            compute_fn=_var99,
            description="60-day rolling historical VaR at 99% confidence"
        ))

        # 4c. Expected Shortfall (CVaR 95%)
        def _es95(d, cols=ret_cols):
            portfolio_ret = sum(d[c] for c in cols) / len(cols)
            def rolling_es(x):
                cutoff = np.percentile(x, 5)
                tail   = x[x <= cutoff]
                return tail.mean() if len(tail) > 0 else cutoff
            es = portfolio_ret.rolling(60).apply(rolling_es, raw=True)
            return {"ES_95": es.fillna(0)}
        self._register(PipelineNode(
            node_id="RISK_ES95", name="Expected Shortfall 95%", layer=4,
            transform_type=TransformType.AGGREGATE,
            input_columns=ret_cols, output_columns=["ES_95"],
            upstream_nodes=ret_ups,
            compute_fn=_es95,
            description="60-day rolling Expected Shortfall at 95% confidence"
        ))

        # 4d. Portfolio volatility (mean vol across stocks)
        def _pvol(d, cols=vol_cols):
            return {"PORTFOLIO_VOL": sum(d[c] for c in cols) / len(cols)}
        self._register(PipelineNode(
            node_id="RISK_PORTFOLIO_VOL", name="Portfolio volatility", layer=4,
            transform_type=TransformType.AGGREGATE,
            input_columns=vol_cols, output_columns=["PORTFOLIO_VOL"],
            upstream_nodes=vol_ups,
            compute_fn=_pvol,
            description="Average 20-day rolling volatility across stocks"
        ))

        # 4e. Stress scenario P&L (shock all prices by -20%)
        def _stress(d, cols=ret_cols):
            portfolio_ret = sum(d[c] for c in cols) / len(cols)
            stress_pnl    = portfolio_ret * (-0.20 / max(float(portfolio_ret.std()), 1e-8))
            return {"STRESS_PNL": stress_pnl.fillna(0)}
        self._register(PipelineNode(
            node_id="RISK_STRESS", name="Stress test P&L", layer=4,
            transform_type=TransformType.CALCULATE,
            input_columns=ret_cols, output_columns=["STRESS_PNL"],
            upstream_nodes=ret_ups,
            compute_fn=_stress,
            description="Simulated P&L under -20% stress scenario"
        ))

        # 4f. Concentration risk (max position / total value)
        pv_cols2 = [f"PORTFOLIO_{i+1}_VALUE" for i in range(len(stock_cols))]
        up_ids2  = [f"ETL_JOIN_{i+1}" for i in range(len(stock_cols))] + ["FEAT_TOTAL_VALUE"]
        def _conc(d, cols=pv_cols2):
            total = sum(d[c] for c in cols)
            max_pos = pd.concat([d[c] for c in cols], axis=1).max(axis=1)
            return {"CONCENTRATION_RISK": (max_pos / total.clip(1e-8))}
        self._register(PipelineNode(
            node_id="RISK_CONCENTRATION", name="Concentration risk", layer=4,
            transform_type=TransformType.CALCULATE,
            input_columns=pv_cols2 + ["PORTFOLIO_TOTAL_VALUE"],
            output_columns=["CONCENTRATION_RISK"],
            upstream_nodes=up_ids2,
            compute_fn=_conc,
            description="Max single position as fraction of portfolio"
        ))

        # ── Layer 5: Reporting ───────────────────────────────────────────────
        # 5a. Regulatory VaR report
        def _reg_var(d):
            var_breach = (d["VAR_99"].abs() > d["VAR_99"].abs().quantile(0.95)).astype(float)
            return {"REG_VAR_REPORT": var_breach}
        self._register(PipelineNode(
            node_id="REPORT_REG_VAR", name="Regulatory VaR Report", layer=5,
            transform_type=TransformType.REPORT,
            input_columns=["VAR_99"], output_columns=["REG_VAR_REPORT"],
            upstream_nodes=["RISK_VAR99"],
            compute_fn=_reg_var,
            description="Flag days where VaR99 exceeds 95th percentile threshold"
        ))

        # 5b. Internal risk dashboard
        def _dashboard(d):
            risk_score = (d["VAR_95"].abs() + d["ES_95"].abs() + d["PORTFOLIO_VOL"]) / 3
            return {"RISK_DASHBOARD_SCORE": risk_score}
        self._register(PipelineNode(
            node_id="REPORT_DASHBOARD", name="Internal Risk Dashboard", layer=5,
            transform_type=TransformType.REPORT,
            input_columns=["VAR_95", "ES_95", "PORTFOLIO_VOL"],
            output_columns=["RISK_DASHBOARD_SCORE"],
            upstream_nodes=["RISK_VAR95", "RISK_ES95", "RISK_PORTFOLIO_VOL"],
            compute_fn=_dashboard,
            description="Composite risk score for internal dashboard"
        ))

        # 5c. Stress test summary
        def _stress_summary(d):
            breach = (d["STRESS_PNL"].abs() > d["STRESS_PNL"].abs().quantile(0.90)).astype(float)
            return {"STRESS_SUMMARY": breach}
        self._register(PipelineNode(
            node_id="REPORT_STRESS", name="Stress Test Summary", layer=5,
            transform_type=TransformType.REPORT,
            input_columns=["STRESS_PNL"],
            output_columns=["STRESS_SUMMARY"],
            upstream_nodes=["RISK_STRESS"],
            compute_fn=_stress_summary,
            description="Flag days with extreme stress scenario losses"
        ))

        # 5d. Limit breach report (vol-adjusted threshold: tighter when vol is elevated)
        def _limit(d):
            vol_norm = d["PORTFOLIO_VOL"].clip(lower=0)
            vol_adj  = 0.40 - 0.10 * (vol_norm / (vol_norm.mean() + 1e-8)).clip(upper=1.0)
            breach   = (d["CONCENTRATION_RISK"] > vol_adj).astype(float)
            return {"LIMIT_BREACH_REPORT": breach}
        self._register(PipelineNode(
            node_id="REPORT_LIMIT", name="Limit Breach Report", layer=5,
            transform_type=TransformType.REPORT,
            input_columns=["CONCENTRATION_RISK", "PORTFOLIO_VOL"],
            output_columns=["LIMIT_BREACH_REPORT"],
            upstream_nodes=["RISK_CONCENTRATION", "RISK_PORTFOLIO_VOL"],
            compute_fn=_limit,
            description="Flag concentration risk exceeding vol-adjusted threshold"
        ))

    def get_topology(self) -> Dict[str, List[str]]:
        """Return adjacency list: node_id -> list of upstream node_ids."""
        return {nid: node.upstream_nodes for nid, node in self.nodes.items()}

    def get_report_nodes(self) -> List[str]:
        return [nid for nid, n in self.nodes.items() if n.layer == 5]

    def get_source_nodes(self) -> List[str]:
        return [nid for nid, n in self.nodes.items() if n.layer == 1]

    def execute(self) -> Dict[str, Dict[str, pd.Series]]:
        """
        Execute the full pipeline in topological order.
        Returns: {node_id: {col: series}}
        """
        import networkx as nx
        G = nx.DiGraph()
        for nid, node in self.nodes.items():
            G.add_node(nid)
            for up in node.upstream_nodes:
                G.add_edge(up, nid)

        order = list(nx.topological_sort(G))
        flat_data: Dict[str, pd.Series] = {}

        for nid in order:
            node = self.nodes[nid]
            result = node.compute(flat_data)
            self.node_data[nid] = result
            flat_data.update(result)

        return self.node_data

    def get_node_outputs(self) -> Dict[str, pd.Series]:
        """Flat dict of all column outputs after execution."""
        flat: Dict[str, pd.Series] = {}
        for nid, col_data in self.node_data.items():
            flat.update(col_data)
        return flat


if __name__ == "__main__":
    from data_generator import MarketDataConfig, MarketDataGenerator
    cfg = MarketDataConfig(seed=42)
    gen = MarketDataGenerator(cfg)
    raw = gen.generate_all()

    pipeline = FinancialRiskPipeline(raw, size="medium")
    print(f"Total nodes: {len(pipeline.nodes)}")
    node_data = pipeline.execute()
    print(f"Executed {len(node_data)} nodes")
    for layer in range(1, 6):
        layer_nodes = [n for n in pipeline.nodes.values() if n.layer == layer]
        print(f"  Layer {layer}: {len(layer_nodes)} nodes")
