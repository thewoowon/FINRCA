"""
FRLG Builder: Financial Risk Lineage Graph
Constructs a column-level directed acyclic graph (DAG) from the pipeline definition.

Node types: Source, Transform, Aggregate, Calculate, Report
Edge types: DirectMap, Aggregate, Calculate, Filter

The FRLG is the foundational data structure for APA-RCA.
Each node carries:
  - node metadata (layer, transform type)
  - anomaly score (populated by anomaly detection layer)
  - lineage edges (directed: upstream -> downstream)
"""

import networkx as nx
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from synfrp.pipeline import FinancialRiskPipeline, TransformType


@dataclass
class FRLGNode:
    """A node in the Financial Risk Lineage Graph."""
    node_id: str
    name: str
    layer: int
    transform_type: TransformType
    output_columns: List[str]
    input_columns: List[str]
    anomaly_score: float = 0.0        # populated by anomaly detector
    is_anomalous: bool   = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "layer": self.layer,
            "transform_type": self.transform_type.value,
            "output_columns": self.output_columns,
            "input_columns": self.input_columns,
            "anomaly_score": self.anomaly_score,
            "is_anomalous": self.is_anomalous,
        }


@dataclass
class FRLGEdge:
    """A directed edge in the FRLG: upstream_node -> downstream_node."""
    source_node_id: str
    target_node_id: str
    source_columns: List[str]   # columns from source used by target
    target_columns: List[str]   # columns in target that depend on source
    edge_type: TransformType    # transform type of the TARGET node

    @property
    def weight_prior(self) -> float:
        """
        Prior edge weight based on transform type.
        Aggregate > Calculate > DirectMap/Filter
        Used as beta in APA-RCA.
        """
        weights = {
            TransformType.AGGREGATE:  1.5,
            TransformType.CALCULATE:  1.2,
            TransformType.REPORT:     1.3,
            TransformType.DIRECT_MAP: 1.0,
            TransformType.FILTER:     0.8,
            TransformType.SOURCE:     1.0,
        }
        return weights.get(self.edge_type, 1.0)


class FRLGBuilder:
    """
    Builds a Financial Risk Lineage Graph from a FinancialRiskPipeline.
    The result is a NetworkX DiGraph augmented with FRLG metadata.
    """

    def __init__(self, pipeline: FinancialRiskPipeline):
        self.pipeline = pipeline
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[str, FRLGNode] = {}
        self.edges: Dict[Tuple[str, str], FRLGEdge] = {}

    def build(self) -> nx.DiGraph:
        """Construct the FRLG from the pipeline definition."""
        self._add_nodes()
        self._add_edges()
        return self.graph

    def _add_nodes(self):
        for nid, pnode in self.pipeline.nodes.items():
            frlg_node = FRLGNode(
                node_id=nid,
                name=pnode.name,
                layer=pnode.layer,
                transform_type=pnode.transform_type,
                output_columns=pnode.output_columns,
                input_columns=pnode.input_columns,
            )
            self.nodes[nid] = frlg_node
            self.graph.add_node(nid, **frlg_node.to_dict())

    def _add_edges(self):
        for nid, pnode in self.pipeline.nodes.items():
            for up_nid in pnode.upstream_nodes:
                if up_nid not in self.pipeline.nodes:
                    continue

                up_node = self.pipeline.nodes[up_nid]
                # Which upstream columns are consumed by this node?
                used_cols = [c for c in up_node.output_columns
                             if c in pnode.input_columns]

                edge = FRLGEdge(
                    source_node_id=up_nid,
                    target_node_id=nid,
                    source_columns=used_cols,
                    target_columns=pnode.output_columns,
                    edge_type=pnode.transform_type,
                )
                self.edges[(up_nid, nid)] = edge
                self.graph.add_edge(
                    up_nid, nid,
                    edge_type=pnode.transform_type.value,
                    weight_prior=edge.weight_prior,
                    source_columns=used_cols,
                    target_columns=pnode.output_columns,
                )

    # ── Anomaly Score Population ─────────────────────────────────────────────

    def set_anomaly_scores(self, scores: Dict[str, float]):
        """
        Populate anomaly scores on graph nodes.
        scores: {node_id: score in [0,1]}
        """
        for nid, score in scores.items():
            if nid in self.nodes:
                self.nodes[nid].anomaly_score = score
                self.nodes[nid].is_anomalous  = score > 0.5
                self.graph.nodes[nid]["anomaly_score"] = score
                self.graph.nodes[nid]["is_anomalous"]  = score > 0.5

    # ── Query Interfaces ─────────────────────────────────────────────────────

    def forward_query(self, node_id: str) -> List[str]:
        """
        Forward query: given a node, return all downstream nodes it can affect.
        Answers: "If this node has an error, what reports are impacted?"
        """
        if node_id not in self.graph:
            return []
        return list(nx.descendants(self.graph, node_id))

    def backward_query(self, node_id: str) -> List[str]:
        """
        Backward query: given a node, return all upstream nodes that feed into it.
        Answers: "What are all possible sources of error for this node?"
        """
        if node_id not in self.graph:
            return []
        return list(nx.ancestors(self.graph, node_id))

    def impact_query(self, node_id: str) -> Dict[str, List[str]]:
        """
        Impact query: which report nodes are downstream of this node?
        Returns dict with affected report nodes and their columns.
        """
        downstream = self.forward_query(node_id)
        report_nodes = [n for n in downstream
                        if self.graph.nodes[n].get("layer") == 5]
        return {
            "downstream_nodes": downstream,
            "affected_reports": report_nodes,
            "affected_columns": [
                col
                for rn in report_nodes
                for col in self.graph.nodes[rn].get("output_columns", [])
            ]
        }

    def get_propagation_paths(
        self, source: str, target: str
    ) -> List[List[str]]:
        """
        Return all simple paths from source to target in the FRLG.
        Used for trace completeness evaluation.
        """
        try:
            return list(nx.all_simple_paths(self.graph, source, target, cutoff=10))
        except nx.NetworkXNoPath:
            return []

    # ── Graph Statistics ─────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        G = self.graph
        return {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_layers": 5,
            "nodes_per_layer": {
                l: len([n for n in G.nodes if G.nodes[n].get("layer") == l])
                for l in range(1, 6)
            },
            "source_nodes": [n for n in G.nodes if G.nodes[n].get("layer") == 1],
            "report_nodes": [n for n in G.nodes if G.nodes[n].get("layer") == 5],
            "avg_in_degree": np.mean([d for _, d in G.in_degree()]),
            "avg_out_degree": np.mean([d for _, d in G.out_degree()]),
            "is_dag": nx.is_directed_acyclic_graph(G),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from synfrp import MarketDataConfig, MarketDataGenerator, FinancialRiskPipeline

    cfg = MarketDataConfig(seed=42)
    raw = MarketDataGenerator(cfg).generate_all()
    pipeline = FinancialRiskPipeline(raw)
    pipeline.execute()

    builder = FRLGBuilder(pipeline)
    G = builder.build()
    summary = builder.summary()

    print("=== FRLG Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n=== Forward Query from SRC_STOCK_1_PRICE ===")
    fwd = builder.forward_query("SRC_STOCK_1_PRICE")
    print(f"  Downstream nodes ({len(fwd)}): {fwd[:8]}...")

    print("\n=== Backward Query from REPORT_REG_VAR ===")
    bwd = builder.backward_query("REPORT_REG_VAR")
    print(f"  Upstream nodes ({len(bwd)}): {bwd[:8]}...")
