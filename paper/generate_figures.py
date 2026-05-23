"""
FINRCA Figure Generator — NeurIPS / arXiv style
Produces publication-quality PDF + PNG figures from experiment results.

Figure 1: Top-5 Accuracy by Anomaly Type × Method (grouped bar, medium pipeline)
Figure 2: Ablation Study — Top-K accuracy + MRR by variant
Figure 3: Scalability — accuracy vs size (bar) + latency vs nodes (line)
Figure 4: Overall Top-K comparison across all 4 methods (all 120 runs)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "experiments" / "results"
OUT     = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ── NeurIPS / arXiv style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "pdf.fonttype":     42,   # embed TrueType fonts
    "ps.fonttype":      42,
})

# ── Color palette (colorblind-friendly) ───────────────────────────────────────
METHOD_COLORS = {
    "APA-RCA-v2":     "#1A237E",   # dark navy (primary proposed)
    "APA-RCA":        "#2166AC",   # blue (v1)
    "AnomalyOnly":    "#E07B54",   # terracotta
    "BFS":            "#5BAD6F",   # green
    "Vanilla-RWR":    "#AAAAAA",   # gray
    "Structural-RWR": "#C62828",   # red
    "PC+RWR":         "#6A1B9A",   # purple
}
METHOD_ORDER = ["APA-RCA-v2", "APA-RCA", "AnomalyOnly", "BFS", "Vanilla-RWR",
                "Structural-RWR", "PC+RWR"]
METHOD_LABELS = {
    "APA-RCA-v2":     "APA-RCA v2 (Proposed)",
    "APA-RCA":        "APA-RCA v1",
    "AnomalyOnly":    "Anomaly-Score Only",
    "BFS":            "BFS",
    "Vanilla-RWR":    "Vanilla RWR",
    "Structural-RWR": "Structural-RWR",
    "PC+RWR":         "PC+RWR",
}

ATYPE_LABELS = {
    "A1_source_corruption":  "A1\nSource\nCorruption",
    "A2_etl_logic_error":    "A2\nETL\nError",
    "A3_feature_calc_error": "A3\nFeature\nCalc Error",
    "A4_aggregation_error":  "A4\nAggregation\nError",
    "A5_downstream_masking": "A5\nDownstream\nMasking",
    "A6_multi_source":       "A6\nMulti-Source\nAmbiguity",
    "A7_coincidental":       "A7\nCoincidental\nAnomaly",
}
ATYPE_ORDER = [
    "A1_source_corruption",
    "A2_etl_logic_error",
    "A3_feature_calc_error",
    "A4_aggregation_error",
    "A5_downstream_masking",
    "A6_multi_source",
    "A7_coincidental",
]

def _save(fig, stem):
    """Save figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 1: Top-5 Accuracy by Anomaly Type ─────────────────────────────────
def fig_anomaly_type_breakdown():
    df = pd.read_csv(RESULTS / "main_results.csv")
    medium = df[df["pipeline_size"] == "medium"]

    pivot = (medium.groupby(["method", "anomaly_type"])["top5"]
             .mean()
             .reset_index()
             .pivot(index="anomaly_type", columns="method", values="top5"))

    fig, ax = plt.subplots(figsize=(9, 4.0))

    n_groups  = len(ATYPE_ORDER)
    n_methods = len(METHOD_ORDER)
    bar_w = 0.11
    x = np.arange(n_groups)

    for i, method in enumerate(METHOD_ORDER):
        if method not in pivot.columns:
            continue
        vals = [pivot.loc[at, method] if at in pivot.index else 0.0
                for at in ATYPE_ORDER]
        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.0%}", ha="center", va="bottom",
                        fontsize=7, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([ATYPE_LABELS[a] for a in ATYPE_ORDER],
                       linespacing=1.25)
    ax.set_ylabel("Top-5 Accuracy")
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Top-5 Accuracy by Anomaly Type  (medium pipeline, 30 trials each)",
                 pad=8, fontsize=9)
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    _save(fig, "fig1_anomaly_type_breakdown")


# ── Figure 2: Ablation Study ──────────────────────────────────────────────────
def fig_ablation():
    df = pd.read_csv(RESULTS / "ablation_results.csv")

    ABLATION_ORDER = [
        "APA-RCA (Full)",
        "w/o Transform Weight",
        "w/o Attenuation",
        "w/o Anomaly Score",
    ]
    ABLATION_COLORS = ["#2166AC", "#5BAD6F", "#E07B54", "#AAAAAA"]
    ABLATION_XLABELS = {
        "APA-RCA (Full)":       "Full\nAPA-RCA",
        "w/o Transform Weight": "w/o\nTransform\nWeight",
        "w/o Attenuation":      "w/o\nAttenuation",
        "w/o Anomaly Score":    "w/o\nAnomaly\nScore",
    }

    agg = (df.groupby("method")
           .agg(top1=("top1", "mean"), top3=("top3", "mean"),
                top5=("top5", "mean"), mrr=("reciprocal_rank", "mean"))
           .reset_index())
    methods_present = [m for m in ABLATION_ORDER if m in agg["method"].values]
    x = np.arange(len(methods_present))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    # Left: Top-1 / Top-3 / Top-5 grouped bars
    bar_w = 0.22
    metric_styles = [
        ("top1", "Top-1", 0.50),
        ("top3", "Top-3", 0.75),
        ("top5", "Top-5", 1.00),
    ]
    for j, (metric, label, alpha) in enumerate(metric_styles):
        vals = [float(agg.loc[agg["method"] == m, metric].iloc[0])
                if m in agg["method"].values else 0.0
                for m in methods_present]
        offset = (j - 1) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=label,
                      color="#2166AC", alpha=alpha,
                      edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([ABLATION_XLABELS.get(m, m) for m in methods_present],
                       linespacing=1.2)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("(a)  Top-K Accuracy by Ablation Variant", pad=6)
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")
    ax.set_axisbelow(True)

    # Right: MRR bar
    mrr_vals = [float(agg.loc[agg["method"] == m, "mrr"].iloc[0])
                if m in agg["method"].values else 0.0
                for m in methods_present]
    bars2 = ax2.bar(x, mrr_vals, 0.48,
                    color=ABLATION_COLORS[:len(methods_present)],
                    edgecolor="white", linewidth=0.4, zorder=3)
    for bar, v in zip(bars2, mrr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([ABLATION_XLABELS.get(m, m) for m in methods_present],
                        linespacing=1.2)
    ax2.set_ylabel("MRR (Mean Reciprocal Rank)")
    ax2.set_ylim(0, max(mrr_vals) * 1.35 + 0.04)
    ax2.set_title("(b)  MRR by Ablation Variant", pad=6)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Figure 2: Ablation Study — Contribution of Each APA-RCA Component"
        "  (medium pipeline, 210 runs per variant)",
        y=1.02, fontsize=9,
    )
    fig.tight_layout(pad=0.8)
    _save(fig, "fig2_ablation")


# ── Figure 3: Scalability ─────────────────────────────────────────────────────
def fig_scalability():
    # Use main_results.csv APA-RCA-v2 data (70 runs per size, all 7 anomaly types)
    df_main = pd.read_csv(RESULTS / "main_results.csv")
    df = df_main[df_main["method"] == "APA-RCA-v2"].copy()

    SIZE_ORDER  = ["small", "medium", "large"]
    SIZE_LABELS = ["Small\n(~45 nodes)", "Medium\n(~63 nodes)", "Large\n(~100+ nodes)"]
    SIZE_NODES  = [45, 63, 100]

    agg = (df.groupby("pipeline_size")
           .agg(top5=("top5", "mean"), mrr=("reciprocal_rank", "mean"),
                latency=("elapsed_ms", "mean"), latency_max=("elapsed_ms", "max"))
           .reindex(SIZE_ORDER).reset_index())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))

    # Left: Accuracy vs pipeline size
    x = np.arange(len(SIZE_ORDER))
    bar_w = 0.32
    b1 = ax1.bar(x - bar_w / 2, agg["top5"], bar_w, label="Top-5",
                 color="#2166AC", alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)
    b2 = ax1.bar(x + bar_w / 2, agg["mrr"],  bar_w, label="MRR",
                 color="#5BAD6F", alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)

    for bar, v in zip(b1, agg["top5"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.010,
                 f"{v:.0%}", ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(b2, agg["mrr"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(SIZE_LABELS, linespacing=1.2)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 0.95)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_title("(a)  Accuracy vs Pipeline Size", pad=6)
    ax1.legend(loc="lower right", framealpha=0.92, edgecolor="#cccccc")
    ax1.set_axisbelow(True)

    # Right: Latency vs node count
    ax2.plot(SIZE_NODES, agg["latency"], "o-", color="#2166AC",
             linewidth=2, markersize=6, label="Avg Latency", zorder=4)
    ax2.fill_between(SIZE_NODES,
                     agg["latency"] * 0.75, agg["latency_max"],
                     alpha=0.14, color="#2166AC", label="Avg–Max range")
    ax2.axhline(2.0, color="#D62728", linestyle="--", linewidth=1.1,
                label="2 ms SLA", zorder=3)

    for xn, yv in zip(SIZE_NODES, agg["latency"]):
        ax2.annotate(f"{yv:.2f} ms", (xn, yv),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8, color="#2166AC")

    ax2.set_xlabel("Approximate Node Count")
    ax2.set_ylabel("Query Latency (ms)")
    ax2.set_ylim(0, 2.6)
    ax2.set_xticks(SIZE_NODES)
    ax2.set_title("(b)  Query Latency vs Pipeline Size", pad=6)
    ax2.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Figure 3: Scalability Analysis — APA-RCA v2  (210 runs per pipeline size, all 7 anomaly types)",
        y=1.02, fontsize=9,
    )
    fig.tight_layout(pad=0.8)
    _save(fig, "fig3_scalability")


# ── Figure 4: Overall Top-K comparison ───────────────────────────────────────
def fig_overall_comparison():
    df = pd.read_csv(RESULTS / "main_results.csv")

    agg = (df.groupby("method")
           .agg(top1=("top1", "mean"), top3=("top3", "mean"),
                top5=("top5", "mean"), mrr=("reciprocal_rank", "mean"))
           .reset_index())

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(METHOD_ORDER))
    bar_w = 0.21

    metric_styles = [
        ("top1", "Top-1", 0.45),
        ("top3", "Top-3", 0.72),
        ("top5", "Top-5", 1.00),
    ]
    for j, (metric, label, alpha) in enumerate(metric_styles):
        vals = [float(agg.loc[agg["method"] == m, metric].iloc[0])
                if m in agg["method"].values else 0.0
                for m in METHOD_ORDER]
        offset = (j - 1) * bar_w
        color = "#1A237E" if j == 0 else "#2166AC" if j == 1 else "#4A90D9"
        bars = ax.bar(x + offset, vals, bar_w, label=label,
                      color=color, alpha=min(alpha + 0.1, 1.0),
                      edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            if v >= 0.012:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.007,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    # Subtle background highlight for APA-RCA v2 column
    ax.axvspan(-0.42, 0.42, alpha=0.055, color="#1A237E", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Accuracy  (all 210 runs)")
    ax.set_ylim(0, 0.95)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        "Figure 4: Overall Top-K Accuracy Comparison"
        "  (210 runs, 7 anomaly types × 3 pipeline sizes)",
        pad=8, fontsize=9,
    )
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    _save(fig, "fig4_overall_comparison")


if __name__ == "__main__":
    print("Generating FINRCA figures (NeurIPS style)...")
    fig_anomaly_type_breakdown()
    fig_ablation()
    fig_scalability()
    fig_overall_comparison()
    print(f"\nDone. Figures saved to: {OUT}")
