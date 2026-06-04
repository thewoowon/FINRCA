"""
Generate thesis figures 5, 6, 7 for ML-enhanced pipeline results.
Academic paper style (not slide style).

Output: paper/figures/fig5_rshb_ml.{pdf,png}
        paper/figures/fig6_ml_ablation.{pdf,png}
        paper/figures/fig7_scorer_comparison.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

GREY   = "#555555"
BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#4dac26"
GOLD   = "#f4a582"

def save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  saved → {path}")
    plt.close(fig)


# ── Figure 5: RSHB — 4방법 비교 (Top-1 / Top-3 / Top-5 / MRR) ───────────────

def fig5_rshb_ml():
    # 6 methods: baseline + ablations + v1 best + v2 + v3 (Table 6.5)
    methods = [
        "APA-RCA v2\n(Baseline)",
        "z-score +\nAPA + GNN",
        "VAE +\nAPA-RCA v2",
        "VAE+APA+GNN\n(v1)",
        "CVAE+APA\n+ResGCN (v2)",
        "HybridTSD+APA\n+ResGCN (v3)",
    ]
    top1 = [32.9, 31.4, 43.3, 54.8, 51.0, 49.5]
    top3 = [67.6, 49.5, 60.0, 65.7, 74.8, 80.0]
    top5 = [82.4, 68.6, 60.0, 71.9, 74.8, 83.3]
    mrr  = [0.503, 0.445, 0.533, 0.620, 0.626, 0.639]

    PURPLE = "#7b3294"

    x = np.arange(len(methods))
    w = 0.18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                    gridspec_kw={"width_ratios": [4, 1]})

    # ── Left: grouped bar Top-1/3/5 ──────────────────────────────────────────
    bar_colors_top = ["#aec6e8", "#6baed6", "#2166ac"]
    for i, (vals, label, c) in enumerate(zip(
            [top1, top3, top5],
            ["Top-1", "Top-3", "Top-5"],
            bar_colors_top)):
        bars = ax1.bar(x + (i - 1) * w, vals, w, label=label,
                       color=c, edgecolor="white", linewidth=0.6)
        if label == "Top-1":
            for bar, v in zip(bars, vals):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1.2, f"{v:.1f}%",
                         ha="center", va="bottom", fontsize=7,
                         fontweight="bold", color="#2166ac")

    # Annotations
    ax1.annotate(
        "+21.9 pp\n(Top-1)",
        xy=(x[3] - w, top1[3]),
        xytext=(x[3] - w - 0.65, top1[3] + 9),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
        fontsize=8, color=ORANGE, fontweight="bold",
    )
    ax1.annotate(
        "+14.3 pp\n(Top-3)",
        xy=(x[5], top3[5]),
        xytext=(x[5] + 0.30, top3[5] + 5),
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
        fontsize=8, color=PURPLE, fontweight="bold",
    )

    # Shade v1 / v2 / v3 columns
    for xi, fc, lbl in [(3, "#e8f0fe", "v1"), (4, "#e8f5e9", "v2"),
                         (5, "#f3e5f5", "v3")]:
        ax1.axvspan(xi - 0.46, xi + 0.46, alpha=0.18, color=fc, zorder=0)
        ax1.text(xi, 101, lbl, ha="center", va="bottom",
                 fontsize=8, color=GREY, style="italic")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 106)
    ax1.set_title("(a) Top-K Accuracy on RSHB (210 trials)", pad=8)
    ax1.legend(loc="upper left", fontsize=8)

    # ── Right: MRR bars ───────────────────────────────────────────────────────
    mrr_colors = [GREY, GREY, GREY, BLUE, GREEN, PURPLE]
    bars2 = ax2.bar(range(len(methods)), mrr, color=mrr_colors,
                    edgecolor="white", linewidth=0.6, width=0.6)
    for bar, v in zip(bars2, mrr):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.006, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7,
                 fontweight="bold" if v >= 0.620 else "normal")

    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(
        ["Base", "+GNN\n(z)", "VAE\n+APA", "v1", "v2", "v3"], fontsize=7)
    ax2.set_ylabel("MRR")
    ax2.set_ylim(0, 0.76)
    ax2.set_title("(b) MRR", pad=8)

    fig.suptitle(
        "Figure 5. ML-Enhanced Pipeline Results on RSHB (Real Market Data, 210 Trials)",
        y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_rshb_ml")


# ── Figure 6: ML Ablation — VAE+GNN vs z-score+GNN ───────────────────────────

def fig6_ml_ablation():
    methods = [
        "APA-RCA v2\n(Baseline)",
        "APA-RCA v2\n+ GNN\n(z-score)",
        "VAE +\nAPA-RCA v2",
        "VAE + APA-RCA v2\n+ GNN",
    ]
    top1 = [32.9, 31.4, 43.3, 54.8]
    top3 = [67.6, 49.5, 60.0, 65.7]

    x = np.arange(len(methods))
    w = 0.32

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bar_colors_t1 = [GREY, ORANGE, GREY, BLUE]
    bar_colors_t3 = ["#aaaaaa", "#f4a582", "#aaaaaa", "#6baed6"]

    b1 = ax.bar(x - w/2, top1, w, color=bar_colors_t1,
                edgecolor="white", linewidth=0.6, label="Top-1")
    b3 = ax.bar(x + w/2, top3, w, color=bar_colors_t3,
                edgecolor="white", linewidth=0.6, label="Top-3", alpha=0.9)

    # Value labels
    for bar, v, c in zip(b1, top1, bar_colors_t1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if c == BLUE else "normal",
                color=c if c != GREY else "#333333")

    # Annotation: z-score GNN regresses
    ax.annotate("",
        xy=(x[1] - w/2, top1[1] + 1),
        xytext=(x[0] - w/2, top1[0] - 2),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))
    ax.text(x[1] - w/2 + 0.02, top1[1] - 6,
            "−1.5 pp\n(regresses)", ha="center", fontsize=8,
            color=ORANGE, style="italic")

    # Annotation: VAE+GNN best
    ax.annotate("★ Best\n+21.9 pp",
        xy=(x[3] - w/2, top1[3]),
        xytext=(x[3] - w/2 + 0.4, top1[3] + 8),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
        fontsize=9, color=BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left")
    ax.set_title(
        "Figure 6. Ablation Study: VAE+GNN vs. z-score+GNN (RSHB, 210 Trials)",
        fontsize=11, fontweight="bold", pad=10)

    # Summary box
    summary = ("Key finding: Adding GNN without VAE (z-score+GNN) regresses below baseline.\n"
               "VAE is essential for cross-domain GNN feature quality.")
    ax.text(0.5, -0.22, summary, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4fa",
                      edgecolor="#bbccdd", linewidth=1))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig6_ml_ablation")


# ── Figure 7: Scorer Comparison — VAE vs IF vs OCSVM ─────────────────────────

def fig7_scorer_comparison():
    scorers = ["IF\n+ APA-RCA + GNN",
               "OCSVM\n+ APA-RCA + GNN",
               "VAE\n+ APA-RCA + GNN"]

    real_top1  = [0.5,  0.0,  54.8]
    # minimum visible bar heights for display
    vis_top1   = [3.5,  2.0,  54.8]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bar_colors = [GREY, GREY, BLUE]
    bars = ax.bar(range(3), vis_top1, color=bar_colors,
                  edgecolor="white", linewidth=0.6, width=0.5)

    # Value labels
    for i, (bar, real_v, vis_v) in enumerate(zip(bars, real_top1, vis_top1)):
        label = f"{real_v:.1f}%"
        weight = "bold" if real_v > 50 else "normal"
        color  = BLUE if real_v > 50 else "#333333"
        ax.text(bar.get_x() + bar.get_width() / 2,
                vis_v + 1.5, label,
                ha="center", va="bottom",
                fontsize=11, fontweight=weight, color=color)

    # Note for minimum bar height
    ax.text(0.5, 9, "※ IF / OCSVM bars inflated for visibility\n"
                    "   (actual: 0.5% and 0.0%)",
            ha="center", fontsize=8, color="#666666", style="italic")

    # Brace / bracket annotation
    ax.annotate("",
        xy=(2, real_top1[2]),
        xytext=(1, 0),
        arrowprops=dict(arrowstyle="<->",
                        color=BLUE, lw=1.5,
                        connectionstyle="arc3,rad=0.0"))
    ax.text(1.5, 30, "Δ = 54.3 pp", ha="center", fontsize=9,
            color=BLUE, fontweight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(scorers, fontsize=9.5)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_ylim(0, 72)
    ax.set_title(
        "Figure 7. Unsupervised Scorer Comparison on RSHB (210 Trials)",
        fontsize=11, fontweight="bold", pad=10)

    # Theory box at bottom
    theory_text = (
        "IF / OCSVM: boundary-based — no temporal pattern learning → distribution shift breaks cross-domain transfer\n"
        "VAE: manifold-based — reconstruction error is stable across distribution shift → enables cross-domain GNN"
    )
    ax.text(0.5, -0.20, theory_text, transform=ax.transAxes,
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4fa",
                      edgecolor="#bbccdd", linewidth=1))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig7_scorer_comparison")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating thesis figures 5-7...")
    fig5_rshb_ml()
    fig6_ml_ablation()
    fig7_scorer_comparison()
    print("Done.")
