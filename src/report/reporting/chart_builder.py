"""generates matplotlib/seaborn figures for the report.
All charts are saved as 150 DPI PNG files into ``reports/assets/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

from ..analysis.dataset_analyser import DatasetStats
from ..cost_model.llm_cost import CostReport

logger = logging.getLogger(__name__)

# Design tokens
sns.set_theme(style="darkgrid", context="notebook")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#0f3460",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0",
        "text.color": "#e0e0e0",
        "grid.color": "#0f3460",
        "grid.linewidth": 0.6,
    }
)

_BIP_COLOR = "#F7931A"   # Bitcoin orange
_BOLT_COLOR = "#3B82F6"  # Lightning blue
_ACCENT = "#e94560"


def _save(fig: plt.Figure, assets_dir: Path, name: str) -> Path:
    """Tighten layout, save and close the figure. Returns the output path."""
    out = assets_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"    Saved → {out.name}")
    return out


# Token distribution histogram
def chart_token_distribution(df: pd.DataFrame, assets_dir: Path) -> Path:
    """Overlaid histogram with KDE for token counts."""
    fig, ax = plt.subplots(figsize=(11, 5))

    # Use Seaborn for better handling of multiple categories
    sns.histplot(
        data=df,
        x="token_count",
        hue="spec_type",
        bins=30,
        kde=True,
        palette={"BIP": _BIP_COLOR, "BOLT": _BOLT_COLOR},
        alpha=0.6,
        edgecolor="#ffffff20",
        ax=ax
    )

    ax.set_xlabel("Token Count (cl100k_base)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Token Distribution — BIPs & BOLTs", fontsize=13, pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    # Customising legend
    ax.get_legend().set_title("Spec Type")

    return _save(fig, assets_dir, "token_distribution.png")


# BIPs vs BOLTs comparison
def chart_bips_vs_bolts(
    stats: DatasetStats, df: pd.DataFrame, assets_dir: Path
) -> Path:
    """Side-by-side bars comparing record count and average token count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["BIPs", "BOLTs"]
    colors = [_BIP_COLOR, _BOLT_COLOR]

    # Subplot 1: Raw Counts
    counts = [stats.total_bips, stats.total_bolts]
    bars1 = ax1.bar(labels, counts, color=colors, width=0.45, edgecolor="#00000060")
    ax1.set_title("Total Specification Count", fontsize=12)
    
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{int(bar.get_height())}", ha="center", va="bottom", fontweight="bold")

    # Subplot 2: Average Tokens
    avgs = [df[df["spec_type"] == "BIP"]["token_count"].mean(), 
            df[df["spec_type"] == "BOLT"]["token_count"].mean()]
    
    bars2 = ax2.bar(labels, avgs, color=colors, width=0.45, edgecolor="#00000060")
    ax2.set_title("Average Tokens per Spec", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Corpus Comparison: BIP vs BOLT Metrics", fontsize=14, fontweight="bold", y=1.02)
    return _save(fig, assets_dir, "bips_vs_bolts.png")


# Cost comparison bar chart
def chart_cost_comparison(cost_report: CostReport, assets_dir: Path) -> Path:
    """Visualises one-time processing cost per model."""
    data = cost_report.total_one_time_cost
    models = list(data.keys())
    costs = list(data.values())

    fig, ax = plt.subplots(figsize=(11, 5))
    
    # Use a gradient-style palette based on cost
    palette = sns.color_palette("flare", n_colors=len(models))
    bars = ax.bar(models, costs, color=palette, edgecolor="#00000060", width=0.6)

    ax.set_title("LLM One-Time Processing Cost (USD)", fontsize=13, pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
    plt.xticks(rotation=15, ha="right")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(costs) * 0.02),
                f"${bar.get_height():.2f}", ha="center", fontweight="bold", size=10)

    return _save(fig, assets_dir, "cost_comparison.png")


# Per-Record token count (Top 50)
def chart_token_per_spec(df: pd.DataFrame, assets_dir: Path) -> Path:
    """Horizontal bar chart of top 50 record."""
    top_50 = df.nlargest(50, "token_count").iloc[::-1] # Reverse for descending view

    fig, ax = plt.subplots(figsize=(10, 12))

    palette = {"BIP": _BIP_COLOR, "BOLT": _BOLT_COLOR}
    sns.barplot(
        data=top_50,
        y="identifier",
        x="token_count",
        hue="spec_type",
        palette=palette,
        dodge=False,
        ax=ax
    )

    ax.set_title("Top 50 Specifications by Token Volume", fontsize=13, pad=12)
    ax.set_xlabel("Token Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", title="Spec Type")

    return _save(fig, assets_dir, "token_per_spec.png")


# Orchestrator
def build_all_charts(
    df: pd.DataFrame,
    stats: DatasetStats,
    cost_report: CostReport,
    assets_dir: Path,
) -> dict[str, Path]:
    """Generate all charts and return mapping."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating analytical visualisations...")

    return {
        "token_distribution": chart_token_distribution(df, assets_dir),
        "bips_vs_bolts": chart_bips_vs_bolts(stats, df, assets_dir),
        "cost_comparison": chart_cost_comparison(cost_report, assets_dir),
        "token_per_spec": chart_token_per_spec(df, assets_dir),
    }