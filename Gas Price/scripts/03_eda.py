"""
03_eda.py
=========
Exploratory Data Analysis — produces 4 publication-quality plots:

  1. Time-series of U.S. average gas prices (1993–2025)
  2. Regional price comparison (all 7 regions overlaid)
  3. Monthly seasonality box-plot
  4. Feature correlation heat-map

Output: plots/eda_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

FEAT_CSV = DATA_DIR / "gas_prices_features.csv"

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]
REGION_COLORS = {}   # filled once data is loaded


# ── Helpers ───────────────────────────────────────────────────────────────────
def style_axis(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)


def annotate_events(ax):
    """Mark key historical gas-price events on a time-series axis."""
    events = [
        ("2001-09-17", "9/11"),
        ("2005-08-29", "Katrina"),
        ("2008-07-14", "Peak\n$4.11"),
        ("2020-04-27", "COVID\nTrough"),
        ("2022-06-13", "Record\n$5.00"),
    ]
    for date_str, label in events:
        x = pd.Timestamp(date_str)
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(x, ax.get_ylim()[1] * 0.97, label,
                fontsize=7, color="dimgray", ha="center", va="top",
                rotation=0, bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))


# ── Plot 1: U.S. Average time-series ─────────────────────────────────────────
def plot_national_timeseries(df: pd.DataFrame):
    us = df[df["region"] == "U.S. Average"].copy()
    us = us.sort_values("date")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(us["date"], us["price"], alpha=0.15, color=PALETTE[0])
    ax.plot(us["date"], us["price"], linewidth=1.4, color=PALETTE[0], label="Weekly price")

    # 52-week rolling average
    roll = us["price"].rolling(52, center=True).mean()
    ax.plot(us["date"], roll, linewidth=2.0, color="#d62728",
            linestyle="--", label="52-week rolling avg")

    style_axis(ax,
               title="U.S. Average Retail Gasoline Prices (1993–2025)",
               xlabel="Year",
               ylabel="Price ($/gallon)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    annotate_events(ax)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out = PLOTS_DIR / "eda_01_national_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Plot 2: Regional comparison ───────────────────────────────────────────────
def plot_regional_comparison(df: pd.DataFrame):
    regions = df["region"].unique()
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, region in enumerate(regions):
        sub = df[df["region"] == region].sort_values("date")
        lw  = 2.2 if region == "U.S. Average" else 1.0
        ls  = "-"  if region == "U.S. Average" else "-"
        ax.plot(sub["date"], sub["price"],
                linewidth=lw, linestyle=ls,
                color=PALETTE[i % len(PALETTE)],
                label=region, alpha=0.85)

    style_axis(ax,
               title="Gas Prices by EIA Region (1993–2025)",
               xlabel="Year",
               ylabel="Price ($/gallon)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    plt.tight_layout()
    out = PLOTS_DIR / "eda_02_regional_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Plot 3: Monthly seasonality ───────────────────────────────────────────────
def plot_monthly_seasonality(df: pd.DataFrame):
    us = df[df["region"] == "U.S. Average"].copy()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    monthly_stats = us.groupby("month")["price"].agg(["mean","median","std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = monthly_stats["month"].values - 1       # 0-indexed for bar positions
    bars = ax.bar(x, monthly_stats["mean"], color=PALETTE[0], alpha=0.75,
                  width=0.6, label="Monthly mean")
    ax.errorbar(x, monthly_stats["mean"],
                yerr=monthly_stats["std"],
                fmt="none", color="black", capsize=4, linewidth=1.2, label="±1 std")
    ax.plot(x, monthly_stats["median"], "D--", color=PALETTE[3],
            markersize=6, linewidth=1.5, label="Monthly median")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    style_axis(ax,
               title="Seasonal Price Pattern — U.S. Monthly Averages (all years)",
               xlabel="Month",
               ylabel="Price ($/gallon)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "eda_03_monthly_seasonality.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Plot 4: Correlation heat-map ──────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = [
        "price", "time_trend", "year", "month", "week_of_year",
        "month_sin", "month_cos", "week_sin", "week_cos",
        "lag_1w", "lag_4w", "lag_52w",
        "roll_mean_4w", "roll_std_4w",
    ]
    sub = df[numeric_cols].dropna()
    corr = sub.corr()

    n = len(corr)
    fig, ax = plt.subplots(figsize=(11, 9))

    # Manual heat-map (no seaborn)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [c.replace("_", "\n") for c in corr.columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    out = PLOTS_DIR / "eda_04_correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n── Exploratory Data Analysis ──")
    df = pd.read_csv(FEAT_CSV, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows for EDA")

    # Summary stats
    us = df[df["region"] == "U.S. Average"]
    print(f"\nU.S. Average price stats:")
    stats = us["price"].describe()
    for k, v in stats.items():
        print(f"  {k:8s}: ${v:.3f}")

    print("\nGenerating plots …")
    plot_national_timeseries(df)
    plot_regional_comparison(df)
    plot_monthly_seasonality(df)
    plot_correlation_heatmap(df)
    print("EDA complete.")


if __name__ == "__main__":
    main()
