"""
05_model_evaluation.py
======================
Visualises model performance with 3 publication-quality plots:

  1. Predicted vs Actual scatter (both models, U.S. Average)
  2. Time-series of true vs predicted prices (2023–2025 test window)
  3. Residual distribution histograms + QQ-like comparison

Input : data/test_predictions.csv
        data/model_metrics.csv
Output: plots/eval_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PRED_CSV    = DATA_DIR / "test_predictions.csv"
METRICS_CSV = DATA_DIR / "model_metrics.csv"

LR_COLOR = "#1f77b4"
RF_COLOR = "#ff7f0e"
TRUE_COLOR = "#2ca02c"


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=9)


def dollar_fmt(ax, axis="y"):
    fmt = mticker.FormatStrFormatter("$%.2f")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)


# ── Plot 1: Predicted vs Actual scatter ──────────────────────────────────────
def plot_scatter(df: pd.DataFrame, metrics: pd.DataFrame):
    us = df[df["region"] == "U.S. Average"].copy()
    y_true = us["price"].values
    y_lr   = us["pred_lr"].values
    y_rf   = us["pred_rf"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    lims = (y_true.min() * 0.97, y_true.max() * 1.02)

    for ax, y_pred, color, label, key in zip(
        axes,
        [y_lr, y_rf],
        [LR_COLOR, RF_COLOR],
        ["Linear Regression", "Random Forest"],
        ["Linear Regression", "Random Forest"],
    ):
        row = metrics[metrics["model"] == key].iloc[0]
        residuals = y_true - y_pred

        ax.scatter(y_true, y_pred, alpha=0.45, s=14, color=color, linewidths=0)
        ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")

        # Residual band (±1 RMSE)
        ax.fill_between(lims,
                        [l - row["RMSE"] for l in lims],
                        [l + row["RMSE"] for l in lims],
                        alpha=0.10, color=color, label=f"±1 RMSE")

        ax.set_xlim(lims); ax.set_ylim(lims)
        dollar_fmt(ax, axis="both")
        ax.set_xlabel("Actual Price ($/gal)", fontsize=10)
        ax.set_ylabel("Predicted Price ($/gal)", fontsize=10)
        ax.set_title(
            f"{label}\nR² = {row['R2']:.4f}   RMSE = ${row['RMSE']:.4f}   "
            f"MAE = ${row['MAE']:.4f}   MAPE = {row['MAPE']:.2f}%",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=8)
        clean_ax(ax)

    fig.suptitle("Predicted vs Actual Gas Prices — Test Set (2023–2025)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "eval_01_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Plot 2: Time-series predictions ──────────────────────────────────────────
def plot_timeseries(df: pd.DataFrame):
    us = df[df["region"] == "U.S. Average"].sort_values("date").copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(us["date"], us["price"],   color=TRUE_COLOR, linewidth=2.2,
            label="Actual price",       zorder=5)
    ax.plot(us["date"], us["pred_lr"], color=LR_COLOR,   linewidth=1.5,
            linestyle="--", label="Linear Regression prediction", zorder=4)
    ax.plot(us["date"], us["pred_rf"], color=RF_COLOR,   linewidth=1.5,
            linestyle=":",  label="Random Forest prediction",     zorder=4)

    ax.fill_between(us["date"], us["price"], us["pred_rf"],
                    alpha=0.10, color=RF_COLOR, label="RF residual band")

    ax.set_title("Gas Price Predictions vs Actuals — Test Period (2023–2025)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Price ($/gallon)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend(fontsize=9, loc="upper right")
    clean_ax(ax)

    plt.tight_layout()
    out = PLOTS_DIR / "eval_02_timeseries_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Plot 3: Residual analysis ─────────────────────────────────────────────────
def plot_residuals(df: pd.DataFrame, metrics: pd.DataFrame):
    us     = df[df["region"] == "U.S. Average"].copy()
    res_lr = us["price"].values - us["pred_lr"].values
    res_rf = us["price"].values - us["pred_rf"].values

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    # ── Histogram LR ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(res_lr, bins=35, color=LR_COLOR, alpha=0.75, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=1.4, linestyle="--")
    ax1.set_title("Linear Regression — Residual Distribution", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Residual ($/gal)")
    ax1.set_ylabel("Count")
    clean_ax(ax1)

    # ── Histogram RF ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(res_rf, bins=35, color=RF_COLOR, alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1.4, linestyle="--")
    ax2.set_title("Random Forest — Residual Distribution", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Residual ($/gal)")
    ax2.set_ylabel("Count")
    clean_ax(ax2)

    # ── Residuals over time ──
    ax3 = fig.add_subplot(gs[1, :])
    dates = pd.to_datetime(us["date"])
    ax3.plot(dates, res_lr, color=LR_COLOR, linewidth=1.0, label="LR residuals", alpha=0.8)
    ax3.plot(dates, res_rf, color=RF_COLOR, linewidth=1.0, label="RF residuals", alpha=0.8)
    ax3.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax3.fill_between(dates, res_lr, 0, alpha=0.08, color=LR_COLOR)
    ax3.fill_between(dates, res_rf, 0, alpha=0.08, color=RF_COLOR)
    ax3.set_title("Residuals Over Time (Test Period)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Residual ($/gal)")
    ax3.legend(fontsize=9)
    clean_ax(ax3)

    fig.suptitle("Model Residual Analysis — U.S. Average (2023–2025)",
                 fontsize=13, fontweight="bold", y=1.01)
    out = PLOTS_DIR / "eval_03_residuals.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Plot 4: Model comparison bar chart ───────────────────────────────────────
def plot_metrics_bar(metrics: pd.DataFrame):
    kpis   = ["MAE", "RMSE", "R2", "MAPE"]
    labels = metrics["model"].values
    colors = [LR_COLOR, RF_COLOR]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, kpi in zip(axes, kpis):
        vals = metrics[kpi].values
        bars = ax.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
        for bar, val in zip(bars, vals):
            suffix = "%" if kpi == "MAPE" else ("" if kpi == "R2" else "$")
            fmt    = f"{val:.4f}" if kpi == "R2" else f"{val:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(kpi, fontsize=11, fontweight="bold")
        ax.set_ylabel(kpi, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=8)

    fig.suptitle("Model Performance Comparison — Test Set (2023–2025)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "eval_04_metrics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n── Model Evaluation & Visualisation ──")
    df      = pd.read_csv(PRED_CSV, parse_dates=["date"])
    metrics = pd.read_csv(METRICS_CSV)

    print("\nMetrics summary:")
    print(metrics[["model", "MAE", "RMSE", "R2", "MAPE"]].to_string(index=False))

    print("\nGenerating evaluation plots …")
    plot_scatter(df, metrics)
    plot_timeseries(df)
    plot_residuals(df, metrics)
    plot_metrics_bar(metrics)
    print("\nAll evaluation plots saved to plots/")


if __name__ == "__main__":
    main()
