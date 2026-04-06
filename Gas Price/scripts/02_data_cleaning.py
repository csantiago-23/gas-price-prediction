"""
02_data_cleaning.py
===================
Cleans the raw gas price dataset and engineers features for ML modelling.

Feature engineering includes:
  - Calendar features (year, month, week) with cyclical sin/cos encoding
  - Lag features: 1-week, 4-week, 52-week (year-ago) prices
  - Rolling statistics: 4-week mean and std
  - Linear time trend
  - Region one-hot encoding

Input : data/gas_prices_raw.csv
Output: data/gas_prices_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "data"
RAW_CSV    = DATA_DIR / "gas_prices_raw.csv"
CLEAN_CSV  = DATA_DIR / "gas_prices_features.csv"


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


# ── 2. Basic cleaning ─────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Drop rows with missing price or date
    df = df.dropna(subset=["date", "price"])

    # Remove physically impossible prices
    df = df[(df["price"] > 0.50) & (df["price"] < 8.00)]

    # IQR outlier removal per region
    def remove_outliers(group):
        q1, q3 = group["price"].quantile(0.01), group["price"].quantile(0.99)
        iqr = q3 - q1
        return group[(group["price"] >= q1 - 3 * iqr) &
                     (group["price"] <= q3 + 3 * iqr)]

    df = df.groupby("region", group_keys=False).apply(remove_outliers)
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    print(f"Cleaning removed {before - len(df)} rows → {len(df):,} remain")
    return df


# ── 3. Feature engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Calendar features (raw) ---
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"]  = df["date"].dt.dayofyear

    # --- Cyclical encoding (captures periodicity for ML) ---
    df["month_sin"]  = np.sin(2 * np.pi * df["month"]        / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"]        / 12)
    df["week_sin"]   = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]   = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # --- Linear time trend (weeks since dataset start) ---
    t0 = df["date"].min()
    df["time_trend"] = (df["date"] - t0).dt.days / 7

    # --- Lag & rolling features (per region to avoid cross-region leakage) ---
    def add_lags(group):
        group = group.sort_values("date").copy()
        group["lag_1w"]  = group["price"].shift(1)
        group["lag_4w"]  = group["price"].shift(4)
        group["lag_52w"] = group["price"].shift(52)
        group["roll_mean_4w"] = group["price"].shift(1).rolling(4).mean()
        group["roll_std_4w"]  = group["price"].shift(1).rolling(4).std()
        return group

    df = df.groupby("region", group_keys=False).apply(add_lags)

    # --- Region one-hot encoding ---
    region_dummies = pd.get_dummies(df["region"], prefix="region", dtype=int)
    df = pd.concat([df, region_dummies], axis=1)

    # Drop rows missing lag/rolling features (first ~52 weeks per region)
    before = len(df)
    df = df.dropna()
    print(f"Feature engineering: dropped {before - len(df)} rows with NaN lags → "
          f"{len(df):,} model-ready rows")

    df = df.reset_index(drop=True)
    return df


# ── 4. Save ───────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, path: Path = CLEAN_CSV):
    df.to_csv(path, index=False)
    feature_cols = [c for c in df.columns if c not in ("date", "region", "price")]
    print(f"Saved features CSV → {path.name}")
    print(f"  Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n── Data Cleaning & Feature Engineering ──")
    df_raw   = load_raw()
    df_clean = clean(df_raw)
    df_feat  = engineer_features(df_clean)
    save(df_feat)
    return df_feat


if __name__ == "__main__":
    main()
