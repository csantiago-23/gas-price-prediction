"""
01_data_collection.py
=====================
Loads real EIA weekly retail gasoline price data (Regular Conventional)
downloaded from the U.S. Energy Information Administration, melts the
wide-format table into a tidy long-format dataset, and saves it as
data/gas_prices_raw.csv.

Source : U.S. EIA — Weekly Retail Gasoline and Diesel Prices
URL    : https://www.eia.gov/dnav/pet/pet_pri_gnd_a_epmr_pte_dpgal_w.htm
Coverage: Aug 1990 – Mar 2026  |  Weekly  |  20 regions/cities

Output: data/gas_prices_raw.csv  (~37,000 rows in long format)
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_EIA  = DATA_DIR / "gaspricesexcel.csv"
OUT_CSV  = DATA_DIR / "gas_prices_raw.csv"

# Friendly short names for the 20 EIA series
REGION_MAP = {
    "Weekly U.S. Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":                  "U.S. Average",
    "Weekly East Coast Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":             "East Coast",
    "Weekly New England (PADD 1A) Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":  "New England",
    "Weekly Central Atlantic (PADD 1B) Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)": "Central Atlantic",
    "Weekly Lower Atlantic (PADD 1C) Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":   "Lower Atlantic",
    "Weekly Midwest Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":                "Midwest",
    "Weekly Gulf Coast Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":             "Gulf Coast",
    "Weekly Rocky Mountain Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":         "Rocky Mountain",
    "Weekly West Coast Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":             "West Coast",
    "Weekly Colorado Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":               "Colorado",
    "Weekly Florida Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":                "Florida",
    "Weekly New York Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":               "New York",
    "Weekly Minnesota Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":              "Minnesota",
    "Weekly Ohio Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":                   "Ohio",
    "Weekly Texas Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":                  "Texas",
    "Weekly Washington Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":             "Washington",
    "Weekly Cleveland, OH Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":          "Cleveland, OH",
    "Weekly Denver, CO Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":             "Denver, CO",
    "Weekly Miami, FL Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":              "Miami, FL",
    "Weekly Seattle, WA Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)":            "Seattle, WA",
}


def load_eia_csv(path: Path = RAW_EIA) -> pd.DataFrame:
    """
    Read the EIA export CSV (3-row header: metadata, sourcekeys, column names)
    and return a clean wide-format DataFrame indexed by date.
    """
    # Row 0: "Back to Contents ..."  — skip
    # Row 1: Sourcekey codes         — skip
    # Row 2: Human-readable headers  — use as column names
    df = pd.read_csv(path, skiprows=2, header=0)
    df = df.rename(columns={df.columns[0]: "date"})

    # Drop the trailing empty column Numbers sometimes appends
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Parse dates and drop any non-date rows (e.g. footer)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Coerce all price columns to float
    price_cols = [c for c in df.columns if c != "date"]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide format (one column per region) to tidy long format:
      date | region | price
    Drop rows where price is NaN (series not yet reported for that date).
    """
    df = df.rename(columns=REGION_MAP)
    region_cols = list(REGION_MAP.values())

    long = df.melt(
        id_vars="date",
        value_vars=region_cols,
        var_name="region",
        value_name="price",
    )
    long = long.dropna(subset=["price"])
    long = long.sort_values(["date", "region"]).reset_index(drop=True)
    return long


def main():
    print("Loading real EIA weekly gasoline price data …")
    wide = load_eia_csv()
    print(f"  Wide format : {len(wide):,} weeks × {len(wide.columns)-1} regions")
    print(f"  Date range  : {wide['date'].min().date()} → {wide['date'].max().date()}")

    long = melt_to_long(wide)
    long.to_csv(OUT_CSV, index=False)

    print(f"\n  Long format : {len(long):,} rows → {OUT_CSV.name}")
    print(f"  Regions     : {long['region'].nunique()} — {', '.join(sorted(long['region'].unique()))}")
    print(f"  Price range : ${long['price'].min():.3f} – ${long['price'].max():.3f}")
    return long


if __name__ == "__main__":
    main()
