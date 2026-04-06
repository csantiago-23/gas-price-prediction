# U.S. Gas Price Prediction — ML Pipeline

**Python | NumPy | Pandas | Matplotlib | 2026**

This project is an end-to-end machine learning pipeline that predicts U.S. gas prices using real weekly data from the U.S. Energy Information Administration (EIA).

I built this from scratch to better understand how time-series data behaves in practice — from cleaning messy real-world data to engineering features and training models without relying on high-level ML libraries like scikit-learn.

---

## Results

| Model | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Linear Regression | 0.9686 | $0.089 | $0.061 | 1.90% |
| Random Forest | 0.9648 | $0.094 | $0.067 | 2.04% |

> Tested on 3,275 observations across 20 regions, January 2023 – March 2026.

---

## What I Did

- Cleaned and reshaped ~30k rows of raw EIA data
- Built 20+ features including lag variables and rolling averages
- Explored trends like seasonality and regional differences
- Implemented Linear Regression and Random Forest from scratch (NumPy only)
- Evaluated models using multiple metrics and visualizations

---

## Dataset

- **Source:** U.S. Energy Information Administration (EIA) — Weekly Retail Gasoline and Diesel Prices
- **URL:** https://www.eia.gov/dnav/pet/pet_pri_gnd_a_epmr_pte_dpgal_w.htm
- **Coverage:** August 1990 – March 2026 | Weekly | Regular Conventional Gasoline
- **Regions:** 20 — U.S. Average, East Coast, New England, Central Atlantic, Lower Atlantic, Midwest, Gulf Coast, Rocky Mountain, West Coast, and 11 state/city series

---
## Why This Project Matters

One of the most interesting takeaways was that a simple linear model outperformed a more complex Random Forest.

This showed that gas prices are highly driven by recent historical values (lag features), making the problem more linear than expected. Instead of overcomplicating the model, the data itself pointed toward a simpler solution

## How to Run

**Requirements:** Python 3.8+, NumPy, Pandas, Matplotlib 

```bash
# Clone and navigate to project
cd "Gas Price"

# Run the full pipeline in one command
python run_pipeline.py

# Or run individual steps
python scripts/01_data_collection.py
python scripts/02_data_cleaning.py
python scripts/03_eda.py
python scripts/04_model_training.py
python scripts/05_model_evaluation.py
```

---

## Pipeline Walkthrough

### Step 1 — Data Collection
The EIA exports data in wide format (one column per region). We rename all 20 columns to short identifiers and melt the table into tidy long format: one row per `(date, region, price)` triplet. This transforms 1,859 weeks × 20 regions into 29,631 observations.

### Step 2 — Data Cleaning & Feature Engineering
Validation removes duplicates and physically impossible prices (< $0.50 or > $8.00). An IQR-based outlier check runs per region. Then 21 features are engineered:

- **Calendar:** year, month, week of year
- **Cyclical encoding:** sin/cos transforms of month and week (so December and January are adjacent, not far apart)
- **Time trend:** weeks elapsed since dataset start
- **Lag features:** price 1 week ago, 4 weeks ago, 52 weeks ago (year-over-year)
- **Rolling statistics:** 4-week rolling mean and standard deviation
- **Region indicators:** one-hot encoding of all 20 regions

### Step 3 — Exploratory Data Analysis
Four plots explore the data before modelling:
- National time-series with annotated historical events (9/11, Katrina, 2008 peak, COVID, 2022 record)
- Regional price comparison across all 20 series
- Monthly seasonality box plot (summer premium, winter trough)
- Feature correlation heat-map (built without seaborn)

### Step 4 — Model Training
Both models are implemented from scratch using only NumPy, with no ML libraries.

**Linear Regression** solves the closed-form normal equation:
```
β = (XᵀX + αI)⁻¹ Xᵀy
```
A small ridge penalty (α = 0.001) prevents numerical instability when features are correlated. Features are standardised (mean=0, std=1) before fitting.

**Random Forest** builds 80 decision trees, each trained on a bootstrap sample (random rows with replacement) using a random 60% feature subset per split. Trees use MSE-minimising binary splits (CART algorithm) up to depth 6. Final prediction is the mean across all trees. Decision trees handle raw feature scales — no standardisation needed.

The train/test split is **time-based** (train ≤ 2022, test 2023–2026) to prevent future data from leaking into training.

### Step 5 — Evaluation
Four evaluation plots cover:
- Predicted vs actual scatter with ±1 RMSE band
- Time-series of true vs predicted prices over the test window
- Residual distribution histograms and residuals over time
- Side-by-side metric bar charts (MAE, RMSE, R², MAPE)

---

## Key Findings

**Linear Regression outperforms Random Forest** (R² 0.9686 vs 0.9648), which reveals that the price-prediction relationship is fundamentally linear. The lag features — especially `lag_1w` (last week's price) — are so strongly correlated with the target that the added complexity of an ensemble model provides no benefit here. This is a meaningful result, not a failure of the Random Forest.

**The strongest predictors** are the lag features. Gas prices exhibit strong autocorrelation: this week's price is the best predictor of next week's price. The year-ago lag (`lag_52w`) captures seasonal cycles. The rolling mean captures medium-term trend momentum.

**Seasonality is real but modest.** Summer prices average about $0.10–$0.15/gallon higher than winter prices — a consistent pattern driven by summer-blend fuel requirements and increased driving demand.
<img width="2400" height="1600" alt="U S  Gas Price Analysis   ML Prediction Dashboard " src="https://github.com/user-attachments/assets/2ac51566-61a1-47b7-b1bd-1dd97799b20d" />


---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading, reshaping, feature engineering |
| NumPy | Matrix math, model implementation |
| Matplotlib | All visualisations (8 publication-quality plots) |
| EIA Open Data | Real weekly gasoline price data |
| Tableau Public | Dashboard

---
## Notes

This project was a way for me to go beyond using libraries and actually understand how models work under the hood. Building everything from scratch helped me get more comfortable with the math and tradeoffs behind different approaches.

## Author

**Chris Santiago** — csantiago@ucsb.edu
