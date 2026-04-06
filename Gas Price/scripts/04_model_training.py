"""
04_model_training.py
====================
Trains two regression models from scratch using only NumPy + Pandas:

  1. LinearRegression  — closed-form normal equation (OLS)
  2. RandomForestRegressor — bagged decision-tree ensemble

Both models expose a scikit-learn-compatible API (.fit / .predict).
Trained models are serialised to data/models/ as NumPy .npz archives.

Input : data/gas_prices_features.csv
Output: data/models/linear_regression.npz
        data/models/random_forest_params.npz
        data/model_metrics.csv
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
FEAT_CSV   = DATA_DIR / "gas_prices_features.csv"
METRICS_CSV= DATA_DIR / "model_metrics.csv"

# Feature columns used for training
FEATURE_COLS = [
    "time_trend", "year", "month_sin", "month_cos",
    "week_sin", "week_cos",
    "lag_1w", "lag_4w", "lag_52w",
    "roll_mean_4w", "roll_std_4w",
    # region dummies added dynamically
]
TARGET = "price"
TRAIN_CUTOFF = "2022-12-31"   # train on ≤2022, test on 2023–2025


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Linear Regression (OLS, normal equation)
# ─────────────────────────────────────────────────────────────────────────────
class LinearRegression:
    """
    Ordinary Least Squares regression solved via the normal equation:
        β = (X'X)⁻¹ X'y
    Uses regularisation (ridge) for numerical stability.
    """
    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha          # L2 regularisation strength
        self.coef_  = None
        self.intercept_ = None
        self._feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        self._feature_names = feature_names
        # Prepend bias column
        n, d = X.shape
        X_b = np.column_stack([np.ones(n), X])
        # Ridge normal equation: β = (X'X + αI)⁻¹ X'y
        A = X_b.T @ X_b
        A += self.alpha * np.eye(d + 1)
        b = X_b.T @ y
        beta = np.linalg.solve(A, b)
        self.intercept_ = beta[0]
        self.coef_       = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def save(self, path: Path):
        np.savez(path,
                 coef=self.coef_,
                 intercept=np.array([self.intercept_]),
                 alpha=np.array([self.alpha]))
        print(f"  Linear Regression saved → {path.name}")

    @classmethod
    def load(cls, path: Path) -> "LinearRegression":
        data = np.load(path)
        m = cls(alpha=float(data["alpha"][0]))
        m.coef_       = data["coef"]
        m.intercept_  = float(data["intercept"][0])
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Decision Tree (CART-style, MSE criterion) — building block of Random Forest
# ─────────────────────────────────────────────────────────────────────────────
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value   # leaf prediction (mean of samples)

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressor:
    """
    CART decision-tree regressor using mean-squared-error splits.
    Supports max_depth, min_samples_split, and max_features (for RF).
    """
    def __init__(self, max_depth=6, min_samples_split=20, max_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features   # None → use all features
        self.root              = None
        self._n_features       = None

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        self._n_features = X.shape[1]
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n = len(y)
        # Stopping criteria
        if depth >= self.max_depth or n < self.min_samples_split or np.var(y) < 1e-8:
            return _Node(value=float(np.mean(y)))

        # Sample feature subset (for Random Forest)
        n_feats = self._n_features
        if self.max_features is not None:
            n_feats = max(1, int(self.max_features * self._n_features))
        feat_idx = np.random.choice(self._n_features, n_feats, replace=False)

        best_feat, best_thresh, best_gain = None, None, -np.inf
        parent_mse = np.var(y) * n

        for f in feat_idx:
            vals = X[:, f]
            # Candidate split thresholds: percentiles (fast & robust)
            candidates = np.percentile(vals, np.linspace(10, 90, 15))
            candidates = np.unique(candidates)

            for thresh in candidates:
                left  = y[vals <= thresh]
                right = y[vals  > thresh]
                if len(left) < 2 or len(right) < 2:
                    continue
                mse_split = np.var(left) * len(left) + np.var(right) * len(right)
                gain = parent_mse - mse_split
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, f, thresh

        if best_feat is None:
            return _Node(value=float(np.mean(y)))

        mask  = X[:, best_feat] <= best_thresh
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, row, node):
        if node.is_leaf:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Random Forest Regressor
# ─────────────────────────────────────────────────────────────────────────────
class RandomForestRegressor:
    """
    Ensemble of DecisionTreeRegressors trained on bootstrap samples.
    Predictions are the mean across all trees (bagging).
    """
    def __init__(self, n_estimators=80, max_depth=6,
                 min_samples_split=20, max_features=0.6, random_state=42):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state
        self.trees_            = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        np.random.seed(self.random_state)
        n = len(y)
        self.trees_ = []
        for i in range(self.n_estimators):
            idx  = np.random.choice(n, n, replace=True)   # bootstrap sample
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            if (i + 1) % 20 == 0:
                print(f"    … trained {i+1}/{self.n_estimators} trees")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([t.predict(X) for t in self.trees_], axis=1)
        return preds.mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    residuals = y_true - y_pred
    mae  = np.mean(np.abs(residuals))
    mse  = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot
    mape = np.mean(np.abs(residuals / y_true)) * 100

    metrics = dict(model=label, MAE=mae, MSE=mse, RMSE=rmse, R2=r2, MAPE=mape)
    print(f"\n  {label}:")
    print(f"    MAE  = ${mae:.4f}")
    print(f"    RMSE = ${rmse:.4f}")
    print(f"    R²   = {r2:.4f}")
    print(f"    MAPE = {mape:.2f}%")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n── Model Training ──")

    # 1. Load features
    df = pd.read_csv(FEAT_CSV, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows")

    # Collect region-dummy columns
    region_cols = [c for c in df.columns if c.startswith("region_")]
    all_features = FEATURE_COLS + region_cols

    # 2. Train / test split (time-based)
    train = df[df["date"] <= TRAIN_CUTOFF]
    test  = df[df["date"]  > TRAIN_CUTOFF]
    print(f"Train: {len(train):,} rows (up to {TRAIN_CUTOFF})")
    print(f"Test : {len(test):,}  rows (2023–2025)")

    X_train = train[all_features].values.astype(np.float64)
    y_train = train[TARGET].values.astype(np.float64)
    X_test  = test[all_features].values.astype(np.float64)
    y_test  = test[TARGET].values.astype(np.float64)

    # 3. Feature standardisation (fit on train only)
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - X_mean) / X_std
    X_test_s  = (X_test  - X_mean) / X_std

    # 4. Train Linear Regression
    print("\nTraining Linear Regression …")
    t0 = time.time()
    lr = LinearRegression(alpha=1e-3)
    lr.fit(X_train_s, y_train, feature_names=all_features)
    print(f"  Done in {time.time()-t0:.1f}s")
    lr.save(MODELS_DIR / "linear_regression.npz")

    # 5. Train Random Forest
    print("\nTraining Random Forest (80 trees) …")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=80, max_depth=6,
                               min_samples_split=25, max_features=0.6)
    rf.fit(X_train, y_train)      # trees handle raw features (no scaling needed)
    print(f"  Done in {time.time()-t0:.1f}s")

    # 6. Evaluate on test set
    print("\n── Test Set Metrics ──")
    lr_pred = lr.predict(X_test_s)
    rf_pred = rf.predict(X_test)

    metrics = [
        compute_metrics(y_test, lr_pred, "Linear Regression"),
        compute_metrics(y_test, rf_pred, "Random Forest"),
    ]

    # 7. Save metrics and predictions
    pd.DataFrame(metrics).to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics saved → {METRICS_CSV.name}")

    # 8. Save predictions for plotting (attach back to test df)
    test = test.copy()
    test["pred_lr"] = lr_pred
    test["pred_rf"] = rf_pred
    pred_path = DATA_DIR / "test_predictions.csv"
    test[["date", "region", TARGET, "pred_lr", "pred_rf"]].to_csv(pred_path, index=False)
    print(f"Predictions saved → {pred_path.name}")

    # 9. Save scaler params (needed if we want to run lr on new data)
    np.savez(MODELS_DIR / "scaler.npz", mean=X_mean, std=X_std)

    return lr, rf, metrics


if __name__ == "__main__":
    main()
