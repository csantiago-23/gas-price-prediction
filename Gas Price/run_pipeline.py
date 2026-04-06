"""
run_pipeline.py
===============
End-to-end runner for the U.S. Gas Price Prediction pipeline.

Usage:
    python run_pipeline.py

Steps:
    1. Data collection   — generate EIA-style weekly dataset (10,000+ rows)
    2. Data cleaning     — clean, validate, engineer features
    3. EDA               — produce 4 exploratory plots
    4. Model training    — train Linear Regression & Random Forest from scratch
    5. Model evaluation  — produce 4 evaluation/prediction plots
"""

import sys
import time
from pathlib import Path

# Make sure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent / "scripts"))


def run_step(name, fn):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"  ✓ Completed in {elapsed:.1f}s")
    return result


def main():
    print("\n" + "█" * 60)
    print("  U.S. GAS PRICE PREDICTION — ML PIPELINE")
    print("  Python | NumPy | Pandas | Matplotlib")
    print("█" * 60)

    import data_collection   as step1
    import data_cleaning     as step2
    import eda               as step3
    import model_training    as step4
    import model_evaluation  as step5

    run_step("Data Collection",       step1.main)
    run_step("Data Cleaning",         step2.main)
    run_step("Exploratory Analysis",  step3.main)
    run_step("Model Training",        step4.main)
    run_step("Model Evaluation",      step5.main)

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print("  EDA plots    → plots/eda_*.png")
    print("  Eval plots   → plots/eval_*.png")
    print("  Metrics      → data/model_metrics.csv")
    print("  Predictions  → data/test_predictions.csv")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
