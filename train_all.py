#!/usr/bin/env python
"""
Train all models for ML Assignment 2.

- Runs each per-model script as a module (e.g., -m model.knn)
- Appends metrics to model/artifacts/metrics.csv
- Saves serialized models (*.pkl), scaler, and feature schema into model/artifacts/
- Options:
    --fresh  : remove existing metrics.csv before running (start clean)
    --strict : stop on first failure (non-zero exit code from a model script)

Usage (run from heart-ml/):
    python train_all.py
    python train_all.py --fresh
    python train_all.py --strict
    python train_all.py --fresh --strict
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent        # .../heart-ml
MODEL_DIR    = PROJECT_ROOT / "model"
ART_DIR      = MODEL_DIR / "artifacts"
DATA_PATH    = PROJECT_ROOT / "data" / "heart.csv"

MODULES = [
    "model.logistic_regression",
    "model.decision_tree",
    "model.knn",
    "model.naive_bayes",
    "model.random_forest",
    "model.xgboost_model",
]

def main():
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing model/artifacts/metrics.csv before running")
    parser.add_argument("--strict", action="store_true",
                        help="Stop on first failure (non-zero return code)")
    args = parser.parse_args()

    # Sanity checks
    if not DATA_PATH.exists():
        print(f" Dataset not found at: {DATA_PATH}")
        print("   Please place your CSV at data/heart.csv and retry.")
        sys.exit(2)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = ART_DIR / "metrics.csv"
    if args.fresh and metrics_path.exists():
        print(f"Removing existing metrics file: {metrics_path}")
        metrics_path.unlink()

    print(" Starting end-to-end training of all models...\n")

    results = []
    for mod in MODULES:
        print(f"\n=====  Running module: {mod} =====")
        # IMPORTANT: run as module from heart-ml so 'from model.utils ...' works
        rc = subprocess.call([sys.executable, "-m", mod], cwd=str(PROJECT_ROOT))
        print(f"===== {' SUCCESS' if rc==0 else f' FAIL (rc={rc})'} : {mod} =====")
        results.append((mod, rc))
        if args.strict and rc != 0:
            print("\n Strict mode is ON. Stopping on first failure.")
            sys.exit(rc)

    ok = sum(1 for _, rc in results if rc == 0)
    print("\n================ SUMMARY ================\n")
    for mod, rc in results:
        print(f"{'Success' if rc == 0 else 'Fail'} {mod} (rc={rc})")
    print(f"\nCompleted {ok}/{len(MODULES)}")
    sys.exit(0 if ok == len(MODULES) else 1)

if __name__ == "__main__":
    main()