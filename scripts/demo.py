"""Run a small offline forecasting example using explicitly synthetic data.

Usage: python scripts/demo.py [--project housing|office]
No raw files, network access, or output-directory writes are needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

# Keep this small demonstration from spawning a large numerical thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=("housing", "office"), default="housing")
    args = parser.parse_args()
    source = ROOT / f"hk-{args.project}" / "src" / "forecast.py"
    spec = importlib.util.spec_from_file_location("synthetic_demo_forecast", source)
    forecast = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forecast)

    rng = np.random.default_rng(42)
    index = pd.date_range("2000-01-31", periods=100, freq="ME")
    fundamental = 2 + np.cumsum(rng.normal(0, 0.03, len(index)))
    residual = np.zeros(len(index))
    for t in range(1, len(index)):
        residual[t] = 0.65 * residual[t - 1] + rng.normal(0, 0.02)
    x = pd.DataFrame({"synthetic_fundamental": fundamental}, index=index)
    y = pd.Series(4 + 0.4 * fundamental + residual, index=index, name="synthetic_log_index")

    records = []
    for horizon in (1, 3):
        errors = {model: [] for model in ("RW", "RWD", "ECM", "GBM")}
        for origin in (79, 84, 89, 94):
            y_train, x_train = y.iloc[:origin + 1], x.iloc[:origin + 1]
            prediction = {
                "RW": float(y_train.iloc[-1]),
                "RWD": float(y_train.iloc[-1] + horizon * y_train.diff().tail(60).mean()),
                "ECM": forecast._fit_ecm_forecast(y_train, x_train, horizon),
                "GBM": forecast._fit_gbm_forecast(y_train, x_train, horizon),
            }
            for model, value in prediction.items():
                errors[model].append(value - float(y.iloc[origin + horizon]))
        baseline = float(np.sqrt(np.mean(np.square(errors["RW"]))))
        for model, error in errors.items():
            rmse = float(np.sqrt(np.mean(np.square(error))))
            records.append({"horizon": horizon, "model": model, "RMSE": rmse,
                            "relative_to_RW": rmse / baseline, "origins": len(error)})

    print(f"SYNTHETIC DEMO — {args.project} forecast engine; fixed random seed 42")
    print("100 invented monthly observations; four expanding-window origins per horizon.")
    print("Uses the project RW, drift, ECM and gradient-boosting specifications.\n")
    print(pd.DataFrame(records).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nThis checks that the software runs; these are not empirical market results.")
    print("The tiny example does not support model ranking or statistical significance claims.")


if __name__ == "__main__":
    main()
