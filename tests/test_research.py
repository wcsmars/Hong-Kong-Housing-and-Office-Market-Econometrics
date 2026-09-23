"""Offline regression checks: synthetic inputs and bundled result snapshots.

Run from the repository root: python -m unittest discover -s tests -v
These checks do not certify the empirical designs or require raw datasets.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("hk-housing", "hk-office")


def load_module(project: str, name: str):
    # Numba stores the module name in its disk cache. Keep PSY's usual import
    # name so this suite can share cached kernels with src/test_psy.py.
    key = "psy" if name == "psy" else f"public_{project.replace('-', '_')}_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            key, ROOT / project / "src" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


class RecursiveADFTests(unittest.TestCase):
    def test_adf_matches_statsmodels_for_full_and_partial_windows(self):
        psy = load_module(PROJECTS[0], "psy")
        y = np.cumsum(np.random.default_rng(1729).normal(size=180))
        for lag in (0, 1, 4):
            for start, end in ((0, 179), (31, 128)):
                with self.subTest(lag=lag, start=start, end=end):
                    expected = adfuller(
                        y[start:end + 1], maxlag=lag,
                        regression="c", autolag=None,
                    )[0]
                    actual = psy.adf_stat(y, k=lag, s=start, e=end)
                    self.assertAlmostEqual(actual, expected, places=8)

    def test_sadf_uses_requested_lag(self):
        psy = load_module(PROJECTS[0], "psy")
        y = np.cumsum(np.random.default_rng(314).normal(size=120))
        w0 = 30
        for lag in (0, 1, 2):
            with self.subTest(lag=lag):
                expected = max(
                    adfuller(y[:end + 1], maxlag=lag, regression="c", autolag=None)[0]
                    for end in range(w0 - 1, len(y))
                )
                self.assertAlmostEqual(psy.sadf(y, k=lag, w0=w0), expected, places=8)


class BoundsTestTests(unittest.TestCase):
    def test_critical_values_and_p_values_count_only_regressors(self):
        # PSS (2001) case III, 5%: (I(0), I(1)) bounds by number of regressors.
        published = {1: (4.92, 5.72), 2: (3.80, 4.81), 3: (3.23, 4.32)}
        for project in PROJECTS:
            module = load_module(project, "longrun")
            for k, bounds in published.items():
                with self.subTest(project=project, k=k):
                    crit, _ = module.pss_bounds(1.0, k=k)
                    self.assertAlmostEqual(crit.loc[95, "lower"], bounds[0], places=2)
                    self.assertAlmostEqual(crit.loc[95, "upper"], bounds[1], places=2)
                    # A statistic at the 5% critical value has a p-value near 5%.
                    _, pvals = module.pss_bounds(crit.loc[95, "upper"], k=k)
                    self.assertAlmostEqual(pvals["upper"], 0.05, delta=0.005)

    def test_housing_bounds_test_keeps_every_regressor(self):
        # AIC drops the irrelevant first regressor here, so mapping the
        # selected orders onto the leading columns would test "noise" alone.
        rng = np.random.default_rng(0)
        n = 200
        idx = pd.date_range("2000-01-31", periods=n, freq="ME")
        signal = np.cumsum(rng.normal(size=n))
        noise = np.cumsum(rng.normal(size=n))
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.5 * u[t - 1] + rng.normal(scale=0.5)
        y = pd.Series(1.0 + 0.8 * signal + u, index=idx, name="y")
        X = pd.DataFrame({"noise": noise, "signal": signal}, index=idx)
        longrun = load_module(PROJECTS[0], "longrun")
        unconstrained = longrun.ardl_select_order(y, 3, X, 3, trend="c", ic="aic")
        self.assertIsNone(unconstrained.aic.iloc[0][1]["noise"])

        result = longrun.ardl_bounds(y, X, maxlag=3)
        levels = [name for name in result["uecm_res"].model.exog_names
                  if name.endswith(".L1") and not name.startswith("D.")]
        self.assertEqual(levels, ["y.L1", "noise.L1", "signal.L1"])
        self.assertEqual(result["k"], 2)
        self.assertEqual(len(result["ardl_order"]), 3)
        self.assertAlmostEqual(result["bounds_crit"].loc[95, "upper"], 4.81, places=2)


class DistrictDispersionTests(unittest.TestCase):
    def test_balanced_panel_is_not_moved_by_a_late_entrant(self):
        breaks = load_module(PROJECTS[1], "breaks")
        rng = np.random.default_rng(5)
        idx = pd.date_range("2013-01-31", periods=144, freq="ME")
        rel = pd.DataFrame({
            "ar_A_North": 0.2 + 0.01 * rng.normal(size=144),
            "ar_A_South": -0.2 + 0.01 * rng.normal(size=144),
            "ar_A_East": -1.1 + 0.01 * rng.normal(size=144),
        }, index=idx)
        rel.loc[:"2016-12-31", "ar_A_East"] = np.nan  # enters in 2017

        _, observed = breaks.sigma_split(rel)
        _, balanced = breaks.sigma_split(rel.dropna(how="any"))
        self.assertGreater(observed["post2019_mean"] - observed["pre2019_mean"], 0.2)
        self.assertAlmostEqual(balanced["pre2019_mean"], balanced["post2019_mean"], delta=0.01)
        self.assertEqual(balanced["start"], "2017-01-31")
        self.assertEqual(balanced["districts"], ["Central", "North", "South", "East"])


class ForecastRecurrenceTests(unittest.TestCase):
    def test_ecm_uses_latest_change_as_first_lag(self):
        # Distinct final changes (4 and 3) make reversed lag order observable.
        # Fit coefficients are fixed, so this checks iteration rather than
        # comparing two copies of the same estimation routine.
        idx = pd.date_range("2000-01-31", periods=5, freq="ME")
        y = pd.Series([2.0, 3.0, 5.0, 8.0, 12.0], index=idx, name="y")
        x = pd.DataFrame({"x": [1., 2., 3., 4., 5.]}, index=idx)
        long_params = pd.Series({"const": 0.5, "x": 0.4})
        short_params = pd.Series({
            "const": 0.1, "ect_l1": -0.2, "dy_l1": 0.6, "dy_l2": -0.1,
        })

        # Independent scalar recurrence: level=12, equilibrium=.5+.4*5.
        level, latest_change, preceding_change = 12.0, 4.0, 3.0
        expected = {}
        for horizon in range(1, 5):
            next_change = (
                0.1 - 0.2 * (level - 2.5)
                + 0.6 * latest_change - 0.1 * preceding_change
            )
            level += next_change
            preceding_change, latest_change = latest_change, next_change
            expected[horizon] = level
        self.assertAlmostEqual(expected[1], 12.3)
        self.assertAlmostEqual(expected[2], 10.22)

        for project in PROJECTS:
            module = load_module(project, "forecast")
            for horizon in (1, 2, 4):
                long_fit = SimpleNamespace(
                    params=long_params,
                    predict=lambda design: design @ long_params,
                )
                short_fit = SimpleNamespace(params=short_params)
                fitted_models = [
                    SimpleNamespace(fit=lambda: long_fit),
                    SimpleNamespace(fit=lambda: short_fit),
                ]
                with self.subTest(project=project, horizon=horizon):
                    with patch.object(module.sm, "OLS", side_effect=fitted_models):
                        actual = module._fit_ecm_forecast(y, x, horizon)
                    self.assertAlmostEqual(actual, expected[horizon], places=10)


class CalendarValidationTests(unittest.TestCase):
    @staticmethod
    def worksheet(months, all_column):
        data = pd.DataFrame(np.nan, index=range(len(months)), columns=range(30))
        data = data.astype(object)
        data.iloc[0, 1] = "2004"  # The year can be text in RVD workbooks.
        for row, month in enumerate(months):
            data.iloc[row, 5] = month
            data.iloc[row, all_column] = f"({100 + row}.0)"
        return data

    def parse(self, project, months):
        module = load_module(project, "pipeline")
        if project == PROJECTS[0]:
            parser, all_column = module.parse_rvd_monthly, 29
        else:
            parser, all_column = module.parse_grade_monthly, 17
        with patch.object(module.pd, "read_excel", return_value=self.worksheet(months, all_column)):
            return parser("synthetic.xls", "px")

    def test_accepts_complete_calendar_and_text_year(self):
        for project in PROJECTS:
            with self.subTest(project=project):
                result = self.parse(project, [1, 2, 3])
                expected = pd.date_range("2004-01-31", periods=3, freq="ME")
                pd.testing.assert_index_equal(result.index, expected, check_names=False)
                np.testing.assert_array_equal(result["px_ALL"], [100., 101., 102.])

    def test_rejects_duplicate_months(self):
        for project in PROJECTS:
            with self.subTest(project=project):
                with self.assertRaisesRegex(ValueError, "duplicated"):
                    self.parse(project, [1, 2, 2])

    def test_rejects_missing_months(self):
        for project in PROJECTS:
            with self.subTest(project=project):
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.parse(project, [1, 3])


class SavedForecastSnapshotTests(unittest.TestCase):
    def test_saved_summaries_agree_with_forecast_rows(self):
        for project in PROJECTS:
            output = ROOT / project / "output"
            summary = pd.read_csv(output / "oos_summary.csv")
            self.assertFalse(summary.duplicated(["horizon", "model"]).any())
            self.assertEqual(set(summary["horizon"]), {1, 3, 12})
            forecasts = {}
            for horizon in (1, 3, 12):
                rows = pd.read_csv(output / f"oos_h{horizon}.csv", index_col=0)
                self.assertFalse(rows.index.duplicated().any())
                forecasts[horizon] = rows
                baseline = np.sqrt(np.mean((rows["RW"] - rows["actual"]) ** 2))
                for _, saved in summary.loc[summary["horizon"] == horizon].iterrows():
                    with self.subTest(project=project, horizon=horizon, model=saved["model"]):
                        error = rows[saved["model"]] - rows["actual"]
                        self.assertTrue(np.isfinite(error).all())
                        rmse = np.sqrt(np.mean(error ** 2))
                        self.assertAlmostEqual(rmse, saved["RMSE"], places=12)
                        self.assertAlmostEqual(rmse / baseline, saved["rel_RW"], places=12)
                        self.assertEqual(len(rows), saved["n"])

            # Also check the stored DM statistics use the documented spacing.
            module = load_module(project, "forecast")
            actual = module.oos_summary(forecasts, step=2 if project == PROJECTS[0] else 1)
            keys = ["horizon", "model"]
            actual = actual.set_index(keys).sort_index()
            expected = summary.set_index(keys).sort_index()
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
