"""
policy.py — Causal-ish evaluation of Hong Kong's demand-side cooling measures.

Two complementary designs:

1. Interrupted time series (ITS): monthly log-return (and log-volume)
   regressions with post-measure level/step dummies and short pulse windows,
   controls for macro fundamentals, HAC (Newey-West) inference. Identifies
   the *average* shift in dynamics after each measure conditional on
   fundamentals — credible to the extent measures are exogenous to the
   month-t shock (they respond to *past* appreciation, addressed by
   conditioning on lagged returns).

2. Jordà (2005) local projections: dynamic response of cumulative returns
   and volumes h = 0..H months after a measure dummy switches on,
       y_{t+h} - y_{t-1} = α_h + β_h·D_t + Γ_h'W_t + ε_{t+h},
   with W_t lagged returns + macro controls and Newey-West SEs (lag h+1,
   the standard LP correction for the MA(h) error this induces).

The measure set and dates (cross-checked against published announcements and prior studies):
   SSD  2010-11-20  Special Stamp Duty (resale within 24m taxed 5-15%)
   BSD  2012-10-27  Buyer's Stamp Duty (15% on non-PR buyers) + SSD enhanced
   DSD  2013-02-23  Doubled ad-valorem stamp duty (all > HK$2m)
   NRSD 2016-11-05  Flat 15% AVD on second+ homes
   EASE 2024-02-28  Full withdrawal of SSD/BSD/NRSD ("spicy measures" end)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MEASURES = {
    "SSD": "2010-11-30",
    "BSD": "2012-10-31",
    "DSD": "2013-02-28",
    "NRSD": "2016-11-30",
    "EASE": "2024-02-29",
}


def build_policy_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Step dummies (1 from measure month onward) and 6-month pulse dummies."""
    out = pd.DataFrame(index=index)
    for name, date in MEASURES.items():
        d = pd.Timestamp(date)
        out[f"{name}_step"] = (index >= d).astype(float)
        out[f"{name}_pulse6"] = ((index >= d) &
                                 (index < d + pd.DateOffset(months=6))).astype(float)
    return out


def its_regression(dep: pd.Series, controls: pd.DataFrame, dummies: pd.DataFrame,
                   dummy_cols: list[str], n_ar: int = 3, hac_lags: int = 6):
    """ITS with AR terms of the dependent variable + macro controls + dummies."""
    parts = {f"{dep.name}_l{l}": dep.shift(l) for l in range(1, n_ar + 1)}
    Z = pd.concat([pd.DataFrame(parts), controls, dummies[dummy_cols]], axis=1)
    aligned = pd.concat([dep, Z], axis=1).dropna()
    res = sm.OLS(aligned.iloc[:, 0], sm.add_constant(aligned.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return res


def local_projection(y: pd.Series, shock: pd.Series, controls: pd.DataFrame,
                     H: int = 24, n_ar: int = 3):
    """Jordà LP IRF of cumulative Δy on a policy step-change indicator.

    shock should be the first difference of a step dummy (=1 in the
    adoption month). Returns DataFrame h, beta, se, lo90, hi90, lo95, hi95.
    """
    dy = y.diff()
    rows = []
    for h in range(H + 1):
        lhs = (y.shift(-h) - y.shift(1)).rename("cum")
        parts = {"shock": shock}
        for l in range(1, n_ar + 1):
            parts[f"dy_l{l}"] = dy.shift(l)
        Z = pd.concat([pd.DataFrame(parts), controls], axis=1)
        aligned = pd.concat([lhs, Z], axis=1).dropna()
        if len(aligned) < 30:
            break
        res = sm.OLS(aligned["cum"], sm.add_constant(aligned.drop(columns="cum"))
                     ).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        b, s = res.params["shock"], res.bse["shock"]
        rows.append({"h": h, "beta": b, "se": s,
                     "lo90": b - 1.645 * s, "hi90": b + 1.645 * s,
                     "lo95": b - 1.96 * s, "hi95": b + 1.96 * s})
    return pd.DataFrame(rows).set_index("h")
