"""
vacancy.py — The natural vacancy rate and office rent adjustment.

Rent adjustment equation (Hendershott 1996; Wheaton-Torto 1988):
    Δ ln R_t = λ (v* − v_{t−1}) + ε_t  =  α + β v_{t−1} + ε_t,
with λ = −β and v* = α / λ.  Estimated by grade (A, B, C, total) on annual
data: year-end vacancy rates (RVD private_office.xls, 1985+) against
Q4-on-Q4 real rent growth built from the *quarterly* rent index (alignment:
RVD measures vacancy at year-end).

Extensions:
  (i)   delta-method and pairs-bootstrap CIs for v*;
  (ii)  asymmetric adjustment: λ⁺ (v above v*) vs λ⁻ (v below v*), grid-MLE
        with a wild-bootstrap test of λ⁺ = λ⁻;
  (iii) time-varying v*: local-level state-space model
            Δ ln R_t = λ (v*_t − v_{t−1}) + ε_t,   v*_t = v*_{t−1} + η_t,
        estimated by MLE (Kalman filter/smoother, statsmodels MLEModel);
  (iv)  demand-shift controls (real GDP growth) to check robustness of v*.

Outputs: output/vacancy.json, output/vstar_path.csv, output/vacancy_log.txt
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")
LOG: list[str] = []


def log(msg: str):
    LOG.append(str(msg))
    print(msg)


def build_annual() -> pd.DataFrame:
    ann = pd.read_csv(os.path.join(PRO, "annual_stockflow.csv"), index_col=0)
    q = pd.read_csv(os.path.join(PRO, "quarterly_indices.csv"), index_col=0)
    q.index = pd.PeriodIndex(q.index, freq="Q")
    q4 = q[q.index.quarter == 4].copy()
    q4.index = q4.index.year
    out = pd.DataFrame(index=ann.index)
    for g in ("A", "B", "C", "TOT"):
        gq = "ALL" if g == "TOT" else g
        # Q4 real rent (deflate by Q4 CPI), growth Dec-on-Dec
        rr = np.log(q4[f"rt_{gq}"] / q4["cpi"])
        out[f"dlr_{g}"] = rr.diff()
        out[f"vac_{g}"] = ann[f"vacrate_{g}"]
    out["dgdp"] = np.log(ann["gdp_real"]).diff()
    return out


# ----------------------------------------------------------- baseline OLS
def baseline(df: pd.DataFrame, g: str, n_boot: int = 5000, seed: int = 7) -> dict:
    d = pd.DataFrame({
        "y": df[f"dlr_{g}"],
        "vlag": df[f"vac_{g}"].shift(1),
    }).dropna()
    X = sm.add_constant(d["vlag"])
    res = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    a, b = res.params["const"], res.params["vlag"]
    lam = -b
    vstar = a / lam if lam > 0 else np.nan
    # delta method on v* = a / (-b)
    V = res.cov_params().values
    grad = np.array([1.0 / lam, a / lam**2])
    se_v = float(np.sqrt(grad @ V @ grad)) if lam > 0 else np.nan
    # pairs bootstrap
    rng = np.random.default_rng(seed)
    vs = []
    arr = d.values
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        yb, xb = arr[idx, 0], arr[idx, 1]
        Xb = np.column_stack([np.ones(len(xb)), xb])
        try:
            cb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            if cb[1] < 0:
                vs.append(cb[0] / -cb[1])
        except np.linalg.LinAlgError:
            continue
    vs = np.array(vs)
    ci = np.percentile(vs, [2.5, 97.5]) if len(vs) > 100 else [np.nan, np.nan]
    # robustness: control for contemporaneous GDP growth; v* evaluated at the
    # sample-mean GDP growth path (v* = rent-growth-neutral vacancy given
    # average demand growth), not at zero growth
    d2 = pd.DataFrame({"y": df[f"dlr_{g}"], "vlag": df[f"vac_{g}"].shift(1),
                       "dgdp": df["dgdp"]}).dropna()
    r2 = sm.OLS(d2["y"], sm.add_constant(d2[["vlag", "dgdp"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2})
    vstar_ctrl = float((r2.params["const"]
                        + r2.params["dgdp"] * d2["dgdp"].mean())
                       / -r2.params["vlag"]) \
        if r2.params["vlag"] < 0 else np.nan
    return {
        "grade": g, "n": int(res.nobs),
        "lambda": float(lam), "lambda_t": float(-res.tvalues["vlag"]),
        "alpha": float(a), "r2": float(res.rsquared),
        "vstar": float(vstar), "vstar_se_delta": se_v,
        "vstar_ci_boot": [float(ci[0]), float(ci[1])],
        "boot_kept": int(len(vs)), "n_boot": n_boot,
        "vstar_gdp_control": vstar_ctrl,
        "gdp_coef": float(r2.params["dgdp"]), "gdp_t": float(r2.tvalues["dgdp"]),
    }


# ----------------------------------------------------------- asymmetry
def asymmetry(df: pd.DataFrame, g: str, n_boot: int = 2000, seed: int = 11) -> dict:
    """Δln R = λ⁺(v*−v)·1[v>v*] + λ⁻(v*−v)·1[v≤v*]; grid over v*."""
    d = pd.DataFrame({"y": df[f"dlr_{g}"], "vlag": df[f"vac_{g}"].shift(1)}).dropna()
    grid = np.linspace(d["vlag"].quantile(0.15), d["vlag"].quantile(0.85), 61)

    def fit(y, v, vs):
        gap = vs - v
        above = (v > vs).astype(float)
        X = np.column_stack([gap * above, gap * (1 - above)])
        beta, ssr = np.linalg.lstsq(X, y, rcond=None)[:2]
        ssr = ssr[0] if len(ssr) else np.sum((y - X @ beta) ** 2)
        return beta, ssr

    best = None
    for vs in grid:
        beta, ssr = fit(d["y"].values, d["vlag"].values, vs)
        if best is None or ssr < best[2]:
            best = (vs, beta, ssr)
    vs_hat, (lam_p, lam_m), ssr_u = best
    # restricted: single lambda
    gap = vs_hat - d["vlag"].values
    lam_r = float(np.linalg.lstsq(gap[:, None], d["y"].values, rcond=None)[0])
    ssr_r = float(np.sum((d["y"].values - lam_r * gap) ** 2))
    F = (ssr_r - ssr_u) / (ssr_u / (len(d) - 2))
    # wild bootstrap under H0 (symmetric)
    rng = np.random.default_rng(seed)
    resid = d["y"].values - lam_r * gap
    Fb = []
    for _ in range(n_boot):
        w = rng.choice([-1.0, 1.0], len(resid))
        yb = lam_r * gap + resid * w
        bb, ssru_b = None, None
        for vsb in grid:
            beta_b, ssr_b = fit(yb, d["vlag"].values, vsb)
            if ssru_b is None or ssr_b < ssru_b:
                ssru_b, bb = ssr_b, beta_b
        gap_b = vs_hat - d["vlag"].values
        lam_rb = float(np.linalg.lstsq(gap_b[:, None], yb, rcond=None)[0])
        ssrr_b = float(np.sum((yb - lam_rb * gap_b) ** 2))
        Fb.append((ssrr_b - ssru_b) / (ssru_b / (len(d) - 2)))
    p = float(np.mean(np.array(Fb) >= F))
    return {
        "grade": g, "vstar_grid": float(vs_hat),
        "lambda_above": float(lam_p), "lambda_below": float(lam_m),
        "F_sym": float(F), "p_wild": p, "n_boot": n_boot,
    }


# ----------------------------------------------------- time-varying v*
def time_varying(df: pd.DataFrame, g: str, lam: float) -> tuple[dict, pd.DataFrame]:
    """Profile-likelihood two-step: with λ fixed at the OLS estimate, the
    model Δln R_t = λ(v*_t − v_{t−1}) + ε, v*_t = v*_{t−1} + η reduces to a
    local-level model on the adjusted series y_t + λ·v_{t−1} = λ·v*_t + ε.
    (Free-λ MLE is unidentified at λ→0 on n≈40; the two-step pins λ to the
    rent-adjustment speed and lets the Kalman smoother allocate the rest.)"""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    d = pd.DataFrame({"y": df[f"dlr_{g}"], "vlag": df[f"vac_{g}"].shift(1)}).dropna()
    yadj = d["y"] + lam * d["vlag"]
    mod = UnobservedComponents(yadj.values, level="local level")
    res = mod.fit(disp=False, maxiter=500)
    lvl = res.smoothed_state[0] / lam
    lvl_se = np.sqrt(res.smoothed_state_cov[0, 0]) / lam
    path = pd.DataFrame({"year": d.index, "vstar_smoothed": lvl,
                         "se": lvl_se}).set_index("year")
    snr = float(res.params[1] / res.params[0]) if res.params[0] > 0 else np.inf
    info = {
        "grade": g, "lambda_fixed": float(lam),
        "var_eps": float(res.params[0]), "var_eta": float(res.params[1]),
        "signal_noise": snr, "loglike": float(res.llf),
        "vstar_first5_mean": float(lvl[:5].mean()),
        "vstar_last5_mean": float(lvl[-5:].mean()),
        "vstar_2019": float(path.loc[2019, "vstar_smoothed"]) if 2019 in path.index else np.nan,
        "vstar_end": float(lvl[-1]),
        "converged": bool(res.mle_retvals.get("converged", True)),
    }
    return info, path


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build_annual()
    results = {"baseline": [], "asymmetry": [], "time_varying": []}
    paths = {}
    for g in ("A", "B", "C", "TOT"):
        b = baseline(df, g)
        results["baseline"].append(b)
        log(f"[{g}] λ={b['lambda']:.3f} (t={b['lambda_t']:.2f})  "
            f"v*={b['vstar']:.3%} ±{b['vstar_se_delta']:.3%} "
            f"boot95={[round(x, 4) for x in b['vstar_ci_boot']]} "
            f"(n={b['n']}, R²={b['r2']:.2f}; v* w/GDP ctrl={b['vstar_gdp_control']:.3%})")
        a = asymmetry(df, g)
        results["asymmetry"].append(a)
        log(f"    asym: λ⁺={a['lambda_above']:.3f} λ⁻={a['lambda_below']:.3f} "
            f"(F={a['F_sym']:.2f}, wild-p={a['p_wild']:.3f}, v*grid={a['vstar_grid']:.3%})")
        try:
            tv, path = time_varying(df, g, b["lambda"])
            results["time_varying"].append(tv)
            paths[g] = path
            log(f"    TV v*: λ={tv['lambda_fixed']:.3f}  "
                f"v*(first5)={tv['vstar_first5_mean']:.3%} → v*(last5)={tv['vstar_last5_mean']:.3%} "
                f"(2019={tv['vstar_2019']:.3%}, end={tv['vstar_end']:.3%})")
        except Exception as e:
            log(f"    TV v* failed for {g}: {e}")

    allp = pd.concat({g: p["vstar_smoothed"] for g, p in paths.items()}, axis=1)
    allp.columns = [f"vstar_{g}" for g in allp.columns]
    vac = df[[c for c in df.columns if c.startswith("vac_")]]
    allp.join(vac).to_csv(os.path.join(OUT, "vstar_path.csv"))

    with open(os.path.join(OUT, "vacancy.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open(os.path.join(OUT, "vacancy_log.txt"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
