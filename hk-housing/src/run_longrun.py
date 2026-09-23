"""
run_longrun.py — Long-run equilibrium between real prices and fundamentals.

Three theory-grounded specifications (logs, monthly):
  S1 inverted demand:  lp_real ~ linc_real + real_blr + lsk_pc
                       (lsk_pc = log stock per capita: supply relative to
                        demographic demand)
  S2 parsimonious:     lp_real ~ linc_real + real_blr
  S3 user cost:        lpr ~ real_blr   (Poterba / HMS: log(P/R) ≈ −log(uc))

For each: ARDL bounds (full sample AND pre-2019 subsample), DOLS, ECM,
Gregory-Hansen cointegration with endogenous regime shift, Bai-Perron breaks
on the equilibrium error. Outputs: output/longrun.json, equilibrium_path.csv
(per spec), unit_roots.csv.
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import longrun
from pipeline import build_panel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

panel = build_panel()
panel["lsk_pc"] = panel["lstock"] - np.log(panel["population"])

SPECS = {
    "S1": ("lp_real", ["linc_real", "real_blr", "lsk_pc"]),
    "S2": ("lp_real", ["linc_real", "real_blr"]),
    "S3": ("lpr", ["real_blr"]),
    "S3H": ("lpr", ["real_hibor"]),
}
PRE_END = "2019-12-31"   # pre-shock subsample boundary

df = panel[["lp_real", "lpr", "linc_real", "real_blr", "lsk_pc", "lstock"]].dropna()
print(f"core sample: {df.index.min().date()} → {df.index.max().date()} (n={len(df)})")

res = {"sample": [str(df.index.min().date()), str(df.index.max().date())],
       "n": len(df), "specs": {}}

# unit roots ------------------------------------------------------------------
ur = longrun.unit_root_table(df, ["lp_real", "lpr", "linc_real", "real_blr",
                                  "lsk_pc"])
ur.to_csv(os.path.join(OUTD, "unit_roots.csv"))
print("\n--- unit roots (ADF level p / diff p / KPSS p / ZA p) ---")
print(ur[["ADF_level_p", "ADF_diff_p", "KPSS_p", "ZA_p"]].round(3).to_string())

# Johansen on the I(1) demand block -------------------------------------------
jo_tab, jo_beta = longrun.johansen(df[["lp_real", "linc_real", "lsk_pc"]],
                                   det_order=0, k_ar_diff=2)
jo_tab.to_csv(os.path.join(OUTD, "johansen.csv"))
res["johansen"] = jo_tab.reset_index().to_dict("records")
res["johansen_beta"] = jo_beta.round(4).to_dict()

for sname, (yv, xs) in SPECS.items():
    out = {"y": yv, "X": xs}
    sdf = panel[[yv] + xs].dropna()        # per-spec sample (HIBOR starts 1996-07)
    y, X = sdf[yv], sdf[xs]
    y_pre, X_pre = y.loc[:PRE_END], X.loc[:PRE_END]

    for tag, (yy, XX) in {"full": (y, X), "pre2019": (y_pre, X_pre)}.items():
        b = longrun.ardl_bounds(yy, XX, maxlag=6)
        d = longrun.dols(yy, XX, leads_lags=4, hac_lags=6)
        ec = longrun.ecm(yy, XX, d["lr_resid"], n_lags=2)
        out[tag] = {
            "n": int(len(yy.dropna())),
            "bounds_F": round(b["bounds_stat"], 3),
            "bounds_k": b["k"],
            "bounds_cv95": [round(float(b["bounds_crit"].loc[95.0, "lower"]), 3),
                            round(float(b["bounds_crit"].loc[95.0, "upper"]), 3)],
            "bounds_cv90": [round(float(b["bounds_crit"].loc[90.0, "lower"]), 3),
                            round(float(b["bounds_crit"].loc[90.0, "upper"]), 3)],
            "bounds_p": [round(float(b["bounds_pvals"]["lower"]), 4),
                         round(float(b["bounds_pvals"]["upper"]), 4)],
            "ardl_order": [int(v) for v in np.atleast_1d(b["ardl_order"])],
            "dols_beta": d["beta"].round(4).to_dict(),
            "dols_t": (d["beta"] / d["se"]).round(2).to_dict(),
            "ecm_lambda": round(ec["lambda"], 4),
            "ecm_lambda_t": round(ec["lambda_t"], 2),
            "half_life_m": round(ec["half_life_months"], 1),
        }
        if tag == "full":
            eq = pd.DataFrame({yv: y, "fitted": y - d["lr_resid"],
                               "ect": d["lr_resid"]})
            eq.columns = ["lp_real", "fitted", "ect"]  # uniform names for fig4
            eq.to_csv(os.path.join(OUTD, f"equilibrium_path_{sname}.csv"))
            bp = longrun.bai_perron_breaks(d["lr_resid"], max_breaks=5,
                                           min_size=36)
            out["bp_breaks"] = [str(dd.date()) for dd in bp["break_dates"]]

    # Gregory-Hansen with regime shift (and level shift for S3)
    for model in (["CS", "C"] if sname != "S3" else ["C", "CS"]):
        gh = longrun.gregory_hansen(y, X, model=model)
        out[f"gh_{model}"] = {
            "stat": round(gh["stat"], 3),
            "break": str(gh["break_date"].date()),
            "cv05": gh["cv"][0.05], "cv10": gh["cv"][0.10],
            "reject_5pct": bool(gh["reject_5pct"]),
        }

    res["specs"][sname] = out
    print(f"\n=== {sname}: {yv} ~ {' + '.join(xs)} ===")
    for tag in ["full", "pre2019"]:
        o = out[tag]
        print(f" [{tag}] n={o['n']} boundsF={o['bounds_F']} "
              f"(cv95 {o['bounds_cv95']}, p {o['bounds_p']}) | DOLS {o['dols_beta']} "
              f"| λ={o['ecm_lambda']} (t={o['ecm_lambda_t']}), HL={o['half_life_m']}m")
    for model in ["CS", "C"]:
        k = f"gh_{model}"
        if k in out:
            print(f" [GH-{model}] stat={out[k]['stat']} break={out[k]['break']} "
                  f"cv05={out[k]['cv05']} reject5%={out[k]['reject_5pct']}")
    if "bp_breaks" in out:
        print(f" [Bai-Perron ECT breaks] {out['bp_breaks']}")

with open(os.path.join(OUTD, "longrun.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print("\nsaved output/longrun.json")
