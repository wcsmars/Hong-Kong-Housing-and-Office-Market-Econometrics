"""
run_policy.py — The footprint of demand-side cooling measures.

Designs:
 1. ITS: Δlog(price) and Δlog(volume) on policy step dummies + 6-month pulse
    dummies, controlling for lagged dependent, Δ(1m HIBOR), Δlog real income,
    and US rate changes. HAC(6) inference.
 2. Jordà local projections (h = 0..24): cumulative log price and log volume
    responses to (a) a pooled "tightening adoption" indicator (SSD, BSD, DSD,
    NRSD adoption months), (b) the single 2024 withdrawal event. 90/95% HAC
    bands. Pooling raises power: 4 adoption events vs 1 withdrawal — the
    asymmetry in precision is itself reported.
Outputs: output/policy_its_price.txt, policy_its_volume.txt,
         output/lp_{tightening,ease}_{price,volume}.csv, policy.json
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import policy
from pipeline import build_panel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

panel = build_panel()
panel["dlp"] = panel["lp_nom"].diff()          # monthly log price change
panel["dlv"] = panel["lvol"].diff()            # monthly log volume change
panel["d_hibor"] = panel["hibor1m"].diff()
panel["d_ff"] = panel["fedfunds"].diff()
panel["dlinc"] = panel["linc_real"].diff()

dummies = policy.build_policy_dummies(panel.index)
controls = panel[["d_hibor", "d_ff", "dlinc"]]

res = {}

# 1 ---------------------------------------------------------------------- ITS
step_cols = [c for c in dummies.columns if c.endswith("_step")]
pulse_cols = [c for c in dummies.columns if c.endswith("_pulse6")]

for dep_name, dep in [("price", panel["dlp"]), ("volume", panel["dlv"])]:
    fit = policy.its_regression(dep.rename("y"), controls, dummies,
                                step_cols + pulse_cols, n_ar=3, hac_lags=6)
    with open(os.path.join(OUTD, f"policy_its_{dep_name}.txt"), "w") as f:
        f.write(str(fit.summary()))
    keep = {c: {"b": round(float(fit.params[c]), 4),
                "t": round(float(fit.tvalues[c]), 2),
                "p": round(float(fit.pvalues[c]), 3)}
            for c in step_cols + pulse_cols}
    res[f"its_{dep_name}"] = keep
    print(f"\n--- ITS {dep_name} (monthly log change) ---")
    for c, v in keep.items():
        star = "*" if v["p"] < 0.10 else " "
        print(f"  {c:>12}: {v['b']: .4f} (t={v['t']: .2f}){star}")

# 2 ---------------------------------------------------------- local projections
tight_adopt = (dummies[[f"{m}_step" for m in ["SSD", "BSD", "DSD", "NRSD"]]]
               .diff().max(axis=1).fillna(0.0).rename("shock"))
ease_adopt = dummies["EASE_step"].diff().fillna(0.0).rename("shock")

for shock_name, shock in [("tightening", tight_adopt), ("ease", ease_adopt)]:
    for dep_name, dep in [("price", panel["lp_nom"]), ("volume", panel["lvol"])]:
        irf = policy.local_projection(dep, shock, controls, H=24, n_ar=3)
        irf.to_csv(os.path.join(OUTD, f"lp_{shock_name}_{dep_name}.csv"))
        peak = irf["beta"].abs().idxmax()
        res[f"lp_{shock_name}_{dep_name}"] = {
            "h12_beta": round(float(irf.loc[12, "beta"]), 4) if 12 in irf.index else None,
            "h12_sig10": bool(abs(irf.loc[12, "beta"]) > 1.645 * irf.loc[12, "se"]) if 12 in irf.index else None,
            "peak_h": int(peak),
            "peak_beta": round(float(irf.loc[peak, "beta"]), 4),
        }
        print(f"LP {shock_name}->{dep_name}: h=12 beta="
              f"{res[f'lp_{shock_name}_{dep_name}']['h12_beta']}, "
              f"peak at h={peak}")

with open(os.path.join(OUTD, "policy.json"), "w") as f:
    json.dump(res, f, indent=2)
print("\nsaved output/policy.json")
