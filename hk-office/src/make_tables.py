"""
make_tables.py — Emit the summary tables as Markdown from output/*.json|csv.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")
TAB = os.path.join(ROOT, "tables")


def w(name: str, text: str):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)
    print("wrote", name)


def t1_data():
    panel = pd.read_csv(os.path.join(PRO, "monthly_panel.csv"),
                        index_col=0, parse_dates=True)
    ann = pd.read_csv(os.path.join(PRO, "annual_stockflow.csv"), index_col=0)
    rows = [
        ("Rent index, Grade A/B/C/all (monthly)", "1993-01 – 2026-04", "RVD his_data_8"),
        ("Price index, Grade A/B/C/all (monthly)", "1993-01 – 2026-04 (5–8 unpublished months)", "RVD his_data_9"),
        ("Rent & price indices by grade (quarterly)", "1978Q1 / 1986Q1 – 2026Q1", "RVD his_data_8/9"),
        ("Grade A district rent indices (monthly)", "1993-01 – 2026-04", "RVD his_data_10"),
        ("Average rents/prices, grade × district (monthly)", "1999-01 – 2026-04", "RVD his_data_6/7"),
        ("Completions, stock, vacancy, take-up by grade (annual)", "1985 – 2025", "RVD private_office"),
        ("1m HIBOR (monthly avg)", "1996-07 – 2026-05", "HKMA API"),
        ("Best Lending Rate (monthly avg)", "1980-01 – 2026-05", "HKMA API"),
        ("Composite CPI (monthly)", "1980-10 – 2026-04", "C&SD"),
        ("US 10y constant maturity, Fed funds (monthly)", "1953 – 2026-05", "FRED"),
        ("Real GDP, constant LCU (annual)", "1961 – 2024", "World Bank API"),
    ]
    md = "| Series | Coverage | Source |\n|---|---|---|\n"
    md += "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    yldrow = panel[["yld_A_mean", "yld_B_mean", "yld_C_mean"]].dropna()
    md += ("\n\nConstructed office yields (12·rent/price from RVD levels), "
           f"monthly 1999-01 – {yldrow.index.max():%Y-%m}: "
           f"Grade A mean {panel['yld_A_mean'].mean():.2%}, "
           f"B {panel['yld_B_mean'].mean():.2%}, C {panel['yld_C_mean'].mean():.2%}.")
    w("t1_data.md", md)


def t2_unit_roots():
    ur = pd.read_csv(os.path.join(OUT, "unit_roots.csv"))
    keep = ["series", "n", "ADF", "ADF_p", "DFGLS_p", "PP_p", "KPSS", "KPSS_p", "ZA", "ZA_p"]
    ur = ur[[c for c in keep if c in ur.columns]]
    w("t2_unit_roots.md", ur.round(3).to_markdown(index=False))


def t3_cointegration():
    jo = pd.read_csv(os.path.join(OUT, "johansen.csv"))
    lr = json.load(open(os.path.join(OUT, "longrun.json")))
    md = "**Johansen trace tests (monthly, det=const):**\n\n"
    md += jo.round(3).to_markdown(index=False)
    md += "\n\n**Single-equation estimates of lpr on real rate:**\n\n"
    md += ("| spec | F (PSS) | p (I(1)) | EC coef (t) | LR semi-elast. | half-life | DOLS β (t) |\n"
           "|---|---|---|---|---|---|---|\n")
    for y, x in (("lpr_A", "real_hibor"), ("lpr_ALL", "real_hibor"),
                 ("lpr_A", "real_gs10")):
        b = lr[f"bounds_{y}_{x}"]; d = lr[f"dols_{y}_{x}"]
        md += (f"| {y} ~ {x} | {b['F_PSS']:.2f} | {b['F_p_upper']:.3f} | "
               f"{b['ec_coef']:.4f} ({b['ec_t']:.2f}) | "
               f"{b['longrun_semi_elasticity']:.4f} | {b['half_life_months']:.0f}m | "
               f"{d['beta']:.4f} ({d['t']:.2f}) |\n")
    cs = lr["campbell_shiller"]
    md += (f"\n**Campbell–Shiller decomposition** (quarterly {cs['sample'][0]}–{cs['sample'][1]}, "
           f"VAR({cs['var_lags']}), ρ={cs['rho']:.4f}, n={cs['n']}): "
           f"expected rent growth {cs['share_rent_growth']:.2f} "
           f"[{cs['ci_rent'][0]:.2f}, {cs['ci_rent'][1]:.2f}]; "
           f"expected returns {cs['share_expected_returns']:.2f} "
           f"[{cs['ci_returns'][0]:.2f}, {cs['ci_returns'][1]:.2f}], "
           f"of which risk-free {cs['share_riskfree']:.2f} "
           f"[{cs['ci_riskfree'][0]:.2f}, {cs['ci_riskfree'][1]:.2f}] and "
           f"risk premium {cs['share_risk_premium']:.2f} "
           f"({cs['n_boot_ok']} bootstrap replications).")
    w("t3_cointegration.md", md)


def t4_vacancy():
    v = json.load(open(os.path.join(OUT, "vacancy.json")))
    md = ("| grade | λ (t) | v* | delta-SE | bootstrap 95% | v* (GDP ctrl) | "
          "λ⁺ / λ⁻ (wild-p) | TV v*: first5 → last5 |\n"
          "|---|---|---|---|---|---|---|---|\n")
    tv = {t["grade"]: t for t in v["time_varying"]}
    asym = {a["grade"]: a for a in v["asymmetry"]}
    for b in v["baseline"]:
        g = b["grade"]; a = asym[g]; t = tv.get(g)
        md += (f"| {g} | {b['lambda']:.2f} ({b['lambda_t']:.2f}) | "
               f"{b['vstar']:.1%} | ±{b['vstar_se_delta']:.1%} | "
               f"[{b['vstar_ci_boot'][0]:.1%}, {b['vstar_ci_boot'][1]:.1%}] | "
               f"{b['vstar_gdp_control']:.1%} | "
               f"{a['lambda_above']:.2f} / {a['lambda_below']:.2f} ({a['p_wild']:.2f}) | "
               + (f"{t['vstar_first5_mean']:.1%} → {t['vstar_last5_mean']:.1%} |"
                  if t else "– |") + "\n")
    md += ("\nΔln(real rent, Q4/Q4) on lagged year-end vacancy, annual 1986–2025 "
           "(n=39). HAC(2) t-stats; pairs bootstrap 5,000 reps; asymmetry by "
           "grid-search threshold with 2,000 wild-bootstrap reps; time-varying "
           "v* from a local-level Kalman smoother with λ fixed at the OLS value.")
    w("t4_vacancy.md", md)


def t5_stockflow():
    s = json.load(open(os.path.join(OUT, "stockflow.json")))
    sup, dem = s["supply"], s["demand"]
    md = ("**Supply (completions):** "
          f"ln comp on ln real price (lag {sup['lag_years']}y): elasticity "
          f"{sup['price_elasticity']:.2f} (t={sup['price_t']:.2f}), real BLR "
          f"semi-elasticity {sup['rate_semi']:.3f} (t={sup['rate_t']:.2f}), "
          f"R²={sup['r2']:.2f}, n={sup['n']}. The negative price coefficient "
          "indicates completions are governed by land-disposal policy and the "
          "construction lag, not by market prices — supply is treated as "
          "exogenous in the simulation.\n\n"
          "**Demand (occupied stock, DOLS):** "
          f"income elasticity {dem['income_elasticity']:.3f} (t={dem['income_t']:.1f}), "
          f"rent elasticity {dem['rent_elasticity']:.3f} (t={dem['rent_t']:.1f}); "
          f"ECM adjustment {dem['ecm_coef']:.3f} (t={dem['ecm_t']:.1f}), n={dem['n']}.\n\n"
          "**Scenario summary (total-market vacancy):**\n\n")

    def fmt(recs):
        sc = pd.DataFrame(recs)
        sc["vac_2030"] = sc["vac_2030"].map("{:.1%}".format)
        sc["vac_end"] = sc["vac_end"].map("{:.1%}".format)
        sc["norm_year"] = sc["norm_year"].fillna(">2035")
        return sc.to_markdown(index=False)

    md += fmt(s["scenarios"])
    md += (f"\n\nPipeline base = mean completions of the last five observed "
           f"years (2021–2025), {s['pipeline_mean_5y']:,.0f} m²/yr; absorbed "
           f"end-2025 demand residual {s['resid_2025_logpts']:+.3f} log "
           "points (occupancy ~7.6% below the historical schedule, treated "
           "as permanent in the baseline).\n\n"
           f"**Sensitivity — adjustment speed at the 95% upper bound "
           f"(κ = {s['kappa_upper']:.3f} vs floor 0.10):**\n\n")
    md += fmt(s["scenarios_kappa_upper"])
    md += ("\n\n**Sensitivity — transitory demand shock "
           "(2025 residual decays linearly over 5 years):**\n\n")
    md += fmt(s["scenarios_demand_recovery"])
    w("t5_stockflow.md", md)


def t6_breaks():
    b = json.load(open(os.path.join(OUT, "breaks.json")))
    md = "**Bai–Perron mean breaks (BIC, 15% trim):**\n\n"
    for k in ("bp_rent_growth", "bp_central_premium"):
        r = b[k]
        md += (f"- {r['series']}: {r['n_breaks']} breaks at "
               f"{', '.join(r['break_dates'])}; segment means "
               f"{[round(m, 4) for m in r['segment_means']]} (n={r['n_obs']})\n")
    m = b["markov"]
    md += (f"\n**Markov-switching (Grade A real rent growth, %/m):** calm regime "
           f"μ={m['mu_regime0_pct_m']:.2f}, σ={m['sigma_regime0']:.2f} "
           f"(expected duration {m['exp_duration0_m']:.0f}m); crisis regime "
           f"μ={m['mu_regime1_pct_m']:.2f}, σ={m['sigma_regime1']:.2f} "
           f"({m['exp_duration1_m']:.0f}m); n={m['n']}.\n")
    c = b["convergence"]

    def sigma(key: str, label: str) -> str:
        s = c[key]
        return (f"{label} ({len(s['districts'])} districts, {s['start'][:7]}+) "
                f"{s['pre2019_mean']:.3f} → {s['post2019_mean']:.3f} "
                f"(Levene p={s['levene_p']:.4f})")

    md += (f"\n**District convergence (Grade A average rents; five districts "
           f"from 1999-01, Mong Kok with gaps, Kwun Tong from 2013-02):** "
           f"σ-dispersion pre-2019 → 2019+: "
           f"{sigma('sigma_balanced', 'balanced panel')}; "
           f"{sigma('sigma_balanced_ex_kwuntong', 'balanced, excluding Kwun Tong')}; "
           f"{sigma('sigma_all_observed', 'all observed districts, mix changes')}; "
           f"β-convergence {c['beta']:.3f} (t={c['beta_t']:.1f}, half-life "
           f"{c['beta_half_life_m']:.0f}m). Central/KwunTong premium: mean "
           f"{c['premium_mean']:.3f} log-pts, ADF p={c['premium_adf_p']:.2f}.\n\n"
           "Pairwise Engle–Granger with Central (p-values): "
           + ", ".join(f"{d} {v['eg_p']:.3f}" for d, v in c["pairwise_eg"].items()))
    w("t6_breaks.md", md)


def t7_oos():
    p = os.path.join(OUT, "oos_summary.csv")
    if not os.path.exists(p):
        print("skip t7 (forecast race still running)")
        return
    s = pd.read_csv(p)
    piv = s.pivot(index="model", columns="horizon",
                  values=["RMSE", "rel_RW", "DM_p"]).round(4)
    md = ("Rolling-origin OOS, log real Grade A rent index; origins from 60% "
          "of sample, horizons in months; DM–HLN p-values vs random walk.\n\n"
          + piv.to_markdown())
    w("t7_oos.md", md)


def main():
    os.makedirs(TAB, exist_ok=True)
    t1_data(); t2_unit_roots(); t3_cointegration(); t4_vacancy()
    t5_stockflow(); t6_breaks(); t7_oos()


if __name__ == "__main__":
    main()
