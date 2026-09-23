"""
run_forecast.py — rolling out-of-sample race for the Grade A office rent index.

Target: log REAL Grade A rent index (monthly). Fundamental for ECM/GBM:
real 1m HIBOR. Sample 1996-07+ (HIBOR availability); forecast origins from
60% of the sample; horizons 1, 3, 12 months; DM-HLN inference vs random walk.
The local forecast.py uses the same model families as the housing study.

Outputs: output/oos_h{1,3,12}.csv, output/oos_summary.csv,
         output/forecast_log.txt
"""
from __future__ import annotations

import os

import pandas as pd

from forecast import oos_summary, rolling_oos

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = pd.read_csv(os.path.join(PRO, "monthly_panel.csv"),
                        index_col=0, parse_dates=True)
    y = panel["lr_real_A"].rename("y")
    X = panel[["real_hibor"]]
    oos = rolling_oos(y, X, start_frac=0.6, horizons=(1, 3, 12), step=1)
    for h, df in oos.items():
        df.to_csv(os.path.join(OUT, f"oos_h{h}.csv"))
    summ = oos_summary(oos)
    summ.to_csv(os.path.join(OUT, "oos_summary.csv"), index=False)
    lines = [summ.round(4).to_string(index=False)]
    print(lines[0])
    with open(os.path.join(OUT, "forecast_log.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
