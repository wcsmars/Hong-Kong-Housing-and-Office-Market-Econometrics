"""
psy.py — Recursive right-tailed unit-root tests for explosive behaviour.

Implements, from first principles:

  * ADF statistic on an arbitrary subsample window [s, e] with k lagged
    differences (fitted intercept, no trend) — the regression of
    Phillips, Shi & Yu (2015, Int. Econ. Rev.), eq. (1):

        Δy_t = α + β·y_{t-1} + Σ_{i=1..k} ψ_i Δy_{t-i} + ε_t,
        ADF = β̂ / se(β̂).

  * SADF  (Phillips, Wu & Yu, 2011): sup over forward-expanding windows
    anchored at the start of the sample.
  * GSADF (Phillips, Shi & Yu, 2015): sup over all windows [r1, r2] with
    r2 - r1 ≥ r0 (double-sup).
  * BSADF sequence: for each endpoint r2, the backward sup ADF — used for
    real-time date-stamping of bubble origination/termination.
  * Monte-Carlo critical values (null: driftless random walk), including a
    per-endpoint critical-value *sequence* for date-stamping.
  * A wild-bootstrap p-value for the GSADF statistic, robust to
    unconditional heteroskedasticity (in the spirit of Phillips & Shi,
    2020, who show volatility shifts oversize the asymptotic test).

For the k = 0 case (used in all simulation loops, as in PSY's own critical
value computations) the windowed ADF statistic is computed in O(1) per
window from prefix sums, which makes the O(T²)-window double-sup feasible
inside Monte-Carlo loops:

    regression sample for window [s, e]:  t = s+1, …, e   (n = e - s obs)
    x_t = y_{t-1},  d_t = Δy_t
    β̂  = (n·S_xd − S_x S_d) / (n·S_xx − S_x²)
    α̂  = (S_d − β̂ S_x) / n
    RSS = S_dd − α̂ S_d − β̂ S_xd          (since RSS = d'd − b̂'X'd)
    se(β̂) = sqrt( σ̂² n / (n·S_xx − S_x²) ),  σ̂² = RSS/(n−2)

All prefix sums are over the full series, so each window statistic is a
handful of float ops. Everything is JIT-compiled with numba; the Monte-Carlo
loop is parallelised with prange.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# --------------------------------------------------------------------------
# Windowed ADF statistics
# --------------------------------------------------------------------------


@njit(cache=True)
def _prefix_sums(y):
    """Prefix sums of the cross-moments needed for the k=0 windowed ADF.

    Index convention: sums[t] aggregates observations 1..t of the
    regression sample built on (x_t = y[t-1], d_t = y[t] - y[t-1]).
    """
    T = y.shape[0]
    Sx = np.zeros(T)
    Sxx = np.zeros(T)
    Sd = np.zeros(T)
    Sxd = np.zeros(T)
    Sdd = np.zeros(T)
    for t in range(1, T):
        x = y[t - 1]
        d = y[t] - y[t - 1]
        Sx[t] = Sx[t - 1] + x
        Sxx[t] = Sxx[t - 1] + x * x
        Sd[t] = Sd[t - 1] + d
        Sxd[t] = Sxd[t - 1] + x * d
        Sdd[t] = Sdd[t - 1] + d * d
    return Sx, Sxx, Sd, Sxd, Sdd


@njit(cache=True)
def _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, s, e):
    """ADF t-stat on window [s, e] (0-based, inclusive), k = 0, via prefix sums."""
    n = float(e - s)
    if n < 3.0:
        return -np.inf
    sx = Sx[e] - Sx[s]
    sxx = Sxx[e] - Sxx[s]
    sd = Sd[e] - Sd[s]
    sxd = Sxd[e] - Sxd[s]
    sdd = Sdd[e] - Sdd[s]
    det = n * sxx - sx * sx
    if det <= 1e-12:
        return -np.inf
    beta = (n * sxd - sx * sd) / det
    alpha = (sd - beta * sx) / n
    rss = sdd - alpha * sd - beta * sxd
    if rss <= 0.0:
        return -np.inf
    sigma2 = rss / (n - 2.0)
    se = np.sqrt(sigma2 * n / det)
    if se <= 0.0:
        return -np.inf
    return beta / se


@njit(cache=True)
def _adf_window_klags(y, s, e, k):
    """ADF t-stat on window [s, e] with k lagged differences (intercept, no trend).

    Solves the normal equations directly; p = k + 2 regressors.
    """
    p = k + 2
    n = e - s - k  # usable observations: t = s+k+1 .. e
    if n < p + 2:
        return -np.inf
    XtX = np.zeros((p, p))
    Xty = np.zeros(p)
    yty = 0.0
    for t in range(s + k + 1, e + 1):
        d = y[t] - y[t - 1]
        # regressor vector: [1, y_{t-1}, dy_{t-1}, ..., dy_{t-k}]
        row = np.empty(p)
        row[0] = 1.0
        row[1] = y[t - 1]
        for i in range(1, k + 1):
            row[1 + i] = y[t - i] - y[t - i - 1]
        for a in range(p):
            Xty[a] += row[a] * d
            for b in range(a, p):
                XtX[a, b] += row[a] * row[b]
        yty += d * d
    for a in range(p):
        for b in range(a):
            XtX[a, b] = XtX[b, a]
    # solve
    coef = np.linalg.solve(XtX, Xty)
    rss = yty
    for a in range(p):
        rss -= coef[a] * Xty[a]
    if rss <= 0.0:
        return -np.inf
    sigma2 = rss / (n - p)
    XtX_inv = np.linalg.inv(XtX)
    var_b = sigma2 * XtX_inv[1, 1]
    if var_b <= 0.0:
        return -np.inf
    return coef[1] / np.sqrt(var_b)


def adf_stat(y: np.ndarray, k: int = 0, s: int = 0, e: int | None = None) -> float:
    """Public single-window ADF statistic (intercept, k lagged differences)."""
    y = np.asarray(y, dtype=np.float64)
    if e is None:
        e = len(y) - 1
    if k == 0:
        sums = _prefix_sums(y)
        return _adf_window_k0(*sums, s, e)
    return _adf_window_klags(y, s, e, k)


# --------------------------------------------------------------------------
# BSADF / GSADF / SADF
# --------------------------------------------------------------------------


@njit(cache=True)
def _bsadf_sequence_k0(y, w0):
    """BSADF_t for every endpoint t ≥ w0-1, k = 0 (prefix-sum fast path)."""
    T = y.shape[0]
    Sx, Sxx, Sd, Sxd, Sdd = _prefix_sums(y)
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        best = -np.inf
        for s in range(0, e - w0 + 2):
            stat = _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, s, e)
            if stat > best:
                best = stat
        out[e] = best
    return out


@njit(cache=True)
def _bsadf_sequence_klags(y, w0, k):
    T = y.shape[0]
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        best = -np.inf
        for s in range(0, e - w0 + 2):
            stat = _adf_window_klags(y, s, e, k)
            if stat > best:
                best = stat
        out[e] = best
    return out


@njit(cache=True)
def _sadf_sequence_k0(y, w0):
    """Forward-expanding ADF (anchored at 0) for every endpoint — PWY (2011)."""
    T = y.shape[0]
    Sx, Sxx, Sd, Sxd, Sdd = _prefix_sums(y)
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        out[e] = _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, 0, e)
    return out


@njit(cache=True)
def _sadf_sequence_klags(y, w0, k):
    """Forward-expanding ADF with k lagged differences for every endpoint."""
    T = y.shape[0]
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        out[e] = _adf_window_klags(y, 0, e, k)
    return out


def min_window(T: int) -> int:
    """PSY (2015) rule of thumb: r0 = 0.01 + 1.8/√T, w0 = ⌈r0·T⌉."""
    return int(np.ceil(T * (0.01 + 1.8 / np.sqrt(T))))


def bsadf(y: np.ndarray, k: int = 0, w0: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if w0 is None:
        w0 = min_window(len(y))
    if k == 0:
        return _bsadf_sequence_k0(y, w0)
    return _bsadf_sequence_klags(y, w0, k)


def gsadf(y: np.ndarray, k: int = 0, w0: int | None = None) -> float:
    return float(np.max(bsadf(y, k=k, w0=w0)))


def sadf(y: np.ndarray, k: int = 0, w0: int | None = None) -> float:
    y = np.asarray(y, dtype=np.float64)
    if w0 is None:
        w0 = min_window(len(y))
    if k == 0:
        return float(np.max(_sadf_sequence_k0(y, w0)))
    return float(np.max(_sadf_sequence_klags(y, w0, k)))


# --------------------------------------------------------------------------
# Monte-Carlo critical values (null: driftless random walk)
# --------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _mc_null_bsadf(T, w0, n_reps, seed):
    """BSADF_t sequences under H0: y_t = y_{t-1} + ε_t, ε ~ iid N(0,1).

    Returns (n_reps, T) matrix of null BSADF sequences (k = 0, as in PSY's
    own critical-value simulations).
    """
    out = np.full((n_reps, T), -np.inf)
    for r in prange(n_reps):
        np.random.seed(seed + r)
        eps = np.random.standard_normal(T)
        y = np.cumsum(eps)
        Sx, Sxx, Sd, Sxd, Sdd = _prefix_sums(y)
        for e in range(w0 - 1, T):
            best = -np.inf
            for s in range(0, e - w0 + 2):
                stat = _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, s, e)
                if stat > best:
                    best = stat
            out[r, e] = best
    return out


def mc_critical_values(T: int, w0: int | None = None, n_reps: int = 999,
                       quantiles=(0.90, 0.95, 0.99), seed: int = 42):
    """Monte-Carlo critical values for GSADF and the BSADF date-stamping sequence.

    Returns
    -------
    dict with:
      'gsadf_cv'    : {q: scalar} critical values for the global GSADF stat
      'bsadf_cv_seq': {q: (T,) array} per-endpoint critical values for BSADF_t
      'null_gsadf'  : (n_reps,) simulated null GSADF distribution
    """
    if w0 is None:
        w0 = min_window(T)
    null_seq = _mc_null_bsadf(T, w0, n_reps, seed)
    null_gsadf = np.max(null_seq, axis=1)
    gsadf_cv = {q: float(np.quantile(null_gsadf, q)) for q in quantiles}
    bsadf_cv_seq = {}
    for q in quantiles:
        cv = np.full(T, np.nan)
        cv[w0 - 1:] = np.quantile(null_seq[:, w0 - 1:], q, axis=0)
        bsadf_cv_seq[q] = cv
    return {"gsadf_cv": gsadf_cv, "bsadf_cv_seq": bsadf_cv_seq,
            "null_gsadf": null_gsadf, "w0": w0}


# --------------------------------------------------------------------------
# Wild bootstrap (heteroskedasticity-robust GSADF inference)
# --------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _wild_bootstrap_gsadf(eps_hat, w0, n_reps, seed):
    """Wild bootstrap of the GSADF statistic.

    Null is imposed by building y*_t = Σ_{s≤t} w_s ε̂_s with w_s iid N(0,1),
    where ε̂ are the demeaned first differences of the observed series. The
    resampled series inherits the *time profile* of the observed volatility,
    so the bootstrap distribution is robust to unconditional variance shifts
    (Phillips & Shi, 2020).
    Returns (n_reps,) of null GSADF stats and (n_reps, T) BSADF sequences.
    """
    T = eps_hat.shape[0] + 1
    stats = np.empty(n_reps)
    seqs = np.full((n_reps, T), -np.inf)
    for r in prange(n_reps):
        np.random.seed(seed + 7919 * r)
        w = np.random.standard_normal(T - 1)
        y = np.empty(T)
        y[0] = 0.0
        for t in range(1, T):
            y[t] = y[t - 1] + w[t - 1] * eps_hat[t - 1]
        Sx, Sxx, Sd, Sxd, Sdd = _prefix_sums(y)
        best_all = -np.inf
        for e in range(w0 - 1, T):
            best = -np.inf
            for s in range(0, e - w0 + 2):
                stat = _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, s, e)
                if stat > best:
                    best = stat
            seqs[r, e] = best
            if best > best_all:
                best_all = best
        stats[r] = best_all
    return stats, seqs


def wild_bootstrap_gsadf(y: np.ndarray, k: int = 0, w0: int | None = None,
                         n_reps: int = 499, seed: int = 7,
                         quantiles=(0.90, 0.95, 0.99)):
    """Wild-bootstrap p-value for GSADF + bootstrap BSADF critical-value sequence."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if w0 is None:
        w0 = min_window(T)
    eps_hat = np.diff(y) - np.mean(np.diff(y))
    null_stats, null_seqs = _wild_bootstrap_gsadf(eps_hat, w0, n_reps, seed)
    obs = gsadf(y, k=k, w0=w0)
    pval = float((np.sum(null_stats >= obs) + 1) / (n_reps + 1))
    bsadf_cv_seq = {}
    for q in quantiles:
        cv = np.full(T, np.nan)
        cv[w0 - 1:] = np.quantile(null_seqs[:, w0 - 1:], q, axis=0)
        bsadf_cv_seq[q] = cv
    return {"gsadf_obs": obs, "pvalue": pval, "null_gsadf": null_stats,
            "bsadf_cv_seq": bsadf_cv_seq, "w0": w0}


# --------------------------------------------------------------------------
# Date-stamping
# --------------------------------------------------------------------------


def date_stamp(bsadf_seq: np.ndarray, cv_seq: np.ndarray, index,
               min_duration: int | None = None):
    """PSY date-stamping: episodes where BSADF_t > cv_t for ≥ min_duration periods.

    min_duration defaults to ⌈log(T)⌉ observations (PSY's δ·log(T) rule with
    δ = 1), filtering out one-month blips.
    Returns list of dicts {start, end, peak, peak_stat, n_periods}.
    """
    T = len(bsadf_seq)
    if min_duration is None:
        min_duration = int(np.ceil(np.log(T)))
    above = (bsadf_seq > cv_seq) & np.isfinite(bsadf_seq) & np.isfinite(cv_seq)
    episodes = []
    t = 0
    while t < T:
        if above[t]:
            start = t
            while t < T and above[t]:
                t += 1
            end = t - 1  # inclusive
            if end - start + 1 >= min_duration:
                seg = bsadf_seq[start:end + 1]
                peak_rel = int(np.argmax(seg))
                episodes.append({
                    "start": index[start], "end": index[end],
                    "peak": index[start + peak_rel],
                    "peak_stat": float(seg[peak_rel]),
                    "n_periods": int(end - start + 1),
                })
        else:
            t += 1
    return episodes
