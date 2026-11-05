"""Demand forecasting for spare parts with intermittent demand.

What it does, per item x warehouse series:
1. Classifies the demand pattern (smooth, erratic, intermittent, lumpy, new, inactive)
   using the Syntetos-Boylan ADI / CV^2 scheme.
2. Runs a library of methods built for this kind of demand: moving average (the SAP style
   baseline), simple exponential smoothing, Croston, SBA, TSB, ADIDA temporal aggregation,
   a seasonal variant, and one global gradient boosted model trained across all items.
3. Tests every method on rolling windows the model never saw, then picks a method per
   series. Series with thin evidence fall back to the best method for their demand class.
4. Reports honest accuracy on a final holdout window that was not used for selection.
5. Produces the inputs for probabilistic simulation: chance of a sale in a period, the
   item's own size distribution, and a model error spread learned from the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METHOD_LABEL = {
    "ma": "Moving average of recent periods (baseline: 13 weeks or 12 months)",
    "ses": "Simple exponential smoothing",
    "croston": "Croston",
    "sba": "Syntetos-Boylan approximation (SBA)",
    "tsb": "Teunter-Syntetos-Babai (TSB)",
    "adida": "Temporal aggregation (ADIDA)",
    "seasonal": "Seasonal smoothing (category season profile)",
    "ml": "Gradient boosted model (global, all items)",
    "trend": "Yearly average with category growth (trended baseline)",
    "combo": "Combination (average of moving average, smoothing, TSB and ADIDA)",
    "zero": "No demand expected",
}
SIMPLE_ORDER = ["ma", "ses", "sba", "croston", "tsb", "adida", "combo", "trend", "seasonal", "ml"]


@dataclass
class ForecastResult:
    table: pd.DataFrame          # one row per series
    path: np.ndarray             # series x horizon mean forecast per period
    accuracy: pd.DataFrame       # holdout accuracy summary
    method_scores: pd.DataFrame  # selection scores by class and method
    season_index: dict           # category -> 53 week index (or None)
    sim_params: dict             # per series inputs for simulation
    level_sigma: dict            # demand class -> lognormal sigma of model error (set by calibration)
    holdout_detail: pd.DataFrame
    holdout_origin: int | None = None
    holdout_path: np.ndarray | None = None
    holdout_actual: np.ndarray | None = None
    calib_origin: int | None = None
    calib_path: np.ndarray | None = None
    calib_actual: np.ndarray | None = None
    quantile_map: dict | None = None   # demand class -> (target grid, adjusted quantile grid)


# ---------------------------------------------------------------------------- smoothing kernels
# Each kernel walks through time for all series at once and returns the one-step-ahead
# forecast level available at the end of every column: L[:, c] = forecast made with data < c+1.

def _ses(M, start, alpha):
    S, T = M.shape
    L = np.zeros((S, T))
    lvl = np.zeros(S)
    for c in range(T):
        y = M[:, c]
        live = c >= start
        init = c == start
        lvl = np.where(init, y, np.where(live, lvl + alpha * (y - lvl), 0.0))
        L[:, c] = lvl
    return L


def _ma(M, start, n=13):  # n = 12 for monthly planning: the flat 12-month average baseline
    S, T = M.shape
    cs = np.cumsum(M, axis=1)
    L = np.zeros((S, T))
    for c in range(T):
        lo = np.maximum(start, c - n + 1)
        width = np.maximum(c - lo + 1, 1)
        prev = np.where(lo > 0, cs[np.arange(S), np.maximum(lo - 1, 0)], 0.0)
        prev = np.where(lo > 0, prev, 0.0)
        L[:, c] = np.where(c >= start, (cs[:, c] - prev) / width, 0.0)
    return L


def _croston(M, start, alpha, sba=False):
    S, T = M.shape
    L = np.zeros((S, T))
    z = np.zeros(S)
    p = np.ones(S)
    q = np.zeros(S)          # periods since last demand
    seen = np.zeros(S, bool)
    for c in range(T):
        y = M[:, c]
        live = c >= start
        q = np.where(live, q + 1, 0)
        hit = live & (y > 0)
        first = hit & ~seen
        upd = hit & seen
        z = np.where(first, y, np.where(upd, z + alpha * (y - z), z))
        p = np.where(first, q, np.where(upd, p + alpha * (q - p), p))
        seen |= hit
        q = np.where(hit, 0, q)
        f = np.where(seen, z / np.maximum(p, 1e-9), 0.0)
        if sba:
            f = f * (1 - alpha / 2)
        L[:, c] = f
    return L


def _tsb(M, start, a_d, a_p):
    S, T = M.shape
    L = np.zeros((S, T))
    prob = np.zeros(S)
    z = np.zeros(S)
    seen = np.zeros(S, bool)
    for c in range(T):
        y = M[:, c]
        live = c >= start
        hit = live & (y > 0)
        first = hit & ~seen
        upd = live & seen & ~first
        z = np.where(first, y, np.where(upd & hit, z + a_d * (y - z), z))
        # first demand: start the probability at 1 / (periods it took to see a sale)
        prob = np.where(first, 1.0 / np.maximum(c - start + 1, 1),
                        np.where(upd, prob + a_p * (hit.astype(float) - prob), prob))
        seen |= hit
        L[:, c] = np.where(seen, prob * z, 0.0)
    return L


def _adida(M, start, k):
    """Aggregate into blocks of k periods ending at each column, smooth the blocks, divide by k."""
    S, T = M.shape
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(M, axis=1)], axis=1)
    L = np.zeros((S, T))
    alpha = 0.2
    for r in range(k):  # one smoothing chain per block phase
        lvl = np.zeros(S)
        init = np.zeros(S, bool)
        for c in range(r, T, k):
            lo = c - k + 1
            if lo < 0:
                continue
            blk = cs[:, c + 1] - cs[:, lo]
            live = lo >= start
            lvl = np.where(live & ~init, blk, np.where(live, lvl + alpha * (blk - lvl), lvl))
            init |= live
            L[:, c] = lvl / k  # level of the block that ends at column c
    return L


# ---------------------------------------------------------------------------- classification

def dead_periods(fc, bucket):
    """Dead stock threshold in periods (dead_stock_months, or the older dead_after_periods)."""
    if fc.get("dead_stock_months") is not None:
        return max(int(round(float(fc["dead_stock_months"]) * periods_per_year(bucket) / 12)), 1)
    return int(fc.get("dead_after_periods", 52))


def classify(M, start, last_n=104, min_hist=13, dead_after=52, sold_recently=None):
    """sold_recently: series that sold after the history ends (the unfinished month): never dead or at risk."""
    S, T = M.shape
    lo = np.maximum(start, T - last_n)
    rows = []
    for i in range(S):
        y = M[i, lo[i]:]
        n = len(y)
        nz = y[y > 0]
        adi = n / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        hist = T - start[i]
        last_sale = T - 1 - np.max(np.nonzero(M[i] > 0)[0]) if (M[i] > 0).any() else T
        if sold_recently is not None and sold_recently[i]:
            last_sale = 0
        if (M[i] > 0).sum() == 0 or last_sale >= dead_after:
            cls = "inactive"
        elif hist < min_hist:
            cls = "new"
        elif adi < 1.32:
            cls = "smooth" if cv2 < 0.49 else "erratic"
        else:
            cls = "intermittent" if cv2 < 0.49 else "lumpy"
        rows.append((adi, cv2, hist, last_sale, len(nz), cls))
    return pd.DataFrame(rows, columns=["adi", "cv2", "history_periods", "periods_since_sale", "nonzero_periods", "demand_class"])


# ---------------------------------------------------------------------------- seasonality

def periods_per_year(bucket):
    return 12 if bucket == "M" else 52


def _pos_in_year(dates, bucket):
    """0-based position inside the year: month (0-11) or ISO week (0-52)."""
    d = pd.DatetimeIndex(dates)
    if bucket == "M":
        return d.month.to_numpy().astype(int) - 1
    return np.minimum(d.isocalendar().week.to_numpy().astype(int), 53) - 1


def season_profiles(M, keys, items, period_end, bucket):
    """Seasonal index per category (by week or by month of year), kept only where the pattern is real."""
    P = periods_per_year(bucket)
    if M.shape[1] < 2 * P:
        return {}
    cat = keys["item_code"].map(items.set_index("item_code")["category"]).fillna("Uncategorized").to_numpy()
    pos = np.minimum(_pos_in_year(period_end, bucket), P - 1)
    win = 7 if bucket == "W" else 3
    out = {}
    for c in np.unique(cat):
        agg = M[cat == c].sum(axis=0)
        if agg.sum() < 200:
            continue
        s = pd.Series(agg)
        trend = s.rolling(P, center=True, min_periods=int(P * 0.75)).mean()
        ratio = (s / trend).to_numpy()
        ok = np.isfinite(ratio)
        if ok.sum() < P:
            continue
        idx = np.ones(P)
        cnt = np.zeros(P)
        for w, r in zip(pos[ok], ratio[ok]):
            idx[w] += r
            cnt[w] += 1
        raw = np.where(cnt > 0, (idx - 1) / np.maximum(cnt, 1), 1.0)
        # circular smoothing, then shrink toward 1 (few years of evidence = weak season)
        h = win // 2
        ext = np.concatenate([raw[-h:], raw, raw[:h]])
        sm = np.convolve(ext, np.ones(win) / win, mode="valid")
        years = cnt.mean()
        shrink = years / (years + 1.0)
        sm = 1 + shrink * (sm - 1)
        sm = sm / sm.mean()
        # keep only when the season explains a meaningful share of period to period noise
        fitted = sm[pos[ok]]
        resid = ratio[ok] / fitted
        if np.var(ratio[ok]) > 0 and (1 - np.var(resid) / np.var(ratio[ok])) > 0.15 and (sm.max() - sm.min()) > 0.08:
            out[c] = np.append(sm, sm[-1]) if bucket == "W" else sm  # week 53 uses week 52
    return out


def _season_matrix(keys, items, season, period_dates, bucket="W"):
    """Index for each series at each date. Returns None if no category is seasonal."""
    if not season:
        return None
    cat = keys["item_code"].map(items.set_index("item_code")["category"]).fillna("Uncategorized").to_numpy()
    pos = _pos_in_year(period_dates, bucket)
    out = np.ones((len(keys), len(period_dates)))
    for c, idx in season.items():
        m = cat == c
        out[m] = idx[np.minimum(pos, len(idx) - 1)][None, :]
    return out


def _category_growth(M, keys, items, P, shrink=0.5, lo=0.8, hi=1.25):
    """Growth factor per series at each column: category demand in the last P periods over the P before, shrunk and capped."""
    S, T = M.shape
    G = np.ones((S, T))
    if T < 2 * P:
        return G
    cat = keys["item_code"].map(items.set_index("item_code")["category"]).fillna("Uncategorized").to_numpy()
    for c in np.unique(cat):
        m = cat == c
        agg = np.concatenate([[0.0], np.cumsum(M[m].sum(axis=0))])
        g = np.ones(T)
        for col in range(2 * P - 1, T):
            last = agg[col + 1] - agg[col + 1 - P]
            prev = agg[col + 1 - P] - agg[col + 1 - 2 * P]
            if prev > 50 and last > 0:
                g[col] = np.clip(1 + shrink * (last / prev - 1), lo, hi)
        G[m] = g[None, :]
    return G


# ---------------------------------------------------------------------------- global ML model

def _ml_features(M, start, c, cat_codes, wh_codes, price_log, woy_c, P=52):
    """Features describing each series using only columns < c. P = periods per year."""
    S = M.shape[0]
    feats = []
    wins = (1, 4, 13, 26, 52) if P == 52 else (1, 3, 6, 12, 24)
    for n in wins:
        lo = max(c - n, 0)
        feats.append(M[:, lo:c].mean(axis=1) if c > 0 else np.zeros(S))
    for n in (wins[2], wins[4]):
        lo = max(c - n, 0)
        feats.append((M[:, lo:c] > 0).mean(axis=1) if c > 0 else np.zeros(S))
    window = M[:, max(c - P, 0):c]
    nzc = (window > 0).sum(axis=1)
    feats.append(np.where(nzc > 0, window.sum(axis=1) / np.maximum(nzc, 1), 0))
    pos = M[:, :c] > 0
    feats.append(np.where(pos.any(axis=1), np.argmax(pos[:, ::-1], axis=1), 999))  # periods since last sale
    feats.append(c - start)
    feats += [cat_codes, wh_codes, price_log, np.full(S, np.sin(2 * np.pi * woy_c / P)), np.full(S, np.cos(2 * np.pi * woy_c / P))]
    return np.column_stack(feats)


def _ml_levels(M, start, origins, H, keys, items, period_end, train_until, bucket="W"):
    """Fit a Poisson gradient boosted model on windows ending before train_until; predict level at each origin."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    S, T = M.shape
    imap = items.set_index("item_code")
    cat_codes = pd.Categorical(keys["item_code"].map(imap["category"]).fillna("U")).codes.astype(float)
    wh_codes = pd.Categorical(keys["warehouse"]).codes.astype(float)
    price_log = np.log1p(keys["item_code"].map(imap["sell_price"]).fillna(0).to_numpy(float))
    P = periods_per_year(bucket)
    woy = _pos_in_year(period_end, bucket) + 1
    X, Y = [], []
    first_c, step, min_age = (26, 4, 8) if P == 52 else (12, 1, 4)
    for c in range(first_c, train_until - H + 1, step):
        live = (c - start) >= min_age
        if not live.any():
            continue
        F = _ml_features(M, start, c, cat_codes, wh_codes, price_log, woy[c - 1], P)
        X.append(F[live])
        Y.append(M[live, c:c + H].mean(axis=1))
    if not X:
        return None
    X = np.vstack(X)
    Y = np.concatenate(Y)
    if len(Y) < 500:
        return None
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    model.fit(X, Y)
    out = {}
    for c in origins:
        nxt = woy[c - 1] if c - 1 < len(woy) else woy[-1]
        F = _ml_features(M, start, c, cat_codes, wh_codes, price_log, nxt, P)
        out[c] = np.clip(model.predict(F), 0, None)
    return out


# ---------------------------------------------------------------------------- main entry

def run(prep, settings) -> ForecastResult:
    fc = settings["forecast"]
    global METHOD_LABEL
    METHOD_LABEL = dict(METHOD_LABEL, ma="Flat 12-month average (baseline)" if prep.bucket == "M" else "13-week moving average (baseline)")
    M = prep.matrix
    S, T = M.shape
    start = prep.first_idx
    H = int(fc.get("backtest_horizon", 8))
    n_orig = int(fc.get("backtest_origins", 4))
    horizon = int(fc.get("horizon_periods", 26))
    # origins: column index where the test window starts; we forecast using columns < origin
    P = periods_per_year(prep.bucket)
    min_train = 26 if P == 52 else 12
    origins = [T - k * H for k in range(n_orig, 0, -1) if T - k * H > min_train]
    if len(origins) < 2:  # short history: one test window only
        origins = [o for o in [T - H] if o > 4]
    holdout = origins[-1] if origins else None
    select_origins = origins[:-1] if len(origins) > 1 else origins

    dead_after = dead_periods(fc, prep.bucket)
    cls = classify(M, start, last_n=2 * P, min_hist=int(fc.get("min_history_periods", 13)), dead_after=dead_after,
                   sold_recently=getattr(prep, "sold_after_history", None))
    at_risk = int(round(float(fc.get("at_risk_months", 12)) * P / 12))
    cls["at_risk"] = (cls["periods_since_sale"] >= at_risk) & (cls["demand_class"] != "inactive")
    adi_k = np.clip(np.round(cls["adi"].to_numpy()), 2, 13).astype(int)

    season = season_profiles(M, prep.keys, prep.items, prep.period_end, prep.bucket)
    S_hist = _season_matrix(prep.keys, prep.items, season, prep.period_end, prep.bucket)
    if prep.bucket == "M" and prep.first_forecast_start is not None:
        future_dates = prep.future_period_starts(max(horizon, H))
    else:
        future_dates = [prep.as_of + pd.Timedelta(days=prep.period_days * (h + 1)) for h in range(max(horizon, H))]
    bk = prep.bucket

    # one-step level matrices for each method (forecast available after column c).
    # Where a category has a proven season, every method works on deseasonalized demand and the season is put back
    # on the forecast (season_adjust_all); otherwise only the "seasonal" method uses it.
    adj = bool(fc.get("season_adjust_all", True)) and S_hist is not None
    M_raw = M
    if adj:
        M = M / np.maximum(S_hist, 1e-6)
    levels = {}
    # the baseline is always the plain average of raw demand (flat 12-month average when monthly), never adjusted
    levels["ma"] = _ma(M_raw, start, n=13 if P == 52 else 12)
    ma_adj = _ma(M, start, n=13 if P == 52 else 12)
    for a in (0.05, 0.1, 0.2, 0.3):
        levels[f"ses_{a}"] = _ses(M, start, a)
    levels["croston"] = _croston(M, start, 0.1)
    levels["sba"] = _croston(M, start, 0.1, sba=True)
    for ad, ap in ((0.1, 0.05), (0.1, 0.1), (0.2, 0.1), (0.1, 0.2)):
        levels[f"tsb_{ad}_{ap}"] = _tsb(M, start, ad, ap)
    ad_lv = np.zeros((S, T))
    for k in np.unique(adi_k):
        m = adi_k == k
        ad_lv[m] = _adida(M[m], start[m], int(k))
    levels["adida"] = ad_lv
    # forecast combination: the average of simple, robust methods (combinations win most forecasting competitions)
    # (Croston and SBA are left out: they do not decay while an item stops selling)
    levels["combo"] = (ma_adj + levels["ses_0.1"] + levels["tsb_0.1_0.1"] + levels["adida"]) / 4.0
    # trended average: the yearly average carried forward with the category's own growth (last year vs the year
    # before, pooled over the category, halved and capped), so a growing or shrinking product line is not lagged
    G = _category_growth(M_raw, prep.keys, prep.items, P)
    lagp = (P - 1) / 2.0 + 1  # periods between the middle of the averaging year and the first forecast period
    levels["trend"] = _ma(M, start, n=P)
    if S_hist is not None and not adj:
        Md = M / np.maximum(S_hist, 1e-6)
        levels["seasonal"] = _ses(Md, start, 0.1)  # deseasonalized level, reseasonalized below
    M = M_raw

    def level_at(name, c):
        """Mean forecast path for the H periods starting at column c, using data < c."""
        L = levels[name][:, c - 1]
        if name == "trend":
            g = G[:, c - 1][:, None] ** ((lagp + np.arange(H)[None, :]) / P)
            base = L[:, None] * g
            if adj:
                idx = S_hist[:, c:c + H] if c + H <= T else _season_matrix(prep.keys, prep.items, season, future_dates[:H], bk)
                base = base * idx
            return base
        if name == "seasonal" or (adj and name != "ma"):
            idx = S_hist[:, c:c + H] if c + H <= T else _season_matrix(prep.keys, prep.items, season, future_dates[:H], bk)
            return L[:, None] * idx
        return np.repeat(L[:, None], H, axis=1)

    ml_levels = None
    if fc.get("use_ml_model", True) and origins and S >= 30:
        try:
            if P == 12:  # monthly data is small: refit at every origin on everything before it (expanding window)
                ml_levels = {}
                for c in origins:
                    r_ = _ml_levels(M, start, [c], H, prep.keys, prep.items, prep.period_end, train_until=c, bucket=bk)
                    if r_ is None:
                        ml_levels = None
                        break
                    ml_levels[c] = r_[c]
            else:
                ml_levels = _ml_levels(M, start, origins, H, prep.keys, prep.items, prep.period_end, train_until=origins[0], bucket=bk)
        except Exception:  # noqa: BLE001  ML is optional; never block the run
            ml_levels = None

    candidates = [k for k in levels]
    if ml_levels is not None:
        candidates.append("ml")

    # scale: in-sample mean squared one-step naive change, per series (RMSSE denominator)
    def scale_for(c):
        out = np.zeros(S)
        for i in range(S):
            y = M[i, start[i]:c]
            d = np.diff(y)
            out[i] = np.mean(d ** 2) if len(d) > 1 else 0.0
        out[out <= 0] = np.nan
        fallback = np.nanmedian(out) if np.isfinite(out).any() else 1.0
        return np.where(np.isnan(out), fallback, out)

    metric = str(fc.get("selection_metric", "rmsse")).lower()
    scores = {m: [] for m in candidates}
    abserr = {m: [] for m in candidates}  # absolute error of the window total (units), for the baseline guard
    totals = {m: {} for m in candidates}  # origin -> forecast total over H
    actual_tot = {}
    for c in origins:
        act = M[:, c:c + H]
        actual_tot[c] = act.sum(axis=1)
        sc = scale_for(c)
        for m in candidates:
            f = np.repeat(ml_levels[c][:, None], H, axis=1) if m == "ml" else level_at(m, c)
            totals[m][c] = f.sum(axis=1)
            if c in select_origins:
                mse = np.mean((act - f) ** 2, axis=1)
                scores[m].append(np.sqrt(mse / sc))
                abserr[m].append(np.abs(act.sum(axis=1) - f.sum(axis=1)))
                if metric != "rmsse":  # window-total absolute error, scaled like the squared error (comparable across items)
                    scores[m][-1] = np.abs(act.sum(axis=1) - f.sum(axis=1)) / (H * np.sqrt(sc)) if metric == "mae" else \
                        0.5 * scores[m][-1] + 0.5 * np.abs(act.sum(axis=1) - f.sum(axis=1)) / (H * np.sqrt(sc))
    score = {m: np.mean(np.vstack(v), axis=0) for m, v in scores.items() if v}
    aerr = {m: np.sum(np.vstack(v), axis=0) for m, v in abserr.items() if v}

    # collapse parameter variants: pick the best alpha per family per series
    fam = {}
    for m in score:
        fam.setdefault(m.split("_")[0], []).append(m)
    fam_best = {}
    fam_aerr = {}
    for f, ms in fam.items():
        arr = np.vstack([score[m] for m in ms])
        fam_best[f] = (arr.min(axis=0), np.array(ms, dtype=object)[arr.argmin(axis=0)])
        ae = np.vstack([aerr[m] for m in ms])
        fam_aerr[f] = ae[arr.argmin(axis=0), np.arange(arr.shape[1])]

    families = [f for f in SIMPLE_ORDER if f in fam_best]
    F_scores = np.vstack([fam_best[f][0] for f in families])  # families x series

    classes = cls["demand_class"].to_numpy()
    class_winner = {}
    rows_ms = []
    for c_ in np.unique(classes):
        m = classes == c_
        if c_ in ("inactive",) or m.sum() == 0:
            continue
        means = np.nanmean(F_scores[:, m], axis=1)
        unit_err = {f: float(fam_aerr[f][m].sum()) for f in families}
        for f, v in zip(families, means):
            rows_ms.append({"demand_class": c_, "method": METHOD_LABEL[f], "family": f, "mean_scaled_error": v,
                            "abs_error_units": unit_err[f], "series": int(m.sum())})
        w = families[int(np.nanargmin(means))]
        # baseline guard: a class default must also beat the baseline on total units missed in the test windows,
        # otherwise the class keeps the baseline (spare parts pilots showed fancy methods losing on sparse items)
        if "ma" in unit_err and w != "ma" and unit_err[w] > unit_err["ma"]:
            w = "ma"
        class_winner[c_] = w
    method_scores = pd.DataFrame(rows_ms)

    item_override = bool(fc.get("item_override", True))
    chosen = np.empty(S, dtype=object)
    variant = np.empty(S, dtype=object)
    reason = np.empty(S, dtype=object)
    fam_idx = {f: i for i, f in enumerate(families)}
    for i in range(S):
        c_ = classes[i]
        if c_ == "inactive":
            chosen[i], variant[i] = "zero", "zero"
            reason[i] = (f"Dead stock: no sale in the last {int(cls.periods_since_sale[i])} {'months' if P == 12 else 'weeks'}, "
                         f"so no demand is planned. Review for phase-out.")
            continue
        if c_ == "new" or not families:
            chosen[i], variant[i] = "ma", "ma"
            reason[i] = (f"Only {int(cls.history_periods[i])} periods of history, too few to test methods. The plan uses the "
                         f"average since the first sale, which reacts as weeks with and without sales come in.")
            continue
        own = int(np.nanargmin(F_scores[:, i]))
        cw = class_winner.get(c_, families[own])
        own_s, cw_s = F_scores[own, i], F_scores[fam_idx[cw], i]
        own_ae, cw_ae = fam_aerr[families[own]][i], fam_aerr[cw][i]
        if item_override and families[own] != cw and own_s < 0.9 * cw_s and own_ae <= cw_ae:
            chosen[i] = families[own]
            reason[i] = (f"{c_.capitalize()} demand. {METHOD_LABEL[families[own]]} had the lowest error on this item in "
                         f"{len(select_origins)} test windows ({own_s:.2f} vs {cw_s:.2f} for the class default).")
        else:
            chosen[i] = cw
            base_note = " (it also had to beat the baseline on units missed)" if cw != "ma" else ""
            reason[i] = (f"{c_.capitalize()} demand. {METHOD_LABEL[cw]} is the most accurate method for {c_} items "
                         f"in {len(select_origins)} test windows{base_note}." +
                         (" No other method beat it on this item by 10% or more." if item_override else ""))
        variant[i] = fam_best[chosen[i]][1][i] if chosen[i] != "ml" else "ml"

    # ------------------------------------------------------------ honest holdout accuracy
    hold_rows = []
    sigma = {}
    def selected_path(c):
        out = np.zeros((S, H))
        for v in np.unique(variant.astype(str)):
            m = variant.astype(str) == v
            if v == "zero":
                continue
            fp = np.repeat(ml_levels[c][:, None], H, axis=1) if v == "ml" else level_at(v, c)
            out[m] = fp[m]
        return out

    hold_path = selected_path(holdout) if holdout is not None else np.zeros((S, H))
    calib_origin = origins[-2] if len(origins) >= 2 else None
    calib_path = selected_path(calib_origin) if calib_origin is not None else None
    if holdout is not None:
        act = actual_tot[holdout]
        sel_tot = np.zeros(S)
        for i in range(S):
            v = variant[i]
            sel_tot[i] = 0.0 if v == "zero" else totals[v][holdout][i]
        base_tot = totals["ma"][holdout]
        price = prep.keys["item_code"].map(prep.items.set_index("item_code")["sell_price"]).fillna(1).to_numpy(float)
        hold = pd.DataFrame({"demand_class": classes, "actual": act, "forecast": sel_tot, "baseline": base_tot, "price": price})
        # unit errors the way the team scores SAP: per item and period, and over the whole test window
        act_p = M[:, holdout:holdout + H]
        base_p = level_at("ma", holdout)
        hold["abs_err_period"] = np.abs(act_p - hold_path).sum(axis=1)
        hold["abs_err_period_base"] = np.abs(act_p - base_p).sum(axis=1)
        hold_detail = hold.copy()
        for c_, g in list(hold.groupby("demand_class")) + [("All items", hold)]:
            w = g["price"]
            aw = (g["actual"] * w).sum()
            if aw <= 0:
                continue
            wape = (np.abs(g["actual"] - g["forecast"]) * w).sum() / aw
            wape_b = (np.abs(g["actual"] - g["baseline"]) * w).sum() / aw
            bias = ((g["forecast"] - g["actual"]) * w).sum() / aw
            bias_b = ((g["baseline"] - g["actual"]) * w).sum() / aw
            au = g["actual"].sum()
            unit = {}
            if au > 0:
                lab = "month" if P == 12 else "week"
                unit = {f"Units error, item x {lab}, AI": g["abs_err_period"].sum() / au,
                        f"Units error, item x {lab}, baseline": g["abs_err_period_base"].sum() / au,
                        f"Units error, item x {H} {lab}s, AI": np.abs(g["actual"] - g["forecast"]).sum() / au,
                        f"Units error, item x {H} {lab}s, baseline": np.abs(g["actual"] - g["baseline"]).sum() / au}
            hold_rows.append({"Demand class": c_, "Series": len(g), "Error (WAPE) AI": wape, "Error (WAPE) baseline": wape_b, **unit,
                              "Bias AI": bias, "Bias baseline": bias_b,
                              "Error reduction": (wape_b - wape) / wape_b if wape_b > 0 else 0.0})
    else:
        hold_detail = pd.DataFrame()
    accuracy = pd.DataFrame(hold_rows)

    # ------------------------------------------------------------ final forecast path
    path = np.zeros((S, horizon))
    S_fut = _season_matrix(prep.keys, prep.items, season, future_dates[:horizon], bk) if season else None
    final_ml = None
    if ml_levels is not None and (chosen == "ml").any():
        try:
            final_ml = _ml_levels(M, start, [T], H, prep.keys, prep.items, prep.period_end, train_until=T, bucket=bk)[T]
        except Exception:  # noqa: BLE001
            final_ml = None
    for i in range(S):
        v = variant[i]
        if v == "zero":
            continue
        if v == "ml":
            lv = final_ml[i] if final_ml is not None else levels["ma"][i, T - 1]
            path[i] = lv
        elif v == "trend":
            path[i] = levels["trend"][i, T - 1] * G[i, T - 1] ** ((lagp + np.arange(horizon)) / P) * (S_fut[i] if adj else 1.0)
        elif v == "seasonal" or (adj and v != "ma"):
            path[i] = levels[v][i, T - 1] * S_fut[i]
        else:
            path[i] = levels[v][i, T - 1]

    # ------------------------------------------------------------ simulation parameters
    sim = {}
    last104 = np.maximum(start, T - 2 * P)
    for i in range(S):
        y = M[i, last104[i]:]
        nz = y[y > 0]
        if len(nz) == 0:
            nz = np.array([1.0])
        occ = float((y > 0).mean()) if len(y) else 0.0
        sim[i] = {"sizes": nz.astype(float), "occ": max(occ, 1e-3)}

    t = prep.keys.copy()
    t = pd.concat([t, cls], axis=1)
    t["method"] = [METHOD_LABEL.get(c, c) for c in chosen]
    t["method_key"] = chosen
    t["variant"] = variant
    t["why"] = reason
    t["forecast_per_period"] = path[:, 0]
    for n_ in ((4, 8, 13, 26) if P == 52 else (3, 6, 9, 12)):
        if n_ <= horizon:
            t[f"forecast_next_{n_}"] = path[:, :n_].sum(axis=1)
    # last quarter and last year of actual demand (column names kept for the rest of the app)
    t["last_13_actual"] = M[:, -(13 if P == 52 else 3):].sum(axis=1)
    t["last_52_actual"] = M[:, -P:].sum(axis=1)
    t["seasonal_category"] = t["item_code"].map(prep.items.set_index("item_code")["category"]).isin(list(season))
    return ForecastResult(table=t, path=path, accuracy=accuracy, method_scores=method_scores, season_index=season,
                          sim_params=sim, level_sigma=sigma, holdout_detail=hold_detail, holdout_origin=holdout,
                          holdout_path=hold_path, holdout_actual=(M[:, holdout:holdout + H] if holdout is not None else None),
                          calib_origin=calib_origin, calib_path=calib_path,
                          calib_actual=(M[:, calib_origin:calib_origin + H] if calib_origin is not None else None))
