"""Monte Carlo demand simulation and forecast calibration.

Demand in each future period is simulated as: does a sale happen (Bernoulli, at the item's
own recent hit rate) and, if so, how big is it (drawn from the item's own past order sizes,
scaled so the average matches the forecast). On top of that, each simulated future gets a
level multiplier that represents forecast error, sized by calibration so the predicted
ranges contain the real outcome as often as they claim to.

This keeps zeros, spikes and lumpy sizes that normal-distribution formulas wash out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_periods(sizes, occ, mean_path, n_sims, sigma, rng):
    """Return (n_sims, n_periods) simulated demand with the given per-period mean path."""
    n = len(mean_path)
    if n == 0 or np.all(np.asarray(mean_path) <= 0):
        return np.zeros((n_sims, n))
    sizes = np.asarray(sizes, float)
    mean_size = sizes.mean()
    mean_path = np.asarray(mean_path, float)
    occ = float(np.clip(occ, 1e-3, 1.0))
    # if the forecast rate is above what occasional sales can deliver, raise the hit rate first
    occ_t = np.clip(np.maximum(occ, mean_path / max(sizes.max(), 1e-9)), 1e-3, 1.0)
    scale = mean_path / (occ_t * mean_size)
    hit = rng.random((n_sims, n)) < occ_t[None, :]
    draw = sizes[rng.integers(0, len(sizes), (n_sims, n))]
    y = hit * draw * scale[None, :]
    if sigma > 0:
        mult = rng.lognormal(-0.5 * sigma ** 2, sigma, n_sims)
        y = y * mult[:, None]
    return y


def lead_time_demand(sizes, occ, mean_path, period_days, lt_days, n_sims, sigma, rng):
    """Total demand over a random number of days per simulation (lead time plus review)."""
    lt_days = np.asarray(lt_days, float)
    n_per = lt_days / period_days
    max_n = int(np.ceil(n_per.max())) + 1
    path = np.asarray(mean_path, float)
    if len(path) < max_n:  # extend the path with its last value
        path = np.concatenate([path, np.full(max_n - len(path), path[-1] if len(path) else 0.0)])
    y = simulate_periods(sizes, occ, path[:max_n], n_sims, sigma, rng)
    cs = np.concatenate([np.zeros((n_sims, 1)), np.cumsum(y, axis=1)], axis=1)
    k = np.floor(n_per).astype(int)
    frac = n_per - k
    rows = np.arange(n_sims)
    tot = cs[rows, k] + frac * (cs[rows, np.minimum(k + 1, max_n)] - cs[rows, k])
    return tot


TAU = np.array([0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995])


def _pit(samples, actual, rng):
    """Randomized probability integral transform: where the actual fell in the simulated range."""
    lo = (samples < actual[:, None]).mean(axis=1)
    hi = (samples <= actual[:, None]).mean(axis=1)
    return lo + rng.random(len(actual)) * (hi - lo)


def _window_pits(fc, prep, origin, path, actual_tot, idx, sigma, n_sims, rng):
    M, start = prep.matrix, prep.first_idx
    sims = np.zeros((len(idx), n_sims))
    for j, i in enumerate(idx):
        y = M[i, max(start[i], origin - 104):origin]
        nz = y[y > 0]
        sizes = nz if len(nz) else np.array([1.0])
        occ = max(float((y > 0).mean()) if len(y) else 0.0, 1e-3)
        sims[j] = simulate_periods(sizes, occ, path[i], n_sims, sigma, rng).sum(axis=1)
    return _pit(sims, actual_tot[idx], rng)


def adjusted_quantile(fc, demand_class, target):
    """Quantile to read from the simulation so the real hit rate matches the target."""
    qm = (fc.quantile_map or {}).get(demand_class)
    if qm is None:
        return float(target)
    return float(np.clip(np.interp(target, qm[0], qm[1]), 0.5, 0.999))


def calibrate(fc, prep, settings, rng=None):
    """Learn forecast error spread and a quantile correction per demand class.

    Fit on an earlier test window, then measure on the final holdout window that neither
    the method choice nor the calibration saw. Returns (sigma by class, report table).
    """
    rng = rng or np.random.default_rng(11)
    classes = fc.table["demand_class"].to_numpy()
    fc.quantile_map = {}
    if fc.holdout_origin is None or fc.holdout_actual is None:
        fc.level_sigma = {c: 0.25 for c in np.unique(classes)}
        return fc.level_sigma, pd.DataFrame()
    fit_origin = fc.calib_origin if fc.calib_origin is not None else fc.holdout_origin
    fit_path = fc.calib_path if fc.calib_path is not None else fc.holdout_path
    fit_act = (fc.calib_actual if fc.calib_actual is not None else fc.holdout_actual).sum(axis=1)
    hold_act = fc.holdout_actual.sum(axis=1)
    grid = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6]
    n_sims = 400
    rows, sig = [], {}
    for cls in np.unique(classes):
        if cls == "inactive":
            sig[cls] = 0.0
            continue
        idx_fit = np.nonzero((classes == cls) & (fit_path.sum(axis=1) > 0))[0]
        idx_hold = np.nonzero((classes == cls) & (fc.holdout_path.sum(axis=1) > 0))[0]
        if len(idx_fit) < 15 or len(idx_hold) < 15:
            sig[cls] = 0.25
            continue
        if len(idx_fit) > 600:
            idx_fit = rng.choice(idx_fit, 600, replace=False)
        if len(idx_hold) > 600:
            idx_hold = rng.choice(idx_hold, 600, replace=False)
        best, best_err, best_pit = 0.25, 9e9, None
        for sgm in grid:
            pit = _window_pits(fc, prep, fit_origin, fit_path, fit_act, idx_fit, sgm, n_sims, rng)
            err = np.mean([abs((pit <= t).mean() - t) for t in TAU[1:]])
            if err < best_err - 1e-9:
                best, best_err, best_pit = sgm, err, pit
        sig[cls] = best
        mapped = np.maximum.accumulate(np.clip(np.quantile(best_pit, TAU), 0.5, 0.999))
        fc.quantile_map[cls] = (TAU.copy(), mapped)
        pit_h = _window_pits(fc, prep, fc.holdout_origin, fc.holdout_path, hold_act, idx_hold, best, n_sims, rng)
        row = {"Demand class": cls, "Series tested": len(idx_hold), "Error spread": best}
        for t, tp in zip(TAU, mapped):
            if t in (0.8, 0.9, 0.95):
                row[f"Hit rate at {t:.0%} target, raw"] = float((pit_h <= t).mean())
                row[f"Hit rate at {t:.0%} target, calibrated"] = float((pit_h <= tp).mean())
        rows.append(row)
    for cls in np.unique(classes):
        sig.setdefault(cls, 0.25)
    sig.setdefault("new", 0.3)
    fc.level_sigma = sig
    return sig, pd.DataFrame(rows)
