"""Policy replay: run real demand history through stocking policies.

Each item x warehouse is replayed week by week against its actual sales. When the stock
position falls to the reorder level, an order is placed up to the target level, rounded to
the supplier minimum and pack size, and it arrives after a lead time drawn from that
supplier's real history. Every policy sees the same demand and the same lead time draws.
Unmet demand is counted as lost. The first year warms the system up from a random point
in its cycle, and results are measured on the most recent year.

Two uses:
1. Model check: the fill rate the simulator predicts for today's SAP min/max versus the
   fill rate those settings deliver on real history. Agreement means the simulator can be
   trusted to size new settings.
2. Policy comparison: today's SAP settings versus the AI planner on the same history.
   The AI settings were computed with data that includes the replay year, so read the
   comparison as an estimate until a forward test confirms it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def replay(D, s_lvl, S_lvl, moq, mult, lt_list, cost, warm, rng):
    """D: series x weeks actual demand. Returns per series fill, shipped, demand, avg on hand value, orders."""
    n, W = D.shape
    ring = 64
    S_lvl = np.maximum(S_lvl, s_lvl + 1)
    oh = s_lvl + rng.random(n) * (S_lvl - s_lvl)
    oo = np.zeros(n)
    pipe = np.zeros((n, ring))
    shipped = np.zeros(n)
    dem = np.zeros(n)
    ohv = np.zeros(n)
    orders = np.zeros(n)
    so_weeks = np.zeros(n)
    L = np.ones((n, W), int)
    for j in range(n):
        L[j] = np.clip(np.ceil(rng.choice(lt_list[j], W) / 7.0), 1, ring - 1).astype(int)
    rows = np.arange(n)
    for w in range(W):
        slot = w % ring
        arr = pipe[:, slot].copy()
        pipe[:, slot] = 0
        oh += arr
        oo -= arr
        d = D[:, w]
        ship = np.minimum(oh, d)
        oh -= ship
        if w >= warm:
            shipped += ship
            dem += d
            ohv += oh * cost
            so_weeks += d > ship + 1e-9
        ip = oh + oo
        need = ip <= s_lvl
        if need.any():
            q = np.ceil(np.maximum(S_lvl - ip, moq) / mult - 1e-9) * mult
            r = rows[need]
            pipe[r, (w + L[r, w]) % ring] += q[need]
            oo[r] += q[need]
            if w >= warm:
                orders[r] += 1
    m = W - warm
    return {"fill": np.where(dem > 0, shipped / np.maximum(dem, 1e-9), np.nan), "shipped": shipped, "demand": dem,
            "avg_value": ohv / max(m, 1), "orders": orders, "stockout_weeks": so_weeks}


def _inputs(prep, pol, rows):
    t = pol.table
    moq = np.nan_to_num(t["moq"].to_numpy(float)[rows], nan=1.0).clip(1)
    mult = np.nan_to_num(t["order_multiple"].to_numpy(float)[rows], nan=1.0).clip(1)
    lts = [pol.sim_cache[int(t["_sid"].iat[r]) if "_sid" in t else r]["lt"] for r in rows]
    return moq, mult, lts


def run(prep, pol, weeks: int = 52, seed: int = 21, max_series: int = 3000, extra: dict | None = None) -> dict:
    """extra: optional {name: (reorder point array, order-up-to array)} aligned with pol.table rows."""
    if prep.period_days != 7.0:
        return {"note": "Policy replay runs on weekly buckets only."}
    t = pol.table
    T = prep.matrix.shape[1]
    warm = min(52, max(T - weeks, 0))
    total = warm + weeks
    if T < weeks + 13:
        return {"note": "Policy replay needs at least 65 weeks of sales history."}
    ok = (t["forecast_per_period"] > 0) & t["current_min"].notna() & t["current_max"].notna()
    ok &= pd.Series(prep.matrix[:, -weeks:].sum(axis=1) > 0, index=t.index)
    idx = np.nonzero(ok.to_numpy())[0]
    if len(idx) == 0:
        return {"note": "Needs SAP MinStock and MaxStock on the stock file and at least one sale in the replay window."}
    rng = np.random.default_rng(seed)
    if len(idx) > max_series:  # keep every A item, sample the rest
        a = idx[t["abc"].to_numpy()[idx] == "A"]
        rest = np.setdiff1d(idx, a)
        idx = np.sort(np.concatenate([a, rng.choice(rest, max(max_series - len(a), 0), replace=False)])) if len(a) < max_series else a[:max_series]
    D = prep.matrix[idx, -total:]
    cost = t["unit_cost"].to_numpy()[idx]
    price = t["sell_price"].to_numpy()[idx]
    moq, mult, lts = _inputs(prep, pol, idx)
    pols = {
        "Today's SAP min / max": (t["current_min"].to_numpy(float)[idx], t["current_max"].to_numpy(float)[idx]),
        "AI planner": (t["reorder_point"].to_numpy(float)[idx], t["order_up_to"].to_numpy(float)[idx]),
    }
    for name, (s_all, S_all) in (extra or {}).items():
        pols[name] = (np.asarray(s_all, float)[idx], np.asarray(S_all, float)[idx])
    res, per = {}, {}
    for name, (s_l, S_l) in pols.items():
        r = replay(D, s_l, S_l, moq, mult, lts, cost, warm, np.random.default_rng(seed + 1))
        per[name] = r
        rev_d = (r["demand"] * price).sum()
        res[name] = {
            "Fill rate (units)": float(r["shipped"].sum() / max(r["demand"].sum(), 1e-9)),
            "Fill rate (revenue)": float((r["shipped"] * price).sum() / max(rev_d, 1e-9)),
            "Lost revenue": float(((r["demand"] - r["shipped"]) * price).sum()),
            "Average inventory value": float(r["avg_value"].sum()),
            "Purchase orders placed": int(r["orders"].sum()),
            "Item-weeks with a stockout": int(r["stockout_weeks"].sum()),
        }
    df = pd.DataFrame(res)
    sap = per["Today's SAP min / max"]
    pred = t["current_fill_rate"].to_numpy(float)[idx]
    w = sap["demand"] * price
    m = np.isfinite(pred) & np.isfinite(sap["fill"]) & (w > 0)
    check = pd.DataFrame({"abc": t["abc"].to_numpy()[idx][m], "predicted": pred[m], "replayed": sap["fill"][m], "w": w[m]})
    chk = []
    for g, d in list(check.groupby("abc")) + [("All", check)]:
        chk.append({"Group": g, "Items": len(d), "Predicted fill, today's SAP settings": float((d.predicted * d.w).sum() / d.w.sum()),
                    "Replayed fill on real history": float((d.replayed * d.w).sum() / d.w.sum())})
    return {"table": df, "series": int(len(idx)), "weeks": weeks, "model_check": pd.DataFrame(chk)}


def calibrate_kappa(prep, fc, lt_model, tables, settings, inventory_mod, sample=300, seed=4, grid=(0.0, 0.5, 1.0)):
    """Pick how much forecast level error to add inside policy simulation, by matching
    predicted fill for today's SAP settings to replayed fill on real history."""
    if prep.period_days != 7.0 or prep.matrix.shape[1] < 65:
        return float(settings["inventory"].get("model_error_weight", 0.0)), pd.DataFrame()
    stock = tables.get("stock")
    if stock is None or len(stock) == 0 or stock["min_stock"].isna().all() or stock["max_stock"].isna().all():
        return float(settings["inventory"].get("model_error_weight", 0.0)), pd.DataFrame()
    st = stock.dropna(subset=["min_stock", "max_stock"])
    keyset = set(zip(st["item_code"], st["warehouse"].astype(str)))
    t = fc.table
    live = (fc.path[:, :13].mean(axis=1) > 1e-3) & (prep.matrix[:, -52:].sum(axis=1) > 0)
    cand = [i for i in np.nonzero(live)[0] if (t.item_code.iat[i], t.warehouse.iat[i]) in keyset]
    if len(cand) < 30:
        return float(settings["inventory"].get("model_error_weight", 0.0)), pd.DataFrame()
    rng = np.random.default_rng(seed)
    sub = sorted(rng.choice(cand, min(sample, len(cand)), replace=False).tolist())
    T = prep.matrix.shape[1]
    warm = min(52, T - 52)
    rows, best, best_err = [], 0.0, 9e9
    actual = None
    for k in grid:
        pol = inventory_mod.build(prep, fc, lt_model, tables, settings, kappa=k, subset=sub)
        tt = pol.table.copy()
        tt["_sid"] = sub
        pol.table = tt
        if actual is None:
            ridx = np.arange(len(sub))
            moq, mult, lts = _inputs(prep, pol, ridx)
            r = replay(prep.matrix[sub, -(warm + 52):], tt["current_min"].to_numpy(float), tt["current_max"].to_numpy(float),
                       moq, mult, lts, tt["unit_cost"].to_numpy(), warm, np.random.default_rng(seed + 1))
            w = r["demand"] * tt["sell_price"].to_numpy()
            actual = float(np.nansum(r["fill"] * w) / max(np.nansum(w[np.isfinite(r["fill"])]), 1e-9))
            fin = np.isfinite(r["fill"])
        pred = tt["current_fill_rate"].to_numpy(float)
        m = fin & np.isfinite(pred)
        p = float((pred[m] * w[m]).sum() / max(w[m].sum(), 1e-9))
        err = abs(p - actual)
        rows.append({"Model error weight": k, "Predicted fill, today's SAP settings": p, "Replayed fill on real history": actual,
                     "Gap (points)": (p - actual) * 100})
        if err < best_err - 0.002:
            best, best_err = k, err
    return best, pd.DataFrame(rows)
