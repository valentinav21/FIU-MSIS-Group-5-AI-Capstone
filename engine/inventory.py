"""Dynamic safety stock, reorder points and order-up-to levels per item x warehouse.

For each series the engine simulates thousands of possible futures: a random lead time
drawn from that supplier's real history, and random demand over that lead time drawn
from the calibrated forecast. From those futures it reads the reorder point that meets
the service target, measured either as

* fill rate (default): share of units demanded that ship from stock, or
* cycle service: chance of no stockout before a replenishment arrives.

Safety stock is what sits above average lead time demand. The same futures give the
stock-to-service curve, the service today's SAP minimums deliver, and a budget
optimizer that spends a fixed safety stock budget where it serves the most revenue.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import policysim
from .probabilistic import lead_time_demand

GRID = np.array([0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995])
K_PTS = 500  # points in each calibrated lead time demand distribution


@dataclass
class PolicyResult:
    table: pd.DataFrame
    curve: pd.DataFrame          # stock to service curve, all items
    sim_cache: dict              # series index -> calibrated samples and grid reorder points
    measure: str
    kappa: float = 0.0           # weight on forecast level error inside policy simulation


def abc_classes(fc_table, prep, split=(0.8, 0.95)):
    """ABC by trailing 52 period revenue at item x warehouse level."""
    price = fc_table["item_code"].map(prep.items.set_index("item_code")["sell_price"]).fillna(0).to_numpy(float)
    rev = fc_table["last_52_actual"].to_numpy(float) * price
    order = np.argsort(-rev)
    cum = np.cumsum(rev[order]) / max(rev.sum(), 1e-9)
    cls = np.empty(len(rev), dtype=object)
    cls[order] = np.where(cum <= split[0], "A", np.where(cum <= split[1], "B", "C"))
    cls[rev <= 0] = "C"
    return cls, rev


def choose_supplier_default(item, wh, lt_model, tables, items_idx):
    """Supplier used for planning: preferred vendor, else most used on POs, else first sourcing option."""
    pref = items_idx.get(item, {}).get("preferred_supplier")
    if pref and isinstance(pref, str):
        return pref
    po = lt_model.po_lines
    if len(po):
        g = po[po["item_code"] == item]
        if len(g):
            gw = g[g["warehouse"] == wh] if "warehouse" in g else g
            return (gw if len(gw) else g)["supplier_code"].mode().iloc[0]
    isup = tables.get("item_suppliers")
    if isup is not None and len(isup):
        g = isup[isup["item_code"] == item]
        if len(g):
            return g["supplier_code"].iloc[0]
    return None


def calibrated_samples(fc, demand_class, raw):
    """Re-read raw simulated outcomes through the calibration map so stated odds match reality."""
    taus = (np.arange(K_PTS) + 0.5) / K_PTS
    qm = (fc.quantile_map or {}).get(demand_class)
    if qm is not None:
        xp = np.concatenate([[0.0], qm[0], [1.0]])
        fp = np.concatenate([[0.0], qm[1], [0.999]])
        fp = np.maximum.accumulate(np.clip(fp, 0, 0.999))
        read = np.interp(taus, xp, fp)
    else:
        read = taus
    return np.quantile(raw, read)


class ServiceCurve:
    """Fill rate and cycle service for any reorder point, from sorted calibrated samples."""

    def __init__(self, x_sorted, q):
        self.x = x_sorted
        self.q = max(q, 1e-9)
        self.cs = np.concatenate([[0.0], np.cumsum(x_sorted[::-1])])[::-1]  # suffix sums
        self.n = len(x_sorted)

    def short(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        k = np.searchsorted(self.x, r, side="right")
        above = self.n - k
        return (self.cs[k] - r * above) / self.n

    def fill(self, r):
        return np.clip(1 - self.short(r) / self.q, 0, 1)

    def cycle(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        return np.searchsorted(self.x, r, side="right") / self.n

    def rop_for(self, target, measure):
        cands = np.unique(np.concatenate([[0.0], np.ceil(self.x)]))
        vals = self.fill(cands) if measure == "fill_rate" else self.cycle(cands)
        ok = np.nonzero(vals >= target - 1e-12)[0]
        return float(cands[ok[0]] if len(ok) else cands[-1])


def build(prep, fc, lt_model, tables, settings, seed=3, kappa=None, progress=None, subset=None) -> PolicyResult:
    inv = settings["inventory"]
    rng = np.random.default_rng(seed)
    n_sims = int(inv.get("simulations", 2000))
    n_paths = int(inv.get("policy_paths", 100))
    review = float(inv.get("review_period_days", 7))
    h_pct = float(inv.get("holding_cost_pct_per_year", 0.25))
    K = float(inv.get("order_cost_usd", 75))
    sl_map = inv.get("service_level", {})
    measure = str(inv.get("service_measure", "fill_rate"))
    econ = bool(inv.get("economic_service_level", False))
    if econ:
        measure = "cycle"
    lo_b, hi_b = inv.get("economic_bounds", [0.8, 0.99])
    so_pct = float(settings["purchasing"].get("stockout_cost_pct_of_price", 0.3))
    t = fc.table
    S = len(t)
    abc, rev = abc_classes(t, prep, tuple(inv.get("abc_split", [0.8, 0.95])))
    items_idx = prep.items.set_index("item_code").to_dict("index")
    stock = tables.get("stock")
    st_idx = {}
    if stock is not None and len(stock):
        st = stock.copy()
        st["warehouse"] = st["warehouse"].astype(str)
        st_idx = st.drop_duplicates(["item_code", "warehouse"]).set_index(["item_code", "warehouse"]).to_dict("index")
    isup = tables.get("item_suppliers")
    isup_idx = {}
    if isup is not None and len(isup):
        for r in isup.itertuples(index=False):
            isup_idx.setdefault((r.item_code, r.supplier_code), r)
    periods_per_year = 365.25 / prep.period_days
    if kappa is None:
        kappa = float(inv.get("model_error_weight", 0.0))

    # ---------------------------------------------------------------- phase 1: inputs per series
    base, sim_in = [], {}
    for i in (range(S) if subset is None else subset):
        item, wh = t.item_code.iat[i], t.warehouse.iat[i]
        it = items_idx.get(item, {})
        cost = float(it.get("unit_cost") or 0) or 1.0
        price = float(it.get("sell_price") or cost / 0.6)
        cls = t.demand_class.iat[i]
        path = fc.path[i]
        mean_per = float(path[:13].mean()) if len(path) else 0.0
        if mean_per < 1e-3:
            mean_per = 0.0
        sup = choose_supplier_default(item, wh, lt_model, tables, items_idx)
        lt, lt_src = lt_model.samples(item, sup)
        group = "new" if cls == "new" else abc[i]
        target = float(sl_map.get(group, 0.95))
        moq, mult = it.get("moq"), it.get("order_multiple")
        so = isup_idx.get((item, sup))
        if so is not None:
            moq = so.moq if pd.notna(so.moq) else moq
            mult = so.order_multiple if pd.notna(so.order_multiple) else mult
            if pd.notna(so.unit_price):
                cost = float(so.unit_price)
        moq = float(moq) if moq is not None and pd.notna(moq) and moq > 0 else float(settings["purchasing"].get("default_supplier_moq", 1))
        mult = float(mult) if mult is not None and pd.notna(mult) and mult > 0 else 1.0
        ann_demand = mean_per * periods_per_year
        h_unit = cost * h_pct
        if mean_per > 0:
            eoq = np.sqrt(2 * ann_demand * K / h_unit) if h_unit > 0 else 0.0
            cap = float(inv.get("max_order_cover_weeks", 26)) * mean_per * 7.0 / prep.period_days
            q = max(min(eoq, cap), moq, mean_per * review / prep.period_days)
            q = float(np.ceil(q / mult - 1e-9) * mult)
        else:
            q = 0.0
        if econ and ann_demand > 0:
            cycle_years = max(q / ann_demand, review / 365.25)
            cu = price * so_pct + max(price - cost, 0)
            co = h_unit * cycle_years
            target = float(np.clip(cu / (cu + co), lo_b, hi_b))
        cur = st_idx.get((item, wh), {})
        cur_min, cur_max = cur.get("min_stock"), cur.get("max_stock")
        has_cur = cur_min is not None and pd.notna(cur_min) and cur_max is not None and pd.notna(cur_max)
        live = mean_per > 0 and cls != "inactive"
        if live:
            lt_draw = np.clip(rng.choice(lt, n_sims) + rng.uniform(-1.5, 1.5, n_sims), 1, None) + review
            raw = lead_time_demand(fc.sim_params[i]["sizes"], fc.sim_params[i]["occ"], path, prep.period_days,
                                   lt_draw, n_sims, float(fc.level_sigma.get(cls, 0.25)), rng)
            x = np.sort(calibrated_samples(fc, cls, raw))
            week_path = np.asarray(path, float) * (7.0 / prep.period_days)
            occ_w = fc.sim_params[i]["occ"]
            if prep.period_days != 7.0:  # monthly buckets: weekly mean and weekly chance of a sale
                week_path = np.repeat(week_path, 4)
                occ_w = 1 - (1 - min(occ_w, 0.999)) ** (7.0 / prep.period_days)
            sim_in[i] = {"mean_week": week_path, "sizes": fc.sim_params[i]["sizes"],
                         "occ": occ_w, "sigma": float(fc.level_sigma.get(cls, 0.25)),
                         "lt_weeks": np.asarray(lt, float) / 7.0, "q": q, "moq": moq, "mult": mult,
                         "cands": policysim.candidates(x),
                         "extra": [(float(cur_min), float(cur_max))] if has_cur else []}
        else:
            x = np.zeros(K_PTS)
        base.append(dict(i=i, item=item, wh=wh, cls=cls, cost=cost, price=price, sup=sup, lt=lt, lt_src=lt_src,
                         target=target, moq=moq, mult=mult, q=q, mean_per=mean_per, x=x, cur=cur,
                         cur_min=cur_min, cur_max=cur_max, has_cur=has_cur, live=live))

    # ---------------------------------------------------------------- phase 2: simulate policies in batches
    sim_out = {}
    keys = list(sim_in)
    bs = 100
    for k0 in range(0, len(keys), bs):
        chunk = keys[k0:k0 + bs]
        res = policysim.simulate_batch([sim_in[i] for i in chunk], n_paths=n_paths, rng=rng, kappa=kappa)
        for i, r in zip(chunk, res):
            sim_out[i] = r
        if progress:
            progress(min(1.0, (k0 + bs) / max(len(keys), 1)))

    # ---------------------------------------------------------------- phase 3: choose levels and report
    rows, cache = [], {}
    curve_val = np.zeros(len(GRID))
    curve_inv = np.zeros(len(GRID))
    curve_fill_rev = np.zeros(len(GRID))
    for bdat in base:
        i = bdat["i"]
        x, q, cost = bdat["x"], bdat["q"], bdat["cost"]
        mean_ltd = float(x.mean())
        sc = ServiceCurve(x, q if q > 0 else 1.0)
        if bdat["live"]:
            so_ = sim_out[i]
            cands, fill = sim_in[i]["cands"], so_["fill"]
            if measure == "fill_rate":
                rop, fill_at = policysim.rop_for_fill(cands, fill, bdat["target"])
                grid = [policysim.rop_for_fill(cands, fill, g) for g in GRID]
            else:
                rop = sc.rop_for(bdat["target"], "cycle")
                fill_at = float(np.interp(rop, cands, fill))
                grid = [(r_, float(np.interp(r_, cands, fill))) for r_ in (sc.rop_for(g, "cycle") for g in GRID)]
            grid_rop = np.array([g[0] for g in grid])
            grid_fill = np.array([g[1] for g in grid])
            grid_onhand = np.interp(grid_rop, cands, so_["onhand"])
            cyc_at = float(sc.cycle(rop)[0])
            onhand_at = float(np.interp(rop, cands, so_["onhand"]))
            cur_fill = float(so_["extra_fill"][0]) if bdat["has_cur"] else np.nan
            cur_onhand = float(so_["extra_onhand"][0]) if bdat["has_cur"] else np.nan
            cur_cyc = float(sc.cycle(float(bdat["cur_min"]))[0]) if bdat["has_cur"] else np.nan
        else:
            rop, fill_at, cyc_at, onhand_at = 0.0, 1.0, 1.0, 0.0
            grid_rop, grid_fill = np.zeros(len(GRID)), np.ones(len(GRID))
            grid_onhand = np.zeros(len(GRID))
            cur_fill = cur_cyc = cur_onhand = np.nan
            cands, fill = np.array([0.0]), np.array([1.0])
        ss = max(rop - mean_ltd, 0.0)
        curve_val += np.maximum(grid_rop - mean_ltd, 0) * cost
        curve_inv += grid_onhand * cost
        curve_fill_rev += grid_fill * rev[i]
        cur_ss_val = max(float(bdat["cur_min"]) - mean_ltd, 0) * cost if bdat["has_cur"] else np.nan
        lt = bdat["lt"]
        rows.append({
            "item_code": bdat["item"], "warehouse": bdat["wh"], "abc": abc[i], "demand_class": bdat["cls"],
            "revenue_last_52": rev[i], "unit_cost": cost, "sell_price": bdat["price"],
            "supplier": bdat["sup"], "lead_time_median_days": float(np.median(lt)), "lead_time_p90_days": float(np.quantile(lt, 0.9)),
            "lead_time_source": bdat["lt_src"], "service_measure": measure, "service_target": bdat["target"],
            "forecast_per_period": bdat["mean_per"], "mean_lead_time_demand": mean_ltd,
            "safety_stock": ss, "reorder_point": rop, "order_qty": q, "order_up_to": rop + q,
            "moq": bdat["moq"], "order_multiple": bdat["mult"],
            "fill_rate_at_rop": fill_at, "cycle_service_at_rop": cyc_at,
            "safety_stock_value": ss * cost, "avg_inventory_value": onhand_at * cost,
            "current_min": bdat["cur_min"], "current_max": bdat["cur_max"],
            "current_fill_rate": cur_fill, "current_cycle_service": cur_cyc,
            "current_safety_stock_value": cur_ss_val,
            "current_avg_inventory_value": cur_onhand * cost if np.isfinite(cur_onhand) else np.nan,
            "on_hand": float(bdat["cur"].get("on_hand") or 0),
            "weeks_cover_at_rop": (rop / (bdat["mean_per"] * 7 / prep.period_days)) if bdat["mean_per"] > 0 else np.nan,
        })
        cache[i] = {"x": x, "lt": lt, "supplier": bdat["sup"], "grid_rop": grid_rop, "grid_fill": grid_fill, "grid_onhand": grid_onhand,
                    "cost": cost, "q": q, "rev": rev[i], "mean": mean_ltd, "cands": cands, "fill": fill}
    pol = pd.DataFrame(rows)
    pol["why"] = [_explain(r) for r in pol.itertuples(index=False)]
    tot_rev = max(rev.sum(), 1e-9)
    curve = pd.DataFrame({"service_target": GRID, "safety_stock_value": curve_val, "avg_inventory_value": curve_inv,
                          "revenue_weighted_fill_rate": curve_fill_rev / tot_rev})
    return PolicyResult(table=pol, curve=curve, sim_cache=cache, measure=measure, kappa=kappa)


def summary(pol: PolicyResult) -> dict:
    t = pol.table
    live = t["forecast_per_period"] > 0
    w = t["revenue_last_52"]
    has = live & t["current_fill_rate"].notna()
    out = {
        "series": int(len(t)), "planned_series": int(live.sum()),
        "safety_stock_value": float(t.loc[live, "safety_stock_value"].sum()),
        "avg_inventory_value": float(t.loc[live, "avg_inventory_value"].sum()),
        "fill_rate": float((t.loc[live, "fill_rate_at_rop"] * w[live]).sum() / max(w[live].sum(), 1e-9)),
        "current_safety_stock_value": float(t.loc[has, "current_safety_stock_value"].sum()) if has.any() else np.nan,
        "current_avg_inventory_value": float(t.loc[has, "current_avg_inventory_value"].sum()) if has.any() else np.nan,
        "current_fill_rate": float((t.loc[has, "current_fill_rate"] * w[has]).sum() / max(w[has].sum(), 1e-9)) if has.any() else np.nan,
    }
    return out


def optimize_budget(pol: PolicyResult, budget: float, basis: str = "safety_stock") -> pd.DataFrame:
    """Spend a fixed budget where it serves the most revenue (greedy marginal allocation).

    basis = "safety_stock": the budget is safety stock dollars.
    basis = "inventory":    the budget is average inventory dollars (safety stock plus cycle stock).
    Every item starts at the lowest service step; each step up is bought in order of revenue
    served per dollar until the budget runs out.
    """
    t = pol.table
    idx = [i for i in range(len(t)) if t["forecast_per_period"].iat[i] > 0]
    level = {i: 0 for i in idx}

    def val(i, k):
        c = pol.sim_cache[i]
        if basis == "inventory":
            return float(c["grid_onhand"][k]) * c["cost"]
        return max(c["grid_rop"][k] - c["mean"], 0) * c["cost"]

    spent = sum(val(i, 0) for i in idx)
    heap = []

    def push(i):
        k = level[i]
        c = pol.sim_cache[i]
        if k + 1 >= len(GRID):
            return
        d_cost = max(val(i, k + 1) - val(i, k), 0.0)
        gain = (c["grid_fill"][k + 1] - c["grid_fill"][k]) * max(c["rev"], 1e-6)
        heapq.heappush(heap, (-gain / max(d_cost, 1e-6), i, d_cost))

    for i in idx:
        push(i)
    while heap:
        _, i, d_cost = heapq.heappop(heap)
        if spent + d_cost > budget:
            continue
        spent += d_cost
        level[i] += 1
        push(i)
    out = t[["item_code", "warehouse"]].copy()
    out["budget_target"] = np.nan
    out["budget_reorder_point"] = 0.0
    out["budget_fill_rate"] = np.nan
    for i in idx:
        k = level[i]
        out.iat[i, 2] = GRID[k]
        out.iat[i, 3] = pol.sim_cache[i]["grid_rop"][k]
        out.iat[i, 4] = pol.sim_cache[i]["grid_fill"][k]
    rev = t["revenue_last_52"].to_numpy()
    m = np.array(idx, dtype=int)
    out.attrs["fill_rate"] = float(np.nansum(out["budget_fill_rate"].to_numpy()[m] * rev[m]) / max(rev[m].sum(), 1e-9)) if len(m) else np.nan
    out.attrs["spent"] = float(spent)
    out.attrs["basis"] = basis
    return out


def _explain(r) -> str:
    if r.forecast_per_period <= 0:
        return "No demand expected, so no safety stock is held. Consider phasing out remaining stock."
    m = "fill rate" if r.service_measure == "fill_rate" else "cycle service"
    parts = [f"Target {r.service_target:.0%} {m} ({r.abc} item, {r.demand_class} demand)."]
    parts.append(f"Lead time from {r.supplier or 'unknown supplier'}: median {r.lead_time_median_days:.0f} days, "
                 f"1 in 10 orders takes {r.lead_time_p90_days:.0f}+ days ({r.lead_time_source}).")
    parts.append(f"Average demand over lead time plus review is {r.mean_lead_time_demand:.1f} units. A reorder point of "
                 f"{r.reorder_point:.0f} meets the target, so safety stock is {r.safety_stock:.0f} units.")
    if pd.notna(r.current_fill_rate):
        parts.append(f"Today's SAP minimum of {r.current_min:.0f} gives about {r.current_fill_rate:.0%} fill rate.")
    return " ".join(parts)
