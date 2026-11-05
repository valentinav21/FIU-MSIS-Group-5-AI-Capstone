"""Turn loaded tables into clean demand history and shared lookups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIOD_DAYS = {"W": 7.0, "M": 30.4375}


def month_anchor(as_of: pd.Timestamp) -> pd.Timestamp:
    """Last complete calendar month end on or before as_of."""
    me = as_of + pd.offsets.MonthEnd(0)
    return me.normalize() if me.normalize() == as_of.normalize() else (as_of - pd.offsets.MonthEnd(1)).normalize()


def periods_back(dates: pd.Series, as_of: pd.Timestamp, bucket: str, pdays: float) -> pd.Series:
    """0 = the most recent period. Monthly buckets are calendar months ending at the last complete month."""
    if bucket == "M":
        a = month_anchor(as_of)
        return (a.year * 12 + a.month) - (dates.dt.year * 12 + dates.dt.month)
    return np.floor((as_of - dates).dt.days / pdays).astype(int)


def trusted_open_orders(tables, settings, as_of, warehouses=None):
    """Open customer orders that still count as committed: planning warehouses only, and not more than
    stale_open_days past their ship date (older ones are orders nobody closed). Returns (table or None, notes)."""
    oo = tables.get("open_orders")
    if oo is None or len(oo) == 0:
        return oo, []
    notes = []
    o = oo.copy()
    dw = settings["data"].get("default_warehouse") or "A1"
    o["warehouse"] = o["warehouse"].fillna(dw).astype(str)
    plan = [str(w) for w in (warehouses if warehouses is not None else (settings["data"].get("plan_warehouses") or []))]
    if plan:
        o = o[o["warehouse"].isin(plan)]
    days = int(settings["data"].get("stale_open_days", 0) or 0)
    if days > 0 and o["due_date"].notna().any():
        old = (as_of - o["due_date"]).dt.days > days
        if old.any():
            notes.append(f"{int(old.sum())} open sales order lines ({o.loc[old, 'open_qty'].sum():,.0f} units) are more than {days} days "
                         f"past their ship date and are not counted as committed.")
        o = o[~old]
    past = o["due_date"].notna() & (o["due_date"] < as_of)
    if past.any():
        notes.append(f"{int(past.sum())} open sales order lines are past their ship date: treated as backorders due now.")
    return o, notes


@dataclass
class Prepared:
    as_of: pd.Timestamp
    bucket: str
    period_days: float
    lines: pd.DataFrame        # clean demand lines with period index
    keys: pd.DataFrame         # one row per item x warehouse series
    matrix: np.ndarray         # series x periods, oldest first, last column = period ending as_of
    first_idx: np.ndarray      # first period (column) each series existed
    period_end: pd.DatetimeIndex
    items: pd.DataFrame
    warehouses: list
    notes: list
    history_end: pd.Timestamp | None = None      # last day covered by the demand history
    first_forecast_start: pd.Timestamp | None = None  # first day of the first forecast period
    weekly: np.ndarray | None = None  # weekly demand ending at as_of (only filled when planning is monthly)
    sold_after_history: np.ndarray | None = None  # per series: sold in the unfinished month (not in the history)

    def weekly_matrix(self):
        return self.matrix if self.period_days == 7.0 else self.weekly

    def future_period_starts(self, n):
        """First day of each of the next n forecast periods."""
        if self.bucket == "M":
            return [self.first_forecast_start + pd.offsets.MonthBegin(h) if h else self.first_forecast_start for h in range(n)]
        return [self.first_forecast_start + pd.Timedelta(days=self.period_days * h) for h in range(n)]


def as_of_date(tables, settings) -> pd.Timestamp:
    v = str(settings["data"].get("as_of_date") or "").strip()
    if v:
        return pd.Timestamp(v).normalize()
    s = tables.get("sales")
    if s is not None and len(s):
        return pd.Timestamp(s["doc_date"].max()).normalize()
    return pd.Timestamp.today().normalize()


def build_items(tables) -> pd.DataFrame:
    items = tables.get("items")
    sales = tables.get("sales")
    codes = set()
    if items is not None:
        codes |= set(items["item_code"])
    if sales is not None:
        codes |= set(sales["item_code"])
    if items is None or len(items) == 0:
        items = pd.DataFrame({"item_code": sorted(codes)})
    items = items.drop_duplicates("item_code", keep="first").copy()
    missing = sorted(codes - set(items["item_code"]))
    if missing:
        items = pd.concat([items, pd.DataFrame({"item_code": missing})], ignore_index=True)
    for c, d in (("item_name", ""), ("category", "Uncategorized"), ("subcategory", ""), ("brand", "")):
        if c not in items.columns:
            items[c] = d
        items[c] = items[c].fillna(d).astype(str)
    items.loc[items["item_name"] == "", "item_name"] = items["item_code"]
    for c in ("unit_cost", "list_price", "moq", "order_multiple", "lead_time_days"):
        if c not in items.columns:
            items[c] = np.nan
    # price and cost fallbacks from transactions
    if sales is not None and len(sales):
        s = sales[(sales["quantity"] > 0)].copy()
        if "unit_price" in s and s["unit_price"].notna().any():
            px = s.groupby("item_code")["unit_price"].median()
        elif "line_total" in s and s["line_total"].notna().any():
            px = (s["line_total"] / s["quantity"]).groupby(s["item_code"]).median()
        else:
            px = pd.Series(dtype=float)
        items["avg_sell_price"] = items["item_code"].map(px)
    else:
        items["avg_sell_price"] = np.nan
    items["sell_price"] = items["avg_sell_price"].fillna(items["list_price"])
    purchases = tables.get("purchases")
    if purchases is not None and len(purchases) and purchases["unit_price"].notna().any():
        last_cost = purchases.sort_values("po_date").groupby("item_code")["unit_price"].last()
        items["unit_cost"] = items["unit_cost"].fillna(items["item_code"].map(last_cost))
    # Last resort assumptions are flagged so the user can see them
    items["cost_assumed"] = items["unit_cost"].isna()
    items["unit_cost"] = items["unit_cost"].fillna(items["sell_price"] * 0.6)
    items["sell_price"] = items["sell_price"].fillna(items["unit_cost"] / 0.6)
    items["unit_cost"] = items["unit_cost"].fillna(1.0)
    items["sell_price"] = items["sell_price"].fillna(1.0 / 0.6)
    return items.reset_index(drop=True)


def prepare(tables, settings) -> Prepared:
    notes = []
    fc = settings["forecast"]
    bucket = fc.get("bucket", "W").upper()[:1]
    pdays = PERIOD_DAYS.get(bucket, 7.0)
    as_of = as_of_date(tables, settings)
    dw = settings["data"].get("default_warehouse") or "MAIN"
    sales = tables["sales"].copy()
    if "canceled" in sales:
        n = int(sales["canceled"].fillna(False).astype(bool).sum())
        if n:
            notes.append(f"Removed {n} canceled sales lines.")
        sales = sales[~sales["canceled"].fillna(False).astype(bool)]
    neg = sales["quantity"] <= 0
    if neg.any():
        notes.append(f"{int(neg.sum())} sales lines with zero or negative quantity were treated as returns or corrections and left out of demand.")
    sales = sales[~neg]
    sales = sales[sales["doc_date"] <= as_of]
    hs = str(settings["data"].get("history_start") or "").strip()
    if hs:
        sales = sales[sales["doc_date"] >= pd.Timestamp(hs)]
    sales["warehouse"] = sales["warehouse"].fillna(dw).astype(str)
    plan = [str(w) for w in (settings["data"].get("plan_warehouses") or [])]
    if plan and not sales["warehouse"].isin(plan).any():
        notes.append(f"No sales lines are in the planning warehouses ({', '.join(plan)}), so every warehouse is planned. "
                     f"Check plan_warehouses in Settings (warehouses found: {', '.join(sorted(sales['warehouse'].unique())[:8])}).")
        plan = []
    if plan:
        out = ~sales["warehouse"].isin(plan)
        if out.any():
            gone = ", ".join(sorted(sales.loc[out, "warehouse"].unique()))
            notes.append(f"Left {int(out.sum()):,} sales lines from warehouse(s) {gone} out of forecasting and purchasing "
                         f"(planning covers {', '.join(plan)}). Change plan_warehouses in Settings to include them.")
        sales = sales[~out]
    if sales["line_total"].isna().all() and sales["unit_price"].notna().any():
        sales["line_total"] = sales["unit_price"] * sales["quantity"]
    sales["demand_qty"] = sales["quantity"].astype(float)

    # events marked exclude
    ev = tables.get("events")
    if ev is not None and len(ev):
        mask = pd.Series(False, index=sales.index)
        for _, e in ev.iterrows():
            if str(e.get("action") or "exclude").strip().lower() == "keep":
                continue
            m = (sales["doc_date"] >= e["start_date"]) & (sales["doc_date"] <= e["end_date"])
            if e.get("item_code"):
                m &= sales["item_code"] == e["item_code"]
            if e.get("customer_code"):
                m &= sales["customer_code"] == e["customer_code"]
            mask |= m
        if mask.any():
            notes.append(f"Excluded {int(mask.sum())} sales lines that fall inside 'exclude' events from the forecast history.")
            sales.loc[mask, "demand_qty"] = 0.0

    cap = float(fc.get("spike_cap_multiple") or 0)
    if cap > 0:
        med = sales.groupby("item_code")["demand_qty"].transform("median")
        lim = (med * cap).clip(lower=1)
        n = int((sales["demand_qty"] > lim).sum())
        if n:
            notes.append(f"Capped {n} unusually large lines at {cap:g} times the item's median line.")
        sales["demand_qty"] = np.minimum(sales["demand_qty"], lim)

    full = sales  # every line up to as_of, used for the weekly replay history
    if bucket == "M":
        hist_end = month_anchor(as_of)
        late = sales["doc_date"] > hist_end
        if late.any():
            notes.append(f"Monthly planning uses complete months only: {int(late.sum()):,} sales lines after "
                         f"{hist_end:%b %d, %Y} (the current, unfinished month) are left out of the history.")
        sales = sales[~late]
        first_fc = hist_end + pd.Timedelta(days=1)
    else:
        hist_end = as_of
        first_fc = as_of + pd.Timedelta(days=1)
    sales["p_back"] = periods_back(sales["doc_date"], as_of, bucket, pdays).astype(int)  # 0 = most recent period
    max_back = int(sales["p_back"].max()) if len(sales) else 0
    T = min(max_back + 1, 520)
    sales = sales[sales["p_back"] < T]

    # optional return netting
    if settings["data"].get("net_returns_from_demand") and tables.get("returns") is not None and len(tables["returns"]):
        r = tables["returns"].copy()
        r = r[(r["doc_date"] <= hist_end) & (r["quantity"].abs() > 0)]
        r["warehouse"] = r["warehouse"].fillna(dw).astype(str)
        if plan:
            r = r[r["warehouse"].isin(plan)]
        r["p_back"] = periods_back(r["doc_date"], as_of, bucket, pdays).astype(int)
        r = r[r["p_back"] < T]
        r["demand_qty"] = -r["quantity"].abs()
        sales = pd.concat([sales, r[["item_code", "warehouse", "p_back", "demand_qty", "doc_date"]]], ignore_index=True)
        notes.append(f"Netted {len(r)} return lines out of demand.")

    g = sales.groupby(["item_code", "warehouse", "p_back"])["demand_qty"].sum().reset_index()
    keys = g[["item_code", "warehouse"]].drop_duplicates().sort_values(["item_code", "warehouse"]).reset_index(drop=True)
    keys["sid"] = np.arange(len(keys))
    g = g.merge(keys, on=["item_code", "warehouse"])
    M = np.zeros((len(keys), T))
    col = T - 1 - g["p_back"].to_numpy()
    np.add.at(M, (g["sid"].to_numpy(), col), g["demand_qty"].to_numpy())
    M = np.clip(M, 0, None)
    # a series exists from its first positive period onward
    pos = M > 0
    first = np.where(pos.any(axis=1), pos.argmax(axis=1), T - 1)
    if bucket == "M":
        period_end = pd.DatetimeIndex([hist_end - pd.offsets.MonthEnd(T - 1 - i) if i < T - 1 else hist_end for i in range(T)]).normalize()
    else:
        period_end = pd.DatetimeIndex([as_of - pd.Timedelta(days=pdays * (T - 1 - i)) for i in range(T)]).normalize()
    weekly = None
    if bucket != "W":  # the policy replay and simulator check run week by week, so keep a weekly view too
        wb = np.floor((as_of - full["doc_date"]).dt.days / 7.0).astype(int)
        f2 = full.assign(sid=full.set_index(["item_code", "warehouse"]).index.map(
            dict(zip(zip(keys["item_code"], keys["warehouse"]), keys["sid"]))), wb=wb)
        f2 = f2[f2["sid"].notna() & (f2["wb"] < 260)]
        TW = int(f2["wb"].max()) + 1 if len(f2) else 1
        weekly = np.zeros((len(keys), TW))
        np.add.at(weekly, (f2["sid"].astype(int).to_numpy(), TW - 1 - f2["wb"].to_numpy()), f2["demand_qty"].to_numpy(float))
        weekly = np.clip(weekly, 0, None)
    sold_after = np.zeros(len(keys), bool)
    if bucket == "M":
        lt = full[(full["doc_date"] > hist_end) & (full["demand_qty"] > 0)]
        recent = set(zip(lt["item_code"], lt["warehouse"]))
        sold_after = np.array([(a, b) in recent for a, b in zip(keys["item_code"], keys["warehouse"])], bool)
    items = build_items(tables)
    whs = sorted(set(keys["warehouse"]) | (set(tables["stock"]["warehouse"].astype(str)) if tables.get("stock") is not None else set()))
    if plan:
        whs = [w for w in whs if w in plan]
    return Prepared(as_of=as_of, bucket=bucket, period_days=pdays, lines=sales, keys=keys, matrix=M,
                    first_idx=first, period_end=period_end, items=items, warehouses=whs, notes=notes,
                    history_end=hist_end, first_forecast_start=first_fc, weekly=weekly,
                    sold_after_history=sold_after)
