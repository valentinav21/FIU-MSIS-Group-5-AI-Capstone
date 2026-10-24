"""Turn loaded tables into clean demand history and shared lookups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIOD_DAYS = {"W": 7.0, "M": 30.4375}


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

    age = (as_of - sales["doc_date"]).dt.days
    sales["p_back"] = np.floor(age / pdays).astype(int)  # 0 = most recent period
    max_back = int(sales["p_back"].max()) if len(sales) else 0
    T = min(max_back + 1, 520)
    sales = sales[sales["p_back"] < T]

    # optional return netting
    if settings["data"].get("net_returns_from_demand") and tables.get("returns") is not None and len(tables["returns"]):
        r = tables["returns"].copy()
        r = r[(r["doc_date"] <= as_of) & (r["quantity"].abs() > 0)]
        r["warehouse"] = r["warehouse"].fillna(dw).astype(str)
        r["p_back"] = np.floor((as_of - r["doc_date"]).dt.days / pdays).astype(int)
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
    period_end = pd.DatetimeIndex([as_of - pd.Timedelta(days=pdays * (T - 1 - i)) for i in range(T)]).normalize()
    items = build_items(tables)
    whs = sorted(set(keys["warehouse"]) | (set(tables["stock"]["warehouse"].astype(str)) if tables.get("stock") is not None else set()))
    return Prepared(as_of=as_of, bucket=bucket, period_days=pdays, lines=sales, keys=keys, matrix=M,
                    first_idx=first, period_end=period_end, items=items, warehouses=whs, notes=notes)
