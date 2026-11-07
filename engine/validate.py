"""Data health checks. Each check returns a row the app shows on the Data Health page."""
from __future__ import annotations

import pandas as pd

from .schema import TABLES

OK, WARN, FAIL, INFO = "OK", "Check", "Problem", "Info"


def _row(area, check, status, detail, fix=""):
    return {"Area": area, "Check": check, "Status": status, "Detail": detail, "How to fix": fix}


def run_checks(res, settings) -> pd.DataFrame:
    t = res.tables
    rows = []
    for level, table, msg in res.issues:
        st = {"error": FAIL, "warning": WARN}.get(level, INFO)
        rows.append(_row(table, "File loading", st, msg))

    items, sales = t.get("items"), t.get("sales")
    if items is not None:
        d = int(items["item_code"].duplicated().sum())
        rows.append(_row("items", "Duplicate item codes", OK if d == 0 else WARN, f"{d:,} duplicates",
                         "Keep one row per ItemCode; the first row is used."))
        for c, fix in (("unit_cost", "Needed for inventory value and holding cost. Missing cost falls back to the latest PO price, then 60% of selling price."),
                       ("category", "Needed for grouping and segment analysis. Missing values show as Uncategorized.")):
            share = items[c].isna().mean() if c in items else 1
            rows.append(_row("items", f"Items missing {c}", OK if share < 0.05 else WARN, f"{share:.0%} of items", fix))
    if sales is not None and len(sales):
        dmin, dmax = sales["doc_date"].min(), sales["doc_date"].max()
        months = (dmax - dmin).days / 30.44
        rows.append(_row("sales", "History length", OK if months >= 24 else (WARN if months >= 12 else FAIL),
                         f"{dmin:%Y-%m-%d} to {dmax:%Y-%m-%d} ({months:.0f} months)",
                         "24 months or more is best for seasonality and lead times."))
        fut = int((sales["doc_date"] > pd.Timestamp.today() + pd.Timedelta(days=1)).sum())
        if fut:
            rows.append(_row("sales", "Future dated lines", WARN, f"{fut} lines dated after today", "Check the date column."))
        neg = (sales["quantity"] <= 0).mean()
        rows.append(_row("sales", "Zero or negative quantities", OK if neg < 0.02 else WARN, f"{neg:.1%} of lines",
                         "These are excluded from demand. Export returns separately as returns.csv."))
        if items is not None:
            unk = ~sales["item_code"].isin(set(items["item_code"]))
            rows.append(_row("sales", "Items not in product master", OK if unk.mean() < 0.01 else WARN,
                             f"{unk.sum()} lines, {sales.loc[unk, 'item_code'].nunique()} item codes",
                             "Include inactive items in the OITM export."))
        for c, why in (("customer_code", "customer and cross-sell analysis"), ("warehouse", "per warehouse planning"),
                       ("line_total", "revenue, ABC class and pricing")):
            share = sales[c].isna().mean()
            st = OK if share < 0.02 else (WARN if share < 0.5 else (FAIL if c == "line_total" and sales["unit_price"].isna().all() else WARN))
            rows.append(_row("sales", f"Lines missing {c}", st, f"{share:.0%} of lines", f"Needed for {why}."))
        geo = (sales["ship_state"].isna() & sales["ship_country"].isna()).mean()
        rows.append(_row("sales", "Lines with no ship-to state or country", OK if geo < 0.05 else WARN, f"{geo:.0%} of lines",
                         "Needed for geographic analysis. Customer master location is used when the line has none."))
        if sales["doc_entry"].notna().any() and sales["line_num"].notna().any():
            keys = [c for c in ("doc_entry", "line_num", "item_code", "warehouse") if c in sales]
            dup = int(sales.duplicated(keys).sum())
            rows.append(_row("sales", "Duplicate document lines", OK if dup == 0 else WARN, f"{dup:,} duplicates",
                             "The same DocEntry and LineNum appears twice. Check for overlapping exports."))
    stock = t.get("stock")
    if stock is not None and len(stock):
        neg = int((stock["on_hand"] < 0).sum())
        rows.append(_row("stock", "Negative on hand", OK if neg == 0 else WARN, f"{neg:,} item-warehouse rows",
                         "Negative stock is treated as zero available."))
        if "min_stock" in stock and "max_stock" in stock:
            mn = pd.to_numeric(stock["min_stock"], errors="coerce").fillna(0)
            mx = pd.to_numeric(stock["max_stock"], errors="coerce").fillna(0)
            share = float(((mn > 0) | (mx > 0)).mean())
            rows.append(_row("stock", "Rows with SAP MinStock / MaxStock set", OK if share >= 0.5 else INFO, f"{share:.0%} of rows",
                             "0 / 0 means not set in SAP. When most rows are not set, Stock levels compares the planner with "
                             "today's stock on hand instead of today's SAP settings."))
        if sales is not None:
            noso = ~sales["item_code"].isin(set(stock["item_code"]))
            rows.append(_row("stock", "Sold items with no stock row", OK if noso.mean() < 0.05 else WARN,
                             f"{sales.loc[noso, 'item_code'].nunique():,} items", "Assumed zero on hand."))
    po, gr = t.get("purchases"), t.get("receipts")
    if po is not None and len(po):
        if gr is not None and len(gr):
            linked = gr["po_entry"].isin(set(po["po_entry"])).mean() if gr["po_entry"].notna().any() else 0
            rows.append(_row("receipts", "Receipts linked to a PO", OK if linked > 0.9 else (WARN if linked > 0.5 else FAIL),
                             f"{linked:.0%} of receipt lines",
                             "Receipts are joined to the PO line on PO key + line (BaseEntry + BaseLine). Receipts with no PO key "
                             "are matched to the latest earlier PO of the same item and supplier."))
        else:
            rows.append(_row("receipts", "Goods receipts", WARN, "No receipts file",
                             "Without receipts the lead time comes from settings, not history."))
        pr = po["promised_date"].notna().mean()
        rows.append(_row("purchases", "POs with a promised date", OK if pr > 0.8 else (WARN if pr >= 0.2 else INFO), f"{pr:.0%} of PO lines",
                         "With promise dates: on-time % and days late. Without them (FORTPRO): reliability = arriving within 1.5 x "
                         "the supplier's median lead time, and a PO is late once older than 9 in 10 past orders."))
        days = int((settings or {}).get("data", {}).get("stale_open_days", 0) or 0)
        if days and po["open_qty"].notna().any():
            asof = po["po_date"].max()
            o = po[po["open_qty"].fillna(0) > 0]
            old = (asof - o["po_date"]).dt.days > days
            rows.append(_row("purchases", f"Open PO lines older than {days} days", OK if old.sum() == 0 else WARN,
                             f"{int(old.sum())} lines, {o.loc[old, 'open_qty'].sum():,.0f} units",
                             "Not counted as incoming stock. Close or confirm them in SAP."))
    else:
        rows.append(_row("purchases", "Purchase orders", WARN, "No purchase file",
                         "Lead times, supplier scorecards and open PO arrivals need POR1 and PDN1."))
    if stock is not None and len(stock) and po is not None and len(po) and "on_order" in stock and stock["on_order"].notna().any():
        a_, b_ = float(stock["on_order"].fillna(0).sum()), float(po["open_qty"].fillna(0).clip(lower=0).sum())
        if a_ > 0 and abs(a_ - b_) / a_ > 0.05:
            rows.append(_row("stock", "Stock on_order vs open PO lines", INFO, f"{a_:,.0f} vs {b_:,.0f} units",
                             "Incoming stock uses the open PO lines (purchases.open_qty); stock.on_order is only a fallback."))
    plan = [str(w) for w in ((settings or {}).get("data", {}).get("plan_warehouses") or [])]
    if sales is not None and plan:
        out = ~sales["warehouse"].astype(str).isin(plan)
        rows.append(_row("sales", "Sales outside the planning warehouses", INFO, f"{out.mean():.1%} of lines",
                         f"Planning covers {', '.join(plan)} only (Settings). Other warehouses stay in the files but are not forecast."))
    cust = t.get("customers")
    if sales is not None and cust is not None and len(cust):
        miss = ~sales["customer_code"].dropna().isin(set(cust["customer_code"]))
        rows.append(_row("customers", "Customers on sales not in master", OK if miss.mean() < 0.02 else WARN,
                         f"{miss.mean():.0%} of lines", "Include inactive customers in the OCRD export."))
    for key, tab in TABLES.items():
        n = len(t[key]) if key in t else 0
        rows.append(_row(key, f"{tab.title}", INFO if n else (WARN if tab.essential else INFO),
                         f"{n:,} row{'s' if n != 1 else ''} loaded" if n else ("not provided" + (" (essential)" if tab.essential else " (optional)")),
                         f"SAP source: {tab.sap_source}"))
    return pd.DataFrame(rows)


def module_readiness(res, settings=None) -> pd.DataFrame:
    h = res.has
    mods = [
        ("Demand forecast", h("sales"), "sales"),
        ("Safety stock and reorder points", h("sales"), "sales (+ stock, purchases, receipts for accuracy)"),
        ("Lead time model", h("purchases") and h("receipts"), "purchases + receipts"),
        ("Risk alerts 1 to 8 weeks", h("sales") and h("stock"), "sales + stock (+ open POs, open orders)"),
        ("PO recommendations", h("sales") and h("stock"), "sales + stock (+ item_suppliers for multi-supplier)"),
        ("Supplier scorecards", h("purchases") and h("receipts"), "purchases + receipts"),
        ("Customer segments and churn", h("sales") and res.tables.get("sales") is not None and res.tables["sales"]["customer_code"].notna().any(), "sales with customer_code"),
        ("Cross-sell and upsell", h("sales") and res.tables.get("sales") is not None and res.tables["sales"]["customer_code"].notna().any(), "sales with customer_code"),
        ("Markets and geography", h("sales") and (h("customers") or (res.tables.get("sales") is not None and res.tables["sales"]["ship_state"].notna().any())), "customers or ship_state on sales (+ market_potential)"),
        ("Pricing", h("sales") and res.tables.get("sales") is not None and (res.tables["sales"]["unit_price"].notna().any() or res.tables["sales"]["line_total"].notna().any()), "sales with price"),
        ("Unmet web demand", h("web_searches"), "web_searches (FocusPoint)"),
    ]
    bp2 = bool((settings or {}).get("features", {}).get("bp2_enabled", False))
    bp2_mods = {"Customer segments and churn", "Cross-sell and upsell", "Markets and geography", "Pricing", "Unmet web demand"}
    return pd.DataFrame([{"Module": m + (" (BP2, TBD)" if m in bp2_mods and not bp2 else ""),
                          "Ready": ("TBD" if m in bp2_mods and not bp2 else ("Yes" if ok else "No")), "Needs": need}
                         for m, ok, need in mods])
