"""Supplier lead time model and supplier scorecards.

Lead time is learned from real PO and receipt history, not from the static setting in
SAP. Each item and supplier pair gets an empirical distribution. When an item has few
receipts, it borrows the supplier's overall pattern, shifted to the item's own median.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LeadTimeModel:
    po_lines: pd.DataFrame           # one row per PO line with receipt facts
    by_pair: dict                    # (item, supplier) -> np.array of days
    by_supplier: dict                # supplier -> np.array of days
    settings_lt: dict                # item -> lead time setting (days)
    default_days: float
    scorecard: pd.DataFrame
    open_pos: pd.DataFrame

    def samples(self, item, supplier, n=None):
        """Lead time samples in days for this item from this supplier, plus a short label of the source."""
        pair = self.by_pair.get((item, supplier))
        sup = self.by_supplier.get(supplier)
        if pair is not None and len(pair) >= 6:
            return pair, f"{len(pair)} receipts of this item"
        if sup is not None and len(sup) >= 5:
            if pair is not None and len(pair) >= 1:
                shift = np.median(pair) - np.median(sup)
                w = len(pair) / 6.0
                return np.clip(sup + w * shift, 1, None), f"{len(pair)} receipts of this item + supplier pattern ({len(sup)} receipts)"
            return sup, f"supplier pattern ({len(sup)} receipts of all items)"
        base = self.settings_lt.get(item)
        if base is None or not np.isfinite(base) or base <= 0:
            base = self.default_days
            src = "default lead time setting"
        else:
            src = "SAP lead time setting (no receipts yet)"
        # no history: assume a lognormal spread with 25% variation around the setting
        q = np.linspace(0.02, 0.98, 25)
        from scipy.stats import lognorm
        return lognorm.ppf(q, s=0.25, scale=base), src

    def arrival_probabilities(self, po_row, as_of, weeks, grace_days=7):
        """Probability an open PO line arrives in each of the next `weeks` weeks (week 0 = this week)."""
        lt, _ = self.samples(po_row["item_code"], po_row["supplier_code"])
        age = (as_of - po_row["po_date"]).days
        remaining = lt[lt > age] - age
        overdue = len(remaining) == 0
        if overdue:  # older than every lead time seen: expect it within the next four weeks
            p = np.zeros(weeks)
            p[: min(4, weeks)] = 1.0 / min(4, weeks)
            return p, True
        wk = np.floor(remaining / 7).astype(int)
        p = np.bincount(np.clip(wk, 0, weeks), minlength=weeks + 1)[:weeks] / len(remaining)
        late = pd.notna(po_row.get("promised_date")) and as_of > po_row["promised_date"] + pd.Timedelta(days=grace_days)
        return p, bool(late)


def _match_receipts(po: pd.DataFrame, gr: pd.DataFrame) -> pd.DataFrame:
    """Attach receipt quantity and dates to each PO line."""
    po = po.copy()
    po["po_entry"] = po["po_entry"].astype(str)
    po["line_num"] = po["line_num"].fillna(0).astype(int)
    for c in ("received_qty", "first_receipt", "last_receipt", "wavg_receipt"):
        po[c] = 0.0 if c == "received_qty" else pd.NaT
    if gr is None or len(gr) == 0:
        return po
    gr = gr.copy().reset_index(drop=True)
    gr["_rid"] = np.arange(len(gr))
    gr["po_entry"] = gr["po_entry"].astype(str)
    gr["po_line"] = gr["po_line"].fillna(-1).astype(int)
    lines = po[["po_entry", "line_num", "item_code", "supplier_code", "po_date"]]
    # 1) receipts that name their PO
    linked = gr[gr["po_entry"].isin(set(po["po_entry"]))]
    a = linked.merge(lines[["po_entry", "line_num", "item_code"]], on=["po_entry", "item_code"], how="inner")
    a = a[(a["po_line"] < 0) | (a["po_line"] == a["line_num"])]
    a = a.sort_values(["_rid", "line_num"]).drop_duplicates("_rid")
    # 2) receipts with no usable link: most recent earlier PO for the same item (and supplier when known)
    un = gr[~gr["_rid"].isin(set(a["_rid"]))].drop(columns=["po_entry", "po_line"])
    parts = [a[["_rid", "po_entry", "line_num", "quantity", "receipt_date"]]]
    if len(un):
        cand = lines.dropna(subset=["po_date"]).sort_values("po_date")
        un = un.dropna(subset=["receipt_date"]).sort_values("receipt_date")
        by = ["item_code"]
        if un["supplier_code"].notna().all():
            by.append("supplier_code")
        else:
            un = un.drop(columns=["supplier_code"])
            cand = cand.drop(columns=["supplier_code"])
        m = pd.merge_asof(un, cand, left_on="receipt_date", right_on="po_date", by=by, direction="backward")
        m = m.dropna(subset=["po_entry"])
        if len(m):
            m["line_num"] = m["line_num"].astype(int)
            parts.append(m[["_rid", "po_entry", "line_num", "quantity", "receipt_date"]])
    r = pd.concat(parts, ignore_index=True)
    if len(r) == 0:
        return po
    r["_d"] = r["receipt_date"].astype("datetime64[ns]").astype("int64") / 86400e9
    r["_qd"] = r["quantity"] * r["_d"]
    agg = r.groupby(["po_entry", "line_num"]).agg(received_qty=("quantity", "sum"), first_receipt=("receipt_date", "min"),
                                                 last_receipt=("receipt_date", "max"), _qd=("_qd", "sum"))
    agg["wavg_receipt"] = pd.to_datetime(agg["_qd"] / agg["received_qty"].where(agg["received_qty"] > 0), unit="D").dt.normalize()
    po = po.drop(columns=["received_qty", "first_receipt", "last_receipt", "wavg_receipt"])
    po = po.merge(agg.drop(columns="_qd").reset_index(), on=["po_entry", "line_num"], how="left")
    po["received_qty"] = po["received_qty"].fillna(0.0)
    return po


def build(tables, items: pd.DataFrame, as_of: pd.Timestamp, settings) -> LeadTimeModel:
    inv = settings["inventory"]
    default = float(inv.get("default_lead_time_days", 75))
    grace = int(settings["alerts"].get("late_po_grace_days", 7))
    settings_lt = items.set_index("item_code")["lead_time_days"].to_dict() if "lead_time_days" in items else {}
    # item_suppliers lead time settings fill gaps
    isup = tables.get("item_suppliers")
    if isup is not None and len(isup) and isup["lead_time_days"].notna().any():
        for r in isup.dropna(subset=["lead_time_days"]).itertuples(index=False):
            if not np.isfinite(settings_lt.get(r.item_code, np.nan) or np.nan):
                settings_lt[r.item_code] = r.lead_time_days
    po = tables.get("purchases")
    if po is None or len(po) == 0:
        empty = pd.DataFrame()
        return LeadTimeModel(empty, {}, {}, settings_lt, default, empty, empty)
    po = po[po["po_date"] <= as_of].copy()
    po = _match_receipts(po, tables.get("receipts"))
    po["lead_days"] = (po["wavg_receipt"] - po["po_date"]).dt.days
    po.loc[(po["lead_days"] < 0) | (po["lead_days"] > 400), "lead_days"] = np.nan
    if po["open_qty"].isna().all():
        po["open_qty"] = np.where(po["status"].fillna("").str.upper().str.startswith("C"), 0.0,
                                  (po["quantity"] - po["received_qty"]).clip(lower=0))
    po["open_qty"] = po["open_qty"].fillna(0.0).clip(lower=0)
    closed = po["open_qty"] <= 0
    po["fill_rate"] = np.where(closed, (po["received_qty"] / po["quantity"].replace(0, np.nan)).clip(upper=1), np.nan)
    po["days_late"] = (po["last_receipt"] - po["promised_date"]).dt.days
    po["on_time"] = np.where(po["last_receipt"].notna() & po["promised_date"].notna(), po["days_late"] <= grace, np.nan)

    hist = po.dropna(subset=["lead_days"])
    by_pair = {k: g["lead_days"].to_numpy(float) for k, g in hist.groupby(["item_code", "supplier_code"])}
    by_sup = {k: g["lead_days"].to_numpy(float) for k, g in hist.groupby("supplier_code")}

    # ---------------------------------------------------------------- scorecard
    sup_names = {}
    sm = tables.get("suppliers")
    if sm is not None and len(sm):
        sup_names = sm.set_index("supplier_code")["supplier_name"].to_dict()
        sup_country = sm.set_index("supplier_code")["country"].to_dict()
    else:
        sup_country = {}
    last12 = po[po["po_date"] > as_of - pd.Timedelta(days=365)]
    rows = []
    # price competitiveness: supplier price vs cheapest option for the same item in the last 12 months
    px = last12.dropna(subset=["unit_price"]).groupby(["item_code", "supplier_code"])["unit_price"].median().reset_index()
    if len(px):
        px["min_px"] = px.groupby("item_code")["unit_price"].transform("min")
        px["n_sup"] = px.groupby("item_code")["supplier_code"].transform("count")
        px = px[px["n_sup"] > 1]
        px["ratio"] = px["unit_price"] / px["min_px"]
    for sup, g in po.groupby("supplier_code"):
        h = g.dropna(subset=["lead_days"])
        lt = h["lead_days"]
        ot = g["on_time"].dropna()
        fr = g["fill_rate"].dropna()
        g12 = last12[last12["supplier_code"] == sup]
        spend = float((g12["quantity"] * g12["unit_price"]).sum()) if g12["unit_price"].notna().any() else np.nan
        # price change on the same items, latest 12 months vs the 12 before
        prev = po[(po["supplier_code"] == sup) & (po["po_date"] <= as_of - pd.Timedelta(days=365)) &
                  (po["po_date"] > as_of - pd.Timedelta(days=730))]
        trend = np.nan
        if len(prev) and len(g12):
            a = g12.groupby("item_code")["unit_price"].median()
            b = prev.groupby("item_code")["unit_price"].median()
            common = a.index.intersection(b.index)
            if len(common) >= 3:
                trend = float(np.median(a[common] / b[common]) - 1)
        pr = px[px["supplier_code"] == sup]["ratio"] if len(px) else pd.Series(dtype=float)
        lt_cv = float(lt.std() / lt.mean()) if len(lt) > 2 and lt.mean() > 0 else np.nan
        rows.append({
            "Supplier": sup, "Name": sup_names.get(sup, sup), "Country": sup_country.get(sup, ""),
            "PO lines": len(g), "PO lines last 12 months": len(g12), "Spend last 12 months": spend,
            "Median lead time (days)": float(lt.median()) if len(lt) else np.nan,
            "P90 lead time (days)": float(lt.quantile(0.9)) if len(lt) else np.nan,
            "Lead time variability (CV)": lt_cv,
            "On time %": float(ot.mean()) if len(ot) else np.nan,
            "Average days late when late": float(g.loc[g["days_late"] > grace, "days_late"].mean()) if (g["days_late"] > grace).any() else 0.0,
            "Fill rate %": float(fr.mean()) if len(fr) else np.nan,
            "Price change vs prior year": trend,
            "Price vs cheapest option": float(pr.median()) if len(pr) else np.nan,
            "Open PO lines": int((g["open_qty"] > 0).sum()),
        })
    sc = pd.DataFrame(rows)
    if len(sc):
        def pts(series, good_high=True, lo=None, hi=None):
            s = series.astype(float)
            lo = s.quantile(0.05) if lo is None else lo
            hi = s.quantile(0.95) if hi is None else hi
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return pd.Series(0.5, index=s.index)
            x = ((s - lo) / (hi - lo)).clip(0, 1)
            x = x if good_high else 1 - x
            return x.fillna(0.5)
        score = (0.35 * pts(sc["On time %"], True, 0.5, 1.0) + 0.25 * pts(sc["Fill rate %"], True, 0.8, 1.0)
                 + 0.20 * pts(sc["Lead time variability (CV)"], False, 0.05, 0.6)
                 + 0.20 * pts(sc["Price vs cheapest option"], False, 1.0, 1.3))
        sc["Score (0-100)"] = (score * 100).round(0)
        sc["Grade"] = pd.cut(sc["Score (0-100)"], [-1, 55, 70, 85, 101], labels=["D", "C", "B", "A"]).astype(str)
        sc = sc.sort_values("Spend last 12 months", ascending=False, na_position="last").reset_index(drop=True)
    open_pos = po[po["open_qty"] > 0].copy()
    return LeadTimeModel(po, by_pair, by_sup, settings_lt, default, sc, open_pos)
