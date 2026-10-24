"""Purchase order recommendations with multi-supplier choice and SAP Business One draft export.

For every item x warehouse at or near its reorder point, each sourcing option is costed:
landed purchase cost, plus the expected cost of running short before that supplier's
goods arrive (simulated with that supplier's own lead time history), plus the carrying
cost of the safety stock that supplier's reliability requires. The cheapest total wins,
and the explanation says why in plain words.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .inventory import ServiceCurve, calibrated_samples
from .probabilistic import lead_time_demand


def _options(item, planning_sup, tables, lt_model, item_cost):
    opts = {}
    isup = tables.get("item_suppliers")
    if isup is not None and len(isup):
        for r in isup[isup["item_code"] == item].itertuples(index=False):
            opts.setdefault(r.supplier_code, {"price": r.unit_price, "moq": r.moq, "mult": r.order_multiple,
                                              "landed": r.landed_cost_pct})
    po = lt_model.po_lines
    if len(po):
        g = po[po["item_code"] == item].sort_values("po_date")
        for sup, gg in g.groupby("supplier_code"):
            o = opts.setdefault(sup, {"price": np.nan, "moq": np.nan, "mult": np.nan, "landed": np.nan})
            if not pd.notna(o["price"]) and gg["unit_price"].notna().any():
                o["price"] = gg["unit_price"].dropna().iloc[-1]
    if planning_sup and planning_sup not in opts:
        opts[planning_sup] = {"price": np.nan, "moq": np.nan, "mult": np.nan, "landed": np.nan}
    if not opts:
        opts[planning_sup or "UNKNOWN"] = {"price": np.nan, "moq": np.nan, "mult": np.nan, "landed": np.nan}
    for o in opts.values():
        o["price"] = float(o["price"]) if pd.notna(o["price"]) else float(item_cost)
        o["landed"] = float(o["landed"]) if pd.notna(o["landed"]) else 0.0
    return opts


def build(prep, fc, pol, lt_model, tables, settings, seed=9, n_sims=1000):
    rng = np.random.default_rng(seed)
    pur = settings["purchasing"]
    inv = settings["inventory"]
    review = float(inv.get("review_period_days", 7))
    h_pct = float(inv.get("holding_cost_pct_per_year", 0.25))
    so_pct = float(pur.get("stockout_cost_pct_of_price", 0.3))
    rel_w = float(pur.get("supplier_reliability_weight", 1.0))
    default_moq = float(pur.get("default_supplier_moq", 1))
    as_of = prep.as_of
    t = fc.table
    P = pol.table
    names = {}
    sm = tables.get("suppliers")
    if sm is not None and len(sm):
        names = sm.set_index("supplier_code")["supplier_name"].to_dict()
    stock = tables.get("stock")
    st_idx = {}
    if stock is not None and len(stock):
        st = stock.copy()
        st["warehouse"] = st["warehouse"].astype(str)
        st_idx = st.drop_duplicates(["item_code", "warehouse"]).set_index(["item_code", "warehouse"]).to_dict("index")
    oo = tables.get("open_orders")
    oo_tot = {}
    if oo is not None and len(oo):
        o = oo.copy()
        o["warehouse"] = o["warehouse"].fillna(settings["data"].get("default_warehouse", "MIA")).astype(str)
        oo_tot = o.groupby(["item_code", "warehouse"])["open_qty"].sum().to_dict()
    op = lt_model.open_pos if lt_model.open_pos is not None and len(lt_model.open_pos) else None
    po_tot = {}
    if op is not None:
        o = op.copy()
        o["warehouse"] = o["warehouse"].fillna(settings["data"].get("default_warehouse", "MIA")).astype(str)
        po_tot = o.groupby(["item_code", "warehouse"])["open_qty"].sum().to_dict()
    names_items = prep.items.set_index("item_code")["item_name"].to_dict()
    rows = []
    for i in range(len(P)):
        pr = P.iloc[i]
        if pr["forecast_per_period"] <= 0:
            continue
        item, wh = pr["item_code"], pr["warehouse"]
        cur = st_idx.get((item, wh), {})
        on_hand = max(float(cur.get("on_hand") or 0), 0.0)
        committed = float(oo_tot.get((item, wh), cur.get("committed") or 0) or 0)
        on_po = float(po_tot.get((item, wh), cur.get("on_order") or 0) or 0) if op is not None else float(cur.get("on_order") or 0)
        ip = on_hand - committed + on_po
        rop, upto = float(pr["reorder_point"]), float(pr["order_up_to"])
        per_day = pr["forecast_per_period"] / prep.period_days
        soon_days = 2 * review
        if ip <= rop:
            action, order_by = "Order now", as_of
        elif ip - per_day * soon_days <= rop:
            days = max((ip - rop) / max(per_day, 1e-9), 0)
            action, order_by = "Order soon", as_of + pd.Timedelta(days=int(days))
        else:
            continue
        need = max(upto - ip, 1.0)
        opts = _options(item, pr["supplier"], tables, lt_model, pr["unit_cost"])
        cls = pr["demand_class"]
        cost_unit_item = float(pr["unit_cost"])
        price_sell = float(pr["sell_price"])
        cu = price_sell * so_pct + max(price_sell - cost_unit_item, 0)
        ann = pr["forecast_per_period"] * 365.25 / prep.period_days
        evals = []
        for sup, o in opts.items():
            moq = float(o["moq"]) if pd.notna(o["moq"]) and o["moq"] > 0 else max(float(pr["moq"]) if pd.notna(pr["moq"]) else default_moq, 1.0)
            mult = float(o["mult"]) if pd.notna(o["mult"]) and o["mult"] > 0 else (float(pr["order_multiple"]) if pd.notna(pr["order_multiple"]) else 1.0)
            qty = float(np.ceil(max(need, moq) / mult - 1e-9) * mult)
            lt, lt_src = lt_model.samples(item, sup)
            lt_draw = np.clip(rng.choice(lt, n_sims) + rng.uniform(-1.5, 1.5, n_sims), 1, None)
            raw = lead_time_demand(fc.sim_params[i]["sizes"], fc.sim_params[i]["occ"], fc.path[i], prep.period_days,
                                   lt_draw, n_sims, float(fc.level_sigma.get(cls, 0.25)), rng)
            x = np.sort(calibrated_samples(fc, cls, raw))
            sc_now = ServiceCurve(x, max(qty, 1))
            exp_short = float(sc_now.short(max(ip, 0))[0])
            p_short = float(1 - sc_now.cycle(max(ip, 0))[0])
            # safety stock this supplier would need to hit the same target, carried for one order cycle
            raw_r = lead_time_demand(fc.sim_params[i]["sizes"], fc.sim_params[i]["occ"], fc.path[i], prep.period_days,
                                     lt_draw + review, n_sims, float(fc.level_sigma.get(cls, 0.25)), rng)
            xr = np.sort(calibrated_samples(fc, cls, raw_r))
            sc = ServiceCurve(xr, max(qty, 1))
            ss_units = max(sc.rop_for(float(pr["service_target"]), pol.measure) - xr.mean(), 0)
            cycle_years = qty / ann if ann > 0 else 1.0
            unit_landed = o["price"] * (1 + o["landed"])
            purchase = qty * unit_landed
            short_cost = exp_short * cu * rel_w
            carry = ss_units * unit_landed * h_pct * min(cycle_years, 1.0)
            evals.append({"supplier": sup, "qty": qty, "unit_price": o["price"], "unit_landed": unit_landed,
                          "lt_median": float(np.median(lt)), "lt_p90": float(np.quantile(lt, 0.9)), "lt_source": lt_src,
                          "exp_short": exp_short, "p_short": p_short, "short_cost": short_cost, "carry": carry, "purchase": purchase,
                          "total": purchase + short_cost + carry, "moq": moq})
        ev = sorted(evals, key=lambda e: e["total"])
        best = ev[0]
        why = [f"Stock position {ip:.0f} (on hand {on_hand:.0f} - customer orders {committed:.0f} + open POs {on_po:.0f}) "
               f"{'is at or below' if action == 'Order now' else 'will reach'} the reorder point {rop:.0f}. Ordering up to {upto:.0f}."]
        if best["qty"] > need + 0.5:
            why.append(f"Quantity rounded up from {need:.0f} to {best['qty']:.0f} for the supplier minimum or pack size.")
        if len(ev) > 1:
            alt = ev[1]
            d_price = (best["unit_landed"] - alt["unit_landed"]) * best["qty"]
            if best["lt_median"] < alt["lt_median"] - 3 and d_price > 0:
                why.append(f"{best['supplier']} costs ${d_price:,.0f} more than {alt['supplier']} but arrives in about "
                           f"{best['lt_median']:.0f} days instead of {alt['lt_median']:.0f}, cutting expected units short "
                           f"before arrival from {alt['exp_short']:.1f} to {best['exp_short']:.1f} (worth ${alt['short_cost'] - best['short_cost']:,.0f}).")
            else:
                why.append(f"{best['supplier']} has the lowest total cost: ${best['total']:,.0f} vs ${alt['total']:,.0f} "
                           f"from {alt['supplier']} (landed price, shortage risk and safety stock carrying cost included).")
        if best["p_short"] >= 0.5 and best["exp_short"] >= 1:
            why.append(f"Even with this order there is a {best['p_short']:.0%} chance of running short before it arrives "
                       f"(about {best['exp_short']:.1f} units); consider a transfer or expediting.")
        eta = as_of + pd.Timedelta(days=float(best["lt_median"]))
        rows.append({
            "approve": action == "Order now", "action": action, "order_by": order_by.normalize(),
            "item_code": item, "item_name": names_items.get(item, item), "warehouse": wh, "abc": pr["abc"],
            "supplier": best["supplier"], "supplier_name": names.get(best["supplier"], best["supplier"]),
            "quantity": best["qty"], "unit_price": round(best["unit_price"], 4), "line_total": round(best["qty"] * best["unit_price"], 2),
            "expected_arrival": eta.normalize(), "arrival_p90": (as_of + pd.Timedelta(days=float(best["lt_p90"]))).normalize(),
            "stock_position": ip, "reorder_point": rop, "order_up_to": upto,
            "expected_units_short_before_arrival": best["exp_short"],
            "chance_short_before_arrival": best["p_short"],
            "weeks_cover_after_receipt": (ip + best["qty"]) / max(per_day * 7, 1e-9),
            "alternatives": "; ".join(f"{e['supplier']}: {e['qty']:.0f} @ ${e['unit_landed']:.2f} landed, ~{e['lt_median']:.0f} days, total ${e['total']:,.0f}"
                                      for e in ev[1:3]),
            "why": " ".join(why),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["action", "expected_units_short_before_arrival", "line_total"],
                              ascending=[True, False, False]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------- SAP export

def group_drafts(recs: pd.DataFrame, as_of) -> list[dict]:
    """One draft PO per supplier and warehouse."""
    drafts = []
    for k, ((sup, wh), g) in enumerate(recs.groupby(["supplier", "warehouse"]), start=1):
        due = g["expected_arrival"].max()
        drafts.append({"key": k, "supplier": sup, "warehouse": wh, "doc_date": pd.Timestamp(as_of), "due": pd.Timestamp(due),
                       "lines": g.reset_index(drop=True)})
    return drafts


def to_dtw(recs: pd.DataFrame, as_of, series: str = "") -> tuple[str, str]:
    """CSV text for the SAP B1 Data Transfer Workbench, object oDrafts with DocObjectCode 22 (purchase order).

    Two header rows, as in the DTW templates. Dates are YYYYMMDD. Values are quoted where needed.
    """
    import csv
    import io as _io
    head_cols = ["RecordKey", "CardCode", "DocDate", "DocDueDate", "DocObjectCode", "DocType", "Comments", "Series"]
    line_cols = ["ParentKey", "LineNum", "ItemCode", "Quantity", "Price", "WarehouseCode", "ShipDate"]
    hb, lb = _io.StringIO(), _io.StringIO()
    hw, lw = csv.writer(hb, lineterminator="\n"), csv.writer(lb, lineterminator="\n")
    hw.writerow(head_cols)
    hw.writerow(head_cols)
    lw.writerow(line_cols)
    lw.writerow(line_cols)
    for d in group_drafts(recs, as_of):
        hw.writerow([d["key"], d["supplier"], d["doc_date"].strftime("%Y%m%d"), d["due"].strftime("%Y%m%d"), "22",
                     "dDocument_Items", f"AI planner draft {d['doc_date']:%Y-%m-%d} - review before approval", series])
        for j, r in d["lines"].iterrows():
            lw.writerow([d["key"], j, r["item_code"], f"{r['quantity']:g}", f"{r['unit_price']:.4f}", r["warehouse"],
                         pd.Timestamp(r["expected_arrival"]).strftime("%Y%m%d")])
    return hb.getvalue(), lb.getvalue()


def to_service_layer(recs: pd.DataFrame, as_of) -> str:
    """JSON bodies for POST /b1s/v1/Drafts on the SAP B1 Service Layer (one draft per supplier and warehouse)."""
    out = []
    for d in group_drafts(recs, as_of):
        out.append({
            "DocObjectCode": "oPurchaseOrders",
            "CardCode": d["supplier"],
            "DocDate": d["doc_date"].strftime("%Y-%m-%d"),
            "DocDueDate": d["due"].strftime("%Y-%m-%d"),
            "Comments": "AI planner draft - review before approval",
            "DocumentLines": [{"ItemCode": str(r["item_code"]), "Quantity": float(r["quantity"]),
                               "UnitPrice": float(r["unit_price"]), "WarehouseCode": str(r["warehouse"]),
                               "ShipDate": pd.Timestamp(r["expected_arrival"]).strftime("%Y-%m-%d")}
                              for _, r in d["lines"].iterrows()],
        })
    return json.dumps(out, indent=2)


def transfers(pol, tables, settings, transfer_days: float = 5.0) -> pd.DataFrame:
    """Suggest moving stock between warehouses when one is short and another holds more than it needs."""
    P = pol.table
    stock = tables.get("stock")
    if stock is None or len(stock) == 0 or P["warehouse"].nunique() < 2:
        return pd.DataFrame()
    st = stock.copy()
    st["warehouse"] = st["warehouse"].astype(str)
    st = st.drop_duplicates(["item_code", "warehouse"]).set_index(["item_code", "warehouse"])
    rows = []
    for item, g in P[P["forecast_per_period"] > 0].groupby("item_code"):
        if len(g) < 2:
            continue
        info = []
        for _, r in g.iterrows():
            key = (item, r["warehouse"])
            oh = float(st.loc[key, "on_hand"]) if key in st.index else 0.0
            com = float(st.loc[key, "committed"]) if key in st.index and pd.notna(st.loc[key, "committed"]) else 0.0
            onord = float(st.loc[key, "on_order"]) if key in st.index and pd.notna(st.loc[key, "on_order"]) else 0.0
            ip = oh - com + onord
            info.append((r["warehouse"], oh - com, ip, r["reorder_point"], r["order_up_to"], r["forecast_per_period"]))
        for wa, avail_a, ip_a, rop_a, up_a, f_a in info:
            if ip_a > rop_a:
                continue
            need = rop_a - ip_a
            for wb, avail_b, ip_b, rop_b, up_b, f_b in info:
                if wb == wa:
                    continue
                spare = min(avail_b, ip_b - rop_b) - f_b * transfer_days / 7.0
                if spare >= 1:
                    q = float(np.floor(min(spare, need)))
                    if q >= 1:
                        rows.append({"item_code": item, "from_warehouse": wb, "to_warehouse": wa, "quantity": q,
                                     "why": f"{wa} is {need:.0f} units below its reorder point; {wb} holds {spare:.0f} more than it needs."})
    return pd.DataFrame(rows)
