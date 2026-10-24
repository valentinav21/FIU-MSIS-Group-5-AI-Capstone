"""Stockout and overstock early warnings, 1 to 8 weeks ahead, with root causes.

For each item x warehouse the engine projects inventory week by week across hundreds of
simulated futures: today's stock, minus customer orders already promised, minus
simulated demand, plus open POs arriving on dates drawn from that supplier's real lead
time behaviour (given how long each PO has already been open). The share of futures
that run out in a week is the stockout risk for that week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .probabilistic import simulate_periods


def _weekly_path(path, period_days, weeks):
    per_week = np.asarray(path, float) * (7.0 / period_days)
    if period_days == 7.0:
        out = per_week[:weeks]
    else:  # monthly buckets: spread each month evenly over its weeks
        idx = np.minimum((np.arange(weeks) * 7 / period_days).astype(int), len(per_week) - 1)
        out = per_week[idx]
    if len(out) < weeks:
        out = np.concatenate([out, np.full(weeks - len(out), out[-1] if len(out) else 0.0)])
    return out


def build(prep, fc, pol, lt_model, tables, settings, seed=5, n_sims=500):
    al = settings["alerts"]
    W = int(al.get("horizon_weeks", 8))
    thr = float(al.get("stockout_probability_threshold", 0.3))
    grace = int(al.get("late_po_grace_days", 7))
    over_w = float(settings["inventory"].get("overstock_weeks", 26))
    rng = np.random.default_rng(seed)
    as_of = prep.as_of
    t = fc.table
    P = pol.table
    stock = tables.get("stock")
    st_idx = {}
    if stock is not None and len(stock):
        st = stock.copy()
        st["warehouse"] = st["warehouse"].astype(str)
        st_idx = st.drop_duplicates(["item_code", "warehouse"]).set_index(["item_code", "warehouse"]).to_dict("index")
    oo = tables.get("open_orders")
    oo_idx = {}
    if oo is not None and len(oo):
        o = oo.copy()
        o["warehouse"] = o["warehouse"].fillna(settings["data"].get("default_warehouse", "MIA")).astype(str)
        o["wk"] = np.where(o["due_date"].notna(), np.floor((o["due_date"] - as_of).dt.days / 7), 0)
        o["wk"] = o["wk"].fillna(0).clip(lower=0).astype(int)
        for (it, wh), g in o.groupby(["item_code", "warehouse"]):
            arr = np.zeros(W)
            for wk, q in zip(g["wk"], g["open_qty"]):
                if wk < W:
                    arr[wk] += q
            oo_idx[(it, wh)] = arr
    po_idx = {}
    open_pos = lt_model.open_pos if lt_model.open_pos is not None else pd.DataFrame()
    if len(open_pos):
        op = open_pos.copy()
        op["warehouse"] = op["warehouse"].fillna(settings["data"].get("default_warehouse", "MIA")).astype(str)
        for (it, wh), g in op.groupby(["item_code", "warehouse"]):
            po_idx[(it, wh)] = g
    sup_ot = {}
    if len(lt_model.scorecard):
        sup_ot = lt_model.scorecard.set_index("Supplier")["On time %"].to_dict()

    rows = []
    week_cols = [f"W{w + 1}" for w in range(W)]
    M = prep.matrix
    for i in range(len(t)):
        item, wh = t.item_code.iat[i], t.warehouse.iat[i]
        pr = P.iloc[i]
        cur = st_idx.get((item, wh), {})
        on_hand = max(float(cur.get("on_hand") or 0), 0.0)
        due = oo_idx.get((item, wh))
        committed_now = 0.0
        if due is None:
            committed_now = float(cur.get("committed") or 0)
            due = np.zeros(W)
        g = po_idx.get((item, wh))
        mean_w = _weekly_path(fc.path[i], prep.period_days, W)
        if mean_w.sum() <= 0 and on_hand <= 0 and due.sum() <= 0:
            continue
        cls = t.demand_class.iat[i]
        sp = fc.sim_params[i]
        # simulated weekly demand; open customer orders consume the forecast (no double count)
        if mean_w.sum() > 0:
            if prep.period_days == 7.0:
                d = simulate_periods(sp["sizes"], sp["occ"], mean_w, n_sims, float(fc.level_sigma.get(cls, 0.25)), rng)
            else:
                d = rng.poisson(np.tile(mean_w, (n_sims, 1)))
        else:
            d = np.zeros((n_sims, W))
        d = np.maximum(d, due[None, :])
        arr = np.zeros((n_sims, W))
        po_qty, late_pos, po_notes, next_arrival = 0.0, [], [], None
        if g is not None:
            for _, po in g.iterrows():
                q = float(po["open_qty"])
                if q <= 0:
                    continue
                po_qty += q
                p, late = lt_model.arrival_probabilities(po, as_of, W, grace)
                tail = max(1 - p.sum(), 0)
                probs = np.append(p, tail)
                wk = rng.choice(W + 1, n_sims, p=probs / probs.sum())
                hit = wk < W
                arr[np.nonzero(hit)[0], wk[hit]] += q
                med_wk = int(np.searchsorted(np.cumsum(probs), 0.5))
                eta = as_of + pd.Timedelta(days=7 * med_wk + 3) if med_wk < W else None
                if eta is not None and (next_arrival is None or eta < next_arrival):
                    next_arrival = eta
                if late:
                    late_pos.append((str(po["po_entry"]), (as_of - po["promised_date"]).days, q))
                po_notes.append((str(po["po_entry"]), q, med_wk, p.sum()))
        elif float(cur.get("on_order") or 0) > 0:  # on order but no PO lines supplied
            q = float(cur.get("on_order"))
            po_qty = q
            span = max(int(np.ceil(pr["lead_time_median_days"] / 7)), 1)
            wk = rng.integers(0, span, n_sims)
            hit = wk < W
            arr[np.nonzero(hit)[0], wk[hit]] += q
        inv = on_hand - committed_now + np.cumsum(arr - d, axis=1)
        p_out = (inv < 0).mean(axis=0)
        short = np.maximum(-inv.min(axis=1), 0)
        exp_short = float(short.mean())
        price, cost = float(pr["sell_price"]), float(pr["unit_cost"])
        med_end = float(np.median(inv[:, -1]))
        weekly = mean_w.mean()
        cover_end = med_end / weekly if weekly > 0 else (np.inf if med_end > 0 else 0)
        rop, upto = float(pr["reorder_point"]), float(pr["order_up_to"])
        ip_now = on_hand - committed_now - due.sum() + po_qty
        first = next((w for w in range(W) if p_out[w] >= thr), None)
        causes = []
        risk = None
        if first is not None:
            risk = "Stockout risk"
            # root causes, strongest evidence first
            for po_e, days, q in late_pos:
                causes.append(f"PO {po_e} for {q:.0f} units is {days} days past its promised date")
            slow = [n for n in po_notes if n[2] > first]
            if slow and not late_pos:
                causes.append(f"Next PO ({slow[0][0]}, {slow[0][1]:.0f} units) most likely lands in week {slow[0][2] + 1}, after the risk starts in week {first + 1}")
            recent = M[i, -4:].sum()
            prior = fc.path[i][:4].sum()
            if prior > 0 and recent > 2 * prior and recent >= 3:
                causes.append(f"Demand spike: {recent:.0f} units sold in the last 4 periods vs {prior:.1f} forecast")
            if due.sum() > 0 and due.sum() >= on_hand:
                causes.append(f"Open customer orders of {due.sum():.0f} units exceed the {on_hand:.0f} units on hand")
            elif due.sum() > 0 and due.sum() >= on_hand * 0.5:
                causes.append(f"Open customer orders of {due.sum():.0f} units take {due.sum() / on_hand:.0%} of the {on_hand:.0f} on hand")
            if po_qty <= 0 and ip_now <= rop:
                causes.append(f"No open PO and stock position {ip_now:.0f} is at or below the reorder point {rop:.0f}")
            elif ip_now <= rop:
                causes.append(f"Stock position {ip_now:.0f} including open POs is below the reorder point {rop:.0f}")
            ot = sup_ot.get(pr["supplier"])
            if ot is not None and np.isfinite(ot) and ot < 0.7:
                causes.append(f"Supplier {pr['supplier']} delivers on time only {ot:.0%} of the time")
            if not causes:
                causes.append("Normal demand variation against a thin stock position")
            action = ("Expedite the late PO or transfer stock from the other warehouse" if late_pos else
                      ("Place a PO now (see Purchase Orders)" if po_qty <= 0 else "Check with the supplier on the open PO, or order a top-up from a faster supplier"))
        elif weekly > 0 and cover_end > over_w and (med_end - upto) * cost > 250:
            risk = "Overstock"
            excess = (med_end - upto) * cost
            last13 = M[i, -13:].mean()
            prev = M[i, -52:-13].mean() if M.shape[1] >= 52 else last13
            if prev > 0 and last13 < 0.5 * prev:
                causes.append(f"Demand dropped: {last13:.1f} per period in the last 13 vs {prev:.1f} before")
            if po_qty > 0:
                causes.append(f"{po_qty:.0f} units still on open POs")
            if float(pr["moq"]) > 13 * weekly:
                causes.append(f"Minimum order of {pr['moq']:.0f} is more than 13 weeks of demand")
            if not causes:
                causes.append(f"Stock covers about {cover_end:.0f} weeks against a {over_w:.0f} week limit")
            action = "Pause or cut open POs, move stock to the other warehouse, or offer it to customers who buy this line"
        elif weekly <= 0 and on_hand > 0 and on_hand * cost > 100:
            risk = "Dead stock"
            causes.append(f"No demand expected; {on_hand:.0f} units worth ${on_hand * cost:,.0f} on hand")
            action = "Review for markdown, return to supplier, or export channel"
        if risk is None:
            continue
        p_max = float(p_out.max())
        rev_risk = exp_short * price
        excess_val = max((med_end - upto) * cost, 0) if risk == "Overstock" else (on_hand * cost if risk == "Dead stock" else 0.0)
        if risk == "Stockout risk":
            sev = "Critical" if (first <= 1 and p_max >= 0.6) or rev_risk > 5000 else ("High" if p_max >= 0.5 or rev_risk > 1000 else "Medium")
        else:
            sev = "High" if excess_val > 5000 else "Medium"
        row = {"item_code": item, "warehouse": wh, "risk": risk, "severity": sev,
               "first_risk_week": (first + 1) if first is not None else None, "max_stockout_probability": p_max,
               "expected_units_short": exp_short, "revenue_at_risk": rev_risk, "excess_value": excess_val,
               "on_hand": on_hand, "open_customer_orders": float(due.sum() + committed_now), "open_po_qty": po_qty,
               "next_po_eta": next_arrival, "forecast_next_8_weeks": float(mean_w.sum()),
               "projected_end_stock_median": med_end, "reorder_point": rop, "root_causes": " | ".join(causes),
               "suggested_action": action, "abc": pr["abc"], "supplier": pr["supplier"]}
        for w, c in enumerate(week_cols):
            row[c] = float(p_out[w])
        rows.append(row)
    # stock that never sold in the whole history (not in the forecast at all)
    if stock is not None and len(stock):
        seen = set(zip(t.item_code, t.warehouse))
        cost_of = prep.items.set_index("item_code")["unit_cost"].to_dict()
        for (it, wh), cur in st_idx.items():
            if (it, wh) in seen:
                continue
            oh = float(cur.get("on_hand") or 0)
            val = oh * float(cost_of.get(it, 0) or 0)
            if oh > 0 and val > 100:
                row = {"item_code": it, "warehouse": wh, "risk": "Dead stock", "severity": "High" if val > 5000 else "Medium",
                       "first_risk_week": None, "max_stockout_probability": 0.0, "expected_units_short": 0.0,
                       "revenue_at_risk": 0.0, "excess_value": val, "on_hand": oh, "open_customer_orders": float(cur.get("committed") or 0),
                       "open_po_qty": float(cur.get("on_order") or 0), "next_po_eta": None, "forecast_next_8_weeks": 0.0,
                       "projected_end_stock_median": oh, "reorder_point": 0.0,
                       "root_causes": f"No sale at this warehouse in the whole history loaded; {oh:.0f} units worth ${val:,.0f} on hand",
                       "suggested_action": "Review for markdown, return to supplier, transfer, or export channel",
                       "abc": "C", "supplier": None}
                for c in week_cols:
                    row[c] = 0.0
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        sev_rank = {"Critical": 0, "High": 1, "Medium": 2}
        out["_s"] = out["severity"].map(sev_rank)
        out["_v"] = out["revenue_at_risk"] + out["excess_value"]
        out = out.sort_values(["_s", "_v"], ascending=[True, False]).drop(columns=["_s", "_v"]).reset_index(drop=True)
    return out
