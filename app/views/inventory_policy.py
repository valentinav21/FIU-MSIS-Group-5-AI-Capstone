import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import (REPLAY_SAME, sap_notice, sap_ok, stock_kpis, show_df, AQUA, BLUE, ORANGE, base_layout, download_csv, kpi_row, money, need, page_setup, pct,  # noqa: E402
                    require_results, show)

page_setup("Stock levels: safety stock and reorder points")
st.markdown('<p class="fp-note">For every item and warehouse the planner tests reorder points by running two simulated years: '
            "random demand drawn from the calibrated forecast and random lead times drawn from that supplier's real receipts. "
            "It keeps the lowest level that meets the service target. Settings controls the targets and the budget mode.</p>",
            unsafe_allow_html=True)
r = require_results()
P = r.policy
if not need(P, "stock levels"):
    st.stop()
ps = r.policy_summary
tab1, tab2, tab3 = st.tabs(["Service versus inventory", "Proof on last year's demand", "Item settings"])

with tab1:
    same = r.budget.get("same_inventory", {})
    ok = sap_ok(r)
    sap_notice(r)
    kpi_row(stock_kpis(r))
    c = r.policy_curve
    if len(c) and "avg_inventory_value" in c:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c["avg_inventory_value"], y=c["revenue_weighted_fill_rate"], mode="lines+markers", name="Same target for every item",
                                 line=dict(color=BLUE, width=2), marker=dict(size=8),
                                 customdata=c["service_target"], hovertemplate="Target %{customdata:.1%}<br>Inventory $%{x:,.0f}<br>Fill %{y:.1%}<extra></extra>"))
        pts = []
        if ok and np.isfinite(ps.get("current_avg_inventory_value", np.nan)):
            pts.append(("Today's SAP settings", ps["current_avg_inventory_value"], ps["current_fill_rate"], ORANGE))
        if same:
            pts.append(("AI, today's on-hand $", same.get("spent", same["budget"]), same["fill_rate"], AQUA))
        pts.append(("AI at your targets", ps["avg_inventory_value"], ps["fill_rate"], "#6250d6"))
        for name, x, y, col in pts:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=name,
                                     marker=dict(size=13, color=col, line=dict(color="#fcfcfb", width=2)),
                                     hovertemplate=f"{name}<br>Inventory $%{{x:,.0f}}<br>Fill %{{y:.1%}}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(tickprefix="$", tickformat="~s")
        show(base_layout(fig, "Fill rate you get for each inventory dollar", height=440, x_title="Average inventory value", y_title="Fill rate (revenue weighted)"))
        if same and same.get("spent", same["budget"]) < 0.98 * same["budget"]:
            st.caption(f"Today's stock on hand is {money(same['budget'])} at cost on the planned items. Even at the highest "
                       f"service step the planner would hold about {money(same['spent'])}, so the aqua point sits there.")
        st.caption("Points above the blue line beat a one-size-fits-all target: the planner gives more safety stock to items "
                   "where a dollar protects the most revenue, and less where it protects little.")

with tab2:
    rp = r.replay or {}
    if rp.get("table") is None:
        st.info(rp.get("note", "Replay needs at least 65 weeks of sales history."))
    else:
        t = rp["table"]
        simple = rp.get("basis") == "simple" or "Today's SAP min / max" not in t.columns
        _el = rp.get("eligible", rp["series"])
        _smp = (f" (a sample of the {_el:,} that sold in that time, with every A item included)" if _el > rp["series"] else "")
        st.markdown(f"The last **{rp['weeks']} weeks of real demand** for **{rp['series']:,} items x warehouses**{_smp} were replayed "
                    "through each policy, with the same lead time draws, minimum orders and pack sizes for all of them. "
                    "Unmet demand counts as lost.")
        if simple:
            st.caption("SAP min / max are not set for most items, so the AI settings are compared with a simple rule a buyer could set "
                       "by hand: reorder when stock covers the average demand over the lead time, and order one month of average demand. "
                       "No safety stock. Both use the same 12 months of history.")
        if REPLAY_SAME in t.columns and not ok:  # on-hand dollars of every item mean nothing on a small subset
            t = t.drop(columns=REPLAY_SAME)
        fmt = {"Fill rate (units)": pct, "Fill rate (revenue)": pct, "Lost revenue": money, "Average inventory value": money,
               "Purchase orders placed": lambda v: f"{v:,.0f}", "Item-weeks with a stockout": lambda v: f"{v:,.0f}"}
        disp = pd.DataFrame({col: [fmt[i](t.loc[i, col]) for i in t.index] for col in t.columns}, index=t.index)
        show_df(disp.reset_index().rename(columns={"index": "Measure"}))
        mc = rp.get("model_check")
        if mc is not None and len(mc):
            st.markdown("**Model check.** " + ("The simulator predicted the fill rate of the AI settings, and the replay measures what "
                        "they deliver on last year's real demand." if simple else "Before trusting new settings, the simulator predicts "
                        "the fill rate of today's SAP settings, and the replay measures what those settings really delivered.")
                        + " Close numbers mean the simulator can be trusted.")
            show_df(mc, hide_index=True, width="stretch",
                    column_config={c_: st.column_config.NumberColumn(format="percent") for c_ in mc.columns if "fill" in c_.lower()})
        if len(r.kappa_check):
            with st.expander("How much forecast error the simulator adds"):
                kc = r.kappa_check.copy()
                if "Gap (points)" in kc:
                    kc["Gap (points)"] = kc["Gap (points)"].round(1)
                cfg_k = {c_: st.column_config.NumberColumn(format="percent") for c_ in kc.columns if "fill" in c_.lower()}
                cfg_k["Model error weight"] = st.column_config.NumberColumn(format="%.1f",
                                                                            help="0 = trust the forecast fully; 1 = add the full measured forecast error")
                cfg_k["Gap (points)"] = st.column_config.NumberColumn(format="%.1f", help="Predicted minus replayed fill, in percentage points")
                show_df(kc, hide_index=True, width="stretch", column_config=cfg_k)
                st.caption("The planner uses the weight with the smallest gap.")
        st.caption("The AI settings were computed with data that includes the replay year, so read this as an estimate. "
                   "The honest test is a forward pilot: run the planner on a group of items for 8 to 12 weeks and compare.")

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    wh = c1.multiselect("Warehouse", sorted(P["warehouse"].dropna().astype(str).unique()))
    abc = c2.multiselect("ABC", ["A", "B", "C"])
    cls = c3.multiselect("Demand pattern", sorted(P["demand_class"].dropna().astype(str).unique()))
    q = c4.text_input("Search item code")
    d = P[P["forecast_per_period"] > 0].copy()
    if wh:
        d = d[d["warehouse"].isin(wh)]
    if abc:
        d = d[d["abc"].isin(abc)]
    if cls:
        d = d[d["demand_class"].isin(cls)]
    if q:
        d = d[d["item_code"].str.contains(q, case=False, na=False)]
    full_cols = ["item_code", "warehouse", "abc", "demand_class", "supplier", "lead_time_median_days", "lead_time_p90_days",
                 "forecast_per_period", "service_target", "reorder_point", "safety_stock", "order_qty", "order_up_to",
                 "fill_rate_at_rop", "current_min", "current_max", "current_fill_rate", "safety_stock_value", "avg_inventory_value", "why"]
    if "demand_class" in d:
        d = d.assign(demand_class=d["demand_class"].str.capitalize())

    def _pair(a_, b_):
        return [("" if pd.isna(x) else f"{x:,.0f}") + (" / " + f"{y:,.0f}" if pd.notna(y) else "") for x, y in zip(a_, b_)]

    v = d.copy()
    v["lead_time"] = _pair(v["lead_time_median_days"], v["lead_time_p90_days"])
    v["sap_minmax"] = ["not set" if pd.isna(x) else f"{x:,.0f} / {y:,.0f}" for x, y in zip(v["current_min"], v["current_max"])]
    cols = ["item_code", "warehouse", "abc", "demand_class", "supplier", "lead_time", "forecast_per_period", "service_target",
            "reorder_point", "safety_stock", "order_qty", "order_up_to", "fill_rate_at_rop"]
    cols += ["sap_minmax", "current_fill_rate"] if ok else []
    cols += ["safety_stock_value", "avg_inventory_value", "why"]
    n0 = {"format": "%,.0f"}
    show_df(v[cols], hide_index=True, width="stretch", column_config={
        "item_code": st.column_config.TextColumn("Item"),
        "warehouse": st.column_config.TextColumn("Whs"),
        "abc": st.column_config.TextColumn("ABC"),
        "demand_class": st.column_config.TextColumn("Pattern", help="Demand pattern from how often and how evenly the item sells"),
        "lead_time": st.column_config.TextColumn("Lead time (days)", help="Median / slowest 10% of receipts, from real PO and receipt dates"),
        "forecast_per_period": st.column_config.NumberColumn("Forecast per week" if r.bucket == "W" else "Forecast per month", format="%,.1f"),
        "service_target": st.column_config.NumberColumn("Target", format="percent", help="Fill rate the level is set to reach"),
        "reorder_point": st.column_config.NumberColumn("Reorder point", help="Order when stock on hand plus on order drops to this (SAP MinStock)", **n0),
        "safety_stock": st.column_config.NumberColumn("Safety stock", help="Units held beyond expected demand over the lead time", **n0),
        "order_qty": st.column_config.NumberColumn("Order qty", help="Usual order size, after minimum order and pack size", **n0),
        "order_up_to": st.column_config.NumberColumn("Order up to", help="Reorder point plus order quantity (SAP MaxStock)", **n0),
        "fill_rate_at_rop": st.column_config.NumberColumn("Fill at new level", format="percent"),
        "sap_minmax": st.column_config.TextColumn("SAP min / max", help="Today's MinStock / MaxStock in SAP; 0 / 0 shows as not set"),
        "current_fill_rate": st.column_config.NumberColumn("Fill at SAP min", format="percent"),
        "safety_stock_value": st.column_config.NumberColumn("Safety stock $", format="$%,.0f"),
        "avg_inventory_value": st.column_config.NumberColumn("Avg inventory $", format="$%,.0f"),
        "why": st.column_config.TextColumn("Why")})
    if not ok:
        st.caption("SAP min / max columns are left out because they are not set for most items; the download still has them.")
    cols = full_cols
    download_csv(d[cols], "stock_levels.csv")
    st.caption("To load new levels into SAP, set MinStock to the reorder point and MaxStock to the order-up-to level "
               "(OITW, per warehouse), for example with a DTW update of Items > Warehouse info.")
