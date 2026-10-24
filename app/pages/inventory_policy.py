import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import (show_df, AQUA, BLUE, MUTED, ORANGE, base_layout, download_csv, kpi_row, money, need, page_setup, pct,  # noqa: E402
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
    kpi_row([
        ("Fill rate today (simulated)", pct(ps.get("current_fill_rate")), None, "Today's SAP MinStock and MaxStock, run through the same simulation."),
        ("Fill rate, same inventory $", pct(same.get("fill_rate")), f"{(same.get('fill_rate', np.nan) - ps.get('current_fill_rate', np.nan)) * 100:+.1f} pts" if same else None,
         "Today's average inventory dollars, placed item by item where they serve the most revenue."),
        ("Fill rate at your targets", pct(ps.get("fill_rate")), f"{(ps.get('fill_rate', np.nan) - ps.get('current_fill_rate', np.nan)) * 100:+.1f} pts", "Targets from Settings."),
        ("Inventory at your targets", money(ps.get("avg_inventory_value")), f"{money(ps.get('avg_inventory_value', 0) - ps.get('current_avg_inventory_value', 0))} vs today",
         "Average stock value expected at the targets.", "off"),
    ])
    c = r.policy_curve
    if len(c) and "avg_inventory_value" in c:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c["avg_inventory_value"], y=c["revenue_weighted_fill_rate"], mode="lines+markers", name="Same target for every item",
                                 line=dict(color=BLUE, width=2), marker=dict(size=8),
                                 customdata=c["service_target"], hovertemplate="Target %{customdata:.1%}<br>Inventory $%{x:,.0f}<br>Fill %{y:.1%}<extra></extra>"))
        pts = []
        if np.isfinite(ps.get("current_avg_inventory_value", np.nan)):
            pts.append(("Today's SAP settings", ps["current_avg_inventory_value"], ps["current_fill_rate"], ORANGE))
        if same:
            pts.append(("AI, same inventory $ as today", same["budget"], same["fill_rate"], AQUA))
        pts.append(("AI at your targets", ps["avg_inventory_value"], ps["fill_rate"], "#6250d6"))
        for name, x, y, col in pts:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=name,
                                     marker=dict(size=13, color=col, line=dict(color="#fcfcfb", width=2)),
                                     hovertemplate=f"{name}<br>Inventory $%{{x:,.0f}}<br>Fill %{{y:.1%}}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(tickprefix="$", tickformat="~s")
        show(base_layout(fig, "Fill rate you get for each inventory dollar", height=440, x_title="Average inventory value", y_title="Fill rate (revenue weighted)"))
        st.caption("Points above the blue line beat a one-size-fits-all target: the planner gives more safety stock to items "
                   "where a dollar protects the most revenue, and less where it protects little.")

with tab2:
    rp = r.replay or {}
    if rp.get("table") is None:
        st.info(rp.get("note", "Replay needs weekly buckets, SAP MinStock and MaxStock, and a year of history."))
    else:
        t = rp["table"]
        st.markdown(f"The last **{rp['weeks']} weeks of real demand** for **{rp['series']:,} items x warehouses** were replayed through each policy, "
                    "with the same lead time draws for all of them. Unmet demand counts as lost.")
        disp = t.copy()
        fmt = {"Fill rate (units)": pct, "Fill rate (revenue)": pct, "Lost revenue": money, "Average inventory value": money,
               "Purchase orders placed": lambda v: f"{v:,.0f}", "Item-weeks with a stockout": lambda v: f"{v:,.0f}"}
        disp = pd.DataFrame({col: [fmt[i](t.loc[i, col]) for i in t.index] for col in t.columns}, index=t.index)
        show_df(disp.reset_index().rename(columns={"index": "Measure"}))
        mc = rp.get("model_check")
        if mc is not None and len(mc):
            st.markdown("**Model check.** Before trusting new settings, the simulator predicts the fill rate of today's SAP settings, "
                        "and the replay measures what those settings really delivered. Close numbers mean the simulator can be trusted.")
            show_df(mc, hide_index=True, width="stretch",
                         column_config={c_: st.column_config.NumberColumn(format="percent") for c_ in mc.columns if "fill" in c_.lower()})
        if len(r.kappa_check):
            with st.expander("How much forecast error the simulator adds"):
                show_df(r.kappa_check, hide_index=True, width="stretch")
        st.caption("The AI settings were computed with data that includes the replay year, so read this as an estimate. "
                   "The honest test is a forward pilot: run the planner on a group of items for 8 to 12 weeks and compare.")

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    wh = c1.multiselect("Warehouse", sorted(P["warehouse"].unique()))
    abc = c2.multiselect("ABC", ["A", "B", "C"])
    cls = c3.multiselect("Demand pattern", sorted(P["demand_class"].unique()))
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
    cols = ["item_code", "warehouse", "abc", "demand_class", "supplier", "lead_time_median_days", "lead_time_p90_days",
            "forecast_per_period", "service_target", "reorder_point", "safety_stock", "order_qty", "order_up_to",
            "fill_rate_at_rop", "current_min", "current_max", "current_fill_rate", "safety_stock_value", "avg_inventory_value", "why"]
    show_df(d[cols], hide_index=True, width="stretch", height=480, column_config={
        "service_target": st.column_config.NumberColumn("Target", format="percent"),
        "fill_rate_at_rop": st.column_config.NumberColumn("Fill at new level", format="percent"),
        "current_fill_rate": st.column_config.NumberColumn("Fill at SAP min", format="percent"),
        "safety_stock_value": st.column_config.NumberColumn("Safety stock $", format="$%,.0f"),
        "avg_inventory_value": st.column_config.NumberColumn("Avg inventory $", format="$%,.0f"),
        "forecast_per_period": st.column_config.NumberColumn("Forecast per week" if r.bucket == "W" else "Forecast per month", format="%,.2f"),
        "lead_time_median_days": st.column_config.NumberColumn("Lead time median (days)", format="%,.0f"),
        "lead_time_p90_days": st.column_config.NumberColumn("Lead time P90 (days)", format="%,.0f"),
        "why": st.column_config.TextColumn("Why", width="large")})
    download_csv(d[cols], "stock_levels.csv")
    st.caption("To load new levels into SAP, set MinStock to the reorder point and MaxStock to the order-up-to level "
               "(OITW, per warehouse), for example with a DTW update of Items > Warehouse info.")
