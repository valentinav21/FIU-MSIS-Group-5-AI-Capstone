import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, BLUE, base_layout, download_csv, kpi_row, money, need, page_setup, require_results, show  # noqa: E402

page_setup("Risk alerts, 1 to 8 weeks")
st.markdown('<p class="fp-note">Each item and warehouse is projected week by week across hundreds of simulated futures: '
            'stock on hand, minus promised customer orders, minus simulated demand, plus open POs arriving on dates drawn '
            "from that supplier's real behaviour. Root causes come from the evidence behind each risk.</p>", unsafe_allow_html=True)
r = require_results()
a = r.alerts
if need(a, "risk alerts"):
    so = a[a["risk"] == "Stockout risk"]
    kpi_row([("Stockout risks", f"{len(so):,}", f"{(so['severity'] == 'Critical').sum():,} critical", "Chance of running out in some week above the alert threshold."),
             ("Revenue at risk", money(so["revenue_at_risk"].sum()), None, "Expected units short times selling price, next 8 weeks."),
             ("Overstock", f"{(a['risk'] == 'Overstock').sum():,}", money(a.loc[a['risk'] == 'Overstock', 'excess_value'].sum()), "Stock above the order-up-to level with cover beyond the limit in Settings."),
             ("Dead stock", f"{(a['risk'] == 'Dead stock').sum():,}", money(a.loc[a['risk'] == 'Dead stock', 'excess_value'].sum()), "On hand with no demand expected.")])
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.multiselect("Risk type", sorted(a["risk"].unique()), default=["Stockout risk"])
    sev = c2.multiselect("Severity", ["Critical", "High", "Medium"])
    wh = c3.multiselect("Warehouse", sorted(a["warehouse"].unique()))
    q = c4.text_input("Search item")
    d = a.copy()
    if kind:
        d = d[d["risk"].isin(kind)]
    if sev:
        d = d[d["severity"].isin(sev)]
    if wh:
        d = d[d["warehouse"].isin(wh)]
    if q:
        d = d[d["item_code"].str.contains(q, case=False, na=False)]
    wk = [f"W{i}" for i in range(1, 9) if f"W{i}" in d.columns]
    if len(d) and "Stockout risk" in set(d["risk"]):
        top = d[d["risk"] == "Stockout risk"].head(25)
        z = top[wk].to_numpy()
        fig = go.Figure(go.Heatmap(z=z, x=[w.replace("W", "Week ") for w in wk], y=top["item_code"] + " · " + top["warehouse"],
                                   colorscale=[[0, "#f0efec"], [0.3, "#fab219"], [0.6, "#ec835a"], [1, "#d03b3b"]], zmin=0, zmax=1,
                                   hovertemplate="%{y}<br>%{x}: %{z:.0%} chance of running out<extra></extra>",
                                   colorbar=dict(title="Chance", tickformat=".0%")))
        fig.update_yaxes(autorange="reversed")
        show(base_layout(fig, "Chance of running out by week, 25 most urgent", height=520, legend=False))
    cols = ["severity", "risk", "item_code", "warehouse", "abc", "first_risk_week", "max_stockout_probability", "revenue_at_risk",
            "excess_value", "on_hand", "open_customer_orders", "open_po_qty", "next_po_eta", "forecast_next_8_weeks",
            "root_causes", "suggested_action", "supplier"] + wk
    cfg = {"max_stockout_probability": st.column_config.ProgressColumn("Max chance", format="percent", min_value=0, max_value=1),
           "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%,.0f"),
           "excess_value": st.column_config.NumberColumn("Excess value", format="$%,.0f"),
           "root_causes": st.column_config.TextColumn("Root causes", width="large"),
           "suggested_action": st.column_config.TextColumn("Suggested action", width="medium"),
           "next_po_eta": st.column_config.DateColumn("Next PO likely"),
           "forecast_next_8_weeks": st.column_config.NumberColumn("Forecast 8 weeks", format="%,.1f")}
    for w in wk:
        cfg[w] = st.column_config.NumberColumn(w.replace("W", "Wk "), format="percent")
    show_df(d[cols], hide_index=True, width="stretch", height=520, column_config=cfg)
    download_csv(d[cols], "risk_alerts.csv")
