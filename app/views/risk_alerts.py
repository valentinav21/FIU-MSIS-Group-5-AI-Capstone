import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from common import Strip, show_df, BLUE, base_layout, download_csv, kpi_row, money, need, page_setup, require_results, show  # noqa: E402

page_setup("Risk alerts, 1 to 8 weeks")
st.markdown('<p class="fp-note">Each item and warehouse is projected week by week across hundreds of simulated futures: '
            'stock on hand, minus promised customer orders, minus simulated demand, plus open POs arriving on dates drawn '
            "from that supplier's real behaviour. Root causes come from the evidence behind each risk.</p>", unsafe_allow_html=True)
r = require_results()
if r.as_of is not None:
    st.caption(f"Week 1 starts on the data date, {r.as_of:%b %d, %Y}. Run the analysis on a fresh export to move the weeks forward.")
a = r.alerts
if need(a, "risk alerts"):
    so = a[a["risk"] == "Stockout risk"]
    kpi_row([("Stockout risks", f"{len(so):,}", f"{(so['severity'] == 'Critical').sum():,} critical", "Chance of running out in some week above the alert threshold."),
             ("Revenue at risk", money(so["revenue_at_risk"].sum()), None, "Expected units short times selling price, next 8 weeks."),
             ("Overstock", f"{(a['risk'] == 'Overstock').sum():,}", money(a.loc[a['risk'] == 'Overstock', 'excess_value'].sum()), "Stock above the order-up-to level with cover beyond the limit in Settings."),
             ("Dead stock", f"{(a['risk'] == 'Dead stock').sum():,}", money(a.loc[a['risk'] == 'Dead stock', 'excess_value'].sum()),
              f"On hand with no sale for {r.settings.get("forecast", {}).get("dead_stock_months", 24)} months or more (or never sold here)."),
             ("At-risk stock", f"{(a['risk'] == 'At-risk stock').sum():,}", money(a.loc[a['risk'] == 'At-risk stock', 'excess_value'].sum()),
              f"On hand with no sale for {r.settings.get('forecast', {}).get('at_risk_months', 12)} months or more, not yet dead.")])
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.multiselect("Risk type", sorted(a["risk"].dropna().astype(str).unique()), default=["Stockout risk"] if (a["risk"] == "Stockout risk").any() else None)
    sev = c2.multiselect("Severity", ["Critical", "High", "Medium"])
    wh = c3.multiselect("Warehouse", sorted(a["warehouse"].dropna().astype(str).unique()))
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
    # the explanation comes right after the item, so it can be read without scrolling sideways
    # compact columns so the whole table fits: one dollar column (revenue at risk for stockouts, stock value for
    # overstock, dead and at-risk stock) and the eight weekly chances in one column
    d = d.copy()
    d["value_at_stake"] = np.where(d["risk"] == "Stockout risk", d["revenue_at_risk"], d["excess_value"])
    if wk:
        # one shaded square per week (same colors as the heat map above); the download keeps W1 to W8 as numbers
        d["chance_by_week"] = [Strip(row) for row in d[wk].fillna(0).to_numpy().tolist()]
    cols = ["severity", "risk", "item_code", "warehouse", "root_causes", "suggested_action", "value_at_stake", "max_stockout_probability",
            "first_risk_week", "chance_by_week", "on_hand", "open_customer_orders", "open_po_qty",
            "forecast_next_8_weeks", "supplier"]  # the next PO's timing is in Root causes; the date is in the download
    cols = [c for c in cols if c in d.columns]
    cfg = {"max_stockout_probability": st.column_config.ProgressColumn("Max chance", format="percent", min_value=0, max_value=1),
           "first_risk_week": st.column_config.NumberColumn("First risk week", format="%d"),
           "open_customer_orders": st.column_config.NumberColumn("Customer orders", format="%,.0f"),
           "open_po_qty": st.column_config.NumberColumn("On order", format="%,.0f"),
           "on_hand": st.column_config.NumberColumn("On hand", format="%,.0f"),
           "warehouse": st.column_config.TextColumn("Whs"),
           "value_at_stake": st.column_config.NumberColumn("$ at stake", format="$%,.0f",
                                                           help="Revenue at risk for stockouts; stock value for overstock, dead and at-risk stock"),
           "chance_by_week": st.column_config.TextColumn("Weekly chance of running out (%)",
                                                         help="One square per week: weeks 1 to 4 on the top row, 5 to 8 below. "
                                                              "Darker means more likely to run out. Hover a square to see its week."),
           "excess_value": st.column_config.NumberColumn("Excess value", format="$%,.0f"),
           "root_causes": st.column_config.TextColumn("Root causes", width="large"),
           "suggested_action": st.column_config.TextColumn("Suggested action", width="medium"),
           "next_po_eta": st.column_config.DateColumn("Next PO likely"),
           "forecast_next_8_weeks": st.column_config.NumberColumn("Forecast 8 weeks", format="%,.0f")}
    show_df(d[cols], hide_index=True, width="stretch", height=520, column_config=cfg, fit={"widths": (440, 240, 240)})
    full_cols = [c for c in ["severity", "risk", "item_code", "warehouse", "abc", "root_causes", "suggested_action", "revenue_at_risk",
                             "excess_value", "max_stockout_probability", "first_risk_week", "on_hand", "open_customer_orders",
                             "open_po_qty", "next_po_eta", "forecast_next_8_weeks", "supplier"] + wk if c in d.columns]
    download_csv(d[full_cols], "risk_alerts.csv")
