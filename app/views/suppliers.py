import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, BLUE, ORANGE, base_layout, download_csv, need, page_setup, require_results, show  # noqa: E402

page_setup("Supplier scorecards and lead times")
r = require_results()
sc = r.supplier_scorecard
ot = "On time %" if "On time %" in sc else "Within usual lead time %"
st.markdown('<p class="fp-note">Lead times are learned from real PO and receipt dates, not the static setting in SAP. '
            f'The score weighs {"on-time delivery" if ot == "On time %" else "arriving within the usual lead time"} 35%, fill rate 25%, '
            'lead time consistency 20% and price against other suppliers of the same items 20%.</p>', unsafe_allow_html=True)
if ot != "On time %":
    st.info("PO promise dates are blank in this export, so on-time delivery cannot be measured. Reliability is the share of "
            "PO lines that arrived within 1.5 times the supplier's median lead time.")
if need(sc, "supplier scorecards"):
    d = sc.sort_values("Spend last 12 months", ascending=False)
    top = d.head(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=top["Median lead time (days)"], y=top["Supplier"], mode="markers", name="Median",
                             marker=dict(color=BLUE, size=10),
                             error_x=dict(type="data", symmetric=False, array=(top["P90 lead time (days)"] - top["Median lead time (days)"]).clip(lower=0),
                                          arrayminus=np.zeros(len(top)), color=BLUE, thickness=2, width=0),
                             customdata=np.stack([top["P90 lead time (days)"], top[ot]], axis=1),
                             hovertemplate="%{y}<br>Median %{x:.0f} days<br>1 in 10 orders: %{customdata[0]:.0f}+ days<br>"
                                           + ("On time" if ot == "On time %" else "Within usual lead time") + " %{customdata[1]:.0%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category")
    show(base_layout(fig, f"Lead time of the {len(top)} suppliers with the most spend: median, with a line to the slowest 10% of orders",
                     height=max(320, 26 * len(top)),
                     x_title="Days from PO to receipt", legend=False))
    cfg = {c: st.column_config.NumberColumn(format="percent") for c in (ot, "Fill rate %", "Price change vs prior year")}
    cfg["Spend last 12 months"] = st.column_config.NumberColumn(format="$%,.0f")
    cfg["Lead time variability (CV)"] = st.column_config.NumberColumn(
        format="%.2f", help="Spread of lead times relative to the median. Under 0.3 is steady; near 1 means lead times swing widely.")
    cfg["Price vs cheapest option"] = st.column_config.NumberColumn(
        format="%.2fx", help="Median price against the cheapest supplier for the same items. 1.00x = cheapest or sole source.")
    for c in ("Median lead time (days)", "Average days late when late", "Average days beyond usual when late"):
        cfg[c] = st.column_config.NumberColumn(format="%.0f")
    cfg["P90 lead time (days)"] = st.column_config.NumberColumn(
        "Slowest 10% (days)", format="%.0f", help="9 in 10 PO lines arrived within this many days (90th percentile)")
    if ot in sc:
        cfg[ot] = st.column_config.NumberColumn(format="percent", help="On time against the PO promise date" if ot == "On time %" else
                                                "Share of PO lines that arrived within 1.5 times the supplier's median lead time")
    cfg["Fill rate %"] = st.column_config.NumberColumn(format="percent", help="Average share of each PO line's quantity that was received")
    cfg["Price change vs prior year"] = st.column_config.NumberColumn(format="percent", help="Median price change on the same items, last 12 months against the 12 months before")
    cfg["Score (0-100)"] = st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d", help="0 to 100")
    # one Supplier column: FORTPRO's export has no names (the name equals the code). If names are added later,
    # they show next to the code in the same column.
    view = d.copy()
    if "Name" in view:
        nm = view["Name"].astype(str).str.strip()
        has = nm.ne(view["Supplier"].astype(str).str.strip()) & nm.ne("") & nm.str.lower().ne("nan")
        view["Supplier"] = np.where(has, view["Supplier"].astype(str) + " · " + nm, view["Supplier"].astype(str))
        view = view.drop(columns="Name")
    show_df(view, hide_index=True, width="stretch", column_config=cfg)
    download_csv(d, "supplier_scorecard.csv")
    st.caption("Price vs cheapest option compares each supplier's median price with the cheapest supplier for the same item in the "
               "last 12 months. 1.00x means cheapest or sole source. Price-to-performance: a cheap supplier with long, erratic lead "
               "times needs more safety stock, which the purchase order page counts in its supplier choice.")
