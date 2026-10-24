import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, BLUE, ORANGE, base_layout, download_csv, need, page_setup, require_results, show  # noqa: E402

page_setup("Supplier scorecards and lead times")
st.markdown('<p class="fp-note">Lead times are learned from real PO and receipt dates, not the static setting in SAP. '
            'The score weighs on-time delivery 35%, fill rate 25%, lead time consistency 20% and price against other '
            'suppliers of the same items 20%.</p>', unsafe_allow_html=True)
r = require_results()
sc = r.supplier_scorecard
if need(sc, "supplier scorecards"):
    d = sc.sort_values("Spend last 12 months", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Median lead time (days)"], y=d["Supplier"], mode="markers", name="Median",
                             marker=dict(color=BLUE, size=10),
                             error_x=dict(type="data", symmetric=False, array=(d["P90 lead time (days)"] - d["Median lead time (days)"]).clip(lower=0),
                                          arrayminus=np.zeros(len(d)), color=BLUE, thickness=2, width=0),
                             customdata=np.stack([d["P90 lead time (days)"], d["On time %"]], axis=1),
                             hovertemplate="%{y}<br>Median %{x:.0f} days<br>1 in 10 orders: %{customdata[0]:.0f}+ days<br>On time %{customdata[1]:.0%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(base_layout(fig, "Lead time: median, with a line to the slowest 10% of orders", height=max(320, 26 * len(d)),
                     x_title="Days from PO to receipt", legend=False))
    cfg = {c: st.column_config.NumberColumn(format="percent") for c in ("On time %", "Fill rate %", "Price change vs prior year")}
    cfg["Spend last 12 months"] = st.column_config.NumberColumn(format="$%,.0f")
    cfg["Lead time variability (CV)"] = st.column_config.NumberColumn(format="%.2f")
    cfg["Price vs cheapest option"] = st.column_config.NumberColumn(format="%.2fx")
    cfg["Score (0-100)"] = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")
    show_df(d, hide_index=True, width="stretch", column_config=cfg)
    download_csv(d, "supplier_scorecard.csv")
    st.caption("Price vs cheapest option compares each supplier's median price with the cheapest supplier for the same item in the "
               "last 12 months. 1.00x means cheapest or sole source. Price-to-performance: a cheap supplier with long, erratic lead "
               "times needs more safety stock, which the purchase order page counts in its supplier choice.")
