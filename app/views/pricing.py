import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, BLUE, base_layout, download_csv, kpi_row, money, need, page_setup, pct, require_results, show  # noqa: E402

page_setup("Pricing")
st.markdown('<p class="fp-note">Three checks. Price leakage: lines sold well below what similar customers pay for the same item. '
            'Discount discipline: whether bigger customers really get bigger discounts. Price sensitivity: how order size '
            'changes when the same customer pays a different price, measured only where the data supports it.</p>',
            unsafe_allow_html=True)
r = require_results()
S = r.sales
if S is None:
    st.stop()
ps = S.pricing_summary
if ps:
    kpi_row([("Lines below peer price floor", f"{ps.get('Lines below peer price floor', 0):,}", None,
              f"Below the {r.settings['sales']['price_floor_percentile']}th percentile of same-item prices paid by the same price tier, last 12 months."),
             ("Gap to peer median", money(ps.get("Gap to peer median on those lines")), pct(ps.get("Share of revenue")) + " of revenue", None),
             ("Small customers with top discounts", f"{ps.get('Small customers with top-quartile discounts', 0):,}", None,
              "Smallest 25% by revenue, but discounts in the top 25%.")])
tab1, tab2, tab3 = st.tabs(["Price leakage", "Discounts by customer size", "Price sensitivity"])
with tab1:
    if need(S.pricing_leakage, "price leakage"):
        show_df(S.pricing_leakage, hide_index=True, width="stretch", height=480,
                     column_config={"leakage": st.column_config.NumberColumn("Gap to peer median", format="$%,.0f"),
                                    "why": st.column_config.TextColumn("Why", width="large")})
        download_csv(S.pricing_leakage, "price_leakage.csv")
with tab2:
    d = S.discount_check
    if need(d, "discount analysis"):
        fig = go.Figure(go.Bar(x=d["size_band"].astype(str), y=d["avg_discount_pct"] / 100, marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{x}<br>Average discount %{y:.1%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        show(base_layout(fig, "Average line discount by customer size (revenue quartile)", height=320, legend=False))
        st.caption("A healthy pattern rises from left to right. Flat or falling bars mean discounts are not tied to volume.")
with tab3:
    el = S.elasticity
    if need(el, "price sensitivity"):
        show_df(el, hide_index=True, width="stretch",
                     column_config={"price_sensitivity": st.column_config.NumberColumn("Estimate", format="%.2f",
                                                                                       help="Percent change in order size for a 1% price change. -1 means volume moves one for one."),
                                    "low": st.column_config.NumberColumn("Low (95%)", format="%.2f"),
                                    "high": st.column_config.NumberColumn("High (95%)", format="%.2f"),
                                    "reading": st.column_config.TextColumn("Reading", width="large")})
        st.caption(f"The planner never suggests moving a price more than {r.settings['sales']['max_price_change_pct']}% at once. "
                   "Where the reading is unclear, the right next step is a controlled test: change price for one region or "
                   "customer group for 8 weeks and compare with a matched group.")
