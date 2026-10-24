import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, BLUE, ORANGE, base_layout, download_csv, need, page_setup, require_results, show  # noqa: E402

page_setup("Products and web demand")
r = require_results()
S = r.sales
if S is None:
    st.stop()
tab1, tab2, tab3 = st.tabs(["Product lines", "Unmet web demand", "Item trends"])
with tab1:
    c = S.categories
    if need(c, "product lines"):
        d = c.sort_values("revenue_12m")
        fig = go.Figure(go.Bar(x=d["revenue_12m"], y=d["category"], orientation="h", marker=dict(color=BLUE, cornerradius=4),
                               customdata=np.stack([d["growth"].fillna(0), d["margin_pct"].fillna(0)], axis=1),
                               hovertemplate="%{y}<br>$%{x:,.0f}<br>Growth %{customdata[0]:+.0%}<br>Margin %{customdata[1]:.0%}<extra></extra>"))
        show(base_layout(fig, "Revenue last 12 months by product line", height=max(320, 24 * len(d)), legend=False))
        show_df(c, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="$%,.0f") for k in ("revenue_12m", "revenue_prior_12m", "margin_12m")}
                     | {k: st.column_config.NumberColumn(format="percent") for k in ("growth", "margin_pct", "share")})
with tab2:
    u = S.unmet_demand
    st.markdown('<p class="fp-note">Searches on the web store and app that returned nothing. This is demand SAP never sees: '
                'buyers looked for a part, did not find it and bought elsewhere. Needs the FocusPoint search log.</p>', unsafe_allow_html=True)
    if need(u, "unmet web demand"):
        show_df(u, hide_index=True, width="stretch", height=380,
                     column_config={"reading": st.column_config.TextColumn("Reading", width="large"),
                                    "last_seen": st.column_config.DateColumn("Last searched")})
        download_csv(u, "unmet_web_demand.csv")
    sv = S.search_vs_stock
    if sv is not None and len(sv):
        st.markdown("**Searched, but out of stock everywhere**")
        show_df(sv, hide_index=True, width="stretch", height=300)
with tab3:
    p = S.products
    if need(p, "item trends"):
        trend = st.multiselect("Trend", ["Growing", "Stable", "Declining", "New"], default=["Growing", "Declining"])
        d = p[p["trend"].isin(trend)] if trend else p
        show_df(d[["item_code", "item_name", "category", "abc", "trend", "revenue_12m", "revenue_prior_12m", "growth",
                        "margin_pct", "customers_12m", "units_12m"]], hide_index=True, width="stretch", height=480,
                     column_config={"revenue_12m": st.column_config.NumberColumn(format="$%,.0f"),
                                    "revenue_prior_12m": st.column_config.NumberColumn(format="$%,.0f"),
                                    "growth": st.column_config.NumberColumn(format="percent"),
                                    "margin_pct": st.column_config.NumberColumn(format="percent")})
        download_csv(d, "item_trends.csv")
