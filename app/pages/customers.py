import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, BLUE, ORANGE, base_layout, download_csv, kpi_row, money, need, page_setup, pct, require_results, show  # noqa: E402

page_setup("Customers: segments, personas and churn risk")
r = require_results()
S = r.sales
if S is None or not need(S.customers, "customers"):
    st.stop()
c = S.customers
tab1, tab2, tab3, tab4 = st.tabs(["Segments", "Churn risk", "Buying personas", "Customer lookup"])
with tab1:
    seg = S.segments
    st.markdown('<p class="fp-note">Segments use recency, frequency and spend over the last 12 months (RFM), the standard way to '
                'decide who gets which kind of attention.</p>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=seg["revenue_12m"], y=seg["segment"], orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           customdata=np.stack([seg["customers"], seg["share_of_revenue"]], axis=1),
                           hovertemplate="%{y}<br>$%{x:,.0f} · %{customdata[0]} customers · %{customdata[1]:.0%} of revenue<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(base_layout(fig, "Revenue last 12 months by segment", height=340, legend=False))
    show_df(seg, hide_index=True, width="stretch",
                 column_config={"revenue_12m": st.column_config.NumberColumn("Revenue 12m", format="$%,.0f"),
                                "share_of_revenue": st.column_config.NumberColumn("Share", format="percent"),
                                "avg_recency_days": st.column_config.NumberColumn("Avg days since order", format="%,.0f"),
                                "avg_orders_12m": st.column_config.NumberColumn("Avg orders 12m", format="%,.1f"),
                                "what_it_means": st.column_config.TextColumn("What to do", width="large")})
with tab2:
    m = S.churn_metrics
    if "AUC (out of time)" in m:
        kpi_row([("Model accuracy (AUC)", f"{m['AUC (out of time)']:.2f}", f"recency alone {m['AUC of recency alone']:.2f}",
                  "Tested on a later period the model never saw. 0.5 is a coin flip, 1.0 is perfect."),
                 ("Churn rate, all customers", pct(m["Churn rate in test window"]), None, f"Share with no order in the next {m['Window days']} days."),
                 ("Churn rate, riskiest 10%", pct(m["Churn rate in top 10% riskiest"]), f"{m['Lift in top 10%']:.1f}x the average", "How well the model ranks."),
                 ("Revenue at risk", money(c["revenue_at_risk"].sum()) if "revenue_at_risk" in c else "n/a", None, "Churn chance times last 12 months revenue, summed.")])
        d = c[c.get("churn_risk", 0) >= 0.3].sort_values("revenue_at_risk", ascending=False) if "churn_risk" in c else c.head(0)
        cols = [x for x in ["customer_code", "segment", "customer_type", "state", "country", "sales_rep", "churn_risk", "revenue_12m",
                            "revenue_at_risk", "recency_days", "usual_gap_days", "trend_90d", "churn_reason"] if x in d.columns]
        show_df(d[cols], hide_index=True, width="stretch", height=480,
                     column_config={"churn_risk": st.column_config.ProgressColumn("Churn chance", format="percent", min_value=0, max_value=1),
                                    "revenue_12m": st.column_config.NumberColumn("Revenue 12m", format="$%,.0f"),
                                    "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%,.0f"),
                                    "trend_90d": st.column_config.NumberColumn("Last 90 days vs prior 90", format="percent"),
                                    "churn_reason": st.column_config.TextColumn("Why", width="large")})
        download_csv(d[cols], "churn_risk.csv")
    else:
        st.info(m.get("note", "Churn model did not run."))
with tab3:
    pe = S.personas
    if need(pe, "personas"):
        st.markdown('<p class="fp-note">Customers grouped by what they buy (their mix of product lines over two years). Use these '
                    'to tailor offers and catalogs.</p>', unsafe_allow_html=True)
        show_df(pe, hide_index=True, width="stretch",
                     column_config={"revenue_12m": st.column_config.NumberColumn("Revenue 12m", format="$%,.0f"),
                                    "top_categories": st.column_config.TextColumn("What they buy", width="large")})
        st.caption(f"Cluster separation score {pe.attrs.get('silhouette', float('nan')):.2f} (higher is clearer; above 0.25 is distinct).")
with tab4:
    code = st.selectbox("Customer", c.sort_values("revenue_12m", ascending=False)["customer_code"].tolist())
    x = c[c["customer_code"] == code].iloc[0]
    kpi_row([("Revenue 12m", money(x["revenue_12m"]), f"{x['growth_12m']:+.0%} vs prior" if np.isfinite(x.get("growth_12m", np.nan)) else None, None),
             ("Orders 12m", f"{x['orders_12m']:,.0f}", None, None),
             ("Segment", str(x.get("segment", "")), None, None),
             ("Churn chance", pct(x["churn_risk"]) if "churn_risk" in x and np.isfinite(x["churn_risk"]) else "n/a", None,
              x["churn_reason"] if "churn_reason" in x and isinstance(x["churn_reason"], str) else None)])
    st.caption(f"Type {x.get('customer_type', '')} · {x.get('state', '') or ''} {x.get('country', '')} · persona {x.get('persona', '')} · "
               f"top line {x.get('top_category', '')} · last order {x['recency_days']:.0f} days ago")
    rec = S.recommendations
    if len(rec):
        rr = rec[rec["customer_code"] == code]
        if len(rr):
            st.markdown("**Suggested items**")
            show_df(rr[["rank", "item_code", "item_name", "category", "typical_order_value", "why"]], hide_index=True, width="stretch",
                         column_config={"typical_order_value": st.column_config.NumberColumn(format="$%,.0f"),
                                        "why": st.column_config.TextColumn(width="large")})
    up = S.upsell
    if len(up):
        uu = up[up["customer_code"] == code]
        if len(uu):
            st.markdown("**Product lines they under-buy**")
            show_df(uu[["category", "their_spend_12m", "peer_median_spend_12m", "gap", "why"]], hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="$%,.0f") for k in ("their_spend_12m", "peer_median_spend_12m", "gap")})
