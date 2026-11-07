import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from common import show_df, download_csv, kpi_row, money, need, page_setup, pct, require_results  # noqa: E402

page_setup("Cross-sell and upsell")
r = require_results()
S = r.sales
if S is None:
    st.stop()
tab1, tab2, tab3 = st.tabs(["Item suggestions per customer", "Kits and bundles", "Wallet share gaps"])
with tab1:
    m = S.rec_metrics
    if m:
        k = [x for x in m if x.startswith("Hit rate @")][0]
        b = [x for x in m if x.startswith("Popularity baseline")][0]
        kpi_row([("Method chosen", m.get("Method short", m["Method chosen"][:20]), None, m["Method chosen"]),
                 ("Hit rate, top 10 suggestions", pct(m[k]), f"{m['Lift vs popularity']:.2f}x best-seller list",
                  f"Share of customers who bought at least one suggested item they had never bought, in {m['Test window']}. Suggestions were made with data before that window."),
                 ("Best-seller list hit rate", pct(m[b]), None, "Baseline: suggest the most popular items the customer does not buy yet."),
                 ("Customers tested", f"{m['Customers tested']:,}", None, None)])
        with st.expander("All methods tested"):
            st.write({kk: f"{v:.1%}" for kk, v in m.get("All methods (test hit rate)", {}).items()})
    rec = S.recommendations
    if need(rec, "cross-sell suggestions"):
        c1, c2 = st.columns(2)
        cust = c1.text_input("Customer code")
        cat = c2.multiselect("Category", sorted(rec["category"].dropna().unique()))
        d = rec
        if cust:
            d = d[d["customer_code"].str.contains(cust, case=False, na=False)]
        if cat:
            d = d[d["category"].isin(cat)]
        show_df(d[["customer_code", "rank", "item_code", "item_name", "category", "typical_order_value", "expected_value", "why"]].head(5000),
                     hide_index=True, width="stretch", height=520,
                     column_config={"typical_order_value": st.column_config.NumberColumn("Typical order", format="$%,.0f"),
                                    "expected_value": st.column_config.NumberColumn("Expected value", format="$%,.0f",
                                                                                    help="Typical order times the tested chance a single suggestion converts."),
                                    "why": st.column_config.TextColumn("Why", width="large")})
        download_csv(d, "cross_sell.csv")
        st.caption("Items that are out of stock in every warehouse are left out so reps do not sell what cannot ship.")
with tab2:
    bnd = S.bundles
    if need(bnd, "bundles"):
        st.markdown('<p class="fp-note">Item pairs bought by the same customers far more often than chance (lift). Good candidates for '
                    'kits, "frequently bought together" on the web store, and counter displays.</p>', unsafe_allow_html=True)
        show_df(bnd[["a", "a_name", "b", "b_name", "shared_customers", "confidence", "lift"]], hide_index=True, width="stretch", height=520,
                     column_config={"confidence": st.column_config.NumberColumn("Buyers of A who buy B", format="percent"),
                                    "lift": st.column_config.NumberColumn("Lift", format="%.1fx"),
                                    "a_name": st.column_config.TextColumn("Item A", width="medium"),
                                    "b_name": st.column_config.TextColumn("Item B", width="medium")})
        download_csv(bnd, "bundles.csv")
with tab3:
    up = S.upsell
    if need(up, "wallet share gaps"):
        st.markdown('<p class="fp-note">Product lines a customer buys little or none of, while most similar customers (same type) buy '
                    'them steadily. Gap = what a typical peer spends minus what this customer spends, per year.</p>', unsafe_allow_html=True)
        kpi_row([("Customers with gaps", f"{up['customer_code'].nunique():,}", None, None),
                 ("Total gap", money(up["gap"].sum()), None, "Upper bound; a realistic capture is a fraction."),
                 ("Top line", up.groupby("category")["gap"].sum().idxmax(), None, None)])
        show_df(up, hide_index=True, width="stretch", height=480,
                     column_config={k: st.column_config.NumberColumn(format="$%,.0f") for k in ("their_spend_12m", "peer_median_spend_12m", "gap")}
                     | {"peers_buying_share": st.column_config.NumberColumn("Peers buying", format="percent"),
                        "why": st.column_config.TextColumn("Why", width="large")})
        download_csv(up, "upsell_gaps.csv")
