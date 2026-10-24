import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, page_setup  # noqa: E402

page_setup("How it works")

st.markdown("""
### What the planner does

**Purchasing (problem statement 1)**

1. **Forecasts demand** per item and warehouse with methods built for intermittent spare parts demand (Croston, SBA, TSB,
   temporal aggregation, exponential smoothing, a seasonal variant and a gradient boosted model trained across all items).
   Each item gets the method that tested best on its own history; items with thin evidence use the best method for their demand pattern.
2. **Makes the ranges honest.** Demand is simulated with the item's own mix of zero weeks and order sizes. A calibration step,
   learned on one test window and checked on a later one, corrects the ranges so a 95% level really holds 95% of the time.
3. **Learns real lead times** from PO and goods receipt dates, per item and supplier, with the supplier's overall pattern as backup.
4. **Sets safety stock, reorder points and order-up-to levels** by simulating two years of weekly operation for each candidate level,
   with real supplier minimums and pack sizes. It keeps the lowest level that meets the target fill rate. It can also take a budget
   (for example today's inventory dollars) and place it item by item where it serves the most revenue.
5. **Warns 1 to 8 weeks ahead** about stockouts and overstock, with root causes: late POs, demand spikes, no PO placed, promised
   customer orders, unreliable suppliers, demand drops, large minimums.
6. **Recommends purchase orders** with quantity, timing and supplier, comparing landed cost, shortage risk before arrival and the
   safety stock each supplier's reliability needs. Exports drafts for SAP Business One (DTW or Service Layer) for buyer approval.
7. **Scores suppliers** on on-time delivery, fill rate, lead time consistency and price against alternatives.
8. **Proves itself** by checking its predictions for today's SAP settings against a replay of real history, then replaying last year
   through the new settings.

**Sales growth (problem statement 2)**

Customer segments (RFM), buying personas, a churn model tested on a later period, cross-sell suggestions (the recommender is chosen on
a validation window and reported on a later test window against a best-seller baseline), kit and bundle pairs, wallet share gaps
versus peer customers, geographic penetration against market potential, price leakage, discount discipline, price sensitivity
where the data supports it, product trends, and unmet demand from web searches with no result. A single ranked list of sales plays
puts it all in front of each rep.
""")

st.markdown("### How it compares with tools on the market")
NS = "Not stated"
cmp = pd.DataFrame([
    ["Automatic forecast method per item", "Yes", "Yes", "Yes", "No", "Yes, chosen on held-out weeks"],
    ["Probabilistic demand and lead time", NS, "Yes", "Yes (demand)", "No", "Yes, calibrated and checked"],
    ["Service vs inventory trade-off curve", NS, "Yes (economic)", "Yes", "No", "Yes"],
    ["Budget-constrained service allocation", NS, "Yes (economic)", "Yes", "No", "Yes, revenue per dollar"],
    ["Model check on your own history before go-live", NS, NS, NS, NS, "Yes, built in"],
    ["Supplier performance scoring", "Yes", NS, NS, "No", "Yes, incl. price vs alternatives"],
    ["Multi-supplier choice with shortage risk", NS, NS, NS, "No", "Yes, explained per line"],
    ["Churn, cross-sell, wallet share", "No", "No", "No", "Yes", "Yes, tested out of time"],
    ["Price optimization", "No", "No", "No", "Zilliant: Yes", "Leakage and sensitivity, with guardrails"],
    ["Unmet demand from web store searches", NS, NS, NS, NS, "Yes (FocusPoint log)"],
    ["Purchasing and sales in one tool", "No", "No", "No", "No", "Yes"],
    ["Runs on your own computer", "No (cloud)", "No (cloud)", NS, "No (cloud)", "Yes, data stays in house"],
    ["Pricing", "Quote", "Quote", "Quote", "Quote", "Free"],
], columns=["Capability", "Netstock", "Lokad", "ToolsGroup", "Proton.ai / Zilliant", "FORTPRO AI Planner"])
show_df(cmp, hide_index=True, width="stretch")
st.caption("Vendor columns summarize what each vendor's public pages state (read October 2026). 'Not stated' means the page did "
           "not say, not that the product lacks it. Commercial tools bring things this planner does not: vendor support, live "
           "two-way ERP integration, multi-echelon networks and years of production use.")

st.markdown("""
### Glossary
- **Fill rate**: share of units customers ordered that shipped from stock.
- **Reorder point (SAP MinStock)**: when stock on hand plus open POs minus promised orders falls to this level, order.
- **Order-up-to level (SAP MaxStock)**: order enough to bring the stock position back to this level.
- **Safety stock**: the part of the reorder point above average demand over the lead time.
- **WAPE**: weighted absolute percentage error of 8 period totals, weighted by revenue.
- **Demand patterns**: smooth (sells most weeks, steady sizes), erratic (most weeks, varied sizes), intermittent (many zero weeks,
  steady sizes), lumpy (many zero weeks, varied sizes).
- **AUC**: how well the churn model ranks customers; 0.5 is a coin flip, 1.0 is perfect.

### Limits to keep in mind
- Results are only as good as the export. Check Data health first.
- The policy replay uses settings computed with the replay year included. Confirm with a forward pilot on a group of items.
- Price sensitivity is shown only where prices varied enough for the same customer and item; otherwise run a price test.
- Draft POs are drafts. A buyer approves every order in SAP.
""")
