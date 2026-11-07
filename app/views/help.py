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

1. **Forecasts demand** per item and warehouse, by calendar month over a 9-month horizon, as SAP does. Demand is
   sales orders (agreed with FORTPRO), gross of returns. Planning covers A1 (Master, Florida) and H1 (Houston); A2
   (Trading) stays in the files but is left out, as in SAP MRP. The current, unfinished month is left out of the history.
2. **Picks methods by testing them.** Moving average, exponential smoothing, Croston, SBA, TSB, temporal aggregation (ADIDA),
   a combination, a trended yearly average and a gradient boosted model trained across all items are tested on five rolling
   3-month windows. Each demand pattern (smooth, erratic, intermittent, lumpy) gets the method that tested best, and a
   method only replaces the flat 12-month average when it also misses fewer units. Seasonal categories are adjusted for
   their proven season. Accuracy is then reported on a sixth, later window that no choice was made on.
3. **Makes the ranges honest.** Demand is simulated with the item's own mix of zero months and order sizes. A calibration step,
   learned on one test window and checked on a later one, corrects the ranges so a 95% level really holds 95% of the time.
4. **Learns real lead times** from PO and goods receipt dates (receipts joined to the PO line), per item and supplier, with the
   supplier's overall pattern as backup and 105 days (FORTPRO's observed median) when there is nothing else.
5. **Sets safety stock, reorder points and order-up-to levels** by simulating two years of weekly operation for each candidate level,
   with supplier minimums and pack sizes. It keeps the lowest level that meets the target fill rate. It can also take a budget
   (for example today's stock on hand, in dollars) and place it item by item where it serves the most revenue.
6. **Warns 1 to 8 weeks ahead** about stockouts, overstock, at-risk stock (no sale for 12 months) and dead stock (24 months),
   with root causes. Open POs older than 180 days and sales orders more than 180 days past their ship date are not counted.
7. **Recommends purchase orders** with quantity, timing and supplier, comparing landed cost, shortage risk before arrival and the
   safety stock each supplier's reliability needs, and flags suppliers outside the US below the \\$10,000 minimum order.
   Exports drafts for SAP Business One (DTW or Service Layer) for buyer approval.
8. **Scores suppliers** on reliability, fill rate, lead time consistency and price against alternatives. FORTPRO's PO lines
   have no real promise dates, so reliability means arriving within 1.5 times the supplier's usual (median) lead time.
9. **Proves itself** by checking its predictions for today's SAP settings against a weekly replay of real history, then replaying
   last year through the new settings. This needs SAP MinStock / MaxStock; where they are 0 / 0 (not set), the planner
   compares against today's stock on hand instead and says so. `tools/backtest_export.py` writes the Jan 31, 2026 backtest for the Fabric scorecard.

**Assumptions, not FORTPRO data:** holding cost 25% a year, \\$25 per PO line, stockout cost 30% of price, the service targets
by ABC class, the \\$10,000 international minimum and the 60%-of-price cost for items with no cost. Change them in Settings.

**Sales growth (problem statement 2): TBD.** These pages are turned off (set `features: bp2_enabled: true` in
config/settings.yaml to turn them back on).
""")

st.markdown("### How it compares with tools on the market")
NS = "Not stated"
cmp = pd.DataFrame([
    ["Automatic forecast method selection", "Yes", "Yes", "Yes", "No", "Yes, chosen on held-out months, must beat the baseline"],
    ["Probabilistic demand and lead time", NS, "Yes", "Yes (demand)", "No", "Yes, calibrated and checked"],
    ["Service vs inventory trade-off curve", NS, "Yes (economic)", "Yes", "No", "Yes"],
    ["Budget-constrained service allocation", NS, "Yes (economic)", "Yes", "No", "Yes, revenue per dollar"],
    ["Model check on your own history before go-live", NS, NS, NS, NS, "Yes, built in"],
    ["Supplier performance scoring", "Yes", NS, NS, "No", "Yes, incl. price vs alternatives"],
    ["Multi-supplier choice with shortage risk", NS, NS, NS, "No", "Yes, explained per line"],
    ["Churn, cross-sell, wallet share", "No", "No", "No", "Yes", "TBD (sales growth, problem statement 2)"],
    ["Price optimization", "No", "No", "No", "Zilliant: Yes", "TBD (sales growth, problem statement 2)"],
    ["Runs on your own computer", "No (cloud)", "No (cloud)", NS, "No (cloud)", "Yes, data stays in house"],
    ["Pricing", "Quote", "Quote", "Quote", "Quote", "No license fee (built by the FIU team)"],
], columns=["Capability", "Netstock", "Lokad", "ToolsGroup", "Proton.ai / Zilliant", "FORTPRO AI Planner"])
show_df(cmp, column_config={"FORTPRO AI Planner": st.column_config.TextColumn(width="large"), "Capability": st.column_config.TextColumn(width="medium")})
st.caption("Vendor columns summarize what each vendor's public pages state (read October 2026). 'Not stated' means the page did "
           "not say, not that the product lacks it. Commercial tools bring things this planner does not: vendor support, live "
           "two-way ERP integration, multi-echelon networks and years of production use.")

st.markdown("""
### Glossary
- **Fill rate**: share of units customers ordered that shipped from stock.
- **Reorder point (SAP MinStock)**: when stock on hand plus open POs minus promised orders falls to this level, order.
- **Order-up-to level (SAP MaxStock)**: order enough to bring the stock position back to this level.
- **Safety stock**: the part of the reorder point above average demand over the lead time.
- **Units error (item x month)**: sum of |actual - forecast| over every item and month, divided by actual units. The team
  scores SAP this way: SAP's own forecast 1.05, a flat 12-month average 0.86. Lower is better.
- **WAPE**: the same idea on 3-month totals, weighted by revenue.
- **Demand patterns**: smooth (sells most months, steady sizes), erratic (most months, varied sizes), intermittent (many zero
  months, steady sizes), lumpy (many zero months, varied sizes).
- **At-risk / dead stock**: on hand with no sale for 12 / 24 months.

### Limits to keep in mind
- Results are only as good as the export. Check Data health first.
- The policy replay uses settings computed with the replay year included. Confirm with a forward pilot on a group of items.
- Stock is a single snapshot from the export, so the replay simulates stock from SAP's MinStock and MaxStock rather than
  replaying real stock history, and only for items where those are set.
- Service targets, holding cost and stockout cost are assumptions. The inventory the planner recommends rises or falls with
  them; the curve on Stock levels shows the trade-off.
- Where MOQ and pack size are blank in the export, a minimum of 1 is used.
- Draft POs are drafts. A buyer approves every order in SAP.
""")
