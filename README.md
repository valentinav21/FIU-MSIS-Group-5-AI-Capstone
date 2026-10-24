# FORTPRO AI Planner

Local AI planning tool for World Truck Parts, LLC (FORTPRO), built by FIU MSIS AI Capstone Group 5.
It answers both problem statements FORTPRO submitted:

* **BP1 Purchasing:** intermittent demand forecasting, dynamic safety stock and reorder points, 1 to 8 week
  stockout and overstock warnings with root causes, explainable multi-supplier purchase order recommendations
  exported as SAP Business One drafts, and supplier lead time scorecards.
* **BP2 Sales growth (TBD):** turned off for now, since Group 5 is tackling BP1. Set `features: bp2_enabled: true`
  in `config/settings.yaml` to turn it back on. When on: customer segments and personas, churn risk, cross-sell and kit suggestions, wallet share
  gaps, geographic opportunity, price leakage and price sensitivity, product trends, unmet web demand, and one
  ranked list of sales plays.

Everything runs on your own computer. No data leaves it, and nothing writes to SAP.

---

## 1. Start it

**Windows:** double-click `start_windows.bat`.
**Mac or Linux:** open a terminal in this folder and run `bash start_mac_linux.sh`.

The first start installs a private Python environment (a few minutes, once) and builds the synthetic sample data.
Then the app opens in your browser at http://localhost:8501. Keep the black window open while you work.

You need **Python 3.10 or newer** (https://www.python.org/downloads/; on Windows tick *Add python.exe to PATH*).

## 2. Try it on the sample data

On **Overview**, choose *Sample data* and press **Run analysis** (about 2 minutes). The sample is **synthetic**:
item codes, names and categories come from the public fortpro.com catalog, but every quantity, price, customer,
supplier and stock figure is invented. Every page shows a banner while sample data is in use.

## 3. Plug in FORTPRO's data

1. Daniel runs the queries in `tools/sap_export_queries.sql` (or exports the same tables by hand) and saves each
   result as CSV. File names can be the SAP table names: `OITM.csv`, `INV1.csv`, `RIN1.csv`, `OITW.csv`,
   `POR1.csv`, `PDN1.csv`, `ITM2.csv`, plus `suppliers.csv`, `customers.csv`, `open_orders.csv`.
2. Optional but recommended: replace customer and supplier codes with anonymous IDs:
   `python tools/anonymize.py --src <raw folder> --dst <clean folder> --secret "<phrase only FORTPRO knows>"`
3. Put the files in `data/my_data` (or any folder), choose *My SAP export folder* on Overview, and press **Run analysis**.
4. Open **Data health** first. It lists what loaded, what was skipped, how each column was mapped, and which
   modules have what they need.

Standard SAP column names (DocDate, ItemCode, CardCode, WhsCode, OnHand, BaseEntry, U_ fields ...) are recognised
automatically, as are the plain names in `data/templates`. CSV, tab-separated text and Excel all work; SAP date
formats, currency symbols, accounting negatives, leading zeros and DTW second header rows are handled.

| File | SAP source | Needed for | Priority |
|---|---|---|---|
| items | OITM, OITB, OMRC | both | essential |
| sales | OINV/INV1 (or ORDR/RDR1) | both | essential |
| stock | OITW | BP1 | essential |
| purchases | OPOR/POR1 | BP1 | essential |
| receipts | OPDN/PDN1 | BP1 | essential |
| returns | ORIN/RIN1 | both | optional |
| suppliers, item_suppliers | OCRD (S), ITM2 | BP1 | optional |
| open_orders | ORDR/RDR1 open lines | BP1 | optional |
| customers | OCRD (C), OCRG | BP2 | optional |
| web_searches | FocusPoint search log | both | optional |
| market_potential | public data (Group 5) | BP2 | optional |
| events | one-off orders, promotions | both | optional |

The full column list with SAP fields is in `data/templates/DATA_DICTIONARY.csv` and on the Data health page.
With only `items` and `sales` the forecast and stock levels still run; each extra file switches on more modules.

## 4. Pages

| Page | What you do there |
|---|---|
| Overview | Choose data, run, see the headline numbers and the first actions, download the Excel workbook |
| Data health | Check the export, download templates and SAP queries |
| Purchase orders | Review, edit and approve draft POs; export SAP drafts (DTW zip or Service Layer JSON); warehouse transfers |
| Risk alerts | Stockout, overstock and dead stock warnings for the next 8 weeks, with root causes |
| Stock levels | Service versus inventory curve, proof on last year's demand, new MinStock / MaxStock per item |
| Demand forecast | Per item chart with likely range, method and reason; accuracy and calibration |
| Suppliers | Lead time distribution and scorecard per supplier |
| Sales plays | Ranked weekly actions per rep |
| Customers | Segments, churn risk, personas, customer lookup |
| Cross-sell and upsell | Item suggestions per customer, kits and bundles, wallet share gaps |
| Markets | US and export maps, penetration against market potential |
| Pricing | Price leakage, discounts by customer size, price sensitivity |
| Products and web demand | Product line trends, searches with no result, searched but out of stock |
| Settings | Service targets, budget mode, costs, forecast options |
| How it works | Methods, comparison with tools on the market, glossary, limits |

For a weekly scheduled run without the app: `python run_batch.py --folder <export folder> --out <results folder>`.

## 5. How it works, and how it was checked

**Forecast.** Each item and warehouse is classified (smooth, erratic, intermittent, lumpy, new, inactive) and every
method in the library is tested on rolling windows it never saw: 13-period moving average (the static SAP style
baseline), exponential smoothing, Croston, SBA, TSB, temporal aggregation (ADIDA), a seasonal variant and a global
gradient boosted model. The best method per item is used; items with thin evidence use their class's best method.
Accuracy is reported on a final window that neither selection nor calibration saw.

**Ranges you can trust.** Demand is simulated with the item's own share of zero weeks and its own order sizes, plus
forecast error. A quantile calibration learned on one window and checked on the next makes a 95% level hold about
95% of the time.

**Lead times** come from real PO and receipt dates per item and supplier, falling back to the supplier's pattern,
then the SAP setting.

**Stock levels.** Each candidate reorder point is tested by simulating two years of weekly operation with simulated
demand, real minimums and pack sizes, and lead times drawn from history. The lowest level meeting the fill rate target
wins. A budget mode places a fixed number of dollars item by item where it serves the most revenue.

**Self-check.** Before anything is trusted, the simulator predicts the fill rate of today's SAP min/max settings and
compares it with what those settings actually delivered when last year's real demand is replayed. On the synthetic
sample: predicted 84.4%, replayed 86.4% (the simulator errs on the cautious side).

**Results on the synthetic sample** (illustration only; real numbers come from FORTPRO's data):

| Measure | Today's SAP settings | AI planner, same inventory $ | AI planner at targets |
|---|---|---|---|
| Fill rate (units), last 52 weeks replayed | 86.5% | 93.2% | 96.4% |
| Average inventory value | $6.9M | $7.3M | $11.6M |
| Forecast error (WAPE, 8 week totals) | 59.0% (moving average) | 54.8% | |
| Forecast bias | +11.3% | +6.1% | |

Churn model AUC 0.89 on a later period it never saw; cross-sell hit rate 63.6% versus 57.6% for a best-seller list.

**Tests.** `python -m pytest tests` runs 28 tests: method kernels, loader robustness on messy exports, full runs on
complete, minimal (sales and items only) and monthly data, SAP export formats, anonymization, and every app page.
They pass on Python 3.11 with the oldest supported libraries and on Python 3.13 with the newest.

## 6. Limits

* Results are only as good as the export. Check Data health first.
* The policy replay uses settings computed with the replay year included. Confirm with a forward pilot (for example
  200 items for 8 to 12 weeks) before rolling out.
* Price sensitivity is reported only where prices varied enough for the same customer and item; otherwise run a test.
* Draft POs are drafts. Test the DTW or Service Layer import in a test company first; a buyer approves every order in SAP.
* The sample market potential figures are placeholders; replace them with public data (for example FHWA truck
  registrations by state) before using that view.

## 7. Folder map

```
app/            Streamlit app (Home.py and one file per page)
engine/         planning and sales engine (no UI code)
config/         settings.yaml (documented defaults); settings_local.yaml holds changes made in the app
data/my_data    put FORTPRO's export here
data/sample     synthetic sample (rebuilt by tools/generate_synthetic.py)
data/templates  blank CSV templates and the data dictionary
tools/          SAP export queries, anonymizer, template and sample builders
tests/          automated tests
output/         saved results and Excel workbooks
```
