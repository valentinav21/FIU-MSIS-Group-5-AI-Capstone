import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (show_df, ROOT, SAMPLE_FOLDER, is_sample, kpi_row, money, page_setup, pct, results, settings)  # noqa: E402
from engine import io as eio  # noqa: E402
from engine import pipeline, report  # noqa: E402

page_setup("FORTPRO AI Planner")
st.markdown('<p class="fp-note">Purchasing planning (BP1) for World Truck Parts, built for SAP Business One exports. Sales growth (BP2) is TBD. '
            'Every number below is computed on this computer from the files you load.</p>', unsafe_allow_html=True)

cfg = settings()
r = results()

# ---------------------------------------------------------------- data source and run
with st.container(border=True):
    st.subheader("1. Choose data and run")
    c1, c2 = st.columns([3, 2])
    with c1:
        last = r.data_folder if r is not None else cfg["data"]["folder"]
        src = st.radio("Data source", ["My SAP export folder", "Sample data (synthetic, for practice)"],
                       index=1 if is_sample(last) else 0, horizontal=True)
        if src.startswith("My"):
            folder = st.text_input("Folder with your CSV or Excel files",
                                   value=cfg["data"]["folder"] if not is_sample(cfg["data"]["folder"]) else "data/my_data",
                                   help="A full path such as C:\\FORTPRO\\exports, or a folder inside the app such as data/my_data. "
                                        "File names tell the app what each file holds; see Data health for the list.")
            up = st.file_uploader("Or drop files here to copy them into data/my_data", accept_multiple_files=True,
                                  type=["csv", "txt", "tsv", "xlsx", "xls", "parquet"])
            if up:
                dest = ROOT / "data" / "my_data"
                dest.mkdir(parents=True, exist_ok=True)
                for f in up:
                    (dest / Path(f.name).name).write_bytes(f.getbuffer())
                st.success(f"Copied {len(up)} file(s) into data/my_data.")
                folder = "data/my_data"
        else:
            folder = SAMPLE_FOLDER
            if not (ROOT / SAMPLE_FOLDER / "INV1.csv").exists():
                st.warning("The sample data has not been built yet. It takes about 20 seconds.")
                if st.button("Build sample data"):
                    with st.spinner("Building synthetic sample data from the public fortpro.com catalog..."):
                        from tools.generate_synthetic import generate
                        generate(ROOT / SAMPLE_FOLDER)
                    st.rerun()
    with c2:
        p = eio.resolve_folder(folder)
        files = sorted([x.name for x in p.iterdir() if x.is_file() and not x.name.startswith((".", "~"))]) if p.exists() else []
        st.caption(f"Folder: `{p}`")
        if not p.exists():
            st.error("This folder does not exist yet.")
        elif not files:
            st.warning("The folder is empty. Export the tables listed on Data health, then press Run.")
        else:
            st.caption(f"{len(files)} files: " + ", ".join(files[:14]) + (" ..." if len(files) > 14 else ""))
        run = st.button("Run analysis", type="primary", width="stretch", disabled=not files)
        st.caption("Takes 1 to 3 minutes for about 6,000 item and warehouse pairs.")
    if run:
        cfg["data"]["folder"] = folder
        eio.save_settings(cfg)
        bar = st.progress(0.0, text="Starting")
        def prog(f, msg):
            bar.progress(min(max(float(f), 0.0), 1.0), text=msg)
        res = pipeline.run(folder, cfg, progress=prog)
        st.cache_resource.clear()
        if res.errors:
            st.error("Finished with problems: " + " | ".join(res.errors))
        else:
            st.success(f"Done in {res.timings.get('Total', 0):.0f} seconds.")
        r = res

if r is None:
    st.info("No results yet. Pick a data source and press **Run analysis**. To explore first, choose the sample data.")
    st.stop()

if is_sample(r.data_folder):
    st.markdown('<div class="fp-banner"><b>Sample data.</b> Results below use SYNTHETIC data built on the public fortpro.com '
                'catalog. They show how the planner works, not FORTPRO performance.</div>', unsafe_allow_html=True)
for e in r.errors:
    st.error(e)

st.subheader("2. Where things stand")
st.caption(f"Data as of {r.as_of:%B %d, %Y} · run {r.created} · folder {r.data_folder}")
ps, rp = r.policy_summary or {}, r.replay or {}
rt = rp.get("table")
items = []
if ps:
    items.append(("Fill rate, today's SAP settings", pct(ps.get("current_fill_rate")), None,
                  "Share of units demanded that ship from stock, simulated with today's MinStock and MaxStock."))
    same = r.budget.get("same_inventory")
    if same:
        items.append(("Fill rate, same inventory $, AI", pct(same["fill_rate"]),
                      f"{(same['fill_rate'] - ps.get('current_fill_rate', np.nan)) * 100:+.1f} pts",
                      "Same average inventory dollars as today, placed where they serve the most revenue."))
    items.append(("Fill rate, AI at your targets", pct(ps.get("fill_rate")),
                  f"{(ps.get('fill_rate', np.nan) - ps.get('current_fill_rate', np.nan)) * 100:+.1f} pts",
                  "Service targets from Settings (A 98%, B 96%, C 92% by default)."))
    items.append(("Inventory at your targets", money(ps.get("avg_inventory_value")),
                  f"{money(ps.get('avg_inventory_value', 0) - ps.get('current_avg_inventory_value', 0))} vs today",
                  "Average stock value the simulation expects at your targets, versus today's settings.", "off"))
if items:
    kpi_row(items)
items2 = []
if rt is not None:
    a, b = "Today's SAP min / max", "AI planner, same inventory dollars as today"
    if b in rt.columns:
        items2.append(("Lost sales avoided (replay)", money(rt.loc["Lost revenue", a] - rt.loc["Lost revenue", b]), None,
                       "Real demand of the last 52 weeks replayed through both policies at the same inventory dollars. See Stock levels."))
if len(r.po_recs):
    now = r.po_recs[r.po_recs["action"] == "Order now"]
    items2.append(("PO lines to place now", f"{len(now):,}", money(now["line_total"].sum()), "See Purchase orders."))
if len(r.alerts):
    crit = r.alerts[(r.alerts["risk"] == "Stockout risk") & (r.alerts["severity"] == "Critical")]
    items2.append(("Critical stockout risks", f"{len(crit):,}", money(crit["revenue_at_risk"].sum()) + " at risk", "See Risk alerts."))
S = r.sales
if S is not None and len(S.customers) and "revenue_at_risk" in S.customers:
    risky = S.customers[S.customers["churn_risk"] >= 0.5]
    items2.append(("Customers likely to churn", f"{len(risky):,}", money(risky["revenue_12m"].sum()) + " a year", "See Customers."))
if items2:
    kpi_row(items2)

st.subheader("3. Do these first")
acts = []
if len(r.alerts):
    for x in r.alerts[r.alerts["risk"] == "Stockout risk"].head(3).itertuples():
        acts.append(("Stockout risk", f"{x.item_code} at {x.warehouse}", f"{x.max_stockout_probability:.0%} chance of running out by week {int(x.first_risk_week)}. {x.root_causes.split(' | ')[0]}.", x.revenue_at_risk))
if len(r.po_recs):
    for x in r.po_recs.head(2).itertuples():
        acts.append(("Purchase order", f"{x.quantity:,.0f} x {x.item_code} from {x.supplier} for {x.warehouse}", x.why.split(". ")[0] + ".", x.line_total))
if S is not None and len(S.plays):
    for x in S.plays.head(3).itertuples():
        acts.append((x.play, f"Customer {x.customer_code}", x.detail, x.value))
if S is not None and len(S.unmet_demand):
    u = S.unmet_demand.iloc[0]
    acts.append(("Unmet web demand", f"'{u['term']}'", f"Searched {u['searches']} times by {u['accounts']} accounts with no result in the last year.", np.nan))
if acts:
    df = pd.DataFrame(acts, columns=["Type", "What", "Why", "Value ($)"])
    show_df(df, hide_index=True, width="stretch", column_config={"Value ($)": st.column_config.NumberColumn(format="$%,.0f"),
                                                                     "Why": st.column_config.TextColumn(width="large")})

st.subheader("4. Take it with you")
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Prepare Excel workbook"):
        with st.spinner("Writing every result to Excel..."):
            st.session_state["xlsx"] = report.workbook(r)
    if "xlsx" in st.session_state:
        st.download_button("Download workbook", st.session_state["xlsx"], file_name=f"FORTPRO_AI_Planner_{r.as_of:%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with c2:
    st.caption("One sheet per page: PO recommendations, transfers, alerts, stock levels, forecasts, supplier scorecards, "
               "customers, cross-sell, pricing, geography, products, data health.")
with st.expander("Run details"):
    st.write({k: f"{v:.1f} s" for k, v in r.timings.items()})
    for n in r.notes:
        st.write("•", n)
