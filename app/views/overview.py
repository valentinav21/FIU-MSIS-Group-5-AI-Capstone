import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (sap_ok, stock_kpis, signed_money, esc, esc_html, ROOT, SAMPLE_FOLDER, is_sample, kpi_row, money, page_setup, results, settings)  # noqa: E402
from engine import io as eio  # noqa: E402
from engine import pipeline, report  # noqa: E402

cfg = settings()
r = results()
page_setup("FORTPRO AI Planner")
INTRO = ('<p class="fp-note">Purchasing planning (BP1) for World Truck Parts, built for SAP Business One exports. Sales growth (BP2) is TBD. '
         'Every number is computed on this computer from the files you load.</p>')
st.markdown(INTRO, unsafe_allow_html=True)

# ---------------------------------------------------------------- data source and run
st.subheader("Choose data and run")
with st.container(border=True):
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
            if not (ROOT / SAMPLE_FOLDER / "sample_v2.txt").exists():
                st.warning("The sample data has not been built yet (or was built by an older version). It takes about 20 seconds.")
                if st.button("Build sample data"):
                    with st.spinner("Building synthetic sample data from the public fortpro.com catalog..."):
                        from tools.generate_synthetic import generate
                        generate(ROOT / SAMPLE_FOLDER)
                    st.rerun()
    with c2:
        p = eio.resolve_folder(folder)
        files = sorted([x.name for x in p.iterdir() if x.is_file() and not x.name.startswith((".", "~"))
                        and x.name not in ("README_SYNTHETIC.txt", "sample_v2.txt")]) if p.exists() else []
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
    st.info("No results yet. Pick a data source above and press **Run analysis**. To explore first, choose the sample data.")
    st.stop()

if is_sample(r.data_folder):
    st.markdown('<div class="fp-banner"><b>Sample data.</b> Results below use SYNTHETIC data built on the public fortpro.com '
                'catalog. They show how the planner works, not FORTPRO performance.</div>', unsafe_allow_html=True)
for e in r.errors:
    st.error(e)

# ---------------------------------------------------------------- the answer first
_s = r.settings or {}
_pw = _s.get("data", {}).get("plan_warehouses") or []
_pw = (" and ".join(_pw) if len(_pw) <= 2 else ", ".join(_pw)) or "all warehouses"
_he = getattr(r, "history_end", None)
_now = r.po_recs[r.po_recs["action"] == "Order now"] if len(r.po_recs) else r.po_recs
_crit = (r.alerts[(r.alerts["risk"] == "Stockout risk") & (r.alerts["severity"] == "Critical")] if len(r.alerts) else r.alerts)
if len(_now):
    h1 = f"{len(_now):,} order lines are due now, worth {money(_now['line_total'].sum())}."
else:
    h1 = "No purchase orders are due right now."
if len(_crit):
    h2 = f"{len(_crit):,} critical stockout risks in the next 8 weeks, with {money(_crit['revenue_at_risk'].sum())} of sales at stake."
elif len(r.alerts):
    h2 = "No critical stockout risks in the next 8 weeks."
else:
    h2 = ""
ctx = (f"Based on data through {r.as_of:%B} {r.as_of.day}, {r.as_of.year} for warehouses {_pw}"
       + (f", with monthly history through {_he:%B %Y}" if r.bucket == "M" and _he is not None else "") + ". "
       + (lambda t: f"Analysis run {t:%b} {t.day}, {t.year} at {t.strftime('%I:%M %p').lstrip('0')}.")(pd.Timestamp(r.created)))
st.markdown(esc_html(f'<div class="fp-hero"><p class="h1">{h1}</p>' + (f'<p class="h2">{h2}</p>' if h2 else "")
                     + f'<p class="ctx">{ctx}</p></div>'), unsafe_allow_html=True)
l1, l2, _ = st.columns([1, 1, 3])
try:  # links need the app's menu; a page run on its own (tests) has none
    l1.page_link("views/purchase_orders.py", label="Review purchase orders", icon=":material/shopping_cart:")
    l2.page_link("views/risk_alerts.py", label="See risk alerts", icon=":material/warning:")
except Exception:
    pass

st.subheader("Where things stand")
st.caption("Costs, holding rate, stockout cost and service targets are ASSUMPTIONS (Settings), not FORTPRO data. "
           "Dollar figures below depend on them.")
ps, rp = r.policy_summary or {}, r.replay or {}
rt = rp.get("table")
items = stock_kpis(r) if ps else []
if items:
    kpi_row(items)
items2 = []
if rt is not None:
    a, b = "Today's SAP min / max", "AI planner"
    if a in rt.columns and b in rt.columns and sap_ok(r):
        items2.append(("Lost sales avoided (replay)", money(rt.loc["Lost revenue", a] - rt.loc["Lost revenue", b]),
                       f"{signed_money(rt.loc['Average inventory value', b] - rt.loc['Average inventory value', a])} avg inventory",
                       f"Real demand of the last 52 weeks replayed through today's SAP min / max and the AI settings, on the "
                       f"{rp['series']:,} items that have SAP min / max set. See Stock levels.", "off"))
S = r.sales
if S is not None and len(S.customers) and "revenue_at_risk" in S.customers:
    risky = S.customers[S.customers["churn_risk"] >= 0.5]
    items2.append(("Customers likely to churn", f"{len(risky):,}", money(risky["revenue_12m"].sum()) + " a year", "See Customers."))
if items2:
    kpi_row(items2)

st.subheader("Do these first")
acts = []
if len(r.alerts):
    for x in r.alerts[r.alerts["risk"] == "Stockout risk"].head(3).itertuples():
        acts.append(("Stockout risk", f"{x.item_code} at {x.warehouse}", f"Risk starts in week {int(x.first_risk_week)} and peaks at a {x.max_stockout_probability:.0%} chance of running out. {x.root_causes.split(' | ')[0]}.", x.revenue_at_risk))
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
    for kind, what, why, val in acts:
        with st.container(border=True):
            c1, c2 = st.columns([7, 1])
            c1.markdown(esc(f"**{kind}: {what}**"))
            c1.markdown(f'<p class="fp-note" style="margin:0 0 6px 0">{esc_html(why)}</p>', unsafe_allow_html=True)
            if pd.notna(val):
                c2.markdown(esc_html(f'<p style="text-align:right;margin:0"><b>{money(val)}</b><br><span class="fp-note">'
                            f'{"at risk" if kind == "Stockout risk" else "order value"}</span></p>'), unsafe_allow_html=True)
else:
    st.caption("Nothing urgent: no stockout risks above the alert threshold and no orders due now.")

st.subheader("Take it with you")
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Prepare Excel workbook"):
        with st.spinner("Writing every result to Excel..."):
            st.session_state["xlsx"] = report.workbook(r)
    if "xlsx" in st.session_state:
        st.download_button("Download workbook", st.session_state["xlsx"], file_name=f"FORTPRO_AI_Planner_{r.as_of:%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with c2:
    st.caption("One sheet per page: summary, PO recommendations, transfers, old open POs, risk alerts, stock levels, "
               "forecasts and accuracy, supplier scorecards, the replay and model check, data health and column mapping."
               + (" Plus the sales growth sheets." if r.sales is not None else ""))
with st.expander("Run details", icon=":material/info:"):
    st.write({k: f"{v:.1f} s" for k, v in r.timings.items()})
    for n in r.notes:
        st.write("•", n)
