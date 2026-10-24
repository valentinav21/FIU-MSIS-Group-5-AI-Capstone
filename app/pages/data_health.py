import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, ROOT, page_setup, results  # noqa: E402
from engine.schema import TABLES  # noqa: E402

page_setup("Data health")
st.markdown('<p class="fp-note">What the app found in your files, what it could not use, and what to export from SAP Business One.</p>',
            unsafe_allow_html=True)

r = results()
tab1, tab2, tab3 = st.tabs(["Checks", "What to export", "Column mapping"])

with tab1:
    if r is None:
        st.info("Run the analysis on the Overview page to see the checks.")
    else:
        st.subheader("Which modules can run")
        rd = r.readiness.copy()
        show_df(rd, hide_index=True, width="stretch",
                     column_config={"Ready": st.column_config.TextColumn(width="small")})
        st.subheader("Checks")
        h = r.health.copy()
        order = {"Problem": 0, "Check": 1, "OK": 2, "Info": 3}
        h["_o"] = h["Status"].map(order)
        h = h.sort_values(["_o", "Area"]).drop(columns="_o")
        icon = {"Problem": "🔴 Problem", "Check": "🟡 Check", "OK": "🟢 OK", "Info": "ℹ️ Info"}
        h["Status"] = h["Status"].map(icon)
        show_df(h, hide_index=True, width="stretch", height=520,
                     column_config={"Detail": st.column_config.TextColumn(width="medium"),
                                    "How to fix": st.column_config.TextColumn(width="large")})
        if r.notes:
            with st.expander("Notes from the last run"):
                for n in r.notes:
                    st.write("•", n)

with tab2:
    st.markdown("Put one file per table in the data folder. File names can be the SAP table name (`OITM.csv`, `INV1.csv`, "
                "`OITW.csv`, `POR1.csv`, `PDN1.csv`) or the names below. CSV, tab-separated text and Excel all work, and "
                "standard SAP column names are recognised automatically.")
    rows = []
    for k, t in TABLES.items():
        rows.append({"File name": f"{k}.csv", "Contents": t.title, "SAP source": t.sap_source, "Needed for": t.problem,
                     "Priority": "Essential" if t.essential else "Optional", "Required columns": ", ".join(t.required),
                     "What it is": t.desc})
    show_df(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"What it is": st.column_config.TextColumn(width="large")})
    sql = (ROOT / "tools" / "sap_export_queries.sql").read_text(encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "data" / "templates").glob("*.csv")):
            z.write(p, f"templates/{p.name}")
        z.writestr("sap_export_queries.sql", sql)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download templates and data dictionary (zip)", buf.getvalue(), "FORTPRO_data_templates.zip", "application/zip")
    with c2:
        st.download_button("Download SAP export queries (SQL)", sql.encode("utf-8"), "sap_export_queries.sql", "text/plain")
    with st.expander("Preview the SAP export queries"):
        st.code(sql, language="sql")
    st.markdown("**Privacy.** The app needs codes, not names. If customer codes contain names, run "
                "`python tools/anonymize.py --src <raw folder> --dst <clean folder> --secret \"<phrase>\"` first. "
                "Everything runs on this computer; no data is sent anywhere.")

with tab3:
    if r is None or not len(r.mapping):
        st.info("Run the analysis to see how your columns were mapped.")
    else:
        st.caption("How each column in your files was matched to the app's fields. Columns not listed were ignored.")
        show_df(r.mapping, hide_index=True, width="stretch", height=520)
