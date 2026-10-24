import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import kpi_row, money, need, page_setup, require_results, table  # noqa: E402
from engine import purchasing  # noqa: E402

page_setup("Purchase orders")
st.markdown('<p class="fp-note">Draft purchase orders for buyer approval. Each line says why it is suggested, which supplier '
            'wins and what the alternatives cost. Edit quantities, untick lines you do not want, then export drafts for SAP.</p>',
            unsafe_allow_html=True)
r = require_results()
recs = r.po_recs
tab1, tab2 = st.tabs(["Recommended POs", "Warehouse transfers"])
with tab1:
    if need(recs, "purchase orders"):
        c1, c2, c3, c4 = st.columns(4)
        whs = c1.multiselect("Warehouse", sorted(recs["warehouse"].unique()))
        act = c2.multiselect("Action", sorted(recs["action"].unique()), default=["Order now"] if "Order now" in set(recs["action"]) else None)
        sups = c3.multiselect("Supplier", sorted(recs["supplier"].astype(str).unique()))
        abc = c4.multiselect("ABC class", ["A", "B", "C"])
        q = st.text_input("Search item code or name", "")
        d = recs.copy()
        if whs:
            d = d[d["warehouse"].isin(whs)]
        if act:
            d = d[d["action"].isin(act)]
        if sups:
            d = d[d["supplier"].astype(str).isin(sups)]
        if abc:
            d = d[d["abc"].isin(abc)]
        if q:
            d = d[d["item_code"].str.contains(q, case=False, na=False) | d["item_name"].astype(str).str.contains(q, case=False, na=False)]
        kpi_row([("Lines shown", f"{len(d):,}", None, None),
                 ("Value shown", money(d["line_total"].sum()), None, "Quantity times supplier unit price."),
                 ("Suppliers", f"{d['supplier'].nunique():,}", None, None),
                 ("Likely to run short before arrival", f"{(d['chance_short_before_arrival'] >= 0.5).sum():,}", None,
                  "Lines with a 50% or higher chance that stock runs out before this order lands. Consider a transfer or expediting.")])
        show_cols = ["approve", "action", "order_by", "item_code", "item_name", "warehouse", "abc", "supplier", "quantity", "unit_price",
                     "line_total", "expected_arrival", "stock_position", "reorder_point", "order_up_to",
                     "chance_short_before_arrival", "expected_units_short_before_arrival", "weeks_cover_after_receipt", "why", "alternatives"]
        ed = st.data_editor(
            d[show_cols], hide_index=True, width="stretch", height=520, key="po_editor",
            disabled=[c for c in show_cols if c not in ("approve", "quantity")],
            column_config={
                "approve": st.column_config.CheckboxColumn("Approve", help="Ticked lines go into the SAP draft export"),
                "action": st.column_config.TextColumn("Action"),
                "item_code": st.column_config.TextColumn("Item"),
                "warehouse": st.column_config.TextColumn("Whs"),
                "abc": st.column_config.TextColumn("ABC"),
                "supplier": st.column_config.TextColumn("Supplier"),
                "chance_short_before_arrival": st.column_config.ProgressColumn("Chance short before arrival", format="percent", min_value=0, max_value=1),
                "quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1, format="%,.0f"),
                "unit_price": st.column_config.NumberColumn("Unit price", format="$%,.2f"),
                "line_total": st.column_config.NumberColumn("Line total", format="$%,.0f"),
                "stock_position": st.column_config.NumberColumn("Stock position", format="%,.0f"),
                "reorder_point": st.column_config.NumberColumn("Reorder point", format="%,.0f"),
                "order_up_to": st.column_config.NumberColumn("Order up to", format="%,.0f"),
                "expected_units_short_before_arrival": st.column_config.NumberColumn("Units short before arrival", format="%,.1f"),
                "weeks_cover_after_receipt": st.column_config.NumberColumn("Weeks cover after receipt", format="%,.1f"),
                "order_by": st.column_config.DateColumn("Order by"),
                "expected_arrival": st.column_config.DateColumn("Expected arrival"),
                "why": st.column_config.TextColumn("Why", width="large"),
                "alternatives": st.column_config.TextColumn("Other suppliers", width="large"),
                "item_name": st.column_config.TextColumn("Item name", width="medium"),
            })
        approved = ed[ed["approve"] & (ed["quantity"] > 0)].copy()
        approved["line_total"] = approved["quantity"] * approved["unit_price"]
        st.markdown(f"**{len(approved):,} approved lines, {money(approved['line_total'].sum())}, "
                    f"{approved.groupby(['supplier', 'warehouse']).ngroups if len(approved) else 0} draft POs** (one per supplier and warehouse).")
        if len(approved):
            h, l = purchasing.to_dtw(approved, r.as_of, r.settings.get("purchasing", {}).get("po_draft_series", ""))
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("Documents.csv", h)
                z.writestr("Document_Lines.csv", l)
                z.writestr("README.txt", "SAP Business One Data Transfer Workbench import\n"
                           "Business object: oDrafts (Drafts). DocObjectCode 22 = purchase order.\n"
                           "Import Documents.csv as the header file and Document_Lines.csv as the lines file.\n"
                           "Always run the import in simulation mode first, and test in a test company before production.\n"
                           "Drafts are not posted: a buyer opens each draft in SAP, reviews it and adds it as a purchase order.\n")
            c1, c2, c3 = st.columns(3)
            c1.download_button("SAP drafts for DTW (zip)", buf.getvalue(), f"PO_drafts_DTW_{r.as_of:%Y%m%d}.zip", "application/zip",
                               help="Data Transfer Workbench, object oDrafts. Run in simulation mode first.")
            c2.download_button("SAP drafts for Service Layer (JSON)", purchasing.to_service_layer(approved, r.as_of).encode("utf-8"),
                               f"PO_drafts_ServiceLayer_{r.as_of:%Y%m%d}.json", "application/json",
                               help="One body per draft for POST /b1s/v1/Drafts. Test in a test company first.")
            c3.download_button("Approved lines (CSV)", approved.to_csv(index=False).encode("utf-8-sig"), f"PO_lines_{r.as_of:%Y%m%d}.csv", "text/csv")
        st.caption("Drafts never post on their own. A buyer opens each draft in SAP, checks it and turns it into a purchase order, "
                   "which keeps buyer approval in place as the problem statement asks.")
with tab2:
    t = r.transfers
    if need(t, "transfers"):
        st.caption("Move stock where one warehouse is below its reorder point and the other holds more than it needs. "
                   "Transfers are faster than a new PO.")
        table(t, height=480, num_cols=("quantity",), wide=("why",))
