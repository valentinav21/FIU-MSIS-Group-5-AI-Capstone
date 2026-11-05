import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import esc, kpi_row, money, need, page_setup, paged, require_results, table  # noqa: E402
from engine import purchasing  # noqa: E402

page_setup("Purchase orders")
st.markdown('<p class="fp-note">Draft purchase orders for buyer approval. Each line says why it is suggested, which supplier '
            'wins and what the alternatives cost. Edit quantities, untick lines you do not want, then export drafts for SAP.</p>',
            unsafe_allow_html=True)
r = require_results()
recs = r.po_recs
stale = getattr(r, "stale_pos", None)
tab1, tab2, tab3 = st.tabs(["Recommended POs", "Warehouse transfers",
                            f"Old open POs ({len(stale) if stale is not None else 0})"])
with tab1:
    if need(recs, "purchase orders"):
        c1, c2, c3, c4 = st.columns(4)
        whs = c1.multiselect("Warehouse", sorted(recs["warehouse"].dropna().astype(str).unique()))
        act = c2.multiselect("Action", sorted(recs["action"].dropna().astype(str).unique()), default=["Order now"] if "Order now" in set(recs["action"]) else None)
        sups = c3.multiselect("Supplier", sorted(recs["supplier"].dropna().astype(str).unique()))
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
        if "below_supplier_minimum" in d and d["below_supplier_minimum"].any():
            low = d[d["below_supplier_minimum"]]
            mn = float(r.settings.get("purchasing", {}).get("international_min_order_usd", 10000))
            ns = low["supplier"].nunique()
            st.warning(esc(f"**{ns} supplier{'s' if ns != 1 else ''} outside the US {'are' if ns != 1 else 'is'} below the ${mn:,.0f} minimum order.**  \n"
                           f"{len(low):,} lines, {money(low['line_total'].sum())} in total.  \n"
                           "Options: add 'Order soon' lines from the same supplier, buy from a US source, or wait.  \n"
                           "The Why column on each line shows how far below the minimum it is."))
        compact = ["approve", "action", "order_by", "item_code", "item_name", "warehouse", "supplier", "supplier_country",
                   "below_supplier_minimum", "quantity", "unit_price", "line_total", "expected_arrival", "chance_short_before_arrival", "why"]
        full = ["approve", "action", "order_by", "item_code", "item_name", "warehouse", "abc", "supplier", "supplier_country",
                "below_supplier_minimum", "quantity", "unit_price", "line_total", "expected_arrival", "stock_position", "reorder_point", "order_up_to",
                "chance_short_before_arrival", "expected_units_short_before_arrival", "weeks_cover_after_receipt", "why", "alternatives"]
        # ticks and quantity changes are kept per line across pages and filters until the next run
        store = st.session_state.setdefault(f"po_edits_{r.created}", {})
        base = d.copy()
        for i, v in store.items():
            if i in base.index:
                base.at[i, "approve"], base.at[i, "quantity"] = v
        b1, b2, b3 = st.columns([1, 1, 2])
        if b1.button(f"Approve all {len(base):,} shown", width="stretch"):
            for i in base.index:
                store[i] = (base.at[i, "supplier"] != "NO SUPPLIER", float(base.at[i, "quantity"]))
            st.session_state["po_ver"] = st.session_state.get("po_ver", 0) + 1  # fresh editors show the new ticks
            st.rerun()
        if b2.button("Untick all shown", width="stretch"):
            for i in base.index:
                store[i] = (False, float(base.at[i, "quantity"]))
            st.session_state["po_ver"] = st.session_state.get("po_ver", 0) + 1
            st.rerun()
        show_all = b3.toggle("Show every column", False, help="Stock position, reorder point, order up to, cover after receipt and other suppliers.")
        show_cols = [c for c in (full if show_all else compact) if c in d.columns]
        page = paged(base, "po_page", default=50)
        ed = st.data_editor(
            page[show_cols], hide_index=True, width="stretch", height="content", key=f"po_editor_{r.created}_{st.session_state.get('po_ver', 0)}_{page.index[0] if len(page) else 0}",
            disabled=[c for c in show_cols if c not in ("approve", "quantity")],
            column_config={
                "approve": st.column_config.CheckboxColumn("Approve", help="Ticked lines go into the SAP draft export"),
                "action": st.column_config.TextColumn("Action"),
                "item_code": st.column_config.TextColumn("Item"),
                "warehouse": st.column_config.TextColumn("Whs"),
                "abc": st.column_config.TextColumn("ABC"),
                "supplier": st.column_config.TextColumn("Supplier"),
                "supplier_country": st.column_config.TextColumn("Country"),
                "below_supplier_minimum": st.column_config.CheckboxColumn("Below intl. minimum", help="Supplier outside the US and all its 'Order now' lines total less than the minimum order setting"),
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
        for i in ed.index:
            store[i] = (bool(ed.at[i, "approve"]), float(ed.at[i, "quantity"]))
        allv = d.copy()
        for i, v in store.items():
            if i in allv.index:
                allv.at[i, "approve"], allv.at[i, "quantity"] = v
        approved = allv[allv["approve"].astype(bool) & (allv["quantity"] > 0)].copy()
        approved["line_total"] = approved["quantity"] * approved["unit_price"]
        st.markdown(esc(f"**{len(approved):,} approved lines (all pages), {money(approved['line_total'].sum())}, "
                    f"{approved[approved['supplier'] != 'NO SUPPLIER'].groupby(['supplier', 'warehouse']).ngroups if len(approved) else 0} "
                    "draft POs** (one per supplier and warehouse; lines with no supplier are not exported)."))
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
        table(t, unit_cols=("quantity",), wide=("why",))
with tab3:
    days = int(r.settings.get("data", {}).get("stale_open_days", 180))
    if stale is None or len(stale) == 0:
        st.info(f"No open PO lines older than {days} days.")
    else:
        st.caption(f"Open PO lines older than {days} days by PO date. They are not counted as incoming stock: at FORTPRO, 9 in 10 "
                   "orders arrive within about 180 days, so these are most likely overdue or were never closed. "
                   "Confirm with the supplier or close them in SAP.")
        sd = stale.copy()
        sd["age_days"] = (r.as_of - sd["po_date"]).dt.days
        sd["po_date"] = sd["po_date"].dt.strftime("%Y-%m-%d")
        cols = [c for c in ["po_entry", "line_num", "po_date", "age_days", "item_code", "supplier_code", "warehouse", "quantity",
                            "received_qty", "open_qty", "unit_price"] if c in sd.columns]
        sd = sd.sort_values("age_days", ascending=False)[cols]
        kpi_row([("Old open lines", f"{len(sd):,}", None, None), ("Units", f"{sd['open_qty'].sum():,.0f}", None, None),
                 ("Older than a year", f"{(sd['age_days'] > 365).sum():,}", None, None)])
        table(sd, unit_cols=("quantity", "received_qty", "open_qty", "age_days"), money_cols=())
