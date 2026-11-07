import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from html import escape as _esc  # noqa: E402

from common import esc, esc_html, kpi_row, sentences, money, need, page_setup, paged, require_results, show_df, table  # noqa: E402
from engine import purchasing  # noqa: E402


def _h(v):
    return "" if v is None or (isinstance(v, float) and v != v) else _esc(str(v))


def _cols(widths, align):
    """Tight columns. gap="xsmall" is newer than Streamlit 1.50; older versions get "none" (the CSS adds padding)."""
    try:
        return st.columns(widths, vertical_alignment=align, gap="xsmall")
    except Exception:  # noqa: BLE001
        return st.columns(widths, vertical_alignment=align, gap=None)


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
        show_all = b3.toggle("Show stock details and other suppliers", False,
                             help="Adds stock position, reorder point, order-up-to level, cover after receipt and the other suppliers under each reason.")
        page = paged(base, "po_page", default=25, note="Ticks and quantities are kept when you change pages; the export covers every approved line."
                     + (f" Order by and Arrives count from the data date, {r.as_of:%b %d, %Y}." if r.as_of is not None else ""))
        ver = st.session_state.get("po_ver", 0)
        # every field in its own column, and the reasons last, stacked one sentence per line
        # widths in pixels at full size, sized to the content (the list shrinks proportionally on smaller screens)
        W = [72, 128, 175, 100, 46, 46, 96, 70, 96, 86, 84, 84, 84, 62, 502]
        heads = ["Approve", "Item", "Item name", "Action", "ABC", "Whs", "Supplier", "Country", "Quantity",
                 "Unit price", "Line total", "Order by", "Arrives", "Short risk", "Why"]
        # Action only adds information when the list mixes 'Order now' and 'Order soon' lines
        one_action = base["action"].nunique() <= 1
        if one_action:
            j = heads.index("Action")
            W, heads = W[:j] + W[j + 1:], heads[:j] + heads[j + 1:]
            if len(base):
                st.caption(f"All lines shown are **{base['action'].iloc[0]}**. Select both actions in the filter to mix them; an Action column then appears.")
        right = {"Unit price", "Line total", "Short risk"}
        # list rows use a slightly smaller type so all fields and the reasons fit side by side
        center = {"Approve", "Action", "ABC", "Whs", "Country", "Order by", "Arrives", "Why"}
        # same look as every other table: bordered box, shaded header, thin row lines, hover shade
        box = st.container(border=True, key="po_list")
        with box.container(key="po_head"):
            hdr = _cols(W, "bottom")
            for c, t in zip(hdr, heads):
                al = "fp-ctr"  # every header centered over its column, as in the other tables
                c.markdown(f'<div class="fp-h fp-hw {al}" title="{"Chance of running short before this order arrives" if t == "Short risk" else t}">{t}</div>', unsafe_allow_html=True)

        def cell(col, text, al="", muted=False):
            col.markdown(esc_html(f'<div class="fp-c {al}{" fp-m" if muted else ""}">{text}</div>'), unsafe_allow_html=True)

        for i, x in page.iterrows():
            with box.container(key=f"po_row_{i}"):
                C = dict(zip(heads, _cols(W, "top")))
                k = f"{r.created}_{ver}_{i}"
                nos = x["supplier"] == "NO SUPPLIER"
                ap = C["Approve"].checkbox("Approve", value=bool(x["approve"]) and not nos, key=f"ap_{k}", label_visibility="collapsed",
                                      disabled=bool(nos), help="No supplier on file: pick one in SAP" if nos else None)
                cell(C["Item"], f'<b style="white-space:nowrap">{_h(x["item_code"])}</b>')
                cell(C["Item name"], _h(x.get("item_name", "")), muted=True)
                badge = "fp-b-now" if x["action"] == "Order now" else "fp-b-soon"
                if "Action" in C:
                    cell(C["Action"], f'<span class="fp-b {badge}">{_h(x["action"])}</span>', "fp-ctr")
                cell(C["ABC"], _h(x.get("abc", "")), "fp-ctr")
                cell(C["Whs"], _h(x["warehouse"]), "fp-ctr")
                low = bool(x.get("below_supplier_minimum", False))
                cell(C["Supplier"], _h(x["supplier"]) + ('<br><span class="fp-b fp-b-warn">Below min.</span>' if low else ""))
                cell(C["Country"], _h(x.get("supplier_country") or ""), "fp-ctr")
                q = C["Quantity"].number_input("Quantity", min_value=0.0, value=float(x["quantity"]), step=1.0, format="%.0f",
                                         key=f"q_{k}", label_visibility="collapsed")
                cell(C["Unit price"], money(float(x["unit_price"]), 2), "fp-num")
                cell(C["Line total"], f'<b>{money(q * float(x["unit_price"]))}</b>', "fp-num")
                cell(C["Order by"], pd.Timestamp(x["order_by"]).strftime("%m/%d/%y") if pd.notna(x.get("order_by")) else "", "fp-ctr")
                cell(C["Arrives"], pd.Timestamp(x["expected_arrival"]).strftime("%m/%d/%y") if pd.notna(x.get("expected_arrival")) else "", "fp-ctr")
                ch = x.get("chance_short_before_arrival")
                cell(C["Short risk"], f"{ch:.0%}" if pd.notna(ch) else "", "fp-num")
                why = "".join(f'<div class="fp-s">{_h(t)}</div>' for t in sentences(x["why"]))
                if show_all:
                    det = (f'Stock position {x.get("stock_position", 0):,.0f} · reorder point {x.get("reorder_point", 0):,.0f} · '
                           f'order up to {x.get("order_up_to", 0):,.0f} · {x.get("weeks_cover_after_receipt", 0):,.1f} weeks of cover after receipt')
                    why += f'<div class="fp-s fp-m">{_h(det)}</div>'
                    if isinstance(x.get("alternatives"), str) and x["alternatives"].strip():
                        why += "".join(f'<div class="fp-s fp-m">{_h(t)}</div>' for t in sentences("Other suppliers: " + x["alternatives"]))
                C["Why"].markdown(esc_html(f'<div class="fp-c fp-whycol fp-whyc"><div class="fp-whyb">{why}</div></div>'), unsafe_allow_html=True)
                store[i] = (bool(ap), float(q))
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
        st.caption(f"Open PO lines older than {days} days by PO date. They are not counted as incoming stock, because lines this old "
                   "are most likely overdue or were never closed. Confirm with the supplier or close them in SAP.")
        sd = stale.copy()
        sd["age_days"] = (r.as_of - sd["po_date"]).dt.days
        sd["po_date"] = sd["po_date"].dt.strftime("%m/%d/%y")
        if "po_entry" in sd:
            sd["po_entry"] = [str(int(v)) if pd.notna(v) and str(v).replace(".0", "").isdigit() else str(v) for v in sd["po_entry"]]
        cols = [c for c in ["po_entry", "line_num", "po_date", "age_days", "item_code", "supplier_code", "warehouse", "quantity",
                            "received_qty", "open_qty", "unit_price"] if c in sd.columns]
        sd = sd.sort_values("age_days", ascending=False)[cols]
        kpi_row([("Old open lines", f"{len(sd):,}", None, None), ("Units", f"{sd['open_qty'].sum():,.0f}", None, None),
                 ("Older than a year", f"{(sd['age_days'] > 365).sum():,}", None, None)])
        show_df(sd, hide_index=True, column_config={
            "po_entry": st.column_config.TextColumn("PO"), "line_num": st.column_config.NumberColumn("Line", format="%d"),
            "po_date": st.column_config.TextColumn("PO date"), "age_days": st.column_config.NumberColumn("Age (days)", format="%,.0f"),
            "item_code": st.column_config.TextColumn("Item"), "supplier_code": st.column_config.TextColumn("Supplier"),
            "warehouse": st.column_config.TextColumn("Whs"), "quantity": st.column_config.NumberColumn("Ordered", format="%,.0f"),
            "received_qty": st.column_config.NumberColumn("Received", format="%,.0f"),
            "open_qty": st.column_config.NumberColumn("Still open", format="%,.0f"),
            "unit_price": st.column_config.NumberColumn("Unit price", format="$%,.2f")})
