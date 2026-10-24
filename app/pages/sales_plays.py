import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from common import show_df, download_csv, kpi_row, money, need, page_setup, require_results  # noqa: E402

page_setup("Sales plays")
st.markdown('<p class="fp-note">One ranked list of what each rep should do this week: win back customers likely to leave, '
            'grow share where a customer under-buys a line their peers buy, cross-sell items they are likely to need, and review '
            'prices well below what similar customers pay. Value is an estimate of what the play is worth.</p>', unsafe_allow_html=True)
r = require_results()
S = r.sales
if S is None or not need(S.plays, "sales plays"):
    st.stop()
p = S.plays
kpi_row([(k, f"{(p['play'] == k).sum():,}", money(p.loc[p['play'] == k, 'value'].sum()), None) for k in p["play"].unique()])
c1, c2, c3 = st.columns(3)
play = c1.multiselect("Play", sorted(p["play"].unique()))
reps = sorted(p["sales_rep"].dropna().astype(str).unique()) if "sales_rep" in p else []
rep = c2.multiselect("Sales rep", reps)
seg = c3.multiselect("Segment", sorted(p["segment"].dropna().unique()) if "segment" in p else [])
d = p.copy()
if play:
    d = d[d["play"].isin(play)]
if rep:
    d = d[d["sales_rep"].astype(str).isin(rep)]
if seg:
    d = d[d["segment"].isin(seg)]
show_df(d, hide_index=True, width="stretch", height=560,
             column_config={"value": st.column_config.NumberColumn("Value", format="$%,.0f"),
                            "detail": st.column_config.TextColumn("Detail", width="large")})
download_csv(d, "sales_plays.csv")
