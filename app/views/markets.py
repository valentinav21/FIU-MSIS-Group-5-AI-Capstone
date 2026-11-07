import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, base_layout, download_csv, is_sample, need, page_setup, require_results, show  # noqa: E402

ISO3 = {"US": "USA", "DO": "DOM", "CO": "COL", "PA": "PAN", "GT": "GTM", "HN": "HND", "CR": "CRI", "EC": "ECU", "PE": "PER",
        "JM": "JAM", "HT": "HTI", "TT": "TTO", "BS": "BHS", "SV": "SLV", "NI": "NIC", "CL": "CHL", "VE": "VEN", "MX": "MEX",
        "PR": "PRI", "BR": "BRA", "AR": "ARG", "BO": "BOL", "PY": "PRY", "UY": "URY", "BZ": "BLZ", "GY": "GUY", "SR": "SUR",
        "CU": "CUB", "BB": "BRB", "AW": "ABW", "CW": "CUW", "KY": "CYM", "LC": "LCA", "GD": "GRD", "VC": "VCT", "AG": "ATG",
        "DM": "DMA", "KN": "KNA", "CA": "CAN", "ES": "ESP", "NG": "NGA", "GH": "GHA", "SA": "SAU", "AE": "ARE"}
SEQ = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]]

page_setup("Markets and geography")
st.markdown('<p class="fp-note">Revenue by US state and export country. Add a market_potential file (for example registered heavy '
            'trucks or trucking establishments by state, from public sources) to see penetration and untapped regions.</p>',
            unsafe_allow_html=True)
r = require_results()
S = r.sales
if S is None or not need(S.geo, "geography"):
    st.stop()
g = S.geo.copy()
has_pot = "potential" in g.columns
us = g[(g["country"] == "US") & g["region"].astype(str).str.fullmatch(r"[A-Z]{2}")]
if len(us):
    metric = st.radio("Color states by", ["Revenue last 12 months", "Growth vs prior year"] + (["Penetration vs market potential"] if has_pot else []), horizontal=True)
    col = {"Revenue last 12 months": "revenue_12m", "Growth vs prior year": "growth", "Penetration vs market potential": "penetration_index"}[metric]
    z = us[col].astype(float)
    if col == "growth":
        cs = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
        lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
        zmin, zmax = -lim, lim
    elif col == "penetration_index":
        cs = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
        zmin, zmax = 0, 2
    else:
        cs, zmin, zmax = SEQ, None, None
    fig = go.Figure(go.Choropleth(locations=us["region"], z=z, locationmode="USA-states", colorscale=cs, zmin=zmin, zmax=zmax,
                                  marker_line_color="#fcfcfb", colorbar=dict(title=""),
                                  customdata=np.stack([us["revenue_12m"], us["customers"], us["growth"].fillna(0)], axis=1),
                                  hovertemplate="%{location}<br>Revenue $%{customdata[0]:,.0f}<br>%{customdata[1]:.0f} customers<br>Growth %{customdata[2]:+.0%}<extra></extra>"))
    fig.update_geos(scope="usa", bgcolor="#fcfcfb", lakecolor="#fcfcfb")
    show(base_layout(fig, f"United States: {metric.lower()}", height=460, legend=False))
intl = g[g["country"] != "US"].copy()
if len(intl):
    intl["iso3"] = intl["country"].map(ISO3)
    m = intl.dropna(subset=["iso3"])
    fig = go.Figure(go.Choropleth(locations=m["iso3"], z=m["revenue_12m"], colorscale=SEQ, marker_line_color="#fcfcfb",
                                  customdata=np.stack([m["customers"].fillna(0), m["growth"].fillna(0)], axis=1),
                                  hovertext=m["country"],
                                  hovertemplate="%{hovertext}<br>Revenue $%{z:,.0f}<br>%{customdata[0]:.0f} customers<br>Growth %{customdata[1]:+.0%}<extra></extra>"))
    fig.update_geos(projection_type="natural earth", lataxis_range=[-58, 50], lonaxis_range=[-125, -30], bgcolor="#fcfcfb",
                    showcountries=True, countrycolor="#e1e0d9")
    show(base_layout(fig, "Export markets: revenue last 12 months", height=460, legend=False))
cols = [c for c in ["country", "region", "flag", "revenue_12m", "revenue_prior_12m", "growth", "customers", "revenue_per_customer",
                    "potential", "penetration_index", "opportunity"] if c in g.columns]
show_df(g[cols], hide_index=True, width="stretch", height=440,
             column_config={k: st.column_config.NumberColumn(format="$%,.0f") for k in ("revenue_12m", "revenue_prior_12m", "revenue_per_customer", "opportunity")}
             | {"growth": st.column_config.NumberColumn("Growth", format="percent"),
                "penetration_index": st.column_config.NumberColumn("Penetration index", format="%.2f", help="Share of FORTPRO revenue divided by share of market potential. 1.0 = in line.")})
download_csv(g[cols], "geography.csv")
if has_pot:
    st.caption("Opportunity = revenue the region would bring at the median penetration rate, minus revenue today. "
               "Untapped = potential but no customers yet.")
if has_pot and is_sample(r.data_folder):
    st.caption("The sample market potential figures are placeholders. Replace them with public data before using this view.")
