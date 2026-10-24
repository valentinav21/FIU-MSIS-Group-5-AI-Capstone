import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import show_df, BLUE, MUTED, ORANGE, base_layout, download_csv, kpi_row, need, page_setup, pct, require_results, show  # noqa: E402
from engine.inventory import calibrated_samples  # noqa: E402
from engine.probabilistic import simulate_periods  # noqa: E402

page_setup("Demand forecast")
st.markdown('<p class="fp-note">Each item and warehouse gets the method that tested best on its own history, from a library '
            'built for spare parts: Croston, SBA, TSB, temporal aggregation, exponential smoothing, a seasonal variant and a '
            'gradient boosted model trained across all items. Accuracy below is measured on the latest weeks, which neither '
            'method selection nor calibration saw.</p>', unsafe_allow_html=True)
r = require_results()
ft = r.forecast_table
if not need(ft, "forecasts"):
    st.stop()
tab1, tab2 = st.tabs(["Item explorer", "Accuracy and methods"])
with tab2:
    acc = r.forecast_accuracy
    if len(acc):
        allr = acc[acc["Demand class"] == "All items"].iloc[0]
        kpi_row([("Forecast error, AI (WAPE)", pct(allr["Error (WAPE) AI"]), f"{-allr['Error reduction'] * 100:.1f}% vs baseline", "Weighted absolute error of 8 week totals, by revenue. Lower is better."),
                 ("Forecast error, moving average", pct(allr["Error (WAPE) baseline"]), None, "13 period moving average, the kind of static model SAP MRP uses."),
                 ("Bias, AI", pct(allr["Bias AI"]), None, "Positive means forecasts ran high."),
                 ("Bias, moving average", pct(allr["Bias baseline"]), None, None)])
        show_df(acc, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in acc.columns if c not in ("Demand class", "Series")})
        st.caption("WAPE on intermittent items looks high for every method because single weeks are mostly zero. What matters "
                   "for stock is the range, checked next.")
    cal = r.calibration
    if len(cal):
        st.markdown("**Are the ranges honest?** For each demand pattern: how often the real 8 week demand fell at or below the "
                    "forecast's 80th, 90th and 95th percentile. Calibrated values should sit close to 80%, 90% and 95%.")
        show_df(cal, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in cal.columns if "Hit rate" in c})
    mix = ft["method"].value_counts().reset_index()
    mix.columns = ["Method", "Items x warehouses"]
    cls = ft["demand_class"].value_counts().reset_index()
    cls.columns = ["Demand pattern", "Items x warehouses"]
    c1, c2 = st.columns(2)
    c1.dataframe(mix, hide_index=True, width="stretch")
    c2.dataframe(cls, hide_index=True, width="stretch")
    if r.season_categories:
        st.caption("Seasonal profile found in: " + ", ".join(r.season_categories))
with tab1:
    ft = ft.copy()
    ft["label"] = ft["item_code"] + " · " + ft["warehouse"] + " · " + ft["demand_class"]
    names = r.items.set_index("item_code")["item_name"]
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("Search item code or name", "")
    only = c2.selectbox("Show", ["Top sellers first", "All"], index=0)
    d = ft
    if q:
        m = ft["item_code"].str.contains(q, case=False, na=False) | ft["item_code"].map(names).astype(str).str.contains(q, case=False, na=False)
        d = ft[m]
    if only.startswith("Top"):
        d = d.sort_values("last_52_actual", ascending=False)
    if len(d) == 0:
        st.warning("No item matches.")
        st.stop()
    pick = st.selectbox("Item and warehouse", d["label"].head(500).tolist())
    i = int(ft.index[ft["label"] == pick][0])
    row = ft.loc[i]
    st.markdown(f"**{row.item_code}** {names.get(row.item_code, '')} · warehouse **{row.warehouse}** · "
                f"pattern **{row.demand_class}** (sells in {1 / max(row.adi, 1e-9):.0%} of periods) · method **{row.method}**")
    st.info(row.why)
    hist = r.history[i]
    T = len(hist)
    show_n = min(T, 104)
    xh = r.period_end[-show_n:]
    H = r.forecast_path.shape[1]
    xf = [r.as_of + pd.Timedelta(days=r.period_days * (h + 1)) for h in range(H)]
    sp = r.sim_params[i]
    rng = np.random.default_rng(0)
    sims = simulate_periods(sp["sizes"], sp["occ"], r.forecast_path[i], 2000, float(r.level_sigma.get(row.demand_class, 0.25)), rng)
    cum = np.cumsum(sims, axis=1)

    class _F:  # minimal object for the calibration helper
        quantile_map = r.quantile_map
    lo = np.array([np.quantile(calibrated_samples(_F, row.demand_class, sims[:, h]), 0.1) for h in range(H)])
    hi = np.array([np.quantile(calibrated_samples(_F, row.demand_class, sims[:, h]), 0.9) for h in range(H)])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xh, y=hist[-show_n:], name="Actual demand", marker=dict(color="#c3c2b7", cornerradius=3)))
    fig.add_trace(go.Scatter(x=xf + xf[::-1], y=list(hi) + list(lo[::-1]), fill="toself", fillcolor="rgba(42,120,214,0.15)",
                             line=dict(width=0), name="Likely range (10th to 90th percentile)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=r.forecast_path[i], mode="lines", name="Forecast (average)", line=dict(color=BLUE, width=2),
                             hovertemplate="%{x|%b %d, %Y}<br>Forecast %{y:.2f}<extra></extra>"))
    show(base_layout(fig, f"{'Weekly' if r.bucket == 'W' else 'Monthly'} demand: last {show_n} periods and next {H}", height=400, y_title="Units"))
    tot = cum[:, min(7, H - 1)]
    tq = np.quantile(calibrated_samples(_F, row.demand_class, tot), [0.1, 0.5, 0.9])
    kpi_row([("Next 8 periods, average", f"{r.forecast_path[i][:8].sum():,.1f}", None, None),
             ("Likely range, next 8", f"{tq[0]:,.0f} to {tq[2]:,.0f}", None, "Calibrated 10th to 90th percentile of total demand."),
             ("Last 13 periods actual", f"{row.last_13_actual:,.0f}", None, None),
             ("Last 52 periods actual", f"{row.last_52_actual:,.0f}", None, None)])
    cols = ["item_code", "warehouse", "demand_class", "method", "forecast_next_4", "forecast_next_8", "forecast_next_13",
            "forecast_next_26", "last_13_actual", "last_52_actual", "adi", "cv2", "why"]
    cols = [c for c in cols if c in ft.columns]
    with st.expander("All forecasts (table and download)"):
        show_df(ft[cols], hide_index=True, width="stretch", height=420)
        download_csv(ft[cols], "forecasts.csv")
