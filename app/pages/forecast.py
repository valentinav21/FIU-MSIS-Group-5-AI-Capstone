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
r = require_results()
monthly = r.bucket == "M"
U = "month" if monthly else "week"
st.markdown('<p class="fp-note">Methods are tested on rolling windows the model never saw, from a library built for spare parts: '
            'moving average, exponential smoothing, Croston, SBA, TSB, temporal aggregation, a combination, a trended yearly '
            'average, seasonal adjustment and a gradient boosted model trained across all items. A method only replaces the '
            f'baseline when it also misses fewer units. Accuracy below is measured on the latest {U}s, which neither method '
            'selection nor calibration saw.</p>', unsafe_allow_html=True)
ft = r.forecast_table
if not need(ft, "forecasts"):
    st.stop()
tab1, tab2 = st.tabs(["Item explorer", "Accuracy and methods"])
with tab2:
    acc = r.forecast_accuracy
    if len(acc):
        allr = acc[acc["Demand class"] == "All items"].iloc[0]
        im_ai, im_b = f"Units error, item x {U}, AI", f"Units error, item x {U}, baseline"
        if im_ai in acc:
            dd = allr[im_ai] - allr[im_b]
            kpi_row([(f"Units error per item and {U}, AI", f"{allr[im_ai]:.2f}",
                      f"{abs(dd):.3f} {'lower' if dd < 0 else 'higher'} than baseline" if abs(dd) >= 0.0005 else "same as baseline",
                      f"Sum of |actual - forecast| over item x {U} cells, divided by actual units: the measure the team scores SAP with "
                      "(SAP 1.05, flat 12-month average 0.86 on FORTPRO data). Lower is better."),
                     (f"Units error per item and {U}, baseline", f"{allr[im_b]:.2f}", None,
                      "Flat average of the last 12 months." if monthly else "13-week moving average.")])
        win = f"{int(r.settings['forecast'].get('backtest_horizon', 3))} {U}"
        dw = (allr["Error (WAPE) AI"] - allr["Error (WAPE) baseline"]) * 100
        kpi_row([("Forecast error, AI (WAPE)", pct(allr["Error (WAPE) AI"]),
                  f"{abs(dw):.1f} pts {'lower' if dw < 0 else 'higher'} than baseline" if abs(dw) >= 0.05 else "same as baseline",
                  f"Absolute error of {win} totals, weighted by revenue. Lower is better.", "off"),
                 ("Forecast error, baseline", pct(allr["Error (WAPE) baseline"]), None,
                  "Flat 12-month average, the simple benchmark the team scores against." if monthly else "13-week moving average."),
                 ("Bias, AI", pct(allr["Bias AI"]), None, "Positive means forecasts ran high."),
                 ("Bias, baseline", pct(allr["Bias baseline"]), None, None)])
        acc = acc.assign(**{"Demand class": acc["Demand class"].str.capitalize()})
        show_df(acc, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent" if "Units error" not in c else "%.2f")
                                    for c in acc.columns if c not in ("Demand class", "Series")})
        st.caption(f"Error on intermittent items looks high for every method because single {U}s are often zero. What matters "
                   "for stock is the range, checked next.")
    cal = r.calibration
    if len(cal):
        st.markdown(f"**Are the ranges honest?** For each demand pattern: how often the real demand over a test window fell at or below the "
                    "forecast's 80th, 90th and 95th percentile. Calibrated values should sit close to 80%, 90% and 95%.")
        cal = cal.assign(**{"Demand class": cal["Demand class"].str.capitalize()}) if "Demand class" in cal else cal
        show_df(cal, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in cal.columns if "Hit rate" in c})
    mix = ft["method"].value_counts().reset_index()
    mix.columns = ["Method", "Items x warehouses"]
    cls = ft["demand_class"].str.capitalize().value_counts().reset_index()
    cls.columns = ["Demand pattern", "Items x warehouses"]
    st.markdown("**Which methods won, and how demand is spread across patterns**")
    c1, c2 = st.columns([3, 2])
    with c1:
        show_df(mix, column_config={"Method": st.column_config.TextColumn(width="large"),
                                    "Items x warehouses": st.column_config.NumberColumn(format="%,d", width="small")})
    with c2:
        show_df(cls, column_config={"Items x warehouses": st.column_config.NumberColumn(format="%,d", width="small")})
    if r.season_categories:
        st.caption("Seasonal profile found in: " + "; ".join(r.season_categories) + ".")
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
                f"pattern **{row.demand_class}** (sells in {1 / max(row.adi, 1e-9):.0%} of {U}s) · method **{row.method}**")
    st.info(row.why)
    hist = r.history[i]
    T = len(hist)
    show_n = min(T, 104)
    xh = r.period_end[-show_n:]
    H = r.forecast_path.shape[1]
    if monthly and getattr(r, "first_forecast_start", None) is not None:
        xf = [r.period_end[-1] + pd.offsets.MonthEnd(h + 1) for h in range(H)]  # month ends, like the history bars
    else:
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
    show(base_layout(fig, f"{'Weekly' if r.bucket == 'W' else 'Monthly'} demand: last {show_n} {U}s and next {H}", height=400, y_title="Units"))
    nn = min(H, 9 if monthly else 8)
    tot = cum[:, nn - 1]
    tq = np.quantile(calibrated_samples(_F, row.demand_class, tot), [0.1, 0.5, 0.9])
    kpi_row([(f"Next {nn} {U}s, expected total", f"{r.forecast_path[i][:nn].sum():,.0f}", None, "Sum of the average forecast."),
             (f"Likely range, next {nn}", f"{tq[0]:,.0f} to {tq[2]:,.0f}", None, "Calibrated 10th to 90th percentile of total demand."),
             ("Last 3 months actual" if monthly else "Last 13 weeks actual", f"{row.last_13_actual:,.0f}", None, None),
             ("Last 12 months actual" if monthly else "Last 52 weeks actual", f"{row.last_52_actual:,.0f}", None, None)])
    cols = ["item_code", "warehouse", "demand_class", "at_risk", "method", "forecast_next_3", "forecast_next_4", "forecast_next_6",
            "forecast_next_8", "forecast_next_9", "forecast_next_12", "forecast_next_13", "forecast_next_26",
            "last_13_actual", "last_52_actual", "periods_since_sale", "adi", "cv2", "why"]
    cols = [c for c in cols if c in ft.columns]
    with st.expander("All forecasts (table and download)"):
        show_df(ft[cols], hide_index=True, width="stretch", height=420)
        download_csv(ft[cols], "forecasts.csv")
