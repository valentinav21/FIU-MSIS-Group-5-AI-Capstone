import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from common import page_setup, settings  # noqa: E402
from engine import io as eio  # noqa: E402

page_setup("Settings")
st.markdown('<p class="fp-note">Changes are saved to config/settings_local.yaml (config/settings.yaml keeps the documented defaults) '
            'and take effect on the next run.</p>', unsafe_allow_html=True)
cfg = settings()
inv, fc, pur, al, sp, dt = cfg["inventory"], cfg["forecast"], cfg["purchasing"], cfg["alerts"], cfg["sales"], cfg["data"]

with st.form("settings"):
    st.subheader("Service and inventory")
    c1, c2, c3, c4 = st.columns(4)
    sl = dict(inv.get("service_level", {}))
    sl["A"] = c1.slider("Target for A items", 0.80, 0.995, float(sl.get("A", 0.98)), 0.005, format="%.3f", help="A items make up the top 80% of revenue.")
    sl["B"] = c2.slider("Target for B items", 0.70, 0.995, float(sl.get("B", 0.96)), 0.005, format="%.3f")
    sl["C"] = c3.slider("Target for C items", 0.50, 0.99, float(sl.get("C", 0.92)), 0.005, format="%.3f")
    sl["new"] = c4.slider("Target for new items", 0.50, 0.99, float(sl.get("new", 0.90)), 0.005, format="%.3f")
    c1, c2, c3 = st.columns(3)
    measure = c1.selectbox("Service measured as", ["fill_rate", "cycle"], index=0 if inv.get("service_measure", "fill_rate") == "fill_rate" else 1,
                           format_func=lambda v: "Fill rate: share of units shipped from stock" if v == "fill_rate" else "Cycle service: chance of no stockout per cycle")
    modes = ["off", "inventory", "current"]
    cur_mode = str(inv.get("budget_mode", "off"))
    mode = c2.selectbox("Budget mode", modes + ([cur_mode] if cur_mode not in modes else []),
                        index=(modes + [cur_mode]).index(cur_mode) if cur_mode in modes + [cur_mode] else 0,
                        format_func=lambda v: {"off": "Off: use the targets above", "inventory": "Keep today's inventory dollars",
                                               "current": "Keep today's safety stock dollars"}.get(v, f"Inventory budget ${float(v):,.0f}" if v.replace('.', '', 1).isdigit() else v),
                        help="With a budget, the planner places the money item by item where it serves the most revenue, instead of using fixed targets.")
    custom = c3.number_input("Or a fixed inventory budget ($, 0 = not used)", min_value=0.0, value=0.0, step=100000.0)
    c1, c2, c3, c4 = st.columns(4)
    h = c1.number_input("Holding cost per year (% of item cost)", 1.0, 80.0, float(inv.get("holding_cost_pct_per_year", 0.25)) * 100, 1.0)
    k = c2.number_input("Cost per PO line ($)", 0.0, 1000.0, float(inv.get("order_cost_usd", 25)), 5.0)
    rv = c3.number_input("Review every (days)", 1, 60, int(inv.get("review_period_days", 7)))
    mx = c4.number_input("Max weeks of demand per order", 2, 104, int(inv.get("max_order_cover_weeks", 26)))
    c1, c2, c3 = st.columns(3)
    ow = c1.number_input("Overstock when cover exceeds (weeks)", 4, 156, int(inv.get("overstock_weeks", 26)))
    dl = c2.number_input("Fallback lead time (days)", 1, 365, int(inv.get("default_lead_time_days", 75)))
    soc = c3.number_input("Cost of a unit short (% of selling price)", 0.0, 300.0, float(pur.get("stockout_cost_pct_of_price", 0.3)) * 100, 5.0,
                          help="Lost margin and goodwill per unit you cannot ship. Used to choose between suppliers.")
    st.subheader("Forecast")
    c1, c2, c3, c4 = st.columns(4)
    bucket = c1.selectbox("Planning bucket", ["W", "M"], index=0 if fc.get("bucket", "W") == "W" else 1, format_func=lambda v: "Weekly" if v == "W" else "Monthly")
    ml = c2.checkbox("Try the gradient boosted model", bool(fc.get("use_ml_model", True)))
    cap = c3.number_input("Cap one-off lines at x median (0 = off)", 0.0, 100.0, float(fc.get("spike_cap_multiple", 0)), 1.0)
    dead = c4.number_input("Inactive after no sale for (periods)", 13, 260, int(fc.get("dead_after_periods", 52)))
    st.subheader("Data")
    c1, c2, c3 = st.columns(3)
    asof = c1.text_input("As-of date (blank = last sales date)", str(dt.get("as_of_date") or ""))
    hs = c2.text_input("Ignore sales before (YYYY-MM-DD, optional)", str(dt.get("history_start") or ""))
    dw = c3.text_input("Default warehouse code", str(dt.get("default_warehouse") or "MIA"))
    net = st.checkbox("Subtract customer returns from demand", bool(dt.get("net_returns_from_demand", False)))
    st.subheader("Alerts and sales")
    c1, c2, c3, c4 = st.columns(4)
    thr = c1.slider("Alert when chance of running out exceeds", 0.05, 0.9, float(al.get("stockout_probability_threshold", 0.3)), 0.05)
    cw = c2.number_input("Churn window (days)", 30, 365, int(sp.get("churn_window_days", 90)))
    rpc = c3.number_input("Suggestions per customer", 3, 30, int(sp.get("recommendations_per_customer", 10)))
    mpc = c4.number_input("Max price change suggested (%)", 1, 20, int(sp.get("max_price_change_pct", 5)))
    saved = st.form_submit_button("Save settings", type="primary")
if saved:
    inv.update({"service_level": sl, "service_measure": measure, "budget_mode": (str(int(custom)) if custom > 0 else mode),
                "holding_cost_pct_per_year": h / 100, "order_cost_usd": k, "review_period_days": int(rv), "max_order_cover_weeks": int(mx),
                "overstock_weeks": int(ow), "default_lead_time_days": int(dl)})
    pur["stockout_cost_pct_of_price"] = soc / 100
    fc.update({"bucket": bucket, "use_ml_model": bool(ml), "spike_cap_multiple": float(cap), "dead_after_periods": int(dead)})
    dt.update({"as_of_date": asof.strip(), "history_start": hs.strip(), "default_warehouse": dw.strip() or "MIA", "net_returns_from_demand": bool(net)})
    al["stockout_probability_threshold"] = float(thr)
    sp.update({"churn_window_days": int(cw), "recommendations_per_customer": int(rpc), "max_price_change_pct": int(mpc)})
    eio.save_settings(cfg)
    st.success("Saved. Run the analysis again on the Overview page to apply.")
with st.expander("Restore defaults"):
    if st.button("Restore all settings to defaults"):
        eio.reset_settings(keep_folder=cfg["data"]["folder"])
        st.success("Defaults restored.")
