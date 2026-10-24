"""FORTPRO AI Planner: entry point. Run with:  streamlit run app/Home.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="FORTPRO AI Planner", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

from engine import io as eio  # noqa: E402

P = Path(__file__).resolve().parent / "pages"
BP2 = bool(eio.load_settings().get("features", {}).get("bp2_enabled", False))


def bp2_page(file, title, icon):
    """BP2 pages run normally when enabled; otherwise they show a TBD notice."""
    if BP2:
        return st.Page(str(P / file), title=title, icon=icon)

    def tbd():
        st.title(f"{title} (TBD)")
        st.info("**To be determined.** Group 5 is tackling **BP1 Purchasing** for the capstone. "
                "BP2 Sales growth is out of scope for now, so this page is turned off.")
        st.caption("To turn BP2 back on, set features: bp2_enabled: true in config/settings.yaml and run the analysis again.")

    tbd.__name__ = "tbd_" + Path(file).stem
    return st.Page(tbd, title=f"{title} (TBD)", icon=":material/block:", url_path=Path(file).stem)


pages = {
    "Start": [
        st.Page(str(P / "overview.py"), title="Overview", icon=":material/dashboard:", default=True),
        st.Page(str(P / "data_health.py"), title="Data health", icon=":material/fact_check:"),
    ],
    "Purchasing (BP1)": [
        st.Page(str(P / "purchase_orders.py"), title="Purchase orders", icon=":material/shopping_cart:"),
        st.Page(str(P / "risk_alerts.py"), title="Risk alerts", icon=":material/warning:"),
        st.Page(str(P / "inventory_policy.py"), title="Stock levels", icon=":material/inventory_2:"),
        st.Page(str(P / "forecast.py"), title="Demand forecast", icon=":material/insights:"),
        st.Page(str(P / "suppliers.py"), title="Suppliers", icon=":material/local_shipping:"),
    ],
    "Sales growth (BP2)" + ("" if BP2 else " (TBD)"): [
        bp2_page("sales_plays.py", "Sales plays", ":material/bolt:"),
        bp2_page("customers.py", "Customers", ":material/groups:"),
        bp2_page("cross_sell.py", "Cross-sell and upsell", ":material/add_shopping_cart:"),
        bp2_page("markets.py", "Markets", ":material/public:"),
        bp2_page("pricing.py", "Pricing", ":material/sell:"),
        bp2_page("products.py", "Products and web demand", ":material/category:"),
    ],
    "Setup": [
        st.Page(str(P / "settings_page.py"), title="Settings", icon=":material/tune:"),
        st.Page(str(P / "help.py"), title="How it works", icon=":material/help:"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.caption("FORTPRO AI Planner · runs on this computer · data never leaves it")
nav.run()
