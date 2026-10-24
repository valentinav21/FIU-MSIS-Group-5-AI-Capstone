"""FORTPRO AI Planner: entry point. Run with:  streamlit run app/Home.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="FORTPRO AI Planner", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

P = Path(__file__).resolve().parent / "pages"
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
    "Sales growth (BP2)": [
        st.Page(str(P / "sales_plays.py"), title="Sales plays", icon=":material/bolt:"),
        st.Page(str(P / "customers.py"), title="Customers", icon=":material/groups:"),
        st.Page(str(P / "cross_sell.py"), title="Cross-sell and upsell", icon=":material/add_shopping_cart:"),
        st.Page(str(P / "markets.py"), title="Markets", icon=":material/public:"),
        st.Page(str(P / "pricing.py"), title="Pricing", icon=":material/sell:"),
        st.Page(str(P / "products.py"), title="Products and web demand", icon=":material/category:"),
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
