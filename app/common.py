"""Shared helpers for every page: results loading, formatting, charts, empty states."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import io as eio  # noqa: E402
from engine import pipeline  # noqa: E402

# palette (validated categorical order) and reserved status colors
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SAMPLE_FOLDER = "data/sample"


def page_setup(title: str, icon: str = ""):
    st.markdown(
        """<style>
        .block-container {padding-top: 1.6rem; max-width: 1400px;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        .fp-note {color:#52514e; font-size:0.9rem;}
        .fp-banner {background:#fff5e0; border:1px solid #fab219; padding:8px 12px; border-radius:8px; margin-bottom:10px; color:#0b0b0b;}
        .fp-card {border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:12px 14px; background:#ffffff;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title(title)


def _mtime():
    p = pipeline.results_path()
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _load_cached(mtime):
    return pipeline.load_results()


def results():
    return _load_cached(_mtime())


def require_results():
    r = results()
    if r is None:
        st.info("No results yet. Go to **Overview** and press **Run analysis**.")
        st.stop()
    if Path(r.data_folder).as_posix().rstrip("/").endswith(SAMPLE_FOLDER):
        st.markdown('<div class="fp-banner"><b>Sample data.</b> These results come from SYNTHETIC data built on the public '
                    'fortpro.com catalog. They are not FORTPRO numbers. Load your SAP export on the Overview page.</div>',
                    unsafe_allow_html=True)
    return r


def need(df, what: str):
    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        st.info(f"Nothing to show for {what}. The Data Health page lists which files this needs.")
        return False
    return True


# ---------------------------------------------------------------- formatting

def money(v, digits=0):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.2f}M"
    if a >= 1e4:
        return f"{sign}${a / 1e3:,.0f}K"
    return f"{sign}${a:,.{digits}f}"


def signed_money(v):
    """+$5.15M / -$1.20M, for changes."""
    m = money(v)
    return m if m.startswith("-") or m == "n/a" else "+" + m


def pct(v, digits=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def num(v, digits=0):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.{digits}f}"


def colcfg(df: pd.DataFrame, money_cols=(), pct_cols=(), num_cols=(), wide=(), unit_cols=()):
    cfg = {}
    for c in df.columns:
        if c in unit_cols:
            cfg[c] = st.column_config.NumberColumn(c, format="%,.0f")
            continue
        if c in money_cols:
            cfg[c] = st.column_config.NumberColumn(c, format="$%,.0f")
        elif c in pct_cols:
            cfg[c] = st.column_config.NumberColumn(c, format="percent")
        elif c in num_cols:
            cfg[c] = st.column_config.NumberColumn(c, format="%,.1f")
        elif c in wide:
            cfg[c] = st.column_config.TextColumn(c, width="large")
    return cfg


ACRONYMS = {"abc": "ABC", "po": "PO", "pos": "POs", "sap": "SAP", "eta": "ETA", "rfm": "RFM", "cv": "CV", "adi": "ADI",
            "cv2": "CV²", "p90": "P90", "12m": "12m", "90d": "90d", "id": "ID", "sku": "SKU", "moq": "MOQ"}


def label(c) -> str:
    """item_code -> Item code; leaves labels that are already readable alone."""
    c = str(c)
    if c != c.lower() or " " in c:
        return c
    words = c.split("_")
    out = [ACRONYMS.get(w, w) for w in words]
    first = out[0]
    out[0] = first if first in ACRONYMS.values() else first.capitalize()
    return " ".join(out)


def esc(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not read '$...$' as a math formula."""
    return str(text).replace("$", "\\$")


def esc_html(text: str) -> str:
    """Same, for text inside raw HTML (a backslash would show there)."""
    return str(text).replace("$", "&#36;")


PAGE_SIZES = (25, 50, 100, 250)


def paged(df: pd.DataFrame, key: str, default: int = 50) -> pd.DataFrame:
    """Rows for the current page. Tables grow to fit their rows (the page scrolls, not the grid) and big
    tables are split into pages, so the grid never needs its own vertical scrollbar."""
    n = len(df)
    if n <= 60:  # short tables are shown whole
        return df
    c1, c2, c3 = st.columns([1, 1, 4])
    size = c1.selectbox("Rows per page", PAGE_SIZES, index=PAGE_SIZES.index(default) if default in PAGE_SIZES else 0,
                        key=f"{key}_size")
    pages = max(int(np.ceil(n / size)), 1)
    page = c2.number_input(f"Page (of {pages:,})", min_value=1, max_value=pages, value=1, step=1, key=f"{key}_page")
    lo = (int(page) - 1) * size
    c3.markdown(f'<p class="fp-note" style="margin-top:2.1rem">Rows {lo + 1:,} to {min(lo + size, n):,} of {n:,}. '
                'Sorting a column sorts this page; the download has every row.</p>', unsafe_allow_html=True)
    return df.iloc[lo:lo + size]


def _key_for(df, key):
    return key or "tbl_" + str(abs(hash((tuple(map(str, df.columns)), len(df)))) % 10 ** 9)


def _one_decimal_percent(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Streamlit's 'percent' format shows two decimals; show every share the same way: 87.0%."""
    for c, v in list(cfg.items()):
        tc = v.get("type_config", {}) if isinstance(v, dict) else {}
        if tc.get("type") == "number" and tc.get("format") == "percent" and c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce") * 100
            cfg[c] = st.column_config.NumberColumn(v.get("label") or c, format="%.1f%%", help=v.get("help"))
    return d


def show_df(df, column_config=None, height=None, key=None, **_):
    """st.dataframe with readable headers, no index and full width. Grows to fit its rows (paged when long),
    so it scrolls with the page and the fullscreen button shows the whole grid."""
    cfg = {label(k): v for k, v in (column_config or {}).items()}
    k = _key_for(df, key)
    d = paged(df, k).rename(columns={c: label(c) for c in df.columns})
    d = _one_decimal_percent(d.copy(), cfg)
    st.dataframe(d, hide_index=True, width="stretch", height="content", column_config=cfg, key=key)


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Readable column headers: item_code -> Item code."""
    out = df.copy()
    out.columns = [str(c).replace("_", " ").strip().capitalize() if c == str(c).lower() else c for c in out.columns]
    return out


def table(df, height=None, money_cols=(), pct_cols=(), num_cols=(), wide=(), key=None, unit_cols=()):
    d = pretty(df)
    m = [str(c).replace("_", " ").capitalize() if str(c) == str(c).lower() else c for c in money_cols]
    p = [str(c).replace("_", " ").capitalize() if str(c) == str(c).lower() else c for c in pct_cols]
    n = [str(c).replace("_", " ").capitalize() if str(c) == str(c).lower() else c for c in num_cols]
    w = [str(c).replace("_", " ").capitalize() if str(c) == str(c).lower() else c for c in wide]
    k = _key_for(df, key)
    d = paged(d, k).copy()
    u = [str(c).replace("_", " ").capitalize() if str(c) == str(c).lower() else c for c in unit_cols]
    cfg = colcfg(d, m, p, n, w, u)
    d = _one_decimal_percent(d, cfg)
    st.dataframe(d, hide_index=True, width="stretch", height="content", column_config=cfg, key=key)


def download_csv(df, name, label="Download CSV", key=None):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=name, mime="text/csv", key=key or f"dl_{name}")


# ---------------------------------------------------------------- charts

def base_layout(fig, title=None, height=360, x_title=None, y_title=None, legend=True):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height, margin=dict(l=10, r=10, t=44 if title else 10, b=60 if legend else 10),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
        hoverlabel=dict(bgcolor="white", font_size=12), showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", title=x_title, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title, tickfont=dict(color=MUTED))
    return fig


def show(fig):
    # map outlines are served from app/static so maps work without internet
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                                  "topojsonURL": "/app/static/topojson/"})


def bar(df, x, y, title=None, color=BLUE, horizontal=False, fmt=None, height=360, x_title=None, y_title=None):
    if horizontal:
        fig = go.Figure(go.Bar(x=df[y], y=df[x], orientation="h", marker=dict(color=color, cornerradius=4),
                               hovertemplate=f"%{{y}}<br>%{{x:{fmt or ',.0f'}}}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure(go.Bar(x=df[x], y=df[y], marker=dict(color=color, cornerradius=4),
                               hovertemplate=f"%{{x}}<br>%{{y:{fmt or ',.0f'}}}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    return base_layout(fig, title, height, x_title, y_title, legend=False)


def kpi_row(items):
    """items: (label, value, delta, help[, delta_color]). delta_color: normal (up is good), inverse, off."""
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        label, value, delta, help_ = it[:4]
        color = it[4] if len(it) > 4 else ("normal" if delta and any(k in str(delta) for k in ("pts", "x ")) else "off")
        # an arrow only for signed changes (+8.8 pts, -$1.2M); plain notes such as "$8.28M" or "lower than baseline" get none
        arrow = "auto" if delta is not None and str(delta).strip()[:1] in "+-" else "off"
        with c:
            st.metric(label, value, delta=delta, help=help_, border=True, delta_color=color, delta_arrow=arrow)


def settings():
    return eio.load_settings()


def is_sample(folder) -> bool:
    return Path(str(folder)).as_posix().rstrip("/").endswith(SAMPLE_FOLDER)
