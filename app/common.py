"""Shared helpers for every page: results loading, formatting, charts, empty states."""
from __future__ import annotations

import html as _html
import re
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
try:
    import inspect as _inspect
    _METRIC_ARROW = "delta_arrow" in _inspect.signature(st.metric).parameters
except (TypeError, ValueError):
    _METRIC_ARROW = False


def page_setup(title: str, icon: str = ""):
    st.markdown(
        """<style>
        .block-container {padding-top: 1.6rem; max-width: 1680px; padding-left: 2.2rem; padding-right: 2.2rem;}
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {font-size: 2.1rem !important; font-weight: 600;
            white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.2;}
        [data-testid="stMetricLabel"] p {font-size: 0.98rem !important; line-height: 1.3;}
        [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {font-size: 0.95rem !important;}
        .fp-note {color:#52514e; font-size:0.9rem;}
        .fp-h {font-weight:600; font-size:0.84rem; color:#3d3a34; white-space:nowrap;}
        .st-key-po_list {max-width:1640px; margin-left:auto; margin-right:auto; overflow-x:auto !important; padding:0 !important; gap:0 !important; border-color:#e6e4df !important; border-radius:10px !important; overflow:hidden;}
        /* PO list: same frame as the tables. Padding sits on the column row so text never touches the lines */
        .st-key-po_head, [class*="st-key-po_row_"] {padding:0 !important; gap:0 !important;}
        .st-key-po_head {background:#f6f5f2; border-bottom:1px solid #e6e4df;}
        [class*="st-key-po_row_"] {border-bottom:1px solid #efede8;}
        [class*="st-key-po_row_"]:hover {background:#faf9f6;}
        .st-key-po_head [data-testid="stHorizontalBlock"], [class*="st-key-po_row_"] [data-testid="stHorizontalBlock"] {
            align-items:stretch !important; gap:0 !important; min-width:1240px;}  /* below this the list scrolls sideways instead of squeezing */
        .st-key-po_head, [class*="st-key-po_row_"] {min-width:1240px;}
        .st-key-po_head [data-testid="stColumn"], [class*="st-key-po_row_"] [data-testid="stColumn"] {padding:0.55rem 0.38rem;}
        [class*="st-key-po_row_"] [data-testid="stColumn"] {padding-bottom:0.85rem;}
        [class*="st-key-po_row_"] [data-testid="stMarkdownContainer"], [class*="st-key-po_row_"] [data-testid="stElementContainer"] {margin-bottom:0 !important;}
        [class*="st-key-po_row_"] .fp-s {margin-bottom:0.3rem;}
        .st-key-po_head [data-testid="stColumn"] {display:flex; flex-direction:column; justify-content:center; padding-bottom:0.95rem;}
        .st-key-po_head [data-testid="stMarkdownContainer"], .st-key-po_head [data-testid="stElementContainer"] {margin-bottom:0 !important;}
        [class*="st-key-po_row_"] [data-testid="stColumn"]:first-child [data-testid="stVerticalBlock"] {align-items:center;}
        [class*="st-key-po_row_"] [data-testid="stColumn"]:first-child [data-testid="stElementContainer"] {width:auto !important;}
        .st-key-po_head [data-testid="stColumn"] + [data-testid="stColumn"],
        [class*="st-key-po_row_"] [data-testid="stColumn"] + [data-testid="stColumn"] {border-left:1px solid #efede8;}
        hr.fp-line {margin:0.35rem 0 0.6rem 0; border:none; border-top:1px solid #e6e4df;}
        .fp-c {font-size:0.9rem; line-height:1.45;} .fp-m {color:#6f6b62; font-size:0.84rem;}
        .fp-ctr {text-align:center;} .fp-num {text-align:right; white-space:nowrap;}
        .fp-whycol {padding:0 0.5rem;} .fp-b {white-space:nowrap;}
        .fp-h.fp-hw {white-space:normal; word-break:keep-all; overflow-wrap:normal; hyphens:none; font-size:0.84rem; line-height:1.25; padding:0 2px;}
        .fp-ctr {white-space:nowrap;}
        .fp-why {background:#f7f6f3; border-radius:8px; padding:0.45rem 0.75rem; font-size:0.93rem; line-height:1.45; margin-top:-0.2rem;}
        .fp-why-h {display:block; font-weight:600; font-size:0.8rem; color:#6f6b62; margin-bottom:0.15rem; text-transform:uppercase; letter-spacing:0.03em;}
        .fp-b {display:inline-block; padding:0.05rem 0.5rem; border-radius:999px; font-size:0.78rem; font-weight:600; margin-top:0.2rem;}
        .fp-b-now {background:#e6f0fb; color:#1f5fa8;} .fp-b-soon {background:#f1efe9; color:#5f5b52;} .fp-b-warn {background:#fdf1d8; color:#8a5a00;}
        /* the frame hugs the table; columns are only as wide as their content needs (nothing is cut off) */
        .fp-tw {overflow-x:auto; border:1px solid #e6e4df; border-radius:10px; margin:0.25rem auto 1rem auto; width:fit-content; max-width:100%;}
        .fp-t {border-collapse:collapse; width:auto; margin:0 !important; font-size:0.9rem; line-height:1.45;}
        .fp-t th {background:#f6f5f2; text-align:center !important; font-weight:600; font-size:0.84rem; color:#3d3a34; padding:0.55rem 0.5rem;
                  border-bottom:1px solid #e6e4df; white-space:normal; vertical-align:middle; word-break:keep-all;}
        .fp-t th, .fp-t td {border:0 !important;}
        .fp-t td + td, .fp-t th + th {border-left:1px solid #efede8 !important;}
        .fp-t td {padding:0.55rem 0.55rem; border-bottom:1px solid #efede8 !important; vertical-align:top;}
        .fp-t th {border-bottom:1px solid #e6e4df !important;}
        .fp-t tr:last-child td {border-bottom:0 !important;}
        .fp-t tbody tr:hover {background:#faf9f6;}
        .fp-t td.num {text-align:right; white-space:nowrap;}
        .fp-t td.ctr {text-align:center; white-space:nowrap;} .fp-t td.mid {white-space:nowrap;} .fp-t td.wide {min-width:11rem; max-width:20rem;}
        .fp-t th.txt, .fp-t td.txt {min-width:18rem; max-width:32rem;}
        .fp-t th {max-width:9rem;} .fp-t th.txt {max-width:none;}
        /* explanations: every bullet starts at the same place, wrapped lines justified */
        /* smaller screens (laptops, Windows display scaling): tighter type and cells so wide tables still fit */
        @media (max-width: 1800px) {.fp-t {font-size:0.83rem;} .fp-t td {padding:0.42rem 0.32rem;} .fp-t th {padding:0.42rem 0.3rem; font-size:0.78rem;}
            .fp-t th.txt, .fp-t td.txt {min-width:13.5rem;} .fp-t td.wide {min-width:8.5rem;}
            section[data-testid="stSidebar"] {width:236px !important; min-width:236px !important;}}
        @media (max-width: 1450px) {.fp-t {font-size:0.8rem;} .fp-t td, .fp-t th {padding:0.4rem 0.22rem;} .fp-t th {font-size:0.76rem;}
            .fp-t th.txt, .fp-t td.txt {min-width:11rem;} .fp-t td.wide {min-width:7rem; max-width:13rem;}
            .fp-t th {max-width:6.5rem;} .fp-t td.mid, .fp-t td.ctr {white-space:normal;}
            .fp-t th.txt, .fp-t td.txt {min-width:10rem;} .fp-chip {padding:0.1rem 0.4rem;}
            .fp-t .fp-heat {grid-template-columns:auto repeat(4, 1.3rem); gap:2px;} .fp-t .fp-heat span {font-size:0.68rem;}
            .fp-t .fp-heat i {font-size:0.6rem; padding-right:0.15rem;}}
        .fp-t th.fp-tip {cursor:help; text-decoration:underline dotted #9a968d; text-underline-offset:3px;}
        .fp-whyc {text-align:left;} .fp-whyb {display:block; text-align:left;}
        .fp-s {padding-left:0.9rem; text-indent:-0.9rem; margin:0 0 0.2rem 0;} .fp-s::before {content:"•"; display:inline-block; width:0.9rem; text-indent:0; color:#8a877f;}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {white-space:normal !important; overflow:visible !important; text-overflow:clip !important;}
        /* KPI cards: everything centered, and every card in a row the same height (with or without a change chip) */
        [data-testid="stMetric"] {text-align:center; min-height:168px; display:flex; flex-direction:column; justify-content:center;}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {justify-content:center !important; display:flex !important; width:100%;}
        [data-testid="stMetricLabel"] > div {justify-content:center !important;}
        [data-testid="stMetricDelta"] {margin-left:auto !important; margin-right:auto !important; width:fit-content;}
        .fp-hero {margin:0.2rem 0 0.4rem 0; max-width:64rem;}
        .fp-hero p {margin:0;}
        .fp-hero .h1, .fp-hero .h2 {font-size:2.3rem; font-weight:700; line-height:1.2; letter-spacing:-0.012em; color:#1b1a17;}
        .fp-hero .h2 {color:#6f6c64; margin-top:0.35rem;}
        .fp-hero .ctx {font-size:0.95rem; color:#52514e; margin-top:1rem; line-height:1.5;}
        @media (max-width: 1450px) {.fp-hero .h1, .fp-hero .h2 {font-size:1.85rem;}}
        .fp-side {font-size:0.86rem; line-height:1.45; color:#3d3a34; margin-top:0.2rem;}
        .fp-heat {display:inline-grid; grid-template-columns:auto repeat(4, 1.6rem); gap:3px; align-items:center;}
        .fp-heat span {height:1.45rem; line-height:1.45rem; border-radius:3px; font-size:0.74rem; font-weight:600;
                       text-align:center; font-variant-numeric:tabular-nums; cursor:default;}
        .fp-heat i {font-style:normal; font-size:0.68rem; color:#6f6c64; padding-right:0.25rem; white-space:nowrap; text-align:right;}
        .fp-chip {display:inline-block; padding:0.12rem 0.55rem; border-radius:999px; font-size:0.9em; font-weight:600; line-height:1.35; white-space:nowrap;}
        .fp-chip-crit {background:#fbe3e0; color:#9b2a21;} .fp-chip-high {background:#fde8d4; color:#8f450c;}
        .fp-chip-med {background:#faefc8; color:#715300;} .fp-chip-low {background:#e4edf9; color:#1d4c8a;}
        .fp-chip-ok {background:#dff1e5; color:#1d6a3c;} .fp-chip-info {background:#ebeae6; color:#4a4842;}
        .fp-banner {background:#fff5e0; border:1px solid #fab219; padding:8px 12px; border-radius:8px; margin-bottom:10px; color:#0b0b0b;}
        .fp-card {border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:12px 14px; background:#ffffff;}
        </style>""",
        unsafe_allow_html=True,
    )
    if title:
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


PX_PER_CHAR = 7.1      # average width of a character in the grid font (base font 17px)
LINE_PX, PAD_PX = 22, 24  # height of one text line, and the row padding


def fit_long_text(d: pd.DataFrame, cfg: dict, min_chars: int = 45, max_lines: int = 12, widths=(560, 420, 300, 300), skip=()):
    """Make long text readable without opening the cell: long text columns get a fixed wide width and the rows
    get tall enough for the longest text on the page, so it wraps instead of being cut off.
    Returns (data, column config, row height or None)."""
    texty = [c for c in d.columns if (d[c].dtype == object or pd.api.types.is_string_dtype(d[c]))]
    lens = {c: d[c].dropna().astype(str).str.len().max() if d[c].notna().any() else 0 for c in texty}
    long = sorted([c for c in texty if lens[c] > min_chars and c not in skip], key=lambda c: -lens[c])
    # short labels (codes like A1, Order now, Critical) read best centered; long text stays left, numbers stay right
    for c in texty:
        if c in long or c in skip or lens[c] > 14:
            continue
        old = cfg.get(c)
        if old is None or (isinstance(old, dict) and old.get("type_config", {}).get("type") in (None, "text")):
            lab = old.get("label") if isinstance(old, dict) else None
            hlp = old.get("help") if isinstance(old, dict) else None
            w = old.get("width") if isinstance(old, dict) else None
            cfg[c] = st.column_config.TextColumn(lab or c, help=hlp, width=w, alignment="center")
    if not long:
        return d, cfg, None
    d = d.copy()
    lines = 1
    for j, c in enumerate(long[:4]):
        px = widths[min(j, len(widths) - 1)]
        old = cfg.get(c)
        lab = old.get("label") if isinstance(old, dict) else None
        hlp = old.get("help") if isinstance(old, dict) else None
        cfg[c] = st.column_config.TextColumn(lab or c, width=px, help=hlp)
        d[c] = d[c].astype(str).where(d[c].notna(), "").str.replace(" | ", " · ", regex=False)
        cpl = max(int(px / PX_PER_CHAR) - 2, 10)
        lines = max(lines, int(np.ceil(d[c].str.len().max() / cpl)))
    return d, cfg, PAD_PX + LINE_PX * min(lines, max_lines)


# ---------------------------------------------------------------- readable tables (stacked sentences)

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\s+\|\s+|\s+·\s+")


def sentences(text) -> list[str]:
    """Split an explanation into its sentences (and ' | ' separated causes)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return [p.strip() for p in _SPLIT.split(str(text)) if p.strip()]


def _fmt_cell(v, conf):
    """Format one value the way the column config asks, for the HTML table."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (v is pd.NaT):
        return ""
    tc = conf.get("type_config", {}) if isinstance(conf, dict) else {}
    t, f = tc.get("type"), tc.get("format")
    if isinstance(v, (bool, np.bool_)):
        return "✓" if v else ""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%m/%d/%y")
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if f == "percent" or (t == "progress" and (f in (None, "percent"))):
            return f"{x * 100:.1f}%"
        if isinstance(f, str) and "%" in f:
            m = re.match(r"^(.*?)%(,?)\.?(\d*)([fd])(.*)$", f)
            if m:
                pre, comma, dec, kind, post = m.groups()
                dec = 0 if kind == "d" or dec == "" else int(dec)
                body = f"{x:,.{dec}f}" if comma else f"{x:.{dec}f}"
                return pre + body + post.replace("%%", "%")
        return f"{x:,.0f}" if abs(x) >= 100 or x == int(x) else f"{x:,.2f}"
    return str(v)


class Strip(tuple):
    """Weekly chances (0 to 1) shown as a small heat strip of shaded squares, one per week."""


_HEAT = [(0.0, (0xF0, 0xEF, 0xEC)), (0.3, (0xFA, 0xB2, 0x19)), (0.6, (0xEC, 0x83, 0x5A)), (1.0, (0xD0, 0x3B, 0x3B))]  # = heat map


def _heat_color(v: float) -> str:
    v = min(max(float(v), 0.0), 1.0)
    for (a, ca), (b, cb) in zip(_HEAT, _HEAT[1:]):
        if v <= b:
            f = (v - a) / (b - a)
            return "#%02x%02x%02x" % tuple(round(x + (y - x) * f) for x, y in zip(ca, cb))
    return "#d03b3b"


def heat_strip(vals) -> str:
    cells = []
    for i, v in enumerate(vals, 1):
        if i in (1, 5):  # two rows of four: weeks 1 to 4, then 5 to 8
            cells.append(f"<i>wk {i}</i>")
        v = 0.0 if v is None or not np.isfinite(v) else float(v)
        fg = "#ffffff" if v >= 0.8 else "#1b1a17"
        cells.append(f'<span style="background:{_heat_color(v)};color:{fg}" title="Week {i}: {v:.0%} chance of running out">'
                     f'{v * 100:.0f}</span>')
    return f'<div class="fp-heat">{"".join(cells)}</div>'


# columns whose values are a scale get a colored chip, so a long table scans by color first
CHIPS = {
    "Severity": {"Critical": "crit", "High": "high", "Medium": "med", "Low": "low"},
    "Status": {"Problem": "crit", "Check": "med", "OK": "ok", "Info": "info"},
    "Grade": {"A": "ok", "B": "low", "C": "med", "D": "crit"},
    "Ready": {"Yes": "ok", "No": "crit", "TBD": "info"},
}


def html_table(d: pd.DataFrame, cfg: dict | None = None, stacked=(), key: str = "ht", sortable: bool = True):
    """A table that reads like a document: long explanations show one sentence per line, nothing is cut off,
    and the page scrolls, not the table. Columns in `stacked` are split into sentences."""
    cfg = cfg or {}
    slots = None
    if sortable and len(d) > 1:
        # one control row: sort on the left, paging on the right
        c1, c2, c3, c4, c5 = st.columns([2.1, 1.7, 1, 1, 2.6], vertical_alignment="top")
        slots = (c3, c4, c5)
        opts = ["(as listed)"] + [c for c in d.columns if c not in stacked
                                  and not (len(d) and isinstance(d[c].iloc[0], Strip))]
        by = c1.selectbox("Sort by", opts, key=f"{key}_sort",
                          format_func=lambda c: (cfg.get(c) or {}).get("label") or c if isinstance(cfg.get(c), dict) else c)
        if by != "(as listed)":
            asc = c2.radio("Order", ["High to low", "Low to high"], horizontal=True, key=f"{key}_asc") == "Low to high"
            d = d.sort_values(by, ascending=asc, kind="stable", na_position="last")
    d = paged(d, key, note="Sort by reorders every row; the download has every row.", slots=slots)
    num_like = {c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])}
    # one alignment per column: short labels centered, longer text left, numbers right
    def _cls(c):
        if c in num_like:
            return "num"
        if len(d) and isinstance(d[c].iloc[0], Strip):
            return "ctr"
        m = d[c].astype(str).str.len().max() if len(d) else 0
        return "ctr" if m <= 14 else ("mid" if m <= 26 else "wide")
    col_cls = {c: _cls(c) for c in d.columns}
    def _lab(c):
        v = cfg.get(c)
        return (v.get("label") if isinstance(v, dict) and v.get("label") else str(c))
    def _tip(c):
        v = cfg.get(c)
        h = v.get("help") if isinstance(v, dict) else None
        return f' title="{_html.escape(h)}"' if h else ""

    head = "".join(f'<th class="{"txt" if c in stacked else col_cls[c]}{" fp-tip" if _tip(c) else ""}"{_tip(c)}>{_html.escape(_lab(c))}</th>'
                   for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if c in stacked:
                parts = sentences(v)
                inner = "".join(f'<div class="fp-s">{_html.escape(p)}</div>' for p in parts)
                cells.append(f'<td class="txt"><div class="fp-whyc"><div class="fp-whyb">{inner}</div></div></td>')
            elif isinstance(v, Strip):
                cells.append(f'<td class="ctr">{heat_strip(v)}</td>')
            elif _lab(c) in CHIPS and str(v) in CHIPS[_lab(c)]:
                cells.append(f'<td class="ctr"><span class="fp-chip fp-chip-{CHIPS[_lab(c)][str(v)]}">{_html.escape(str(v))}</span></td>')
            else:
                cls = col_cls[c]
                txt = _fmt_cell(v, cfg.get(c))
                nw = ""  # short text columns stay on one line (see .mid); longer ones wrap within .wide
                cells.append(f'<td class="{cls}"{nw}>{_html.escape(txt)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    out = f'<div class="fp-tw"><table class="fp-t"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    st.markdown(out.replace("$", "&#36;"), unsafe_allow_html=True)


def _stacked_cols(d: pd.DataFrame, min_chars=45) -> list:
    """Text columns that hold explanations (long text or more than one sentence)."""
    out = []
    for c in d.columns:
        if len(d) and isinstance(d[c].iloc[0], Strip):
            continue
        if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
            v = d[c].dropna().astype(str)
            if len(v) and (v.str.len().max() > min_chars) and v.map(lambda t: len(sentences(t)) > 1 or len(t) > 80).any():
                out.append(c)
    return out


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


def paged(df: pd.DataFrame, key: str, default: int = 25, note: str | None = None, slots=None) -> pd.DataFrame:
    """Rows for the current page. Tables grow to fit their rows (the page scrolls, not the grid) and big
    tables are split into pages, so the grid never needs its own vertical scrollbar.
    slots: three containers (rows per page, page, note) when the caller lays out the control row itself."""
    n = len(df)
    if n <= 35:  # short tables are shown whole (a page of 25 plus a few)
        return df
    c1, c2, c3 = slots if slots is not None else st.columns([1, 1, 4])
    size = c1.selectbox("Rows per page", PAGE_SIZES, index=PAGE_SIZES.index(default) if default in PAGE_SIZES else 0,
                        key=f"{key}_size")
    pages = max(int(np.ceil(n / size)), 1)
    page = c2.number_input(f"Page (of {pages:,})", min_value=1, max_value=pages, value=1, step=1, key=f"{key}_page")
    lo = (int(page) - 1) * size
    note = note if note is not None else "Sorting a column sorts this page; the download has every row."
    c3.markdown(f'<p class="fp-note" style="margin-top:2.1rem">Rows {lo + 1:,} to {min(lo + size, n):,} of {n:,}. {note}</p>',
                unsafe_allow_html=True)
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


def show_df(df, column_config=None, height=None, key=None, fit=None, **_):
    """st.dataframe with readable headers, no index and full width. Grows to fit its rows (paged when long),
    so it scrolls with the page and the fullscreen button shows the whole grid."""
    cfg = {label(k): v for k, v in (column_config or {}).items()}
    k = _key_for(df, key)
    full = df.rename(columns={c: label(c) for c in df.columns})
    # one table look across the app: document-style table, explanations one sentence per line
    html_table(full, cfg, stacked=_stacked_cols(full), key=k, sortable=len(full) > 15)
    return
    d = paged(df, k).rename(columns={c: label(c) for c in df.columns})
    d = _one_decimal_percent(d.copy(), cfg)
    d, cfg, rh = fit_long_text(d, cfg, **(fit or {}))
    st.dataframe(d, hide_index=True, width="stretch", height="content", column_config=cfg, key=key, row_height=rh)


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
    u = [str(c).replace("_", " ").capitalize() if str(c) == str(c).lower() else c for c in unit_cols]
    html_table(d, colcfg(d, m, p, n, w, u), stacked=_stacked_cols(d), key=k, sortable=len(d) > 15)
    return
    d = paged(d, k).copy()
    cfg = colcfg(d, m, p, n, w, u)
    d = _one_decimal_percent(d, cfg)
    d, cfg, rh = fit_long_text(d, cfg)
    st.dataframe(d, hide_index=True, width="stretch", height="content", column_config=cfg, key=key, row_height=rh)


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
        kw = {"delta_arrow": arrow} if _METRIC_ARROW else {}  # delta_arrow is newer than Streamlit 1.50
        with c:
            st.metric(label, value, delta=delta, help=help_, border=True, delta_color=color, **kw)


def settings():
    return eio.load_settings()


def is_sample(folder) -> bool:
    return Path(str(folder)).as_posix().rstrip("/").endswith(SAMPLE_FOLDER)


# ---------------------------------------------------------------- stock levels: what "today" means
REPLAY_SAME = "AI planner, today's on-hand dollars"


def sap_ok(r) -> bool:
    """True when enough planned items have SAP MinStock / MaxStock set for 'today's SAP settings' to mean something."""
    cov = getattr(r, "sap_coverage", None) or {}
    if "meaningful" in cov:
        return bool(cov["meaningful"])
    return bool(np.isfinite((r.policy_summary or {}).get("current_fill_rate", np.nan)))


def sap_text(r) -> str:
    """One neutral sentence on SAP min / max coverage, or "" when they are maintained."""
    cov = getattr(r, "sap_coverage", None) or {}
    if sap_ok(r) or not cov:
        return ""
    if cov["set"] == 0:
        return (f"SAP MinStock / MaxStock are 0 / 0 (not set) on all {cov['live']:,} planned items, so there are no SAP "
                "settings to compare against.")
    share = "under 1%" if 0 < cov["revenue_share"] < 0.01 else format(cov["revenue_share"], ".0%")
    return (f"SAP MinStock / MaxStock are set on only {cov['set']:,} of {cov['live']:,} planned items ({share} of revenue); "
            "the rest are 0 / 0 (not set), so there are no SAP settings to compare against.")


def sap_notice(r):
    """A single gray line on Stock levels explaining the comparison basis."""
    t = sap_text(r)
    if t:
        st.caption(t + " Comparisons here use today's stock on hand.")


def stock_kpis(r) -> list:
    """The four stock-level cards shared by Overview and Stock levels."""
    ps = r.policy_summary or {}
    ok = sap_ok(r)
    cur = ps.get("current_fill_rate", np.nan) if ok else np.nan
    onhand = ps.get("on_hand_value", np.nan)
    same = r.budget.get("same_inventory") or {}

    def pts(v):
        return f"{(v - cur) * 100:+.1f} pts" if np.isfinite(cur) and v is not None and np.isfinite(v) else None

    items = []
    if ok:
        items.append(("Fill rate, SAP today", pct(cur), None,
                      "Share of units demanded that ship from stock, simulated with today's SAP MinStock and MaxStock "
                      "(items where they are set). The replay tab on Stock levels measures what they really delivered."))
    else:
        items.append(("Stock on hand today", money(onhand), None,
                      "On-hand units in the stock file (the date of the export), at unit cost, on the items the planner plans. "
                      + sap_text(r) + " The cards here compare the planner with this real stock instead."))
    if same:
        tip = "Today's stock on hand, in dollars, placed item by item where it serves the most revenue."
        if same.get("basis") != "on_hand":
            tip = "Today's average inventory dollars, placed item by item where they serve the most revenue."
        elif same.get("spent", same["budget"]) < 0.98 * same["budget"]:
            tip += (f" The planner needs only {money(same['spent'])} of the {money(same['budget'])} even at the highest "
                    "service step; the rest is stock beyond what demand calls for.")
        items.append(("Fill rate, AI, on-hand $", pct(same.get("fill_rate")), pts(same.get("fill_rate")), tip))
    items.append(("Fill rate, AI targets", pct(ps.get("fill_rate")), pts(ps.get("fill_rate")),
                  "Service targets from Settings (A 98%, B 96%, C 92% by default)."))
    base = onhand if np.isfinite(onhand) else ps.get("current_avg_inventory_value", np.nan)
    items.append(("Inventory, AI targets", money(ps.get("avg_inventory_value")),
                  f"{signed_money(ps.get('avg_inventory_value', 0) - base)} vs on hand today" if np.isfinite(base) else None,
                  "Average stock value the simulation expects at your targets, compared with the stock on hand today.", "off"))
    return items
