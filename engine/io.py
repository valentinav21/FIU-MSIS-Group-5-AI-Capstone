"""Load settings and data files, map column names to the data contract, coerce types."""
from __future__ import annotations

import copy
import csv
import io as _io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .schema import TABLES, Table

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"

DEFAULTS = {
    "data": {"folder": "data/my_data", "net_returns_from_demand": False, "as_of_date": "", "history_start": "",
             "default_warehouse": "MIA", "currency": "USD"},
    "forecast": {"bucket": "W", "horizon_periods": 26, "backtest_origins": 4, "backtest_horizon": 8,
                 "min_history_periods": 13, "dead_after_periods": 52, "use_ml_model": True, "spike_cap_multiple": 0},
    "inventory": {"service_measure": "fill_rate", "service_level": {"A": 0.98, "B": 0.96, "C": 0.92, "new": 0.90}, "economic_service_level": False,
                  "economic_bounds": [0.80, 0.99], "holding_cost_pct_per_year": 0.25, "order_cost_usd": 25, "max_order_cover_weeks": 26,
                  "review_period_days": 7, "simulations": 2000, "policy_paths": 100, "budget_mode": "off", "model_error_weight": 0.0, "abc_split": [0.80, 0.95], "overstock_weeks": 26,
                  "default_lead_time_days": 75},
    "alerts": {"horizon_weeks": 8, "stockout_probability_threshold": 0.30, "late_po_grace_days": 7},
    "purchasing": {"default_supplier_moq": 1, "stockout_cost_pct_of_price": 0.30, "supplier_reliability_weight": 1.0,
                   "po_draft_series": ""},
    "sales": {"churn_window_days": 90, "recommendations_per_customer": 10, "min_co_buyers": 5,
              "price_floor_percentile": 25, "max_price_change_pct": 5},
}

# File names accepted for each table (compared after normalizing)
FILE_ALIASES = {
    "items": ["items", "item", "oitm", "itemmaster", "products", "product_master"],
    "sales": ["sales", "sales_lines", "invoices", "invoice_lines", "inv1", "oinv", "sales_orders", "rdr1", "orders"],
    "returns": ["returns", "credit_memos", "rin1", "orin", "rdn1", "ordn"],
    "stock": ["stock", "inventory", "oitw", "stock_by_warehouse", "onhand"],
    "purchases": ["purchases", "purchase_orders", "po_lines", "por1", "opor"],
    "receipts": ["receipts", "goods_receipts", "pdn1", "opdn", "grpo"],
    "suppliers": ["suppliers", "vendors", "supplier_master"],
    "item_suppliers": ["item_suppliers", "itm2", "sourcing", "item_vendors", "oscn"],
    "open_orders": ["open_orders", "open_sales_orders", "backorders"],
    "customers": ["customers", "customer_master", "ocrd_customers"],
    "web_searches": ["web_searches", "searches", "search_log", "focuspoint_searches"],
    "market_potential": ["market_potential", "market", "external_market", "market_size"],
    "events": ["events", "demand_events", "promotions"],
}


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    if "." in s:  # drop a table prefix such as OINV.DocDate
        s = s.split(".")[-1]
    return re.sub(r"[^a-z0-9]", "", s)


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


LOCAL_PATH = ROOT / "config" / "settings_local.yaml"  # changes made in the app; settings.yaml keeps its comments


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path.name} is not valid YAML: {e}") from e


def load_settings(path: Path | None = None) -> dict:
    """Defaults, then config/settings.yaml, then the app's saved changes (settings_local.yaml)."""
    if path:
        return deep_merge(DEFAULTS, _read_yaml(Path(path)))
    return deep_merge(deep_merge(DEFAULTS, _read_yaml(SETTINGS_PATH)), _read_yaml(LOCAL_PATH))


def save_settings(settings: dict, path: Path | None = None):
    path = Path(path) if path else LOCAL_PATH
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False, allow_unicode=True)


def reset_settings(keep_folder: str | None = None):
    if LOCAL_PATH.exists():
        LOCAL_PATH.unlink()
    if keep_folder:
        save_settings(deep_merge(load_settings(), {"data": {"folder": keep_folder}}))


def resolve_folder(folder: str | Path) -> Path:
    p = Path(folder).expanduser()
    return p if p.is_absolute() else (ROOT / p)


@dataclass
class LoadResult:
    tables: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)   # (level, table, message)
    mapping: list = field(default_factory=list)  # (table, file, source column, field)
    files: dict = field(default_factory=dict)

    def add(self, level, table, msg):
        self.issues.append((level, table, msg))

    def has(self, key) -> bool:
        df = self.tables.get(key)
        return df is not None and len(df) > 0


# ---------------------------------------------------------------- reading files

def _read_text_table(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and "\x00" in text[:200]:
                continue
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("could not decode file")
    sample = text[:20000]
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = "\t" if sample.count("\t") > sample.count(",") else ","
    return pd.read_csv(_io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, na_values=[""])


def read_any(path: Path) -> pd.DataFrame | dict:
    ext = path.suffix.lower()
    if ext in (".csv", ".txt", ".tsv"):
        return _read_text_table(path)
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=None, dtype=str)
    if ext == ".parquet":
        return pd.read_parquet(path).astype(str)
    raise ValueError(f"unsupported file type {ext}")


def _match_table_key(stem: str) -> str | None:
    n = _norm(stem)
    for key, aliases in FILE_ALIASES.items():
        if n in {_norm(a) for a in aliases}:
            return key
    return None


# ---------------------------------------------------------------- mapping and coercion

def map_columns(df: pd.DataFrame, table: Table, overrides: dict | None = None):
    """Return (renamed frame, list of (source, target))."""
    overrides = overrides or {}
    src_norm = {}
    for c in df.columns:
        n = _norm(c)
        # SAP user-defined fields start with U_ ; U_Channel matches channel
        if str(c).strip().lower().startswith("u_"):
            n = _norm(str(c).strip()[2:])
        src_norm[c] = n
    used_src, mapping = set(), []
    for col in table.cols:
        candidates = []
        if col.name in overrides and overrides[col.name] in df.columns:
            candidates = [overrides[col.name]]
        else:
            keys = [_norm(col.name)]
            if col.sap:
                keys += [_norm(p) for p in re.split(r"[ /,()]+", col.sap) if "." in p]
            keys += [_norm(s) for s in col.synonyms]
            for k in keys:
                for c, n in src_norm.items():
                    if n == k and c not in used_src:
                        candidates.append(c)
                if candidates:
                    break
        if candidates:
            used_src.add(candidates[0])
            mapping.append((candidates[0], col.name))
    out = df[[s for s, _ in mapping]].copy()
    out.columns = [t for _, t in mapping]
    return out, mapping


_TRUE = {"y", "yes", "true", "1", "t", "tyes"}
# SAP marks canceled documents Y and their cancellation documents C; both stay out of demand
_TRUE_EXTRA = {"canceled": {"c", "canceled", "cancelled"}, "active": {"active", "a"}, "export_only": {"x", "ex", "export"}}


def _to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.strip()
    neg = t.str.match(r"^\(.*\)$")  # accounting negatives (12.5)
    t = t.str.replace(r"[\$,\s%()]", "", regex=True).replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    v = pd.to_numeric(t, errors="coerce")
    v[neg.fillna(False)] *= -1
    return v


def _to_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    t = s.astype(str).str.strip().replace({"": None, "nan": None, "NaT": None, "None": None})
    # SAP B1 often exports yyyymmdd
    is_compact = t.str.fullmatch(r"\d{8}").fillna(False)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if is_compact.any():
        out[is_compact] = pd.to_datetime(t[is_compact], format="%Y%m%d", errors="coerce")
    rest = ~is_compact & t.notna()
    if rest.any():
        try:
            parsed = pd.to_datetime(t[rest], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(t[rest], errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        out[rest] = parsed.dt.normalize()
    return out


def coerce(df: pd.DataFrame, table: Table, res: LoadResult) -> pd.DataFrame:
    df = df.copy()
    for col in table.cols:
        if col.name not in df.columns:
            continue
        raw = df[col.name]
        if col.dtype == "float" or col.dtype == "int":
            v = _to_float(raw)
            bad = raw.notna() & (raw.astype(str).str.strip() != "") & v.isna()
            if bad.any():
                res.add("warning", table.key, f"{int(bad.sum())} values in '{col.name}' are not numbers and were left blank")
            df[col.name] = v
        elif col.dtype == "date":
            v = _to_date(raw)
            bad = raw.notna() & (raw.astype(str).str.strip() != "") & v.isna()
            if bad.any():
                res.add("warning", table.key, f"{int(bad.sum())} values in '{col.name}' are not dates and were left blank")
            df[col.name] = v
        elif col.dtype == "bool":
            truthy = _TRUE | _TRUE_EXTRA.get(col.name, set())
            df[col.name] = raw.astype(str).str.strip().str.lower().isin(truthy) & raw.notna()
        else:
            v = raw.astype(str).str.strip()
            v = v.where(~v.isin(["", "nan", "None", "NaN"]), None)
            # Excel turns codes like 00123 into 123.0; undo the trailing .0
            v = v.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
            df[col.name] = v
    return df


def load_folder(folder: str | Path, settings: dict | None = None) -> LoadResult:
    settings = settings or load_settings()
    folder = resolve_folder(folder)
    res = LoadResult()
    if not folder.exists():
        res.add("error", "-", f"Data folder not found: {folder}")
        return res
    raw_frames: dict[str, list] = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith(("~", ".")) or not path.is_file():
            continue
        if path.suffix.lower() not in (".csv", ".txt", ".tsv", ".xlsx", ".xlsm", ".xls", ".parquet"):
            continue
        if path.suffix.lower() in (".csv", ".txt", ".tsv", ".parquet") and not _match_table_key(path.stem):
            if path.suffix.lower() != ".txt":
                res.add("info", "-", f"Skipped {path.name}: file name does not match a table (see data/templates)")
            continue
        try:
            obj = read_any(path)
        except Exception as e:  # noqa: BLE001
            res.add("error", "-", f"Could not read {path.name}: {e}")
            continue
        if isinstance(obj, dict):  # Excel: each sheet can be a table, or the file name decides
            file_key = _match_table_key(path.stem)
            for sheet, df in obj.items():
                key = _match_table_key(sheet) or (file_key if len(obj) == 1 else None)
                if key:
                    raw_frames.setdefault(key, []).append((f"{path.name}:{sheet}", df))
                else:
                    res.add("info", "-", f"Skipped sheet '{sheet}' in {path.name}: name does not match a table")
        else:
            key = _match_table_key(path.stem)
            if key:
                raw_frames.setdefault(key, []).append((path.name, obj))
            else:
                res.add("info", "-", f"Skipped {path.name}: file name does not match a table (see data/templates)")
    overrides = settings.get("column_overrides", {}) or {}
    for key, parts in raw_frames.items():
        table = TABLES[key]
        frames = []
        for fname, df in parts:
            df = df.dropna(how="all")
            # SAP DTW and some exports carry a second header row of descriptions
            if len(df) and _looks_like_header(df.iloc[0], table):
                df = df.iloc[1:]
            mapped, mp = map_columns(df, table, overrides.get(key))
            for s, t in mp:
                res.mapping.append((key, fname, s, t))
            missing = [c for c in table.required if c not in mapped.columns]
            if missing:
                res.add("error", key, f"{fname}: missing required column(s) {missing}. Found {list(df.columns)[:15]}")
                continue
            frames.append(coerce(mapped, table, res))
            res.files.setdefault(key, []).append(fname)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            for c in table.cols:  # every contract column exists, blank if not supplied
                if c.name not in out.columns:
                    out[c.name] = (np.nan if c.dtype in ("float", "int") else (pd.NaT if c.dtype == "date" else None))
            n0 = len(out)
            out = out.dropna(subset=table.required)
            if len(out) < n0:
                res.add("warning", key, f"{n0 - len(out)} rows dropped because a required field was blank or unreadable")
            res.tables[key] = out.reset_index(drop=True)
    for key, t in TABLES.items():
        if t.essential and key not in res.tables:
            res.add("warning" if key not in ("sales", "items") else "error", key,
                    f"No {t.title.lower()} file found ({key}.csv). Modules that need it will be skipped.")
    return res


def _looks_like_header(row: pd.Series, table: Table) -> bool:
    """True when the first data row is a second header row (SAP DTW templates carry one).

    The row counts as a header only if its values themselves map to the table's required fields.
    """
    vals = [str(v) for v in row.dropna()]
    if len(vals) < 2:
        return False
    fake = pd.DataFrame(columns=vals)
    _, mp = map_columns(fake, table)
    hit = {t for _, t in mp}
    return all(r in hit for r in table.required)
