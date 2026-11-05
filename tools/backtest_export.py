"""Backtest and forward forecast export for the Fabric scorecard (Group 5 data engineering).

Run on your own computer, on the cleaned Silver CSV export (it never leaves the team):

    python tools/backtest_export.py --data "C:\\path\\to\\silver_csv"

It writes, next to this app (folder exports/):
  ai_forecast.csv          backtest: trained only on sales up to the cutoff (default 2026-01-31),
                           monthly forecast Feb 2026 to Oct 2026, A1 and H1 only
  ai_forecast_forward.csv  forward forecast from the latest data, Nov 2026 to Jul 2027
  backtest_check.txt       a local score of the backtest against actual sales orders (same rules as the
                           scorecard: item x month, and 3-month windows), next to a flat 12-month average

Columns of both CSVs: item_code, warehouse, forecast_month (first day of month, yyyy-MM-dd), forecast_qty, method.
Every item x warehouse series that sold at A1 or H1 before the cutoff gets a row for every month, zeros included.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import forecast, io, prep  # noqa: E402
from engine.pipeline import VERSION  # noqa: E402


def month_start(s) -> pd.Timestamp:
    return pd.Timestamp(s).to_period("M").to_timestamp()


def run_forecast(tables, settings, as_of, first_month, n_months):
    """Forecast n_months monthly buckets starting at first_month, using only sales up to as_of."""
    s = copy.deepcopy(settings)
    s["data"]["as_of_date"] = str(pd.Timestamp(as_of).date())
    s["forecast"].update({"bucket": "M", "backtest_horizon": 3, "min_history_periods": 6})  # monthly rules, whatever Settings say
    t = dict(tables)
    t["sales"] = tables["sales"][tables["sales"]["doc_date"] <= pd.Timestamp(as_of)]  # nothing after the cutoff
    P = prep.prepare(t, s)
    skip = (first_month.year - P.first_forecast_start.year) * 12 + (first_month.month - P.first_forecast_start.month)
    if skip < 0:
        raise SystemExit(f"The first month asked for ({first_month:%b %Y}) is before the first month the data allows "
                         f"({P.first_forecast_start:%b %Y}).")
    s["forecast"]["horizon_periods"] = int(skip + n_months)
    F = forecast.run(P, s)
    path = F.path[:, skip:skip + n_months]
    months = [first_month + pd.offsets.MonthBegin(h) if h else first_month for h in range(n_months)]
    key = F.table[["item_code", "warehouse", "method_key"]].reset_index(drop=True)
    out = pd.DataFrame({
        "item_code": np.repeat(key["item_code"].to_numpy(), n_months),
        "warehouse": np.repeat(key["warehouse"].to_numpy(), n_months),
        "forecast_month": np.tile([m.strftime("%Y-%m-%d") for m in months], len(key)),
        "forecast_qty": np.round(path.reshape(-1), 4),
        "method": np.repeat([f"fortpro_ai_v{VERSION}_{m}" for m in key["method_key"]], n_months),
    })
    return out, P, F


def flat_average(P, n_months):
    """Flat average of the last 12 complete months per series (the scorecard's simple baseline)."""
    M = P.matrix
    w = min(12, M.shape[1])
    return np.repeat((M[:, -w:].sum(axis=1) / 12.0)[:, None], n_months, axis=1)


def score(fc: pd.DataFrame, base: pd.DataFrame, sales: pd.DataFrame, plan, last_actual: pd.Timestamp):
    """Item x month and 3-month window errors: sum |actual - forecast| / sum actual."""
    months = sorted(pd.to_datetime(fc["forecast_month"].unique()))
    months = [m for m in months if m + pd.offsets.MonthEnd(0) <= last_actual]  # complete months only
    if not months:
        return None
    s = sales[sales["warehouse"].astype(str).isin(plan) & (sales["quantity"] > 0)].copy()
    if "canceled" in s:
        s = s[~s["canceled"].fillna(False).astype(bool)]
    s["forecast_month"] = s["doc_date"].dt.to_period("M").dt.to_timestamp()
    s = s[s["forecast_month"].isin(months)]
    act = s.groupby(["item_code", "warehouse", "forecast_month"])["quantity"].sum().rename("actual").reset_index()
    rows = []
    for name, f in (("AI planner", fc), ("Flat 12-month average", base)):
        f = f.assign(forecast_month=pd.to_datetime(f["forecast_month"]))
        f = f[f["forecast_month"].isin(months)]
        m = f.merge(act, on=["item_code", "warehouse", "forecast_month"], how="outer").fillna({"forecast_qty": 0.0, "actual": 0.0})
        e_month = (m["actual"] - m["forecast_qty"]).abs().sum() / max(m["actual"].sum(), 1e-9)
        m["win"] = [months.index(x) // 3 for x in m["forecast_month"]]
        g = m.groupby(["item_code", "warehouse", "win"])[["actual", "forecast_qty"]].sum()
        e_win = (g["actual"] - g["forecast_qty"]).abs().sum() / max(g["actual"].sum(), 1e-9)
        bias = (m["forecast_qty"].sum() - m["actual"].sum()) / max(m["actual"].sum(), 1e-9)
        new = m[~m.set_index(["item_code", "warehouse"]).index.isin(f.set_index(["item_code", "warehouse"]).index)]
        rows.append({"Model": name, "Error, item x month": e_month, "Error, item x 3-month window": e_win, "Bias": bias,
                     "Actual units": m["actual"].sum(), "Forecast units": m["forecast_qty"].sum(),
                     "Actual units on series with no forecast (new since cutoff)": new["actual"].sum()})
    return pd.DataFrame(rows), months


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder with the cleaned CSV files (sales.csv, items.csv, ...)")
    ap.add_argument("--cutoff", default="2026-01-31", help="Last day of training data for the backtest")
    ap.add_argument("--months", type=int, default=9, help="Backtest months after the cutoff")
    ap.add_argument("--forward-start", default="2026-11", help="First month of the forward forecast (YYYY-MM)")
    ap.add_argument("--forward-months", type=int, default=9)
    ap.add_argument("--out", default=str(ROOT / "exports"), help="Folder to write the files to")
    a = ap.parse_args(argv)

    settings = io.load_settings()
    plan = [str(w) for w in (settings["data"].get("plan_warehouses") or ["A1", "H1"])]
    print(f"Loading {a.data} ...")
    load = io.load_folder(a.data, settings)
    if not load.has("sales"):
        raise SystemExit("No sales file found in that folder.")
    tables = load.tables
    last_sale = pd.Timestamp(tables["sales"]["doc_date"].max()).normalize()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ backtest
    cutoff = pd.Timestamp(a.cutoff)
    first = month_start(cutoff + pd.Timedelta(days=1))
    print(f"Backtest: training on sales up to {cutoff:%Y-%m-%d}, forecasting {a.months} months from {first:%b %Y} "
          f"for {', '.join(plan)} ...")
    bt, P, F = run_forecast(tables, settings, cutoff, first, a.months)
    bt.to_csv(out / "ai_forecast.csv", index=False)
    base = bt.copy()
    base["forecast_qty"] = np.round(flat_average(P, a.months).reshape(-1), 4)
    base["method"] = "flat_12_month_average"
    lines = [f"FORTPRO AI Planner v{VERSION} backtest check",
             f"Training data up to {cutoff:%Y-%m-%d}; warehouses {', '.join(plan)}; {P.matrix.shape[0]:,} item x warehouse series; "
             f"{bt['forecast_month'].nunique()} months; last sales date in the files {last_sale:%Y-%m-%d}.",
             "Method mix: " + ", ".join(f"{k} {v}" for k, v in F.table["method_key"].value_counts().items()), ""]
    sc = score(bt, base, tables["sales"], plan, last_sale)
    if sc is not None:
        df, months = sc
        lines.append(f"Scored on complete months {months[0]:%b %Y} to {months[-1]:%b %Y} against actual sales orders "
                     f"(a partial last month is left out here; the Fabric scorecard has the full picture).")
        lines.append(df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        lines += ["", "Reference from the data engineer: SAP's own forecast 1.05, flat 12-month average 0.86 (item x month)."]
    lines += ["", "Internal holdout (last 3 months before the cutoff), by demand class:",
              F.accuracy.to_string(index=False, float_format=lambda v: f"{v:,.3f}") if len(F.accuracy) else "n/a"]

    # ------------------------------------------------------------ forward
    fwd_first = month_start(a.forward_start + "-01" if len(a.forward_start) == 7 else a.forward_start)
    as_of = pd.Timestamp(settings["data"].get("as_of_date") or last_sale)
    print(f"Forward: training on sales up to {as_of:%Y-%m-%d}, forecasting {a.forward_months} months from {fwd_first:%b %Y} ...")
    fw, P2, F2 = run_forecast(tables, settings, as_of, fwd_first, a.forward_months)
    fw.to_csv(out / "ai_forecast_forward.csv", index=False)
    lines += ["", f"Forward forecast: data up to {as_of:%Y-%m-%d} (history used through {P2.history_end:%Y-%m-%d}, complete months), "
                  f"{fwd_first:%b %Y} onward, {P2.matrix.shape[0]:,} series, {fw['forecast_qty'].sum():,.0f} units in total.",
              "Note: items.list_price in the export is a 12-month median price up to Oct 2026; the model uses price only as a "
              "minor feature, so its effect on the backtest is small."]
    (out / "backtest_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out / 'ai_forecast.csv'} ({len(bt):,} rows), {out / 'ai_forecast_forward.csv'} ({len(fw):,} rows) "
          f"and {out / 'backtest_check.txt'}.")


if __name__ == "__main__":
    main()
