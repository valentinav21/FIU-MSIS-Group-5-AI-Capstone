"""Engine tests: kernels, loader robustness, end-to-end runs on full, minimal and odd data."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import forecast, io, pipeline, policysim, prep, purchasing  # noqa: E402
from engine.inventory import ServiceCurve  # noqa: E402
from tools.generate_synthetic import generate  # noqa: E402


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("small")
    generate(d, n_items=150, seed=3, n_customers=120)
    return d


def fast_settings():
    s = io.load_settings(ROOT / "config" / "settings.yaml")
    s["inventory"]["simulations"] = 600
    s["inventory"]["policy_paths"] = 40
    return s


# ------------------------------------------------------------------ kernels

def test_croston_matches_hand_calculation():
    y = np.array([[0, 0, 3, 0, 0, 0, 2.0]])
    L = forecast._croston(y, np.array([0]), 0.1)
    # first demand at t=2: size 3, interval 3 -> 1.0; next at t=6: z=2.9, p=3.1 -> 0.9355
    assert L[0, 2] == pytest.approx(1.0)
    assert L[0, 6] == pytest.approx(2.9 / 3.1, rel=1e-6)


def test_tsb_decays_without_demand():
    y = np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]])
    L = forecast._tsb(y, np.array([0]), 0.1, 0.2)
    assert L[0, -1] < L[0, 1] < L[0, 0]


def test_service_curve_fill_and_cycle():
    x = np.sort(np.arange(1, 101, dtype=float))
    sc = ServiceCurve(x, q=50)
    assert sc.cycle(50)[0] == pytest.approx(0.5)
    assert sc.fill(100)[0] == pytest.approx(1.0)
    assert sc.fill(0)[0] == pytest.approx(0.0)  # expected short 50.5 > q
    assert sc.rop_for(0.9, "cycle") == 90


def test_policy_sim_more_stock_serves_more():
    rng = np.random.default_rng(0)
    ser = [dict(mean_week=np.full(30, 2.0), sizes=np.array([1, 2, 3.0]), occ=0.8, sigma=0.0, lt_weeks=np.array([4, 5, 6.0]),
                q=10, moq=1, mult=1, cands=np.array([0, 4, 8, 12, 16, 24.0]), extra=[(8, 18)])]
    out = policysim.simulate_batch(ser, n_paths=60, rng=rng)[0]
    assert np.all(np.diff(out["fill"]) >= -1e-9)
    assert out["fill"][-1] > out["fill"][0]
    r, f = policysim.rop_for_fill(ser[0]["cands"], out["fill"], 0.9)
    assert f >= 0.9 - 1e-6 or r == ser[0]["cands"][-1]


# ------------------------------------------------------------------ loader

def test_loader_handles_messy_exports(tmp_path):
    (tmp_path / "items.csv").write_text("ItemCode;ItemName;AvgPrice\nA1;Brake drum;$1,234.50\n00123;Lamp;(5.00)\n", encoding="cp1252")
    # DTW style second header row, tab separated, yyyymmdd dates, U_ field
    rows = ["DocDate\tItemCode\tQuantity\tCardCode\tU_Channel\tCANCELED",
            "DocDate\tItemCode\tQuantity\tCardCode\tU_Channel\tCANCELED",
            "20250105\tA1\t3\tC1\tPortal\tN", "20250112\t00123\t2\tC2\tPhone\tC", "2025-01-19\tA1\tx\tC1\tPortal\tN"]
    (tmp_path / "INV1.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a table", encoding="utf-8")
    r = io.load_folder(tmp_path)
    it, s = r.tables["items"], r.tables["sales"]
    assert it.loc[it.item_code == "A1", "unit_cost"].iloc[0] == pytest.approx(1234.5)
    assert it.loc[it.item_code == "00123", "unit_cost"].iloc[0] == pytest.approx(-5.0)
    assert set(s["item_code"]) == {"A1", "00123"}  # leading zeros kept, header row dropped
    assert s["channel"].notna().all()
    assert s.loc[s.item_code == "00123", "canceled"].iloc[0]  # SAP cancellation document
    assert len(s) == 2  # unreadable quantity dropped with a warning
    assert any("not numbers" in m for _, _, m in r.issues)


def test_loader_reports_missing_required(tmp_path):
    (tmp_path / "sales.csv").write_text("ItemCode,Quantity\nA,1\n")
    r = io.load_folder(tmp_path)
    assert "sales" not in r.tables
    assert any(lvl == "error" and "missing required" in m for lvl, _, m in r.issues)


def test_excel_workbook_with_named_sheets(tmp_path, small):
    s = pd.read_csv(small / "INV1.csv", dtype=str).head(500)
    it = pd.read_csv(small / "OITM.csv", dtype=str)
    with pd.ExcelWriter(tmp_path / "export.xlsx") as xw:
        s.to_excel(xw, sheet_name="sales", index=False)
        it.to_excel(xw, sheet_name="items", index=False)
    r = io.load_folder(tmp_path)
    assert len(r.tables["sales"]) == 500 and len(r.tables["items"]) == len(it)


# ------------------------------------------------------------------ end to end

def test_full_pipeline_small(small):
    st = fast_settings()
    st["features"]["bp2_enabled"] = True  # keep the BP2 code tested even while it is turned off in the app
    res = pipeline.run(small, st, save=False)
    assert not res.errors, res.errors
    assert len(res.forecast_table) > 0 and len(res.policy) == len(res.forecast_table)
    assert res.policy["reorder_point"].ge(0).all()
    live = res.policy[res.policy["forecast_per_period"] > 0]
    assert (live["order_up_to"] >= live["reorder_point"]).all()
    assert res.replay.get("table") is not None
    assert res.sales is not None and len(res.sales.customers) > 0
    # every recommendation carries a reason
    assert res.po_recs["why"].str.len().gt(20).all()
    h, l = purchasing.to_dtw(res.po_recs.head(10), res.as_of)
    assert h.splitlines()[0] == h.splitlines()[1] and "DocObjectCode" in h
    js = json.loads(purchasing.to_service_layer(res.po_recs.head(10), res.as_of))
    assert js and js[0]["DocObjectCode"] == "oPurchaseOrders"
    from engine import report
    assert len(report.workbook(res)) > 10000


def test_minimal_data_only_sales_and_items(tmp_path, small):
    shutil.copy(small / "INV1.csv", tmp_path / "sales.csv")
    shutil.copy(small / "OITM.csv", tmp_path / "items.csv")
    st = fast_settings()
    st["forecast"].update({"bucket": "W", "backtest_horizon": 8, "min_history_periods": 13, "item_override": True})  # weekly path stays tested
    res = pipeline.run(tmp_path, st, save=False)
    assert not res.errors, res.errors
    assert len(res.forecast_table) and len(res.policy) and res.period_days == 7.0
    assert res.policy["lead_time_source"].str.contains("setting").all()


def test_monthly_bucket_and_no_customers(tmp_path, small):
    s = pd.read_csv(small / "INV1.csv", dtype=str).drop(columns=["CardCode"])
    s.to_csv(tmp_path / "sales.csv", index=False)
    for f in ("OITM.csv", "OITW.csv", "POR1.csv", "PDN1.csv"):
        shutil.copy(small / f, tmp_path / f)
    st = fast_settings()
    st["forecast"]["bucket"] = "M"
    res = pipeline.run(tmp_path, st, save=False)
    assert not res.errors, res.errors
    assert res.period_days > 28
    assert len(res.policy)
    assert res.sales is None  # BP2 is off by default (TBD)
    assert (res.readiness["Ready"] == "TBD").any()


def test_settings_roundtrip(tmp_path):
    s = io.load_settings()
    p = tmp_path / "x.yaml"
    io.save_settings(s, p)
    assert io.load_settings(p)["inventory"]["service_level"] == s["inventory"]["service_level"]


def test_anonymize_is_consistent(tmp_path, small):
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(small / "POR1.csv", src)
    shutil.copy(small / "OITM.csv", src)
    dst = tmp_path / "dst"
    subprocess.run([sys.executable, str(ROOT / "tools" / "anonymize.py"), "--src", str(src), "--dst", str(dst), "--secret", "s"], check=True)
    po = pd.read_csv(dst / "POR1.csv", dtype=str)
    it = pd.read_csv(dst / "OITM.csv", dtype=str)
    assert po["CardCode"].str.startswith("BP").all()
    assert set(it["CardCode"].dropna()) <= set(po["CardCode"]) | set(it["CardCode"].dropna())
    orig = pd.read_csv(small / "POR1.csv", dtype=str)
    m = dict(zip(orig["CardCode"], po["CardCode"]))
    assert len(set(m.values())) == len(set(m.keys()))  # one to one


# ------------------------------------------------------------------ FORTPRO data rules (Silver export, Nov 2026)

def _sales(rows):
    return pd.DataFrame(rows, columns=["doc_date", "item_code", "warehouse", "quantity"]).assign(
        doc_date=lambda d: pd.to_datetime(d["doc_date"]), line_total=np.nan, unit_price=np.nan, customer_code=None)


def test_calendar_months_drop_partial_month_and_other_warehouses():
    st = fast_settings()
    st["data"]["as_of_date"] = "2026-10-21"
    s = _sales([("2026-08-03", "X", "A1", 2), ("2026-09-30", "X", "A1", 3), ("2026-10-05", "X", "A1", 7),
                ("2026-09-15", "X", "A2", 50), ("2026-09-15", "X", "H1", 1)])
    P = prep.prepare({"sales": s, "items": pd.DataFrame({"item_code": ["X"]})}, st)
    assert P.history_end == pd.Timestamp("2026-09-30") and P.period_end[-1] == pd.Timestamp("2026-09-30")
    assert P.first_forecast_start == pd.Timestamp("2026-10-01")
    assert set(P.keys["warehouse"]) == {"A1", "H1"}  # A2 left out
    a1 = P.keys.index[P.keys["warehouse"] == "A1"][0]
    assert P.matrix[a1, -1] == 3 and P.matrix[a1, -2] == 2  # October (partial) not in the history
    assert P.weekly is not None and P.weekly[a1].sum() == 12  # weekly view keeps every line up to as_of
    assert [d.strftime("%Y-%m-%d") for d in P.future_period_starts(3)] == ["2026-10-01", "2026-11-01", "2026-12-01"]


def test_receipts_join_on_po_line_not_item():
    from engine import leadtime
    po = pd.DataFrame({"po_entry": ["9", "9"], "line_num": [0, 1], "item_code": ["X", "X"], "supplier_code": ["V", "V"],
                       "po_date": pd.to_datetime(["2026-01-01", "2026-01-01"]), "quantity": [10.0, 10.0], "open_qty": [0.0, 10.0],
                       "unit_price": 1.0, "warehouse": "A1", "promised_date": pd.NaT, "status": None, "currency": None})
    gr = pd.DataFrame({"receipt_date": pd.to_datetime(["2026-03-01"]), "item_code": ["X"], "quantity": [10.0],
                       "po_entry": ["9"], "po_line": [1], "supplier_code": ["V"], "warehouse": ["A1"]})
    m = leadtime._match_receipts(po, gr)
    assert len(m) == 2  # no duplicated rows
    assert m.loc[m.line_num == 1, "received_qty"].iloc[0] == 10 and m.loc[m.line_num == 0, "received_qty"].iloc[0] == 0


def test_stale_open_pos_are_not_incoming():
    from engine import leadtime
    st = fast_settings()
    po = pd.DataFrame({"po_entry": ["1", "2"], "line_num": [0, 0], "item_code": ["X", "X"], "supplier_code": ["V", "V"],
                       "po_date": pd.to_datetime(["2025-12-01", "2026-09-01"]), "quantity": [5.0, 5.0], "open_qty": [5.0, 5.0],
                       "unit_price": 1.0, "warehouse": "A1", "promised_date": pd.NaT, "status": None, "currency": None})
    L = leadtime.build({"purchases": po, "receipts": None}, pd.DataFrame({"item_code": ["X"]}), pd.Timestamp("2026-10-21"), st)
    assert list(L.open_pos["po_entry"]) == ["2"] and list(L.stale_pos["po_entry"]) == ["1"]
    assert not L.has_promises and L.has_po_lines


def test_stale_open_orders_and_plan_warehouses():
    st = fast_settings()
    oo = pd.DataFrame({"item_code": ["X", "X", "X"], "open_qty": [1.0, 2.0, 4.0], "warehouse": ["A1", "A1", "A2"],
                       "due_date": pd.to_datetime(["2026-10-01", "2026-01-01", "2026-10-01"]), "customer_code": None})
    o, notes = prep.trusted_open_orders({"open_orders": oo}, st, pd.Timestamp("2026-10-21"))
    assert o["open_qty"].tolist() == [1.0] and any("past their ship date" in n for n in notes)


def test_international_minimum_flag():
    st = fast_settings()
    recs = pd.DataFrame({"supplier": ["CN1", "CN1", "US1"], "action": ["Order now", "Order now", "Order now"],
                         "line_total": [3000.0, 2000.0, 100.0], "why": ["a", "b", "c"]})
    sup = pd.DataFrame({"supplier_code": ["CN1", "US1"], "supplier_name": ["CN1", "US1"], "country": ["CN", "US"]})
    out = purchasing.apply_international_minimum(recs, {"suppliers": sup}, st)
    assert out["below_supplier_minimum"].tolist() == [True, True, False]
    assert "10,000" in out["why"].iat[0] and out["why"].iat[2] == "c"


def test_dead_stock_months_convert_to_periods():
    assert forecast.dead_periods({"dead_stock_months": 24}, "M") == 24
    assert forecast.dead_periods({"dead_stock_months": 24}, "W") == 104
    assert forecast.dead_periods({"dead_after_periods": 52}, "W") == 52


def test_backtest_export_uses_only_data_before_cutoff(tmp_path, small):
    from tools import backtest_export as be
    load = io.load_folder(small, fast_settings())
    last = load.tables["sales"]["doc_date"].max()
    cutoff = (last - pd.DateOffset(months=10)) + pd.offsets.MonthEnd(0)
    first = (cutoff + pd.Timedelta(days=1))
    bt, P, F = be.run_forecast(load.tables, fast_settings(), cutoff, first, 9)
    assert list(bt.columns) == ["item_code", "warehouse", "forecast_month", "forecast_qty", "method"]
    assert bt["forecast_month"].nunique() == 9 and bt["forecast_month"].min() == first.strftime("%Y-%m-%d")
    assert set(bt["warehouse"]) <= {"A1", "H1"} and (bt["forecast_qty"] >= 0).all()
    assert P.period_end[-1] == cutoff  # nothing after the cutoff
    # the same run on data that stops at the cutoff gives the same forecast (no leakage)
    t2 = dict(load.tables)
    t2["sales"] = load.tables["sales"][load.tables["sales"]["doc_date"] <= cutoff]
    bt2, _, _ = be.run_forecast(t2, fast_settings(), cutoff, first, 9)
    assert np.allclose(bt["forecast_qty"].to_numpy(), bt2["forecast_qty"].to_numpy())
    be.main(["--data", str(small), "--cutoff", cutoff.strftime("%Y-%m-%d"), "--forward-start",
             (last + pd.offsets.MonthBegin(1)).strftime("%Y-%m"), "--out", str(tmp_path)])
    assert (tmp_path / "ai_forecast.csv").exists() and (tmp_path / "ai_forecast_forward.csv").exists()
    assert "Error, item x month" in (tmp_path / "backtest_check.txt").read_text()


def test_plan_warehouses_fall_back_when_none_match():
    st = fast_settings()
    st["data"]["as_of_date"] = "2026-10-21"
    s = _sales([("2026-08-03", "X", "MIA", 2), ("2026-09-03", "X", "HOU", 1)])
    P = prep.prepare({"sales": s, "items": pd.DataFrame({"item_code": ["X"]})}, st)
    assert set(P.keys["warehouse"]) == {"MIA", "HOU"} and any("every warehouse is planned" in n for n in P.notes)


def test_old_local_settings_do_not_override_new_rules(tmp_path, monkeypatch):
    import yaml
    loc = tmp_path / "settings_local.yaml"
    monkeypatch.setattr(io, "LOCAL_PATH", loc)
    old = io.load_settings()
    old["forecast"]["bucket"], old["data"]["default_warehouse"], old["data"]["folder"] = "W", "MIA", "C:/exports"
    loc.write_text(yaml.safe_dump(old))  # a file written by v1.1 (no _version, every setting)
    s = io.load_settings()
    assert s["forecast"]["bucket"] == "M" and s["data"]["default_warehouse"] == "A1" and s["data"]["folder"] == "C:/exports"
    s["inventory"]["order_cost_usd"] = 40
    io.save_settings(s)
    saved = yaml.safe_load(loc.read_text())
    assert saved["_version"] == io.LOCAL_VERSION and "forecast" not in saved and saved["inventory"] == {"order_cost_usd": 40}
    assert io.load_settings()["inventory"]["order_cost_usd"] == 40


def test_sale_in_unfinished_month_is_never_dead_stock():
    st = fast_settings()
    st["data"]["as_of_date"] = "2026-10-21"
    s = _sales([("2023-01-10", "X", "A1", 2), ("2026-10-05", "X", "A1", 1), ("2023-01-10", "Y", "A1", 2)])
    P = prep.prepare({"sales": s, "items": pd.DataFrame({"item_code": ["X", "Y"]})}, st)
    cls = forecast.classify(P.matrix, P.first_idx, dead_after=24, sold_recently=P.sold_after_history)
    by = dict(zip(P.keys["item_code"], cls["demand_class"]))
    assert by["X"] != "inactive" and by["Y"] == "inactive"


def test_receipt_without_matching_line_falls_back_to_item():
    from engine import leadtime
    po = pd.DataFrame({"po_entry": ["9"], "line_num": [np.nan], "item_code": ["X"], "supplier_code": ["V"],
                       "po_date": pd.to_datetime(["2026-01-01"]), "quantity": [10.0], "open_qty": [0.0], "unit_price": 1.0,
                       "warehouse": "A1", "promised_date": pd.NaT, "status": None, "currency": None})
    gr = pd.DataFrame({"receipt_date": pd.to_datetime(["2026-03-01"]), "item_code": ["X"], "quantity": [10.0],
                       "po_entry": ["9"], "po_line": [3], "supplier_code": ["V"], "warehouse": ["A1"]})
    m = leadtime._match_receipts(po, gr)
    assert len(m) == 1 and m["received_qty"].iloc[0] == 10


def test_items_with_no_supplier_anywhere(tmp_path, small):
    """FORTPRO has items with no preferred vendor, no PO history and no sourcing record: never a blank supplier."""
    for f in ("INV1.csv", "OITW.csv", "suppliers.csv", "open_orders.csv"):
        shutil.copy(small / f, tmp_path / f)
    it = pd.read_csv(small / "OITM.csv", dtype=str)
    sales = pd.read_csv(small / "INV1.csv", dtype=str)
    top = sales["ItemCode"].value_counts().index[:25]
    it.loc[it["ItemCode"].isin(top), "CardCode"] = None
    it.to_csv(tmp_path / "OITM.csv", index=False)
    for f, col in (("POR1.csv", "ItemCode"), ("PDN1.csv", "ItemCode"), ("ITM2.csv", "ItemCode")):
        d = pd.read_csv(small / f, dtype=str)
        d[~d[col].isin(top)].to_csv(tmp_path / f, index=False)
    res = pipeline.run(tmp_path, fast_settings(), save=False)
    assert not res.errors, res.errors
    r = res.po_recs
    assert r["supplier"].map(lambda x: isinstance(x, str) and x != "").all()
    sorted(r["supplier"].astype(str).unique())  # what the page does
    nos = r[r["supplier"] == "NO SUPPLIER"]
    assert len(nos) and not nos["approve"].any() and nos["why"].str.contains("No supplier on file").all()
    h, _ = purchasing.to_dtw(r, res.as_of)
    assert "NO SUPPLIER" not in h
