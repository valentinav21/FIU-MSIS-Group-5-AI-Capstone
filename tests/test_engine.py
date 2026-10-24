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
    res = pipeline.run(tmp_path, fast_settings(), save=False)
    assert not res.errors, res.errors
    assert len(res.forecast_table) and len(res.policy)
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
