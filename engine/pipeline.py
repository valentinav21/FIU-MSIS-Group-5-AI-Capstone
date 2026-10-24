"""Run every module in order and save one results file the app reads."""
from __future__ import annotations

import pickle
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import alerts, forecast, inventory, io, leadtime, prep, probabilistic, purchasing, replay, sales, validate

RESULTS_DIR = io.ROOT / "output"
VERSION = "1.1.0"


@dataclass
class Results:
    version: str = VERSION
    created: str = ""
    data_folder: str = ""
    settings: dict = field(default_factory=dict)
    as_of: pd.Timestamp | None = None
    bucket: str = "W"
    period_days: float = 7.0
    timings: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    health: pd.DataFrame = field(default_factory=pd.DataFrame)
    readiness: pd.DataFrame = field(default_factory=pd.DataFrame)
    mapping: pd.DataFrame = field(default_factory=pd.DataFrame)
    items: pd.DataFrame = field(default_factory=pd.DataFrame)
    keys: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: np.ndarray | None = None
    period_end: pd.DatetimeIndex | None = None
    forecast_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_path: np.ndarray | None = None
    forecast_accuracy: pd.DataFrame = field(default_factory=pd.DataFrame)
    method_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    sim_params: dict = field(default_factory=dict)
    level_sigma: dict = field(default_factory=dict)
    quantile_map: dict = field(default_factory=dict)
    season_categories: list = field(default_factory=list)
    supplier_scorecard: pd.DataFrame = field(default_factory=pd.DataFrame)
    lead_time_samples: dict = field(default_factory=dict)
    policy: pd.DataFrame = field(default_factory=pd.DataFrame)
    policy_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    policy_summary: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    kappa_check: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    po_recs: pd.DataFrame = field(default_factory=pd.DataFrame)
    transfers: pd.DataFrame = field(default_factory=pd.DataFrame)
    replay: dict = field(default_factory=dict)
    sales: sales.SalesResult | None = None
    customers_master: pd.DataFrame = field(default_factory=pd.DataFrame)


def _step(res, name, fn, progress, frac):
    t0 = time.time()
    if progress:
        progress(frac, name)
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"{name}: {e}")
        res.notes.append(f"{name} did not finish: {e}. Other modules still ran.")
        res.timings[name] = time.time() - t0
        traceback.print_exc()
        return None
    res.timings[name] = time.time() - t0
    return out


def run(folder: str | None = None, settings: dict | None = None, progress=None, save=True) -> Results:
    settings = settings or io.load_settings()
    folder = folder or settings["data"]["folder"]
    res = Results(created=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), data_folder=str(folder), settings=settings)
    t_all = time.time()
    if progress:
        progress(0.01, "Loading files")
    load = io.load_folder(folder, settings)
    res.health = validate.run_checks(load, settings)
    res.readiness = validate.module_readiness(load, settings)
    res.mapping = pd.DataFrame(load.mapping, columns=["Table", "File", "Your column", "Mapped to"])
    if not load.has("sales"):
        res.errors.append("No sales file could be loaded. Nothing else can run without it.")
        if save:
            save_results(res)
        return res
    tables = load.tables
    res.customers_master = tables.get("customers", pd.DataFrame())
    P = _step(res, "Preparing demand history", lambda: prep.prepare(tables, settings), progress, 0.05)
    if P is None:
        if save:
            save_results(res)
        return res
    res.as_of, res.bucket, res.period_days = P.as_of, P.bucket, P.period_days
    res.notes += P.notes
    res.items, res.keys, res.history, res.period_end = P.items, P.keys, P.matrix, P.period_end

    F = _step(res, "Forecasting demand", lambda: forecast.run(P, settings), progress, 0.10)
    if F is not None:
        cal = _step(res, "Calibrating forecast ranges", lambda: probabilistic.calibrate(F, P, settings), progress, 0.25)
        if cal is not None:
            res.calibration = cal[1]
        res.forecast_table, res.forecast_path = F.table, F.path
        res.forecast_accuracy, res.method_scores = F.accuracy, F.method_scores
        res.sim_params, res.level_sigma, res.quantile_map = F.sim_params, F.level_sigma, F.quantile_map or {}
        res.season_categories = sorted(F.season_index)

    L = _step(res, "Learning supplier lead times", lambda: leadtime.build(tables, P.items, P.as_of, settings), progress, 0.30)
    if L is not None:
        res.supplier_scorecard = L.scorecard
        res.lead_time_samples = {k: v for k, v in L.by_supplier.items()}

    Pol = None
    if F is not None and L is not None:
        kap = _step(res, "Checking the simulator against history",
                    lambda: replay.calibrate_kappa(P, F, L, tables, settings, inventory), progress, 0.33)
        kappa = kap[0] if kap is not None else 0.0
        if kap is not None:
            res.kappa_check = kap[1]

        def _pol():
            def prog(x):
                if progress:
                    progress(0.38 + 0.30 * x, "Simulating stocking policies")
            return inventory.build(P, F, L, tables, settings, kappa=kappa, progress=prog)
        Pol = _step(res, "Simulating stocking policies", _pol, progress, 0.38)
    if Pol is not None:
        sm = inventory.summary(Pol)
        res.policy_summary = sm
        t = Pol.table
        live = (t["forecast_per_period"] > 0).to_numpy()
        has_cur = live & t["current_avg_inventory_value"].notna().to_numpy()
        # today's inventory dollars: simulated average stock under today's SAP settings, plus the AI
        # plan for items that have no SAP settings, so the comparison covers the same items
        inv_today = float(t.loc[has_cur, "current_avg_inventory_value"].sum() + t.loc[live & ~has_cur, "avg_inventory_value"].sum())
        ss_today = sm.get("current_safety_stock_value", np.nan)
        b_inv = b_ss = None
        if has_cur.any():
            b_inv = inventory.optimize_budget(Pol, inv_today, basis="inventory")
            res.budget["same_inventory"] = {"budget": inv_today, "fill_rate": b_inv.attrs["fill_rate"]}
            res.budget["same_inventory_table"] = b_inv
        if np.isfinite(ss_today):
            b_ss = inventory.optimize_budget(Pol, ss_today, basis="safety_stock")
            res.budget["same_as_today"] = {"budget": ss_today, "fill_rate": b_ss.attrs["fill_rate"]}
        bm = str(settings["inventory"].get("budget_mode", "off")).strip().lower()
        chosen = None
        if bm == "inventory" and b_inv is not None:
            chosen = b_inv
        elif bm == "current" and b_ss is not None:
            chosen = b_ss
        elif bm not in ("off", "", "none", "inventory", "current"):
            try:
                chosen = inventory.optimize_budget(Pol, float(bm), basis="inventory")
            except ValueError:
                res.notes.append(f"budget_mode '{bm}' is not understood; service targets were used.")
        if chosen is not None:
            t.loc[live, "service_target"] = chosen.loc[live, "budget_target"].to_numpy()
            t.loc[live, "reorder_point"] = chosen.loc[live, "budget_reorder_point"].to_numpy()
            t.loc[live, "order_up_to"] = t.loc[live, "reorder_point"] + t.loc[live, "order_qty"]
            t.loc[live, "fill_rate_at_rop"] = chosen.loc[live, "budget_fill_rate"].to_numpy()
            t.loc[live, "safety_stock"] = (t.loc[live, "reorder_point"] - t.loc[live, "mean_lead_time_demand"]).clip(lower=0)
            t.loc[live, "safety_stock_value"] = t.loc[live, "safety_stock"] * t.loc[live, "unit_cost"]
            res.budget["applied"] = {"basis": chosen.attrs["basis"], "spent": chosen.attrs["spent"], "fill_rate": chosen.attrs["fill_rate"]}
            res.notes.append(f"Budget mode on: ${chosen.attrs['spent']:,.0f} of {chosen.attrs['basis'].replace('_', ' ')} placed where it serves the most revenue.")
            res.policy_summary = inventory.summary(Pol)
        res.policy, res.policy_curve = Pol.table, Pol.curve
        A = _step(res, "Projecting 1 to 8 week risks", lambda: alerts.build(P, F, Pol, L, tables, settings), progress, 0.70)
        if A is not None:
            res.alerts = A
        R = _step(res, "Building purchase order recommendations", lambda: purchasing.build(P, F, Pol, L, tables, settings), progress, 0.76)
        if R is not None:
            res.po_recs = R
        T = _step(res, "Checking warehouse transfers", lambda: purchasing.transfers(Pol, tables, settings), progress, 0.82)
        if T is not None:
            res.transfers = T
        extra = {}
        if b_inv is not None and chosen is None:
            s_b = np.where(live, b_inv["budget_reorder_point"].to_numpy(float), Pol.table["reorder_point"].to_numpy(float))
            extra["AI planner, same inventory dollars as today"] = (s_b, s_b + Pol.table["order_qty"].to_numpy(float))
        Rp = _step(res, "Replaying last year of demand", lambda: replay.run(P, Pol, extra=extra), progress, 0.84)
        if Rp is not None:
            res.replay = Rp

    stock_by_item = tables["stock"].groupby("item_code")["on_hand"].sum() if load.has("stock") else None
    S = None
    if settings.get("features", {}).get("bp2_enabled", False):
        S = _step(res, "Analysing customers and sales growth",
                  lambda: sales.run(tables, P.items, settings, P.as_of, stock_by_item), progress, 0.86)
    if S is not None:
        res.sales = S
        res.notes += S.notes
    res.timings["Total"] = time.time() - t_all
    if progress:
        progress(0.98, "Saving results")
    if save:
        save_results(res)
    if progress:
        progress(1.0, "Done")
    return res


def results_path(folder_key: str = "latest") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"results_{folder_key}.pkl"


def save_results(res: Results, key: str = "latest"):
    with open(results_path(key), "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(key: str = "latest") -> Results | None:
    p = results_path(key)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            r = pickle.load(f)
        return r if getattr(r, "version", None) == VERSION else None
    except Exception:  # noqa: BLE001
        return None
