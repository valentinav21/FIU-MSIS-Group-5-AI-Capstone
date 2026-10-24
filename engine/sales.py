"""Sales growth engine (BP2).

Every model here is tested on a later period it was not trained on, and every
recommendation carries its reason in plain words.

* Customer segments (RFM) and buying personas (clusters of category mix)
* Churn risk model with honest out-of-time accuracy and revenue at risk
* Cross-sell recommendations per customer (item-to-item co-purchase), tested against a
  popularity baseline, plus kit / bundle pairs
* Upsell and wallet share gaps versus similar customers
* Geographic opportunity, with market potential when supplied
* Pricing: price leakage against same-item peer prices, discount discipline, and
  category price sensitivity where the data supports an estimate
* Product trends and unmet demand from web searches
* A ranked sales play list that combines all of the above
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class SalesResult:
    customers: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    personas: pd.DataFrame = field(default_factory=pd.DataFrame)
    churn_metrics: dict = field(default_factory=dict)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    rec_metrics: dict = field(default_factory=dict)
    bundles: pd.DataFrame = field(default_factory=pd.DataFrame)
    upsell: pd.DataFrame = field(default_factory=pd.DataFrame)
    geo: pd.DataFrame = field(default_factory=pd.DataFrame)
    pricing_leakage: pd.DataFrame = field(default_factory=pd.DataFrame)
    pricing_summary: dict = field(default_factory=dict)
    elasticity: pd.DataFrame = field(default_factory=pd.DataFrame)
    discount_check: pd.DataFrame = field(default_factory=pd.DataFrame)
    products: pd.DataFrame = field(default_factory=pd.DataFrame)
    categories: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmet_demand: pd.DataFrame = field(default_factory=pd.DataFrame)
    search_vs_stock: pd.DataFrame = field(default_factory=pd.DataFrame)
    plays: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------- helpers

def _lines(tables, items, settings, as_of):
    s = tables["sales"].copy()
    if "canceled" in s:
        s = s[~s["canceled"].fillna(False).astype(bool)]
    s = s[(s["quantity"] > 0) & (s["doc_date"] <= as_of)]
    s = s.dropna(subset=["customer_code"])
    if s["line_total"].isna().all():
        s["line_total"] = s["unit_price"] * s["quantity"]
    s["line_total"] = s["line_total"].fillna(s["unit_price"] * s["quantity"])
    s["unit_net"] = s["line_total"] / s["quantity"]
    im = items.set_index("item_code")
    s["category"] = s["item_code"].map(im["category"]).fillna("Uncategorized")
    s["subcategory"] = s["item_code"].map(im["subcategory"]).fillna("")
    cost = s["item_code"].map(im["unit_cost"])
    if s["gross_profit"].isna().all():
        s["gross_profit"] = s["line_total"] - cost * s["quantity"]
    if s["doc_entry"].isna().all():
        s["doc_entry"] = s["customer_code"].astype(str) + "_" + s["doc_date"].dt.strftime("%Y%m%d")
    cust = tables.get("customers")
    if cust is not None and len(cust):
        cm = cust.drop_duplicates("customer_code").set_index("customer_code")
        s["ship_state"] = s["ship_state"].fillna(s["customer_code"].map(cm["state"]))
        s["ship_country"] = s["ship_country"].fillna(s["customer_code"].map(cm["country"]))
    s["ship_country"] = s["ship_country"].fillna("US")
    return s


def customer_features(s, as_of, cust=None, days=365):
    cur = s[s["doc_date"] > as_of - pd.Timedelta(days=days)]
    prev = s[(s["doc_date"] <= as_of - pd.Timedelta(days=days)) & (s["doc_date"] > as_of - pd.Timedelta(days=2 * days))]
    g = s.groupby("customer_code")
    f = pd.DataFrame({
        "first_purchase": g["doc_date"].min(),
        "last_purchase": g["doc_date"].max(),
    })
    f["recency_days"] = (as_of - f["last_purchase"]).dt.days
    f["tenure_days"] = (as_of - f["first_purchase"]).dt.days
    gc = cur.groupby("customer_code")
    f["orders_12m"] = gc["doc_entry"].nunique()
    f["revenue_12m"] = gc["line_total"].sum()
    f["margin_12m"] = gc["gross_profit"].sum()
    f["lines_12m"] = gc.size()
    f["categories_12m"] = gc["category"].nunique()
    f["items_12m"] = gc["item_code"].nunique()
    f["revenue_prior_12m"] = prev.groupby("customer_code")["line_total"].sum()
    f["revenue_90d"] = s[s["doc_date"] > as_of - pd.Timedelta(days=90)].groupby("customer_code")["line_total"].sum()
    f["revenue_prior_90d"] = s[(s["doc_date"] <= as_of - pd.Timedelta(days=90)) & (s["doc_date"] > as_of - pd.Timedelta(days=180))].groupby("customer_code")["line_total"].sum()
    f = f.fillna({"orders_12m": 0, "revenue_12m": 0, "margin_12m": 0, "lines_12m": 0, "categories_12m": 0, "items_12m": 0,
                  "revenue_prior_12m": 0, "revenue_90d": 0, "revenue_prior_90d": 0})
    # usual gap between orders (median of each customer's own gaps, last 2 years)
    od = s[s["doc_date"] > as_of - pd.Timedelta(days=730)].groupby(["customer_code", "doc_entry"])["doc_date"].min().reset_index()
    od = od.sort_values(["customer_code", "doc_date"])
    od["gap"] = od.groupby("customer_code")["doc_date"].diff().dt.days
    f["usual_gap_days"] = od.groupby("customer_code")["gap"].median()
    f["usual_gap_days"] = f["usual_gap_days"].fillna(f["tenure_days"].clip(lower=30))
    f["recency_ratio"] = f["recency_days"] / f["usual_gap_days"].clip(lower=7)
    f["growth_12m"] = np.where(f["revenue_prior_12m"] > 0, f["revenue_12m"] / f["revenue_prior_12m"] - 1, np.nan)
    f["trend_90d"] = np.where(f["revenue_prior_90d"] > 0, f["revenue_90d"] / f["revenue_prior_90d"] - 1, np.nan)
    f["avg_order_value"] = f["revenue_12m"] / f["orders_12m"].replace(0, np.nan)
    f["margin_pct"] = f["margin_12m"] / f["revenue_12m"].replace(0, np.nan)
    top = cur.groupby(["customer_code", "category"])["line_total"].sum().reset_index().sort_values("line_total", ascending=False)
    f["top_category"] = top.drop_duplicates("customer_code").set_index("customer_code")["category"]
    if "channel" in s and s["channel"].notna().any():
        ch = cur.groupby(["customer_code", "channel"])["line_total"].sum().reset_index().sort_values("line_total", ascending=False)
        f["main_channel"] = ch.drop_duplicates("customer_code").set_index("customer_code")["channel"]
    loc = s.sort_values("doc_date").groupby("customer_code").agg(state=("ship_state", "last"), country=("ship_country", "last"))
    f = f.join(loc)
    if cust is not None and len(cust):
        cm = cust.drop_duplicates("customer_code").set_index("customer_code")
        for c in ("customer_group", "customer_type", "price_tier"):
            f[c] = cm[c].reindex(f.index)
        f["state"] = f["state"].fillna(cm["state"].reindex(f.index))
        f["country"] = f["country"].fillna(cm["country"].reindex(f.index))
    if "sales_employee" in s and s["sales_employee"].notna().any():
        rep = s.sort_values("doc_date").groupby("customer_code")["sales_employee"].last()
        f["sales_rep"] = rep
    return f


SEG_NAMES = [
    ("Champions", "Bought recently, buy often and spend the most. Reward and ask for referrals."),
    ("Loyal", "Buy regularly. Offer the next product line they do not buy yet."),
    ("Promising", "Recent buyers with room to grow. Nurture with targeted offers."),
    ("New", "First purchase in the last 90 days. Onboard and follow up."),
    ("Needs attention", "Above average once, now slipping. Reach out before they lapse."),
    ("At risk", "Used to buy often and spend well, but not lately. Call them."),
    ("Cannot lose", "Big spenders gone quiet. Senior outreach now."),
    ("Hibernating", "Low, infrequent and not recent. Low cost win-back only."),
    ("Lost", "No purchase in a long time. Exclude from paid campaigns."),
]


def rfm(f):
    a = f.copy()
    a = a[a["tenure_days"] >= 0]
    r = pd.qcut(a["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    fr = pd.qcut(a["orders_12m"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m = pd.qcut(a["revenue_12m"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    a["R"], a["F"], a["M"] = r, fr, m
    fm = (a["F"] + a["M"]) / 2
    seg = np.select(
        [
            (a["R"] >= 4) & (fm >= 4.5),
            (a["R"] >= 3) & (fm >= 3.5),
            (a["tenure_days"] <= 90),
            (a["R"] >= 4) & (fm < 3.5),
            (a["R"] == 3) & (fm < 3.5),
            (a["R"] <= 2) & (fm >= 4.5),
            (a["R"] <= 2) & (fm >= 3),
            (a["R"] == 2),
        ],
        ["Champions", "Loyal", "New", "Promising", "Needs attention", "Cannot lose", "At risk", "Hibernating"],
        "Lost",
    )
    a["segment"] = seg
    desc = dict(SEG_NAMES)
    seg_t = a.groupby("segment").agg(customers=("revenue_12m", "size"), revenue_12m=("revenue_12m", "sum"),
                                     avg_recency_days=("recency_days", "mean"), avg_orders_12m=("orders_12m", "mean")).reset_index()
    seg_t["share_of_revenue"] = seg_t["revenue_12m"] / max(seg_t["revenue_12m"].sum(), 1e-9)
    seg_t["what_it_means"] = seg_t["segment"].map(desc)
    order = [n for n, _ in SEG_NAMES]
    seg_t["_o"] = seg_t["segment"].map({n: i for i, n in enumerate(order)})
    return a, seg_t.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def personas(s, f, as_of, rng_seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    cur = s[s["doc_date"] > as_of - pd.Timedelta(days=730)]
    mix = cur.pivot_table(index="customer_code", columns="category", values="line_total", aggfunc="sum", fill_value=0)
    mix = mix[mix.sum(axis=1) > 0]
    if len(mix) < 20 or mix.shape[1] < 2:
        return pd.Series(dtype=str), pd.DataFrame()
    share = mix.div(mix.sum(axis=1), axis=0)
    X = np.sqrt(share.to_numpy())  # Hellinger style transform keeps small shares visible
    best_k, best_s, best_lab = 3, -1, None
    for k in range(3, min(9, len(mix) // 10 + 3)):
        km = KMeans(n_clusters=k, n_init=10, random_state=rng_seed).fit(X)
        sc = silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=rng_seed)
        if sc > best_s:
            best_k, best_s, best_lab = k, sc, km.labels_
    lab = pd.Series(best_lab, index=share.index)
    rows, names = [], {}
    for c in range(best_k):
        m = lab == c
        avg = share[m].mean().sort_values(ascending=False)
        top = [k for k, v in avg.items() if v >= 0.12][:2] or [avg.index[0]]
        broad = (share[m] > 0.05).sum(axis=1).mean()
        name = (" + ".join(top) + " buyers") if broad < 4 else f"Broad buyers ({top[0]} led)"
        base, n = name, 2
        while name in names.values():
            name = f"{base} {n}"
            n += 1
        names[c] = name
        rv = f.reindex(share.index[m])["revenue_12m"]
        rows.append({"persona": name, "customers": int(m.sum()), "revenue_12m": float(rv.sum()),
                     "top_categories": ", ".join(f"{k} {v:.0%}" for k, v in avg.head(4).items()),
                     "avg_categories_bought": float(broad)})
    out = pd.DataFrame(rows).sort_values("revenue_12m", ascending=False).reset_index(drop=True)
    out.attrs["silhouette"] = float(best_s)
    return lab.map(names), out


# ---------------------------------------------------------------------------- churn model

CHURN_FEATS = ["recency_days", "tenure_days", "orders_12m", "revenue_12m", "lines_12m", "categories_12m", "items_12m",
               "revenue_prior_12m", "revenue_90d", "revenue_prior_90d", "usual_gap_days", "recency_ratio", "growth_12m",
               "trend_90d", "avg_order_value", "margin_pct"]


def _snapshot(s, cut, window, cust):
    hist = s[s["doc_date"] <= cut]
    f = customer_features(hist, cut, cust)
    f = f[(f["revenue_12m"] > 0)]  # customers active in the year before the cut
    fut = s[(s["doc_date"] > cut) & (s["doc_date"] <= cut + pd.Timedelta(days=window))]
    bought = set(fut["customer_code"])
    f["churned"] = (~f.index.isin(bought)).astype(int)
    return f


def churn_model(s, f_now, as_of, window, cust):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    c1 = as_of - pd.Timedelta(days=window)
    c0 = as_of - pd.Timedelta(days=2 * window)
    if s["doc_date"].min() > c0 - pd.Timedelta(days=365):
        return None, {"note": "Churn model needs at least 1 year of history plus two churn windows."}
    train = _snapshot(s, c0, window, cust)
    test = _snapshot(s, c1, window, cust)
    if len(train) < 50 or train["churned"].nunique() < 2 or test["churned"].nunique() < 2:
        return None, {"note": "Not enough customers or churn events to train a model."}
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(train[CHURN_FEATS], train["churned"])
    p = clf.predict_proba(test[CHURN_FEATS])[:, 1]
    auc = roc_auc_score(test["churned"], p)
    base_auc = roc_auc_score(test["churned"], test["recency_days"])
    top = np.argsort(-p)[: max(len(p) // 10, 1)]
    prec_top = float(test["churned"].to_numpy()[top].mean())
    rate = float(test["churned"].mean())
    # final model on both snapshots
    both = pd.concat([train, test])
    clf2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0).fit(both[CHURN_FEATS], both["churned"])
    act = f_now[f_now["revenue_12m"] > 0]
    prob = pd.Series(clf2.predict_proba(act[CHURN_FEATS])[:, 1], index=act.index)
    metrics = {"AUC (out of time)": float(auc), "AUC of recency alone": float(base_auc),
               "Churn rate in test window": rate, "Churn rate in top 10% riskiest": prec_top,
               "Lift in top 10%": prec_top / max(rate, 1e-9), "Train customers": int(len(train)),
               "Test customers": int(len(test)), "Window days": int(window)}
    return prob, metrics


def _churn_reason(r):
    why = []
    if r.recency_ratio >= 2:
        why.append(f"no order for {r.recency_days:.0f} days, {r.recency_ratio:.1f}x their usual {r.usual_gap_days:.0f} day gap")
    if pd.notna(r.trend_90d) and r.trend_90d <= -0.3:
        why.append(f"spend in the last 90 days down {-r.trend_90d:.0%}")
    if pd.notna(r.growth_12m) and r.growth_12m <= -0.25:
        why.append(f"12 month spend down {-r.growth_12m:.0%}")
    if r.categories_12m <= 1 and r.orders_12m > 0:
        why.append("buys from one product line only")
    return "; ".join(why) or "pattern similar to customers who stopped buying"


# ---------------------------------------------------------------------------- cross-sell

def _item_similarity(s, min_co):
    cust_idx = {c: i for i, c in enumerate(s["customer_code"].unique())}
    items = s["item_code"].unique()
    item_idx = {c: i for i, c in enumerate(items)}
    pair = s[["customer_code", "item_code"]].drop_duplicates()
    X = sparse.csr_matrix((np.ones(len(pair)), (pair["customer_code"].map(cust_idx), pair["item_code"].map(item_idx))),
                          shape=(len(cust_idx), len(items)))
    co = (X.T @ X).tocsr()  # item x item shared buyers
    n = np.asarray(X.sum(axis=0)).ravel()
    co = co.tocoo()
    m = (co.row != co.col) & (co.data >= min_co)
    r, c, d = co.row[m], co.col[m], co.data[m]
    cos = d / np.sqrt(n[r] * n[c])
    conf = d / n[r]
    sim = sparse.csr_matrix((cos, (r, c)), shape=(len(items), len(items)))
    return X, sim, items, cust_idx, n, (r, c, d, conf)


STRATEGY_LABEL = {
    "cf": "Customers who buy what you buy (item-to-item co-purchase)",
    "catpop": "Best sellers in the product lines you already buy",
    "hybrid": "Blend of co-purchase and best sellers in your lines",
    "pop": "Overall best sellers (baseline)",
}


def _fit(hist, min_co):
    """Everything the recommenders need from a history window."""
    X, sim, items, cust_idx, n, extra = _item_similarity(hist, min_co)
    item_pos = {it: j for j, it in enumerate(items)}
    cats = hist.drop_duplicates("item_code").set_index("item_code")["category"].reindex(items).fillna("?").to_numpy()
    cat_codes, cat_names = pd.factorize(cats)
    # customer x category spend share
    sp = hist.groupby(["customer_code", "category"])["line_total"].sum().reset_index()
    sp = sp[sp["customer_code"].isin(cust_idx)]
    cmap = {c: k for k, c in enumerate(cat_names)}
    sp = sp[sp["category"].isin(cmap)]
    A = sparse.csr_matrix((sp["line_total"].to_numpy(float), (sp["customer_code"].map(cust_idx), sp["category"].map(cmap))),
                          shape=(len(cust_idx), len(cat_names)))
    rs = np.asarray(A.sum(axis=1)).ravel()
    A = sparse.diags(1 / np.where(rs > 0, rs, 1)) @ A
    # category x item: item share of its category's buyers
    tot = np.bincount(cat_codes, weights=n, minlength=len(cat_names))
    Cm = sparse.csr_matrix((n / np.maximum(tot[cat_codes], 1), (cat_codes, np.arange(len(items)))), shape=(len(cat_names), len(items)))
    return {"X": X.tocsr(), "sim": sim, "items": items, "cust_idx": cust_idx, "n": n, "A": A.tocsr(), "C": Cm, "extra": extra}


def _scores(m, rows, strategy):
    X = m["X"][rows]
    if strategy == "pop":
        sc = np.tile(m["n"].astype(float), (len(rows), 1))
    elif strategy == "cf":
        sc = (X @ m["sim"]).toarray()
    elif strategy == "catpop":
        sc = (m["A"][rows] @ m["C"]).toarray()
    else:  # hybrid: each part scaled to its row maximum
        a = (X @ m["sim"]).toarray()
        b = (m["A"][rows] @ m["C"]).toarray()
        a = a / np.maximum(a.max(axis=1, keepdims=True), 1e-12)
        b = b / np.maximum(b.max(axis=1, keepdims=True), 1e-12)
        sc = 0.5 * a + 0.5 * b
    sc[X.toarray() > 0] = -np.inf  # never suggest what they already buy
    return sc


def _hit_rate(m, fut, k, strategy, chunk=500):
    fut_sets = fut.groupby("customer_code")["item_code"].apply(set).to_dict()
    inv = {v: c for c, v in m["cust_idx"].items()}
    users = hits = 0
    ids = [m["cust_idx"][c] for c in fut_sets if c in m["cust_idx"]]
    for k0 in range(0, len(ids), chunk):
        rows = ids[k0:k0 + chunk]
        sc = _scores(m, rows, strategy)
        top = np.argsort(-sc, axis=1)[:, :k]
        own = m["X"][rows]
        for r_, ci in enumerate(rows):
            owned = set(m["items"][own.getrow(r_).indices])
            new = fut_sets[inv[ci]] - owned
            if not new:
                continue
            users += 1
            hits += bool(set(m["items"][top[r_]]) & new)
    return hits / users if users else np.nan, users


def recommend(s, as_of, per_cust, min_co, items_df, stock_ok=None):
    """Pick the best recommender on a validation window, report it on a later test window,
    then produce suggestions for every customer with a reason for each."""
    win = pd.Timedelta(days=730)
    t_test = as_of - pd.Timedelta(days=90)
    t_val = as_of - pd.Timedelta(days=180)
    metrics = {}
    best = "hybrid"
    try:
        h_val = s[(s["doc_date"] <= t_val) & (s["doc_date"] > t_val - win)]
        f_val = s[(s["doc_date"] > t_val) & (s["doc_date"] <= t_test)]
        h_tst = s[(s["doc_date"] <= t_test) & (s["doc_date"] > t_test - win)]
        f_tst = s[s["doc_date"] > t_test]
        if len(h_val) and len(f_val) and len(f_tst):
            mv = _fit(h_val, min_co)
            val = {st: _hit_rate(mv, f_val, per_cust, st)[0] for st in STRATEGY_LABEL}
            best = max(["cf", "catpop", "hybrid"], key=lambda st: (val[st] if np.isfinite(val[st]) else -1))
            mt = _fit(h_tst, min_co)
            tst = {st: _hit_rate(mt, f_tst, per_cust, st) for st in STRATEGY_LABEL}
            short = {"cf": "Co-purchase", "catpop": "Line best sellers", "hybrid": "Blend", "pop": "Best sellers"}
            metrics = {"Method chosen": STRATEGY_LABEL[best], "Method short": short[best], "Customers tested": tst[best][1],
                       f"Hit rate @{per_cust}": tst[best][0],
                       f"Popularity baseline hit rate @{per_cust}": tst["pop"][0],
                       "Lift vs popularity": tst[best][0] / max(tst["pop"][0], 1e-9),
                       "Test window": f"{t_test:%Y-%m-%d} to {as_of:%Y-%m-%d}",
                       "All methods (test hit rate)": {STRATEGY_LABEL[k]: v[0] for k, v in tst.items()}}
    except Exception:  # noqa: BLE001
        pass
    hist = s[s["doc_date"] > as_of - win]
    m = _fit(hist, min_co)
    r, c, d, conf = m["extra"]
    items, n, X, sim = m["items"], m["n"], m["X"], m["sim"]
    conf_m = sparse.csr_matrix((conf, (r, c)), shape=sim.shape)
    shared_m = sparse.csr_matrix((d.astype(float), (r, c)), shape=sim.shape)
    inv = {v: k_ for k_, v in m["cust_idx"].items()}
    im = items_df.set_index("item_code")
    price = s.groupby("item_code")["unit_net"].median()
    qty = s.groupby(["customer_code", "item_code"])["quantity"].median().groupby("item_code").median()
    rate = metrics.get(f"Hit rate @{per_cust}", 0.1) / max(per_cust, 1)  # chance a single suggestion converts
    cat_of = hist.drop_duplicates("item_code").set_index("item_code")["category"]
    share_rows = hist.groupby(["customer_code", "category"])["line_total"].sum()
    rows = []
    all_ids = list(range(X.shape[0]))
    for k0 in range(0, len(all_ids), 500):
        ids = all_ids[k0:k0 + 500]
        sc = _scores(m, ids, best)
        top = np.argsort(-sc, axis=1)[:, : per_cust * 2]
        for r_, ci in enumerate(ids):
            own = X.getrow(ci).indices
            cust = inv[ci]
            got = 0
            for j in top[r_]:
                if not np.isfinite(sc[r_, j]) or sc[r_, j] <= 0 or got >= per_cust:
                    continue
                item = items[j]
                if stock_ok is not None and item in stock_ok and not stock_ok[item]:
                    continue
                got += 1
                col = sim[own, j].toarray().ravel() if len(own) else np.array([])
                if len(col) and col.max() > 0:
                    src = own[int(np.argmax(col))]
                    why = (f"Buys {items[src]} ({im['item_name'].get(items[src], '')}). {conf_m[src, j]:.0%} of the {n[src]:.0f} "
                           f"customers who buy it also buy {item} ({shared_m[src, j]:.0f} customers).")
                    because = items[src]
                else:
                    cat = cat_of.get(item, "")
                    sh = share_rows.get((cust, cat), 0) / max(share_rows.get(cust, pd.Series(dtype=float)).sum(), 1e-9)
                    why = (f"{cat} is {sh:.0%} of this customer's spend, and {item} is one of the most bought {cat} "
                           f"items they do not buy yet ({n[j]:.0f} customers buy it).")
                    because = cat
                val = float(price.get(item, np.nan) * qty.get(item, 1))
                rows.append({"customer_code": cust, "rank": got, "item_code": item, "item_name": im["item_name"].get(item, item),
                             "category": im["category"].get(item, ""), "score": float(sc[r_, j]), "typical_order_value": val,
                             "expected_value": val * rate, "because": because, "why": why})
    rec_df = pd.DataFrame(rows)
    pairs = pd.DataFrame({"a": items[r], "b": items[c], "shared_customers": d, "confidence": conf})
    pairs["lift"] = pairs["confidence"] / (n[c] / X.shape[0])
    pairs = pairs[pairs["a"] < pairs["b"]].sort_values(["lift", "shared_customers"], ascending=False)
    pairs["a_name"] = pairs["a"].map(im["item_name"])
    pairs["b_name"] = pairs["b"].map(im["item_name"])
    pairs = pairs[pairs["shared_customers"] >= max(min_co, 8)].head(300).reset_index(drop=True)
    return rec_df, metrics, pairs


# ---------------------------------------------------------------------------- upsell / wallet share

def wallet_gaps(s, f, as_of, peer_col):
    cur = s[s["doc_date"] > as_of - pd.Timedelta(days=365)]
    spend = cur.pivot_table(index="customer_code", columns="category", values="line_total", aggfunc="sum", fill_value=0)
    peer = f.reindex(spend.index)[peer_col].fillna("Other")
    rows = []
    for grp, idx in peer.groupby(peer).groups.items():
        sub = spend.loc[idx]
        if len(sub) < 8:
            continue
        for cat in sub.columns:
            buyers = sub[cat] > 0
            share = buyers.mean()
            if share < 0.35:
                continue
            med = sub.loc[buyers, cat].median()
            gap = (med - sub[cat]).clip(lower=0)
            # only customers who are active overall and clearly under-buy this line
            tot = sub.sum(axis=1)
            mask = (gap > 0.5 * med) & (tot > sub.sum(axis=1).median() * 0.5)
            for c in sub.index[mask]:
                rows.append({"customer_code": c, "peer_group": grp, "category": cat, "their_spend_12m": float(sub.at[c, cat]),
                             "peer_median_spend_12m": float(med), "gap": float(gap[c]), "peers_buying_share": float(share),
                             "why": f"{share:.0%} of {grp} customers buy {cat}, typically ${med:,.0f} a year. This customer spent ${sub.at[c, cat]:,.0f}."})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("gap", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------- geography

def geography(s, f, as_of, market):
    cur = s[s["doc_date"] > as_of - pd.Timedelta(days=365)]
    prev = s[(s["doc_date"] <= as_of - pd.Timedelta(days=365)) & (s["doc_date"] > as_of - pd.Timedelta(days=730))]
    def reg(d):
        return np.where(d["ship_country"].fillna("US") == "US", d["ship_state"].fillna("Unknown"), d["ship_country"])
    cur = cur.assign(region=reg(cur))
    prev = prev.assign(region=reg(prev))
    g = cur.groupby(["ship_country", "region"]).agg(revenue_12m=("line_total", "sum"), customers=("customer_code", "nunique"),
                                                    orders=("doc_entry", "nunique")).reset_index()
    pv = prev.groupby("region")["line_total"].sum()
    g["revenue_prior_12m"] = g["region"].map(pv).fillna(0)
    g["growth"] = np.where(g["revenue_prior_12m"] > 0, g["revenue_12m"] / g["revenue_prior_12m"] - 1, np.nan)
    g["revenue_per_customer"] = g["revenue_12m"] / g["customers"].clip(lower=1)
    g = g.rename(columns={"ship_country": "country"})
    if market is not None and len(market):
        mk = market.copy()
        mk["region"] = mk["region"].astype(str)
        g = g.merge(mk[["country", "region", "potential"]], on=["country", "region"], how="outer")
        g["revenue_12m"] = g["revenue_12m"].fillna(0)
        g["customers"] = g["customers"].fillna(0)
        g["potential_share"] = g["potential"] / g["potential"].sum()
        g["revenue_share"] = g["revenue_12m"] / g["revenue_12m"].sum()
        g["penetration_index"] = g["revenue_share"] / g["potential_share"]
        med_rate = (g["revenue_12m"] / g["potential"]).replace([np.inf, -np.inf], np.nan).median()
        g["revenue_at_median_penetration"] = g["potential"] * med_rate
        g["opportunity"] = (g["revenue_at_median_penetration"] - g["revenue_12m"]).clip(lower=0)
        g["flag"] = np.where(g["customers"] == 0, "Untapped",
                             np.where(g["penetration_index"] < 0.5, "Underserved", np.where(g["penetration_index"] > 1.5, "Strong", "In line")))
    else:
        g["flag"] = np.where(g["growth"] > 0.15, "Growing", np.where(g["growth"] < -0.15, "Shrinking", "Stable"))
    return g.sort_values("revenue_12m", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------- pricing

def pricing(s, f, as_of, settings):
    sp = settings["sales"]
    pct = float(sp.get("price_floor_percentile", 25)) / 100
    cur = s[s["doc_date"] > as_of - pd.Timedelta(days=365)].copy()
    cur["peer"] = cur["customer_code"].map(f["price_tier"] if "price_tier" in f and f["price_tier"].notna().any() else
                                           (f["customer_type"] if "customer_type" in f else pd.Series(dtype=str))).fillna("All")
    grp = cur.groupby(["item_code", "peer"])["unit_net"]
    cur["p_floor"] = grp.transform(lambda x: x.quantile(pct))
    cur["p_med"] = grp.transform("median")
    cur["n_peer"] = grp.transform("size")
    low = cur[(cur["n_peer"] >= 8) & (cur["unit_net"] < cur["p_floor"])].copy()
    low["leakage"] = (low["p_med"] - low["unit_net"]) * low["quantity"]
    leak = low.groupby(["customer_code", "peer"]).agg(lines=("leakage", "size"), leakage=("leakage", "sum"),
                                                     items=("item_code", "nunique")).reset_index()
    leak = leak.sort_values("leakage", ascending=False).reset_index(drop=True)
    leak["why"] = [f"{r.lines} lines on {r.items} items priced below the {pct:.0%} mark of what other {r.peer} customers pay for the same item."
                   for r in leak.itertuples()]
    summary = {"Revenue last 12 months": float(cur["line_total"].sum()),
               "Lines below peer price floor": int(len(low)),
               "Gap to peer median on those lines": float(low["leakage"].sum()),
               "Share of revenue": float(low["leakage"].sum() / max(cur["line_total"].sum(), 1e-9))}
    # discount discipline: discount should rise with customer size, not fall
    disc = pd.DataFrame()
    if cur["discount_pct"].notna().any():
        dd = cur.groupby("customer_code").agg(avg_discount=("discount_pct", "mean"), revenue=("line_total", "sum")).reset_index()
        dd["size_band"] = pd.qcut(dd["revenue"].rank(method="first"), 4, labels=["Smallest 25%", "Lower mid", "Upper mid", "Largest 25%"])
        disc = dd.groupby("size_band", observed=True).agg(customers=("customer_code", "size"), avg_discount_pct=("avg_discount", "mean"),
                                                          revenue=("revenue", "sum")).reset_index()
        big_disc_small = dd[(dd["size_band"] == "Smallest 25%") & (dd["avg_discount"] > dd["avg_discount"].quantile(0.75))]
        summary["Small customers with top-quartile discounts"] = int(len(big_disc_small))
    # category price sensitivity: how order size responds when the same customer pays a different
    # price for the same item in a different month. Customer x item fixed effects remove the
    # "big customers get bigger discounts" mix effect; month effects remove seasonality and trend.
    el_rows = []
    s2 = s[s["doc_date"] > as_of - pd.Timedelta(days=730)].copy()
    s2["month"] = s2["doc_date"].dt.to_period("M")
    m = s2.groupby(["category", "customer_code", "item_code", "month"]).agg(q=("quantity", "sum"), p=("unit_net", "median")).reset_index()
    m = m[(m["q"] > 0) & (m["p"] > 0)]
    for cat, g in m.groupby("category"):
        g = g.copy()
        key = g["customer_code"].astype(str) + "|" + g["item_code"].astype(str)
        g = g[key.map(key.value_counts()) >= 3]
        if len(g) < 60:
            el_rows.append({"category": cat, "observations": len(g), "price_sensitivity": np.nan, "low": np.nan, "high": np.nan,
                            "reading": "Too few repeat purchases to measure. Run a controlled price test."})
            continue
        key = g["customer_code"].astype(str) + "|" + g["item_code"].astype(str)
        g["lq"] = np.log(g["q"])
        g["lp"] = np.log(g["p"])
        g["lq_d"] = g["lq"] - g.groupby(key)["lq"].transform("mean")
        g["lp_d"] = g["lp"] - g.groupby(key)["lp"].transform("mean")
        g["lq_d"] -= g.groupby("month")["lq_d"].transform("mean")
        g["lp_d"] -= g.groupby("month")["lp_d"].transform("mean")
        var = (g["lp_d"] ** 2).sum()
        if var <= 1e-6 or g["lp_d"].std() < 0.01:
            el_rows.append({"category": cat, "observations": len(g), "price_sensitivity": np.nan, "low": np.nan, "high": np.nan,
                            "reading": "Prices barely move for the same customer and item. Run a controlled price test."})
            continue
        b_ = (g["lp_d"] * g["lq_d"]).sum() / var
        resid = g["lq_d"] - b_ * g["lp_d"]
        se = np.sqrt((resid ** 2).sum() / max(len(g) - 2, 1) / var)
        lo, hi = b_ - 1.96 * se, b_ + 1.96 * se
        if hi < -1:
            reading = "Price sensitive: increases likely lose volume"
        elif lo > -1 and hi < 0.5:
            reading = "Low sensitivity: small increases likely keep most volume (confirm with a test)"
        else:
            reading = "Unclear: range too wide to act on without a test"
        el_rows.append({"category": cat, "observations": len(g), "price_sensitivity": float(b_), "low": float(lo), "high": float(hi),
                        "reading": reading})
    el = pd.DataFrame(el_rows)
    return leak, summary, el, disc


# ---------------------------------------------------------------------------- products and web demand

def products(s, items, as_of):
    cur = s[s["doc_date"] > as_of - pd.Timedelta(days=365)]
    prev = s[(s["doc_date"] <= as_of - pd.Timedelta(days=365)) & (s["doc_date"] > as_of - pd.Timedelta(days=730))]
    g = cur.groupby("item_code").agg(revenue_12m=("line_total", "sum"), units_12m=("quantity", "sum"),
                                     margin_12m=("gross_profit", "sum"), customers_12m=("customer_code", "nunique"))
    g["revenue_prior_12m"] = prev.groupby("item_code")["line_total"].sum()
    g = g.fillna({"revenue_prior_12m": 0})
    g["growth"] = np.where(g["revenue_prior_12m"] > 0, g["revenue_12m"] / g["revenue_prior_12m"] - 1, np.nan)
    first = s.groupby("item_code")["doc_date"].min()
    g["first_sale"] = first
    g["trend"] = np.select([g["first_sale"] > as_of - pd.Timedelta(days=365), g["growth"] > 0.2, g["growth"] < -0.2],
                           ["New", "Growing", "Declining"], "Stable")
    g["margin_pct"] = g["margin_12m"] / g["revenue_12m"].replace(0, np.nan)
    im = items.set_index("item_code")
    g["item_name"] = g.index.map(im["item_name"])
    g["category"] = g.index.map(im["category"])
    g = g.sort_values("revenue_12m", ascending=False)
    cum = g["revenue_12m"].cumsum() / max(g["revenue_12m"].sum(), 1e-9)
    g["abc"] = np.where(cum <= 0.8, "A", np.where(cum <= 0.95, "B", "C"))
    cat = g.groupby("category").agg(items_sold=("revenue_12m", "size"), revenue_12m=("revenue_12m", "sum"),
                                    revenue_prior_12m=("revenue_prior_12m", "sum"), margin_12m=("margin_12m", "sum")).reset_index()
    cat["growth"] = np.where(cat["revenue_prior_12m"] > 0, cat["revenue_12m"] / cat["revenue_prior_12m"] - 1, np.nan)
    cat["margin_pct"] = cat["margin_12m"] / cat["revenue_12m"].replace(0, np.nan)
    cat["share"] = cat["revenue_12m"] / max(cat["revenue_12m"].sum(), 1e-9)
    return g.reset_index(), cat.sort_values("revenue_12m", ascending=False).reset_index(drop=True)


def web_demand(web, stock, items, as_of):
    if web is None or len(web) == 0:
        return pd.DataFrame(), pd.DataFrame()
    w = web[web["search_date"] > as_of - pd.Timedelta(days=365)].copy()
    w["term"] = w["search_term"].astype(str).str.strip().str.lower()
    zero = w[w["results_count"].fillna(1) <= 0]
    un = zero.groupby("term").agg(searches=("term", "size"), accounts=("customer_code", "nunique"),
                                  last_seen=("search_date", "max")).reset_index().sort_values("searches", ascending=False)
    un["reading"] = np.where(un["accounts"] >= 5, "Repeated by many accounts: strong signal to source this part",
                             "Watch: few accounts so far")
    svs = pd.DataFrame()
    hit = w.dropna(subset=["item_code"])
    if len(hit) and stock is not None and len(stock):
        oh = stock.groupby("item_code")["on_hand"].sum()
        g = hit.groupby("item_code").agg(searches=("item_code", "size"), accounts=("customer_code", "nunique")).reset_index()
        g["on_hand_all_warehouses"] = g["item_code"].map(oh).fillna(0)
        g["item_name"] = g["item_code"].map(items.set_index("item_code")["item_name"])
        svs = g[g["on_hand_all_warehouses"] <= 0].sort_values("searches", ascending=False).reset_index(drop=True)
    return un.reset_index(drop=True), svs


# ---------------------------------------------------------------------------- orchestration

def run(tables, items, settings, as_of, stock_by_item=None) -> SalesResult:
    out = SalesResult()
    sp = settings["sales"]
    s = _lines(tables, items, settings, as_of)
    if len(s) == 0:
        out.notes.append("No sales lines with a customer code; sales modules skipped.")
        return out
    cust = tables.get("customers")
    f = customer_features(s, as_of, cust)
    seg, seg_t = rfm(f)
    f = f.join(seg[["R", "F", "M", "segment"]])
    out.segments = seg_t
    try:
        lab, pers = personas(s, f, as_of)
        f["persona"] = lab
        out.personas = pers
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"Personas skipped: {e}")
    try:
        prob, met = churn_model(s, f, as_of, int(sp.get("churn_window_days", 90)), cust)
        out.churn_metrics = met
        if prob is not None:
            f["churn_risk"] = prob
            f["revenue_at_risk"] = f["churn_risk"] * f["revenue_12m"]
            f["churn_reason"] = [_churn_reason(r) for r in f.itertuples()]
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"Churn model skipped: {e}")
    out.customers = f.reset_index().rename(columns={"index": "customer_code"})
    try:
        stock_ok = None
        if stock_by_item is not None:
            stock_ok = (stock_by_item > 0).to_dict()
        rec, met, pairs = recommend(s, as_of, int(sp.get("recommendations_per_customer", 10)), int(sp.get("min_co_buyers", 5)), items, stock_ok)
        out.recommendations, out.rec_metrics, out.bundles = rec, met, pairs
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"Cross-sell skipped: {e}")
    peer_col = "customer_type" if "customer_type" in f and f["customer_type"].notna().any() else ("persona" if "persona" in f else "segment")
    out.upsell = wallet_gaps(s, f, as_of, peer_col)
    out.geo = geography(s, f, as_of, tables.get("market_potential"))
    try:
        out.pricing_leakage, out.pricing_summary, out.elasticity, out.discount_check = pricing(s, f, as_of, settings)
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"Pricing skipped: {e}")
    out.products, out.categories = products(s, items, as_of)
    out.unmet_demand, out.search_vs_stock = web_demand(tables.get("web_searches"), tables.get("stock"), items, as_of)
    out.plays = sales_plays(out)
    return out


def sales_plays(r: SalesResult, top=300) -> pd.DataFrame:
    rows = []
    c = r.customers.set_index("customer_code") if len(r.customers) else pd.DataFrame()
    rep = c["sales_rep"] if "sales_rep" in c else pd.Series(dtype=str)
    if "churn_risk" in c:
        risky = c[(c["churn_risk"] >= 0.5) & (c["revenue_12m"] > 0)].sort_values("revenue_at_risk", ascending=False).head(150)
        for cc, x in risky.iterrows():
            rows.append({"customer_code": cc, "play": "Win back", "value": float(x["revenue_at_risk"]), "sales_rep": rep.get(cc),
                         "detail": f"{x['churn_risk']:.0%} churn risk, ${x['revenue_12m']:,.0f} last 12 months. {x['churn_reason']}."})
    if len(r.upsell):
        g = r.upsell.groupby("customer_code").agg(gap=("gap", "sum"), cats=("category", lambda v: ", ".join(list(v)[:3]))).reset_index()
        for x in g.sort_values("gap", ascending=False).head(150).itertuples():
            rows.append({"customer_code": x.customer_code, "play": "Grow wallet share", "value": float(x.gap) * 0.25,
                         "sales_rep": rep.get(x.customer_code), "detail": f"Under-buys {x.cats} vs similar customers; gap ${x.gap:,.0f} a year (value shown assumes 25% captured)."})
    if len(r.recommendations):
        g = r.recommendations.groupby("customer_code").agg(v=("expected_value", "sum"), items=("item_code", lambda v: ", ".join(list(v)[:3]))).reset_index()
        for x in g.sort_values("v", ascending=False).head(150).itertuples():
            rows.append({"customer_code": x.customer_code, "play": "Cross-sell", "value": float(x.v), "sales_rep": rep.get(x.customer_code),
                         "detail": f"Offer {x.items} (see Cross-sell tab for the reason on each)."})
    if len(r.pricing_leakage):
        for x in r.pricing_leakage.head(80).itertuples():
            rows.append({"customer_code": x.customer_code, "play": "Price review", "value": float(x.leakage) * 0.5,
                         "sales_rep": rep.get(x.customer_code), "detail": x.why + f" Gap to peer median ${x.leakage:,.0f} (value shown assumes half recovered)."})
    p = pd.DataFrame(rows)
    if len(p):
        p = p.sort_values("value", ascending=False).head(top).reset_index(drop=True)
        if len(c):
            for col in ("segment", "customer_type", "state", "country"):
                if col in c:
                    p[col] = p["customer_code"].map(c[col])
    return p
