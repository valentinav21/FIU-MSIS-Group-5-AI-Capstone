"""Generate a realistic SYNTHETIC dataset shaped like a FORTPRO SAP Business One export.

Nothing here is FORTPRO data. Item codes, names and categories come from the public
fortpro.com catalog crawl (Oct 10, 2026). Every quantity, price, customer, supplier,
date and stock figure is invented, so the app can be tested end to end before the
real export arrives.

Files are written with standard SAP Business One column names (DocDate, ItemCode,
CardCode, WhsCode ...) to prove the loader maps real SAP exports without editing.

Usage:  python tools/generate_synthetic.py [--out data/sample] [--items 3169] [--seed 7]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED_CATALOG = ROOT / "tools" / "web_catalog_seed.xlsx"

START = pd.Timestamp("2023-01-02")
AS_OF = pd.Timestamp("2026-10-16")

CAT_PRICE = {  # median list price in USD by top category (invented)
    "Brake System": 85, "Engine Components": 140, "Suspension": 95, "Lighting": 45, "Driveline & Axles": 110,
    "Electrical": 70, "Exterior Parts": 160, "Interior Parts": 60, "Exhaust System": 120, "Accessories": 40,
    "Cabin, Engine & Hood Components": 75, "Steerings": 130, "Transmission": 55, "Hydraulics & PTO": 650,
    "Lucas Oil Products": 25, "Brand page only": 80,
}
US_STATES = {"FL": 0.30, "TX": 0.20, "GA": 0.07, "LA": 0.05, "AL": 0.03, "NC": 0.03, "SC": 0.02, "TN": 0.03,
             "OK": 0.02, "AR": 0.02, "MS": 0.02, "CA": 0.03, "AZ": 0.02, "NM": 0.01, "IL": 0.02, "OH": 0.02,
             "PA": 0.02, "NJ": 0.02, "NY": 0.02, "VA": 0.02, "MO": 0.01}
EXPORT = {"DO": 0.16, "CO": 0.12, "PA": 0.08, "GT": 0.08, "HN": 0.07, "CR": 0.06, "EC": 0.07, "PE": 0.06,
          "JM": 0.05, "HT": 0.04, "TT": 0.04, "BS": 0.03, "SV": 0.05, "NI": 0.03, "CL": 0.03, "VE": 0.03}
HOU_STATES = {"TX", "LA", "OK", "AR", "NM", "AZ", "CA", "MO", "MS", "IL"}
CHANNELS_US = ["Portal", "Phone", "Counter", "Email", "App", "Marketplace"]
CHANNELS_US_P = [0.30, 0.30, 0.15, 0.13, 0.04, 0.08]


def _norm(p):
    p = np.asarray(p, float)
    return p / p.sum()


def load_catalog(n_items: int | None, rng) -> pd.DataFrame:
    p = pd.read_excel(SEED_CATALOG, sheet_name="Products", dtype={"SKU": str})
    p = p.rename(columns={"SKU": "ItemCode", "Product Name": "ItemName", "Top Category": "TopCategory",
                          "Subcategory": "Subcategory", "Primary Category": "PrimaryCategory", "Brand": "Brand"})
    p["Brand"] = p["Brand"].replace({"Not tagged": ""})
    is_f = p["ItemCode"].str.match(r"^F\d{6}")
    p.loc[is_f & (p["Brand"] == ""), "Brand"] = "FORTPRO"
    p["Subcategory"] = p["Subcategory"].fillna(p["TopCategory"])
    p["PrimaryCategory"] = p["PrimaryCategory"].fillna(p["Subcategory"])
    if n_items and n_items < len(p):
        p = p.sample(n_items, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    return p[["ItemCode", "ItemName", "TopCategory", "Subcategory", "PrimaryCategory", "Brand"]].reset_index(drop=True)


def generate(out: Path, n_items: int | None = None, seed: int = 7, n_customers: int = 650):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    items = load_catalog(n_items, rng)
    n = len(items)

    # ------------------------------------------------------------- item economics
    med = items["TopCategory"].map(CAT_PRICE).fillna(80).to_numpy()
    list_price = np.round(med * rng.lognormal(0, 0.55, n), 2).clip(3)
    margin = rng.uniform(0.30, 0.55, n)
    cost = np.round(list_price * (1 - margin) * 0.9, 2)
    pop = rng.lognormal(0, 1.6, n)  # popularity: heavy long tail
    pop[rng.random(n) < 0.06] *= 0.02  # near-dead items
    lumpy = rng.random(n) < 0.35       # some items sell in big lumps
    size_mu = np.where(lumpy, rng.uniform(1.0, 2.2, n), rng.uniform(0.0, 0.8, n))
    pack = rng.choice([1, 1, 1, 2, 4, 5, 10, 12], n)
    items["ListPrice"] = list_price
    items["AvgPrice"] = cost
    items["OrdrMulti"] = pack
    items["MinOrdrQty"] = (pack * rng.choice([1, 2, 5, 10], n)).astype(int)

    # ------------------------------------------------------------- suppliers
    sup_rows = []
    imp_countries = ["CN", "CN", "CN", "TW", "TW", "IN", "TR", "BR", "CN", "KR", "CN", "TH", "MX", "CN"]
    for i, c in enumerate(imp_countries):
        overland = c == "MX"
        sup_rows.append(dict(CardCode=f"V{100 + i}", CardName=f"Import supplier {i + 1}", Country=c,
                             lt_med=rng.uniform(28, 40) if overland else rng.uniform(60, 95),
                             lt_sig=rng.uniform(0.12, 0.35), late_tail=rng.uniform(0.02, 0.18),
                             price_mult=rng.uniform(0.9, 1.05), fill=rng.uniform(0.9, 1.0)))
    for i in range(8):
        sup_rows.append(dict(CardCode=f"V{200 + i}", CardName=f"US distributor {i + 1}", Country="US",
                             lt_med=rng.uniform(5, 16), lt_sig=rng.uniform(0.15, 0.4), late_tail=rng.uniform(0.01, 0.08),
                             price_mult=rng.uniform(1.12, 1.35), fill=rng.uniform(0.93, 1.0)))
    sup = pd.DataFrame(sup_rows)
    imp = sup[sup.Country != "US"].reset_index(drop=True)
    dom = sup[sup.Country == "US"].reset_index(drop=True)
    own = items["Brand"].eq("FORTPRO").to_numpy()
    tops = sorted(items["TopCategory"].unique())
    top_to_imp = {t: imp.CardCode.iloc[i % len(imp)] for i, t in enumerate(tops)}
    primary = np.where(own, items["TopCategory"].map(top_to_imp), dom.CardCode.to_numpy()[rng.integers(0, len(dom), n)])
    # some own-brand items have a second import source
    items["CardCode"] = primary
    isup = []
    sup_idx = sup.set_index("CardCode")
    for k in range(n):
        p0 = primary[k]
        s = sup_idx.loc[p0]
        isup.append(dict(ItemCode=items.ItemCode[k], VendorCode=p0, Price=round(cost[k] * s.price_mult, 2),
                         MinOrdrQty=int(items.MinOrdrQty[k]), OrdrMulti=int(pack[k]), LeadTime=int(round(s.lt_med)),
                         LandedPct=0.14 if s.Country != "US" else 0.03))
        if rng.random() < 0.30:
            alt = (dom if s.Country != "US" and rng.random() < 0.6 else imp).sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            if alt.CardCode != p0:
                isup.append(dict(ItemCode=items.ItemCode[k], VendorCode=alt.CardCode, Price=round(cost[k] * alt.price_mult, 2),
                                 MinOrdrQty=int(max(pack[k], items.MinOrdrQty[k] // (2 if alt.Country == "US" else 1))),
                                 OrdrMulti=int(pack[k]), LeadTime=int(round(alt.lt_med)),
                                 LandedPct=0.14 if alt.Country != "US" else 0.03))
    isup = pd.DataFrame(isup)
    items["LeadTime"] = items["CardCode"].map(sup_idx["lt_med"]).round().astype(int)

    # ------------------------------------------------------------- customers
    types = rng.choice(["Fleet", "Repair shop", "Parts dealer", "Export dealer", "Owner operator"], n_customers,
                       p=[0.22, 0.30, 0.16, 0.17, 0.15])
    states, countries = [], []
    for t in types:
        if t == "Export dealer":
            countries.append(rng.choice(list(EXPORT), p=_norm(list(EXPORT.values()))))
            states.append("")
        else:
            countries.append("US")
            states.append(rng.choice(list(US_STATES), p=_norm(list(US_STATES.values()))))
    cust = pd.DataFrame({"CardCode": [f"C{10000 + i}" for i in range(n_customers)], "CustomerType": types,
                         "Country": countries, "State": states})
    cust["GroupName"] = cust["CustomerType"].map({"Fleet": "Fleet", "Repair shop": "Shops", "Parts dealer": "Resellers",
                                                  "Export dealer": "Export", "Owner operator": "Retail"})
    size = {"Fleet": 1.6, "Repair shop": 1.0, "Parts dealer": 2.4, "Export dealer": 1.4, "Owner operator": 0.25}
    rate = cust["CustomerType"].map(size).to_numpy() * rng.lognormal(0, 0.8, n_customers) * 0.45  # orders per week
    disc = cust["CustomerType"].map({"Fleet": 0.08, "Repair shop": 0.05, "Parts dealer": 0.15, "Export dealer": 0.12,
                                     "Owner operator": 0.0}).to_numpy() + rng.uniform(-0.03, 0.04, n_customers)
    disc = disc.clip(0, 0.25)
    cust["ListNum"] = np.select([disc >= 0.12, disc >= 0.06], ["Tier 1", "Tier 2"], "Tier 3")
    weeks = int((AS_OF - START).days // 7) + 1
    start_w = np.where(rng.random(n_customers) < 0.25, rng.integers(0, weeks - 20, n_customers), 0)
    churn_w = np.where(rng.random(n_customers) < 0.18, rng.integers(30, weeks + 40, n_customers), weeks + 999)
    growth = rng.normal(0.0, 0.25, n_customers)  # yearly growth per customer
    cat_size = items["TopCategory"].value_counts().reindex(tops).to_numpy(float)
    persona = rng.dirichlet(0.6 * cat_size / cat_size.mean() + 0.02, n_customers)  # buyers lean to big lines
    top_idx = items["TopCategory"].map({t: i for i, t in enumerate(tops)}).to_numpy()
    cust["first_week"] = start_w
    first_dates = [START + pd.Timedelta(weeks=int(w)) + pd.Timedelta(days=int(rng.integers(0, 6))) for w in start_w]
    cust["CreateDate"] = pd.to_datetime(first_dates)
    whs_of = np.where((cust["State"].isin(HOU_STATES)).to_numpy(), "HOU", "MIA")

    # companions: items in the same subcategory are bought together
    sub_codes = items.groupby("Subcategory").indices
    item_sub = items["Subcategory"].to_numpy()
    cat_items = {i: np.where(top_idx == i)[0] for i in range(len(tops))}
    cat_w = {i: _norm(pop[v]) for i, v in cat_items.items()}
    sub_w = {s: _norm(pop[v]) for s, v in sub_codes.items()}

    # ------------------------------------------------------------- orders
    rows = []
    doc = 100000
    season = lambda w: 1 + 0.10 * np.sin(2 * np.pi * (w / 52.18 - 0.15))
    reps = ["Rep A", "Rep B", "Rep C", "Rep D", "Rep E", "Export desk"]
    for w in range(weeks):
        active = (w >= start_w) & (w < churn_w)
        lam = rate * active * season(w) * np.exp(growth * w / 52.18) * (1 + 0.04 * w / 52.18)
        k = rng.poisson(lam)
        for c in np.nonzero(k)[0]:
            for _ in range(k[c]):
                doc += 1
                day = START + pd.Timedelta(weeks=w, days=int(rng.integers(0, 6)))
                if day > AS_OF:
                    continue
                nl = 1 + rng.poisson(1.3 if types[c] != "Export dealer" else 4.0)
                cat = rng.choice(len(tops), p=persona[c])
                first = rng.choice(cat_items[cat], p=cat_w[cat])
                lines = [first]
                for _ in range(nl - 1):
                    if rng.random() < 0.45:
                        s = item_sub[lines[0]]
                        lines.append(rng.choice(sub_codes[s], p=sub_w[s]))
                    else:
                        cc = rng.choice(len(tops), p=persona[c])
                        lines.append(rng.choice(cat_items[cc], p=cat_w[cc]))
                ch = rng.choice(CHANNELS_US, p=CHANNELS_US_P) if types[c] != "Export dealer" else "Export"
                rep = "Export desk" if ch == "Export" else reps[c % 5]
                for ln, it in enumerate(dict.fromkeys(lines)):
                    mult = 3.0 if types[c] in ("Export dealer", "Parts dealer") else 1.0
                    q = max(1, int(round(rng.lognormal(size_mu[it], 0.6) * mult)))
                    rows.append((doc, ln, day, it, q, c, ch, rep))
    s = pd.DataFrame(rows, columns=["DocEntry", "LineNum", "DocDate", "it", "Quantity", "c", "U_Channel", "SlpName"])
    s["ItemCode"] = items.ItemCode.to_numpy()[s.it]
    s["CardCode"] = cust.CardCode.to_numpy()[s.c]
    s["WhsCode"] = whs_of[s.c]
    # Houston also ships some Southeast orders when Miami is busy
    flip = rng.random(len(s)) < 0.05
    s.loc[flip, "WhsCode"] = np.where(s.loc[flip, "WhsCode"] == "MIA", "HOU", "MIA")
    # price changes once a year (+3%), customer tier discount, some deal noise
    yr = (s["DocDate"].dt.year - 2023).to_numpy()
    lp = list_price[s.it] * (1.03 ** yr)
    dpc = (disc[s.c] + rng.normal(0, 0.02, len(s))).clip(0, 0.35)
    s["DiscPrcnt"] = np.round(dpc * 100, 2)
    s["Price"] = np.round(lp * (1 - dpc), 2)
    s["LineTotal"] = np.round(s["Price"] * s["Quantity"], 2)
    s["GrssProfit"] = np.round(s["LineTotal"] - cost[s.it] * (1.02 ** yr) * s["Quantity"], 2)
    s["State"] = cust.State.to_numpy()[s.c]
    s["Country"] = cust.Country.to_numpy()[s.c]
    s["CANCELED"] = np.where(rng.random(len(s)) < 0.01, "Y", "N")
    # export lines leave from Miami through the forwarder
    s.loc[s["U_Channel"] == "Export", "WhsCode"] = "MIA"
    s = s.sort_values(["DocDate", "DocEntry", "LineNum"]).reset_index(drop=True)

    # ------------------------------------------------------------- returns
    ret = s[(s.CANCELED == "N") & (rng.random(len(s)) < 0.02)].copy()
    ret["DocDate"] = ret["DocDate"] + pd.to_timedelta(rng.integers(3, 30, len(ret)), unit="D")
    ret = ret[ret["DocDate"] <= AS_OF]
    ret["Quantity"] = np.maximum(1, (ret["Quantity"] * rng.uniform(0.3, 1.0, len(ret))).round()).astype(int)
    ret["LineTotal"] = np.round(ret["Quantity"] * ret["Price"], 2)
    ret["U_Reason"] = rng.choice(["Wrong part", "Damaged", "Not needed", "Warranty"], len(ret), p=[0.45, 0.2, 0.25, 0.1])

    # ------------------------------------------------------------- replenishment simulation (weekly)
    whs = ["MIA", "HOU"]
    ok = s[s.CANCELED == "N"]
    week_idx = ((ok["DocDate"] - START).dt.days // 7).to_numpy()
    wi = np.where(ok["WhsCode"].to_numpy() == "MIA", 0, 1)
    D = np.zeros((n, 2, weeks))
    np.add.at(D, (ok.it.to_numpy(), wi, week_idx), ok["Quantity"].to_numpy())
    avg = D[:, :, :26].mean(axis=2) + D.mean(axis=2) * 0.5 + 0.02
    lt_w = items["CardCode"].map(sup_idx["lt_med"]).to_numpy() / 7.0
    # the "current SAP" static policy: min = demand over lead time + 2 weeks, max = min + 8 weeks
    min_s = np.ceil(avg * (lt_w[:, None] + 2))
    max_s = np.ceil(min_s + avg * 8)
    on_hand = max_s.copy()
    po_rows, gr_rows = [], []
    po_doc = 500000
    gr_doc = 700000
    opts_by_item = {}
    for r in isup.itertuples(index=False):
        opts_by_item.setdefault(r.ItemCode, []).append((r.VendorCode, float(r.Price)))
    sv_by = {r.CardCode: r for r in sup.itertuples(index=False)}
    codes = items.ItemCode.to_numpy()
    on_order = np.zeros_like(on_hand)
    arrivals = {}  # week -> list of (item, wh, qty)
    for w in range(weeks):
        for (it, h, q) in arrivals.pop(w, []):
            on_hand[it, h] += q
            on_order[it, h] -= q
        on_hand -= D[:, :, w]
        ip = on_hand + on_order
        need = np.argwhere(ip <= min_s)
        for it, h in need:
            if rng.random() < 0.15:  # buyers do not always act the same week
                continue
            q = max(max_s[it, h] - ip[it, h], items.MinOrdrQty[it])
            q = int(np.ceil(q / pack[it]) * pack[it])
            opts = opts_by_item[codes[it]]
            vend, base_price = opts[0] if (len(opts) == 1 or rng.random() < 0.8) else opts[1]
            sv = sv_by[vend]
            lt = rng.lognormal(np.log(sv.lt_med), sv.lt_sig)
            if rng.random() < sv.late_tail:
                lt += rng.uniform(20, 60)
            po_doc += 1
            po_date = START + pd.Timedelta(weeks=w, days=int(rng.integers(0, 5)))
            price = base_price * (1.02 ** (po_date.year - 2023))
            promised = po_date + pd.Timedelta(days=int(round(sv.lt_med)))
            arrive = po_date + pd.Timedelta(days=float(lt))
            filled = q if rng.random() < sv.fill else int(round(q * rng.uniform(0.5, 0.9)))
            parts = [filled] if (rng.random() > 0.12 or filled < 2) else [int(filled * 0.6), filled - int(filled * 0.6)]
            recv_total, pending = 0, False
            for pi, pq in enumerate(parts):
                a = (arrive + pd.Timedelta(days=21 * pi)).normalize()
                aw = max(int((a - START).days // 7), w + 1)
                if a <= AS_OF:
                    gr_doc += 1
                    gr_rows.append(dict(DocEntry=gr_doc, DocDate=a, ItemCode=codes[it], Quantity=pq,
                                        BaseEntry=po_doc, BaseLine=0, CardCode=vend, WhsCode=whs[h]))
                    recv_total += pq
                else:
                    pending = True
                arrivals.setdefault(aw, []).append((it, h, pq))
                on_order[it, h] += pq
            open_q = q - recv_total if pending else 0
            po_rows.append(dict(DocEntry=po_doc, LineNum=0, DocDate=po_date, ItemCode=codes[it], CardCode=vend,
                                Quantity=q, Price=round(price, 2), Currency="USD", WhsCode=whs[h], ShipDate=promised,
                                OpenQty=int(max(0, open_q)), LineStatus="O" if open_q > 0 else "C"))
    po = pd.DataFrame(po_rows)
    gr = pd.DataFrame(gr_rows)
    on_hand = np.maximum(on_hand, 0)

    # ------------------------------------------------------------- open orders (last 10 days of demand still unshipped)
    recent = s[(s.CANCELED == "N") & (s.DocDate > AS_OF - pd.Timedelta(days=10))].sample(frac=0.35, random_state=seed)
    oo = pd.DataFrame({"ItemCode": recent.ItemCode, "OpenQty": recent.Quantity, "WhsCode": recent.WhsCode,
                       "ShipDate": recent.DocDate + pd.Timedelta(days=7), "CardCode": recent.CardCode})
    committed = oo.groupby(["ItemCode", "WhsCode"])["OpenQty"].sum()

    st = []
    openpo = po[po.OpenQty > 0].groupby(["ItemCode", "WhsCode"])["OpenQty"].sum()
    for it in range(n):
        for h, wc in enumerate(whs):
            key = (items.ItemCode[it], wc)
            if D[it, h].sum() == 0 and on_hand[it, h] == 0 and key not in openpo:
                continue
            st.append(dict(ItemCode=items.ItemCode[it], WhsCode=wc, OnHand=int(on_hand[it, h]),
                           IsCommited=int(committed.get(key, 0)), OnOrder=int(openpo.get(key, 0)),
                           MinStock=int(min_s[it, h]), MaxStock=int(max_s[it, h])))
    stock = pd.DataFrame(st)

    # ------------------------------------------------------------- web searches (last 12 months)
    wk = 52
    n_search = 26000
    sd = AS_OF - pd.to_timedelta(rng.integers(0, wk * 7, n_search), unit="D")
    hit_items = rng.choice(n, n_search, p=_norm(pop + 0.05))
    terms = items.ItemCode.to_numpy()[hit_items].astype(object)
    res_cnt = rng.integers(1, 12, n_search)
    clicked = terms.copy()
    not_carried = ["Donaldson P551000 filter", "toolbox 48 inch", "engine oil pressure sensor", "torque rod bushing kit",
                   "wheel ABS sensor", "axle shaft oil seal", "cab air spring", "coolant temperature sensor",
                   "fuel pressure sensor", "NTN wheel bearing", "lubricant 77 grease", "turbo intake hose"]
    nc_w = _norm([9, 6, 7, 8, 9, 5, 4, 6, 3, 5, 2, 3])
    miss = rng.random(n_search) < 0.09
    terms[miss] = rng.choice(not_carried, miss.sum(), p=nc_w)
    res_cnt[miss] = 0
    clicked[miss] = ""
    acct = np.where(rng.random(n_search) < 0.6, cust.CardCode.to_numpy()[rng.integers(0, n_customers, n_search)], "")
    web = pd.DataFrame({"SearchDate": sd.normalize(), "SearchTerm": terms, "ResultsCount": res_cnt,
                        "ClickedItem": clicked, "Account": acct}).sort_values("SearchDate")

    # ------------------------------------------------------------- market potential placeholder
    mp = [dict(country="US", region=k, potential=round(v * 1000 * rng.uniform(0.6, 1.6)), source="SYNTHETIC placeholder")
          for k, v in US_STATES.items()]
    for extra in ["KY", "IN", "MI", "WI", "MN", "IA", "KS", "NE", "CO", "UT", "NV", "OR", "WA", "MD", "MA", "CT", "WV"]:
        mp.append(dict(country="US", region=extra, potential=round(rng.uniform(8, 40)), source="SYNTHETIC placeholder"))
    for k, v in EXPORT.items():
        mp.append(dict(country=k, region=k, potential=round(v * 600 * rng.uniform(0.5, 2.0)), source="SYNTHETIC placeholder"))
    mp = pd.DataFrame(mp)

    # ------------------------------------------------------------- write SAP-shaped files
    oitm = items.rename(columns={"TopCategory": "ItmsGrpNam", "PrimaryCategory": "U_SubFamily", "Brand": "FirmName"})
    oitm["validFor"] = "Y"
    oitm.loc[pop[: len(oitm)] < 0.03, "validFor"] = "N"
    oitm[["ItemCode", "ItemName", "ItmsGrpNam", "U_SubFamily", "FirmName", "AvgPrice", "ListPrice", "validFor",
          "MinOrdrQty", "OrdrMulti", "LeadTime", "CardCode"]].to_csv(out / "OITM.csv", index=False)
    s[["DocEntry", "LineNum", "DocDate", "ItemCode", "Quantity", "Price", "DiscPrcnt", "LineTotal", "GrssProfit",
       "CardCode", "WhsCode", "U_Channel", "State", "Country", "SlpName", "CANCELED"]].to_csv(out / "INV1.csv", index=False)
    ret[["DocDate", "ItemCode", "Quantity", "CardCode", "WhsCode", "LineTotal", "U_Reason"]].rename(
        columns={"U_Reason": "reason"}).to_csv(out / "RIN1.csv", index=False)
    stock.to_csv(out / "OITW.csv", index=False)
    po.to_csv(out / "POR1.csv", index=False)
    gr.to_csv(out / "PDN1.csv", index=False)
    sup[["CardCode", "CardName", "Country"]].to_csv(out / "suppliers.csv", index=False)
    isup.rename(columns={"LandedPct": "landed_cost_pct"}).to_csv(out / "ITM2.csv", index=False)
    oo.to_csv(out / "open_orders.csv", index=False)
    cust[["CardCode", "GroupName", "CustomerType", "State", "Country", "CreateDate", "ListNum"]].to_csv(out / "customers.csv", index=False)
    web.to_csv(out / "web_searches.csv", index=False)
    mp.to_csv(out / "market_potential.csv", index=False)
    big = s[(s.DocDate.dt.year == 2025) & (s.CANCELED == "N")].nlargest(1, "Quantity").iloc[0]
    pd.DataFrame([dict(item_code=big.ItemCode, customer_code=big.CardCode, start_date=big.DocDate.date(),
                       end_date=big.DocDate.date(), action="exclude",
                       note="SYNTHETIC example: one-off fleet project order, kept out of the forecast")]
                 ).to_csv(out / "events.csv", index=False)
    (out / "README_SYNTHETIC.txt").write_text(
        "SYNTHETIC DATA. Not FORTPRO data.\n"
        "Item codes, names and categories come from the public fortpro.com catalog (crawl of Oct 10, 2026).\n"
        "All quantities, prices, customers, suppliers, dates and stock are invented by tools/generate_synthetic.py\n"
        f"(seed {seed}). Use it to learn the app and to test before the real SAP export arrives.\n", encoding="utf-8")
    return {"items": len(items), "sales_lines": len(s), "customers": n_customers, "po_lines": len(po),
            "receipts": len(gr), "stock_rows": len(stock), "searches": len(web)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(generate(Path(a.out), a.items or None, a.seed))
