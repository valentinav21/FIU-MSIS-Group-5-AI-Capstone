"""Batch simulation of (s, S) stocking policies.

Textbook fill-rate formulas assume smooth demand and fixed lead times and badly misjudge
spare parts. Instead, each candidate reorder point is tested by running the warehouse
forward for two years of simulated weeks: simulated demand (zeros, lumps and all), weekly
review, orders up to the target level rounded to the supplier minimum and pack size,
and lead times drawn from the supplier's real history. All candidates for an item share
the same simulated demand and lead times, so their results are directly comparable.
"""
from __future__ import annotations

import numpy as np

from .probabilistic import simulate_periods

QLEVELS = np.array([0.0, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])


def candidates(x_sorted, extra=None):
    """Reorder point candidates spread over the lead time demand distribution."""
    q = np.quantile(x_sorted, QLEVELS[1:]) if len(x_sorted) else np.zeros(len(QLEVELS) - 1)
    top = q[-1] * 1.25 + 1
    c = np.unique(np.ceil(np.concatenate([[0.0], q, [top]])))
    return c


def simulate_batch(series, weeks=104, warm=26, n_paths=100, rng=None, block=13, kappa=1.0):
    """Simulate many items at once.

    series: list of dicts with keys
        mean_week (array of weekly mean demand, length >= 1), sizes, occ, sigma,
        lt_weeks (array of lead time samples in weeks), q (order quantity), moq, mult,
        cands (array of reorder points), extra (list of (s, S) pairs, e.g. today's SAP min/max)
    Returns list of dicts: fill (per candidate), onhand (per candidate), extra_fill, extra_onhand.
    """
    rng = rng or np.random.default_rng(0)
    B = len(series)
    if B == 0:
        return []
    C = max(len(s["cands"]) + len(s.get("extra", [])) for s in series)
    N = n_paths
    ring = 64
    D = np.zeros((B, N, weeks), np.float32)
    L = np.ones((B, N, weeks), np.int32)
    sL = np.full((B, C), 1e12, np.float32)   # unused lanes never order
    SL = np.full((B, C), 1e12, np.float32)
    moq = np.ones(B, np.float32)
    mult = np.ones(B, np.float32)
    for b, s in enumerate(series):
        mw = np.asarray(s["mean_week"], float)
        path = np.resize(mw, weeks) if len(mw) < weeks else mw[:weeks]
        if len(mw) < weeks:
            path = np.concatenate([mw, np.full(weeks - len(mw), mw[-1])])
        d = simulate_periods(s["sizes"], s["occ"], path, N, 0.0, rng)
        sig = float(s.get("sigma", 0.0)) * kappa
        if sig > 0:  # forecast level error, refreshed every `block` weeks as the forecast would be updated
            nb = int(np.ceil(weeks / block))
            m = rng.lognormal(-0.5 * sig ** 2, sig, (N, nb))
            d = d * np.repeat(m, block, axis=1)[:, :weeks]
        D[b] = d
        L[b] = np.clip(np.ceil(rng.choice(s["lt_weeks"], (N, weeks))), 1, ring - 1).astype(np.int32)
        k = len(s["cands"])
        sL[b, :k] = s["cands"]
        SL[b, :k] = np.asarray(s["cands"]) + max(s["q"], 1)
        for j, (a, bb) in enumerate(s.get("extra", [])):
            sL[b, k + j] = a
            SL[b, k + j] = max(bb, a + 1)
        moq[b] = max(s["moq"], 1)
        mult[b] = max(s["mult"], 1)
    # start each lane at a random point of its own cycle
    oh = (sL[:, :, None] + rng.random((B, C, N)).astype(np.float32) * (SL - sL)[:, :, None])
    oh = np.where(sL[:, :, None] > 1e11, 0, oh).astype(np.float32)
    oo = np.zeros((B, C, N), np.float32)
    pipe = np.zeros((B, C, N, ring), np.float32)
    shipped = np.zeros((B, C, N), np.float32)
    dem = np.zeros((B, C, N), np.float32)
    ohsum = np.zeros((B, C, N), np.float32)
    bi, ci, ni = np.meshgrid(np.arange(B), np.arange(C), np.arange(N), indexing="ij")
    moq3, mult3 = moq[:, None, None], mult[:, None, None]
    for w in range(weeks):
        slot = w % ring
        arr = pipe[:, :, :, slot]
        oh += arr
        oo -= arr
        pipe[:, :, :, slot] = 0
        d = D[:, None, :, w]
        ship = np.minimum(oh, d)
        oh -= ship
        if w >= warm:
            shipped += ship
            dem += d
            ohsum += oh
        ip = oh + oo
        need = ip <= sL[:, :, None]
        if need.any():
            q = np.maximum(SL[:, :, None] - ip, moq3)
            q = np.ceil(q / mult3 - 1e-6) * mult3
            q = np.where(need, q, 0).astype(np.float32)
            lt = L[:, None, :, w]
            tgt = (w + np.broadcast_to(lt, need.shape)) % ring
            m = need
            pipe[bi[m], ci[m], ni[m], tgt[m]] += q[m]
            oo += q
    nm = weeks - warm
    fill = shipped.sum(axis=2) / np.maximum(dem.sum(axis=2), 1e-9)
    avg_oh = ohsum.sum(axis=2) / (N * nm)
    out = []
    for b, s in enumerate(series):
        k = len(s["cands"])
        f = np.maximum.accumulate(fill[b, :k])  # more stock never serves less
        out.append({"fill": f, "onhand": avg_oh[b, :k],
                    "extra_fill": fill[b, k:k + len(s.get("extra", []))],
                    "extra_onhand": avg_oh[b, k:k + len(s.get("extra", []))]})
    return out


def rop_for_fill(cands, fill, target):
    """Smallest reorder point whose simulated fill rate meets the target (linear between candidates)."""
    cands = np.asarray(cands, float)
    fill = np.asarray(fill, float)
    ok = np.nonzero(fill >= target - 1e-9)[0]
    if len(ok) == 0:
        return float(cands[-1]), float(fill[-1])
    j = ok[0]
    if j == 0:
        return float(cands[0]), float(fill[0])
    f0, f1 = fill[j - 1], fill[j]
    c0, c1 = cands[j - 1], cands[j]
    if f1 - f0 <= 1e-9:
        return float(c1), float(f1)
    r = c0 + (target - f0) / (f1 - f0) * (c1 - c0)
    r = float(np.ceil(r))
    return r, float(np.interp(r, cands, fill))
