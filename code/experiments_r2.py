"""Round-2 review analyses (added after the second simulated peer review; protocol of v2 unchanged).

Adds, on the same evaluation seeds 10-19:
  * SPC-DS: univariate Shewhart chart on linked, speed-normalised channels (bp_norm instead of raw back-pressure,
    plus the two incoming-inspection values), thresholded at 2% FAR exactly like the other detectors;
  * CCS: a rule-based current control strategy (fixed RH and oven alarm limits, a 3-sigma back-pressure limit and
    3-sigma incoming-inspection trend limits from qualification data), always combined with end-of-batch QC;
  * a second chance level matched on the false-alarm budget (alarm rate on QC-passing sub-lots);
  * per-fault-type timely detection, NC line-hours and attribution on alarmed sub-lots;
  * false-alarm cost sensitivity, a permutation null for lot clustering, and a tested lot-block threshold arm;
  * the same additions for unseen fault types (leave-one-fault-out), and a redesigned nonlinear scenario.
Usage: python3 experiments_r2.py main | lofo | nonlinear2
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
import experiments as E, sim

R = "../results/r2_"
COSTS = [0.0, 0.5, 1.0, 2.0, 4.0]
DETS = ["SPC", "SPC-DS", "CCS", "T2", "LR-DS", "XGB-DS-u", "XGB-DS", "Hybrid"]


def spcds_frame(snap, ds):
    X = snap.drop(columns=["bp_mean"]).copy()
    X["bp_norm"] = ds["bp_norm"].values
    for c in E.LOT_FEATS: X[c] = ds[c].values
    return X


def ccs_alarm(ds, ref):
    """Rule-based current control strategy: fixed environmental and oven limits, 3-sigma limits on normalised
    back-pressure and on incoming-inspection values estimated from QC-passing qualification (training) data."""
    P = ds[ref]
    def hi(c): return P[c].mean() + 3 * P[c].std()
    def lo(c): return P[c].mean() - 3 * P[c].std()
    rule = {
        "rh": (ds.rh_mean > 45).values,
        "oven": ((ds.oven_T_mean < 40) | (ds.oven_T_mean > 44)).values,
        "bp": (ds.bp_norm > hi("bp_norm")).values,
        "goods_in_mem": (ds.mem_coa_obs > hi("mem_coa_obs")).values,
        "goods_in_rea": (ds.rea_act_obs < lo("rea_act_obs")).values,
    }
    a = np.zeros(len(ds), bool)
    for v in rule.values(): a |= np.nan_to_num(v, nan=0).astype(bool)
    gi = rule["goods_in_mem"] | rule["goods_in_rea"]
    return a, gi


def chance_levels(sub, eps, idx_te, rate_total, rate_fa, rng, n=100, types=None):
    out = {}
    for name, rate in [("total", rate_total), ("fa", rate_fa)]:
        det, ncm = [], []
        for _ in range(n):
            ra = rng.random(len(idx_te)) < rate
            em = E.episode_metrics(sub, ra, eps, idx_te, types=types)
            det.append(em.detected.mean() if len(em) else np.nan); ncm.append(em.nc_model.sum() if len(em) else np.nan)
        out[f"chance_{name}_ep"] = float(np.nanmean(det)); out[f"chance_{name}_nc"] = float(np.nanmean(ncm))
    return out


def lot_cluster(sub, alarm, clean, te, rng, n=1000):
    fa = alarm & clean & te
    k = int(fa.sum())
    if k < 2: return dict(n_fa=k, top_share=np.nan, null_share=np.nan, p=np.nan)
    def share(mask):
        return max(sub.loc[mask, c].value_counts(normalize=True).max() for c in ("mem_lot", "rea_lot"))
    obs = share(fa)
    pool = np.where(clean & te)[0]; null = []
    for _ in range(n):
        m = np.zeros(len(sub), bool); m[rng.choice(pool, k, replace=False)] = True; null.append(share(m))
    null = np.array(null)
    return dict(n_fa=k, top_share=float(obs), null_share=float(null.mean()), p=float((null >= obs).mean()))


def lotblock_threshold(s, sub, va, ok, rng, B=300):
    """Lowest threshold whose lot-block bootstrap 95% upper bound on validation FAR is at most 2%."""
    vm = va & ok; sv = s[vm]
    lots = (sub.rea_lot.values[vm] * 1000 + sub.mem_lot.values[vm]); ul, inv = np.unique(lots, return_inverse=True)
    size = np.bincount(inv)
    picks = rng.integers(0, len(ul), (B, len(ul)))
    for q in np.linspace(0.98, 0.9995, 60):
        th = float(np.quantile(sv, q)); cnt = np.bincount(inv, weights=(sv >= th).astype(float), minlength=len(ul))
        far_b = cnt[picks].sum(1) / size[picks].sum(1)
        if np.quantile(far_b, 0.95) <= E.FAR: return th, float(q)
    return float(np.quantile(sv, 0.9995)), 0.9995


def summarise(sub, eps, idx_te, a_te, score_te, y_te, clean_te, ok_te, rng, chance=True):
    em = E.episode_metrics(sub, a_te, eps, idx_te)
    fa = int((a_te & clean_te).sum())
    r = dict(far_ok=float(a_te[ok_te].mean()), far_clean=float(a_te[clean_te].mean()), alarm_rate=float(a_te.mean()),
             false_alarms=fa, ep_detect=float(em.detected.mean()), nc_model=int(em.nc_model.sum()), nc_sq=int(em.nc_sq.sum()),
             early=float((em.delay < 0).mean()), n_episodes=len(em),
             precision=float((a_te & (y_te == 1)).sum() / max(1, a_te.sum())), recall=float((a_te & (y_te == 1)).sum() / max(1, (y_te == 1).sum())))
    r["pr_auc"] = float(average_precision_score(y_te, score_te)) if score_te is not None else np.nan
    for c in COSTS: r[f"net_{c}"] = r["nc_sq"] - r["nc_model"] - c * fa
    for t in E.CAUSES:
        e2 = em[em.type == t]
        r[f"ep_{t}"] = float(e2.detected.mean()) if len(e2) else np.nan
        r[f"nc_{t}"] = int(e2.nc_model.sum()) if len(e2) else 0; r[f"ncsq_{t}"] = int(e2.nc_sq.sum()) if len(e2) else 0
        r[f"n_{t}"] = len(e2)
    if chance: r.update(chance_levels(sub, eps, idx_te, r["alarm_rate"], r["far_ok"], rng))
    return r


def alarmed_attr(sub_te, A_te, alarm_te):
    m = alarm_te & (sub_te.nc.values == 1)
    if m.sum() == 0: return {}
    return E.attribution_accuracy(sub_te[m].reset_index(drop=True), A_te[m].reset_index(drop=True))[0]


def main_seed(seed):
    r, ctx = E.run_seed(seed, chance=False)
    sub = ctx["sub"]; ds = ctx["feats"]["ds"]; snap = ctx["feats"]["snap"]; eps = ctx["eps"]
    tr = (sub.split == "train").values; va = (sub.split == "val").values; te = (sub.split == "test").values
    y = sub.nc.values; ok = sub.ok.values; clean = sub.clean.values; idx_te = np.where(te)[0]
    rng = np.random.default_rng(seed + 777)
    scores, alarms = dict(ctx["scores"]), dict(ctx["alarms"])
    # SPC-DS
    X = spcds_frame(snap, ds); sp = E.Shewhart().fit(X[tr].reset_index(drop=True), y[tr], ok[tr])
    s = sp.score(X); th = E.threshold(s[va], ok[va]); scores["SPC-DS"] = s; alarms["SPC-DS"] = s >= th
    # CCS
    a_ccs, gi = ccs_alarm(ds, tr & ok); alarms["CCS"] = a_ccs; scores["CCS"] = None
    out = {}
    for k in DETS:
        out[k] = summarise(sub, eps, idx_te, alarms[k][te], None if scores[k] is None else scores[k][te], y[te], clean[te], ok[te], rng)
    out["GoodsIn"] = summarise(sub, eps, idx_te, gi[te], None, y[te], clean[te], ok[te], rng)
    # attribution: all NC sub-lots and alarmed NC sub-lots
    sub_te = sub[te].reset_index(drop=True); att = {}
    A_spc = sp.attrib(X[te]).reset_index(drop=True)
    fT, fL = ctx["fT"], ctx["fL"]
    A_lr = ctx["models"]["LR-DS"][0].attrib(ds).reset_index(drop=True); A_t2 = ctx["models"]["T2"][0].attrib(ds).reset_index(drop=True)
    A_h = pd.DataFrame(np.where((fL >= fT)[:, None], A_lr.values, A_t2.values), columns=ds.columns)
    A_nv = E.naive_attrib(ds, tr & ok)
    for k, A in [("SPC-DS", A_spc), ("LR-DS", A_lr[te].reset_index(drop=True)), ("Hybrid", A_h[te].reset_index(drop=True)),
                 ("Naive-z", A_nv[te].reset_index(drop=True))]:
        al = alarms["Hybrid" if k == "Naive-z" else k][te]
        att[k] = dict(all=E.attribution_accuracy(sub_te, A)[0], alarmed=alarmed_attr(sub_te, A, al))
    # lot clustering null
    clus = {k: lot_cluster(sub, alarms[k], clean, te, rng) for k in ["SPC", "SPC-DS", "LR-DS", "Hybrid"]}
    # lot-block threshold arm
    lb = {}
    for k in ["LR-DS", "Hybrid"]:
        thb, q = lotblock_threshold(scores[k], sub, va, ok, rng)
        a = scores[k] >= thb
        res = summarise(sub, eps, idx_te, a[te], scores[k][te], y[te], clean[te], ok[te], rng, chance=False)
        res["quantile"] = q; res["far_ok_std"] = float(alarms[k][te][ok[te]].mean()); lb[k] = res
    overlap = int(sub.cause.str.contains(r"\+").sum()); overlap_te = int(sub.cause[te].str.contains(r"\+").sum())
    lots_val = dict(mem=int(np.unique(sub.mem_lot.values[va]).size), rea=int(np.unique(sub.rea_lot.values[va]).size))
    return dict(seed=seed, res=out, att=att, clus=clus, lotblock=lb, overlap=overlap, overlap_te=overlap_te, lots_val=lots_val)


def lofo_seed(seed):
    sub, snap, ds, eps, dq = E.prepare(seed)
    tr = (sub.split == "train").values; va = (sub.split == "val").values; te = (sub.split == "test").values
    y = sub.nc.values; ok = sub.ok.values; idx_te = np.where(te)[0]; out = []
    X = spcds_frame(snap, ds)
    for k in E.CAUSES:
        hasK = sub.cause.str.contains(k).values; trk = tr & ~hasK; vak = va & ~hasK
        t2 = E.HotellingT2().fit(ds[trk].reset_index(drop=True), y[trk], ok[trk]); s2 = t2.score(ds); A2 = t2.attrib(ds)
        lr = E.LogReg().fit(ds[trk].reset_index(drop=True), y[trk], ok[trk], ds[vak], y[vak]); sl = lr.score(ds); Al = lr.attrib(ds)
        xm = E.XGB(seed, mono=True).fit(ds[trk].reset_index(drop=True), y[trk], ok[trk], ds[vak], y[vak]); sx = xm.score(ds)
        sp = E.Shewhart().fit(X[trk].reset_index(drop=True), y[trk], ok[trk]); ss = sp.score(X); As = sp.attrib(X)
        cT = E.ecdf_ref(s2[vak & ok]); cL = E.ecdf_ref(sl[vak & ok]); fT, fL = cT(s2), cL(sl); sh = np.maximum(fT, fL)
        A_h = pd.DataFrame(np.where((fL >= fT)[:, None], Al.reset_index(drop=True).values, A2.reset_index(drop=True).values), columns=ds.columns)
        a_ccs, _ = ccs_alarm(ds, trk & ok)
        tgt = te & hasK & (y == 1)
        if tgt.sum() == 0: continue
        for name, s, A in [("LR-DS", sl, Al), ("XGB-DS", sx, None), ("T2", s2, A2), ("Hybrid", sh, A_h), ("SPC-DS", ss, As), ("CCS", None, None)]:
            a_full = a_ccs if s is None else (s >= E.threshold(s[vak], ok[vak]))
            a = a_full[te]
            em = E.episode_metrics(sub, a, eps, idx_te, types=[k])
            rng = np.random.default_rng(seed * 11 + len(out))
            ch = chance_levels(sub, eps, idx_te, a.mean(), a[ok[te]].mean(), rng, n=100, types=[k])
            row = dict(seed=seed, held_out=k, model=name, ep_detect=float(em.detected.mean()) if len(em) else np.nan,
                       n_ep=len(em), n_det=int(em.detected.sum()) if len(em) else 0, nc_model=int(em.nc_model.sum()) if len(em) else 0,
                       nc_sq=int(em.nc_sq.sum()) if len(em) else 0, **ch)
            if A is not None:
                msk = tgt & a_full
                if msk.sum():
                    Ad = A.reset_index(drop=True) if isinstance(A, pd.DataFrame) else A
                    acc = E.attribution_accuracy(sub[msk].reset_index(drop=True), Ad[msk].reset_index(drop=True))[0]
                    row["attr"] = acc.get("overall"); row["n_attr"] = int(msk.sum())
            out.append(row)
    return out


def stress_seed(seed, simfun=None, **kw):
    r, ctx = E.run_seed(seed, simfun=simfun, chance=False, **kw)
    sub = ctx["sub"]; ds = ctx["feats"]["ds"]; snap = ctx["feats"]["snap"]; eps = ctx["eps"]
    tr = (sub.split == "train").values; va = (sub.split == "val").values; te = (sub.split == "test").values
    y = sub.nc.values; ok = sub.ok.values; clean = sub.clean.values; idx_te = np.where(te)[0]
    rng = np.random.default_rng(seed + 4242)
    scores, alarms = dict(ctx["scores"]), dict(ctx["alarms"])
    X = spcds_frame(snap, ds); sp = E.Shewhart().fit(X[tr].reset_index(drop=True), y[tr], ok[tr])
    s = sp.score(X); scores["SPC-DS"] = s; alarms["SPC-DS"] = s >= E.threshold(s[va], ok[va])
    a_ccs, _ = ccs_alarm(ds, tr & ok); alarms["CCS"] = a_ccs; scores["CCS"] = None
    out = {}
    for k in ["SPC", "SPC-DS", "CCS", "T2", "LR-DS", "XGB-DS-u", "Hybrid"]:
        out[k] = summarise(sub, eps, idx_te, alarms[k][te], None if scores[k] is None else scores[k][te], y[te], clean[te], ok[te], rng, chance=False)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "main":
        allr = []
        for s in E.SEEDS: allr.append(main_seed(s)); print("r2 main", s, flush=True)
        json.dump(allr, open(R + "main.json", "w"), indent=1, default=float)
    elif mode == "lofo":
        out = []
        for s in E.SEEDS: out += lofo_seed(s); print("r2 lofo", s, flush=True)
        pd.DataFrame(out).to_csv(R + "lofo.csv", index=False)
    elif mode.startswith("stress_"):
        cond = mode.split("_", 1)[1]
        import sim_nonlinear2 as N2, sim_shared as SH
        simfun, kw = {"mild": (None, dict(severity=0.7)), "shared": (SH.simulate, {}), "nonlinear2": (N2.simulate, {})}[cond]
        rows = []
        for s in E.SEEDS:
            out = stress_seed(s, simfun, **kw)
            for k, v in out.items(): rows.append(dict(cond=cond, seed=s, model=k, **v))
            print("stress", cond, s, flush=True)
        pd.DataFrame(rows).to_csv(R + f"stress_{cond}.csv", index=False)
    elif mode == "nonlinear2":
        import sim_nonlinear2 as N2
        rows = []
        for s in E.SEEDS:
            r, ctx = E.run_seed(s, simfun=N2.simulate, chance=False, light=True)
            base = sim.simulate(seed=s)[0]; nl = ctx["sub"]
            changed = float(((base.nc != nl.nc)).sum() / max(1, ((base.nc == 1) | (nl.nc == 1)).sum()))
            for k, v in r["res"].items(): rows.append(dict(seed=s, model=k, changed_share=changed, **v))
            print("r2 nonlinear2", s, flush=True)
        pd.DataFrame(rows).to_csv(R + "nonlinear2.csv", index=False)
