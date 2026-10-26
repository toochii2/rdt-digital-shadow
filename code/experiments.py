"""Experiments v2 (post-review): causal sensors, operational thresholds, nested tuning, timely detection,
chance baselines, false-alarm cost, naive attribution baseline, hybrid attribution, fresh seeds."""
import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from scipy.stats import kendalltau
import xgboost as xgb
import shap
import sim

warnings.filterwarnings("ignore")
FAR = 0.02          # target false-alarm rate on validation sub-lots that passed QC
LAB_DELAY = 4       # h from end of 8-h batch to release-test result
RESPONSE = 1        # h from alarm to effective containment
TIMELY = 2          # an episode counts as detected if alarmed no later than first NC sub-lot + TIMELY h
FA_COST = 0.5       # line-hours of production lost per false-alarm investigation (assumption)
CAUSES = sim.FAULT_TYPES
SEEDS = list(range(10, 20))   # fresh evaluation seeds (design iterations used seeds 0-9)
MONO = {"rh_mean": 1, "rh_max": 1, "rh_exposure_min": 1, "bp_norm": 1, "bp_norm_trend3": 1,
        "hours_since_clean": 1, "mem_coa_obs": 1, "oven_T_min": -1, "rea_act_obs": -1}
SIGN = {"oven_T_mean": -1, "oven_T_min": -1, "oven_T_slope": -1, "oven_T_trend3": -1, "rea_act_obs": -1}
LOT_FEATS = ["mem_coa_obs", "rea_act_obs"]


def prepare(seed, simfun=None, **kw):
    f = simfun or sim.simulate
    sub, sens, eps = f(seed=seed, **kw)
    snap, ds, dq, comp = sim.build_features(sub, sens)
    sub["clean"] = (sub.cause == "") & (sub.nc == 0)     # ground truth, used ONLY for evaluation
    sub["ok"] = sub.nc == 0                              # operationally known after QC
    return sub, snap, ds, eps, dq


class Shewhart:
    def fit(self, X, y, ref, Xv=None, yv=None):
        P = X[ref]; self.med = P.median(); Pf = P.fillna(self.med); self.mu = Pf.mean(); self.sd = Pf.std() + 1e-9; return self
    def z(self, X): return ((X.fillna(self.med) - self.mu) / self.sd)
    def score(self, X): return self.z(X).abs().max(1).values
    def attrib(self, X): return self.z(X).abs()


class HotellingT2:
    def fit(self, X, y, ref, Xv=None, yv=None):
        P = X[ref]; self.med = P.median(); Pf = P.fillna(self.med); self.mu = Pf.mean(); self.sd = Pf.std() + 1e-9
        Z = ((Pf - self.mu) / self.sd).values; self.Pinv = np.linalg.inv(LedoitWolf().fit(Z).covariance_); return self
    def z(self, X): return ((X.fillna(self.med) - self.mu) / self.sd).values
    def score(self, X):
        Z = self.z(X); return np.einsum("ij,jk,ik->i", Z, self.Pinv, Z)
    def attrib(self, X):
        Z = self.z(X); return pd.DataFrame(Z * (Z @ self.Pinv), columns=X.columns)


class LogReg:
    GRID = [0.05, 0.5, 5.0]
    def __init__(self, C=None): self.C = C
    def _fit(self, X, y, C):
        self.med = X.median(); Xf = X.fillna(self.med); self.mu = Xf.mean(); self.sd = Xf.std() + 1e-9
        self.m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit((Xf - self.mu) / self.sd, y)
    def fit(self, X, y, ref, Xv=None, yv=None):
        if self.C is None and Xv is not None:     # nested tuning on validation PR-AUC
            best = None
            for C in self.GRID:
                self._fit(X, y, C); a = average_precision_score(yv, self.score(Xv))
                if best is None or a > best[0]: best = (a, C)
            self.C = best[1]
        self._fit(X, y, self.C or 0.5); return self
    def z(self, X): return (X.fillna(self.med) - self.mu) / self.sd
    def score(self, X): return self.m.decision_function(self.z(X))
    def attrib(self, X): return self.z(X) * self.m.coef_[0]


class XGB:
    GRID = [(2, 300), (3, 300), (5, 150)]
    def __init__(self, seed=0, mono=False, params=None): self.seed = seed; self.mono = mono; self.params = params
    def _fit(self, X, y, depth, n):
        pos = max(1, y.sum()); neg = len(y) - pos; extra = {}
        if self.mono: extra["monotone_constraints"] = "(" + ",".join(str(MONO.get(c, 0)) for c in X.columns) + ")"
        self.m = xgb.XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=2, scale_pos_weight=neg / pos, eval_metric="logloss",
                                   random_state=self.seed, n_jobs=2, verbosity=0, **extra)
        self.m.fit(X, y)
    def fit(self, X, y, ref, Xv=None, yv=None):
        if self.params is None and Xv is not None:
            best = None
            for d_, n_ in self.GRID:
                self._fit(X, y, d_, n_); a = average_precision_score(yv, self.score(Xv))
                if best is None or a > best[0]: best = (a, (d_, n_))
            self.params = best[1]
        self._fit(X, y, *(self.params or (3, 300))); self.expl = shap.TreeExplainer(self.m); return self
    def score(self, X): return self.m.predict_proba(X)[:, 1]
    def attrib(self, X): return pd.DataFrame(self.expl.shap_values(X), columns=X.columns)


def threshold(s_val, ok_val): return float(np.quantile(s_val[ok_val], 1 - FAR))
def ecdf_ref(ref):
    ref = np.sort(ref); return lambda s: np.searchsorted(ref, s, side="right") / len(ref)


def episode_metrics(sub, alarm, eps, idx_test, types=None):
    t0, t1 = idx_test[0], idx_test[-1]; rows = []
    for e in eps:
        if types and e["type"] not in types: continue
        a, d = e["start"], e["dur"]
        if a < t0 or a + d > t1: continue
        win = np.arange(a, a + d)
        ncw = win[(sub.nc.values[win] == 1) & np.array([e["type"] in c for c in sub.cause.values[win]])]
        if len(ncw) == 0: continue
        onset = ncw[0]
        t_sq = (onset // sim.BATCH_SIZE + 1) * sim.BATCH_SIZE - 1 + LAB_DELAY
        cand = win[(win <= onset + TIMELY)]
        al = cand[alarm[cand - t0]] if alarm is not None else np.array([], int)
        t_al = int(al[0]) if len(al) else None
        contain_sq = t_sq + RESPONSE
        contain = min(t_al + RESPONSE, contain_sq) if t_al is not None else contain_sq
        rows.append(dict(type=e["type"], detected=t_al is not None, delay=(t_al - onset) if t_al is not None else np.nan,
                         lead_vs_release=(t_sq - t_al) if t_al is not None else np.nan,
                         nc_sq=int((ncw <= contain_sq).sum()), nc_model=int((ncw <= contain).sum())))
    return pd.DataFrame(rows)


def attribution_accuracy(sub_te, A):
    mask = (sub_te.nc.values == 1) & np.isin(sub_te.cause.values, CAUSES)
    if mask.sum() == 0: return {}, np.array([]), np.array([])
    A = A[mask]; truth = sub_te.cause.values[mask]
    G = pd.DataFrame({g: A[[c for c in cols if c in A.columns]].clip(lower=0).sum(1) if any(c in A.columns for c in cols) else 0.0
                      for g, cols in sim.FEATURE_GROUPS.items()})
    predg = np.array(list(sim.FEATURE_GROUPS))[G.values.argmax(1)]
    out = {"overall": float((predg == truth).mean()), "n": int(mask.sum())}
    for c in CAUSES:
        m = truth == c
        if m.sum(): out[c] = float((predg[m] == c).mean())
    return out, predg, truth


def naive_attrib(ds, ref):
    P = ds[ref]; med = P.median(); mu = P.fillna(med).mean(); sd = P.fillna(med).std() + 1e-9
    return ((ds.fillna(med) - mu) / sd) * pd.Series({c: SIGN.get(c, 1.0) for c in ds.columns})


def run_seed(seed, simfun=None, chance=True, light=False, **kw):
    sub, snap, ds, eps, dq = prepare(seed, simfun, **kw)
    tr = (sub.split == "train").values; va = (sub.split == "val").values; te = (sub.split == "test").values
    y = sub.nc.values; ok = sub.ok.values; clean = sub.clean.values
    idx_te = np.where(te)[0]
    feats = {"snap": snap, "ds": ds}
    models = {"SPC": (Shewhart(), "snap"), "T2": (HotellingT2(), "ds"), "LR-snap": (LogReg(), "snap"),
              "XGB-snap": (XGB(seed), "snap"), "LR-DS": (LogReg(), "ds"), "XGB-DS-u": (XGB(seed), "ds"),
              "XGB-DS": (XGB(seed, mono=True), "ds")}
    res, scores, alarms, probs, thr = {}, {}, {}, {}, {}
    for k, (m, f) in models.items():
        X = feats[f]
        m.fit(X[tr].reset_index(drop=True), y[tr], ok[tr], X[va], y[va])
        s = m.score(X); thr[k] = threshold(s[va], ok[va])
        scores[k] = s; alarms[k] = s >= thr[k]
        probs[k] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[va], y[va]).predict(s)
    cT = ecdf_ref(scores["T2"][va & ok]); cL = ecdf_ref(scores["LR-DS"][va & ok])
    fT, fL = cT(scores["T2"]), cL(scores["LR-DS"])
    sh = np.maximum(fT, fL) + 1e-6 * fL
    thr["Hybrid"] = threshold(sh[va], ok[va]); scores["Hybrid"] = sh; alarms["Hybrid"] = sh >= thr["Hybrid"]
    probs["Hybrid"] = IsotonicRegression(out_of_bounds="clip").fit(sh[va], y[va]).predict(sh)
    rng = np.random.default_rng(seed + 12345)
    for k in scores:
        a = alarms[k][te]; yt = y[te]
        tp = int((a & (yt == 1)).sum()); fp_clean = int((a & clean[te]).sum())
        rec = tp / max(1, (yt == 1).sum()); prec = tp / max(1, a.sum())
        em = episode_metrics(sub, a, eps, idx_te)
        r = dict(pr_auc=average_precision_score(yt, scores[k][te]), roc_auc=roc_auc_score(yt, scores[k][te]),
                 recall=rec, precision=prec, f1=2 * prec * rec / max(1e-9, prec + rec),
                 far=float(a[clean[te]].mean()), far_ok=float(a[ok[te]].mean()), alarm_rate=float(a.mean()),
                 brier=brier_score_loss(yt, probs[k][te]), false_alarms=fp_clean,
                 ep_detect=float(em.detected.mean()), ep_delay=float(em.delay.mean()),
                 lead_vs_release=float(em.lead_vs_release.mean()),
                 nc_sq=int(em.nc_sq.sum()), nc_model=int(em.nc_model.sum()), n_episodes=len(em))
        r["net_saving"] = r["nc_sq"] - r["nc_model"] - FA_COST * fp_clean
        fa = alarms[k] & clean & te
        r["fa_top_lot_share"] = float(max(sub.loc[fa, c].value_counts(normalize=True).max() if fa.sum() else 0 for c in ("mem_lot", "rea_lot")))
        if chance:   # random detector matched on overall test alarm rate
            det, ncm = [], []
            for _ in range(100):
                ra = rng.random(te.sum()) < r["alarm_rate"]
                emr = episode_metrics(sub, ra, eps, idx_te); det.append(emr.detected.mean()); ncm.append(emr.nc_model.sum())
            r["chance_ep_detect"] = float(np.mean(det)); r["chance_nc"] = float(np.mean(ncm))
        res[k] = r
    em_q = episode_metrics(sub, (y == 1)[te], eps, idx_te)
    hourly_qc_nc = int(em_q.nc_model.sum())
    # attribution
    att = {}
    sub_te = sub[te].reset_index(drop=True)
    for k, (m, f) in models.items():
        att[k] = attribution_accuracy(sub_te, m.attrib(feats[f][te]).reset_index(drop=True))[0]
    att["Naive-z"] = attribution_accuracy(sub_te, naive_attrib(ds, tr & ok)[te].reset_index(drop=True))[0]
    lr_fire = fL >= fT
    A_lr = models["LR-DS"][0].attrib(ds).reset_index(drop=True); A_t2 = models["T2"][0].attrib(ds).reset_index(drop=True)
    A_h = pd.DataFrame(np.where(lr_fire[:, None], A_lr.values, A_t2.values), columns=ds.columns)
    att["Hybrid"] = attribution_accuracy(sub_te, A_h[te].reset_index(drop=True))[0]
    msk = te & alarms["Hybrid"] & (y == 1)
    hybrid_lr_share = float(lr_fire[msk].mean()) if msk.sum() else np.nan
    # ablations (LR, tuned)
    abl = {}
    if light:
        return dict(seed=seed, res=res, att=att, abl={}, farchk={}, hourly_qc_nc=hourly_qc_nc, tuned={},
                    hybrid_lr_share=hybrid_lr_share, dq=dq, prevalence=float(y.mean()), test_prev=float(y[te].mean()),
                    n_nc_by_cause={}), dict(sub=sub, scores=scores, alarms=alarms, eps=eps, thr=thr)
    for name, X in [("LR-snap+lot", pd.concat([snap, ds[LOT_FEATS]], axis=1)),
                    ("LR-DS-sensor", ds[[c for c in ds.columns if c not in LOT_FEATS + ["hours_since_clean"]]])]:
        m = LogReg().fit(X[tr].reset_index(drop=True), y[tr], ok[tr], X[va], y[va]); abl[name] = average_precision_score(y[te], m.score(X)[te])
    # FAR estimate: sub-lot vs lot-block bootstrap upper bound on validation vs realised test FAR
    farchk = {}
    for k in ["LR-DS", "Hybrid", "T2"]:
        s = scores[k]; vm = va & ok; a_v = (s >= thr[k])[vm]
        lots = sub.rea_lot.values[vm] * 1000 + sub.mem_lot.values[vm]; ul = np.unique(lots)
        b1 = [rng.choice(a_v, len(a_v)).mean() for _ in range(500)]
        b2 = [np.concatenate([a_v[lots == l] for l in rng.choice(ul, len(ul))]).mean() for _ in range(500)]
        farchk[k] = dict(test=float((s >= thr[k])[te & ok].mean()), ub_sublot=float(np.quantile(b1, .95)), ub_lot=float(np.quantile(b2, .95)))
    tuned = dict(LR_DS_C=models["LR-DS"][0].C, XGB_DS=models["XGB-DS"][0].params, XGB_DSu=models["XGB-DS-u"][0].params)
    return dict(seed=seed, res=res, att=att, abl=abl, farchk=farchk, hourly_qc_nc=hourly_qc_nc, tuned=tuned,
                hybrid_lr_share=hybrid_lr_share, dq=dq, prevalence=float(y.mean()), test_prev=float(y[te].mean()),
                n_nc_by_cause={c: int(((sub.cause.values == c) & (y == 1) & te).sum()) for c in CAUSES}), \
        dict(sub=sub, scores=scores, alarms=alarms, probs=probs, models=models, feats=feats, eps=eps, thr=thr, fT=fT, fL=fL)


def lofo_seed(seed):
    sub, snap, ds, eps, dq = prepare(seed)
    tr = (sub.split == "train").values; va = (sub.split == "val").values; te = (sub.split == "test").values
    y = sub.nc.values; ok = sub.ok.values; out = []
    for k in CAUSES:
        hasK = sub.cause.str.contains(k).values; trk = tr & ~hasK; vak = va & ~hasK
        # all detectors retrained without fault type k (T2 reference excludes it too)
        t2 = HotellingT2().fit(ds[trk].reset_index(drop=True), y[trk], ok[trk]); s2 = t2.score(ds); A2 = t2.attrib(ds)
        lr = LogReg().fit(ds[trk].reset_index(drop=True), y[trk], ok[trk], ds[vak], y[vak]); sl = lr.score(ds); Al = lr.attrib(ds)
        xm = XGB(seed, mono=True).fit(ds[trk].reset_index(drop=True), y[trk], ok[trk], ds[vak], y[vak]); sx = xm.score(ds)
        cT = ecdf_ref(s2[vak & ok]); cL = ecdf_ref(sl[vak & ok]); fT, fL = cT(s2), cL(sl); sh = np.maximum(fT, fL)
        A_h = pd.DataFrame(np.where((fL >= fT)[:, None], Al.reset_index(drop=True).values, A2.reset_index(drop=True).values), columns=ds.columns)
        tgt = te & hasK & (y == 1)
        if tgt.sum() == 0: continue
        for name, s in [("LR-DS", sl), ("XGB-DS", sx), ("T2", s2), ("Hybrid", sh)]:
            th = threshold(s[vak], ok[vak]); a = (s >= th)[te]
            em = episode_metrics(sub, a, eps, np.where(te)[0], types=[k])
            rng = np.random.default_rng(seed * 7 + len(out))
            ch = np.mean([episode_metrics(sub, rng.random(te.sum()) < a.mean(), eps, np.where(te)[0], types=[k]).detected.mean() for _ in range(50)])
            row = dict(seed=seed, held_out=k, model=name, recall=float((s[tgt] >= th).mean()),
                       ep_detect=float(em.detected.mean()) if len(em) else np.nan, chance=float(ch), n_ep=len(em))
            if name == "Hybrid":
                msk = tgt & (s >= th)
                if msk.sum():
                    acc = attribution_accuracy(sub[msk].reset_index(drop=True), A_h[msk].reset_index(drop=True))[0]
                    row["attr"] = acc.get("overall"); row["t2_share"] = float((fT >= fL)[msk].mean())
            out.append(row)
    return out


def stability_seed(seed, B=10):
    sub, snap, ds, eps, dq = prepare(seed)
    tr = (sub.split == "train").values; va = (sub.split == "val").values; te = (sub.split == "test").values
    y = sub.nc.values; ok = sub.ok.values
    ud = np.unique(sub.day.values[tr]); rng = np.random.default_rng(seed + 999); sub_te = sub[te].reset_index(drop=True)
    out = {}
    for name, ctor in [("XGB-DS", lambda b: XGB(seed * 100 + b, mono=True)), ("LR-DS", lambda b: LogReg()), ("T2", lambda b: HotellingT2())]:
        imps, tops = [], []
        for b in range(B):
            pick = rng.choice(ud, len(ud), replace=True)
            ix = np.concatenate([np.where(tr & (sub.day.values == d))[0] for d in pick])
            m = ctor(b).fit(ds.iloc[ix].reset_index(drop=True), y[ix], ok[ix], ds[va], y[va])
            A = m.attrib(ds[te]).reset_index(drop=True); imps.append(A.abs().mean().values)
            tops.append(attribution_accuracy(sub_te, A)[1])
        taus = [kendalltau(imps[i], imps[j])[0] for i in range(B) for j in range(i + 1, B)]
        T = np.array(tops); agree = np.mean([(T[:, i] == pd.Series(T[:, i]).mode()[0]).mean() for i in range(T.shape[1])])
        out[name] = dict(kendall_tau=float(np.mean(taus)), local_agreement=float(agree))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]; R = "../results/v2_"
    if mode == "main":
        allr = []
        for s in SEEDS: r, _ = run_seed(s); allr.append(r); print("seed", s, flush=True)
        json.dump(allr, open(R + "main.json", "w"), indent=1, default=float)
    elif mode in ("nonlinear", "shared"):
        mod = __import__("sim_nonlinear" if mode == "nonlinear" else "sim_shared")
        allr = []
        for s in SEEDS: r, _ = run_seed(s, simfun=mod.simulate, chance=False, light=True); allr.append(r); print(mode, s, flush=True)
        json.dump(allr, open(R + mode + ".json", "w"), indent=1, default=float)
    elif mode == "lowprev":
        allr = []
        for s in SEEDS:
            r, _ = run_seed(s, chance=False, light=True, rates={k: v * 0.3 for k, v in sim.BASE_RATES.items()}); allr.append(r); print("lowprev", s, flush=True)
        json.dump(allr, open(R + "lowprev.json", "w"), indent=1, default=float)
    elif mode == "lofo":
        out = []
        for s in SEEDS: out += lofo_seed(s); print("lofo", s, flush=True)
        pd.DataFrame(out).to_csv(R + "lofo.csv", index=False)
    elif mode == "stability":
        out = {s: stability_seed(s) for s in SEEDS}; json.dump(out, open(R + "stability.json", "w"), indent=1)
    elif mode == "robust":
        rows = []
        for cond, kw in [("miss_20", dict(miss_rate=0.20)), ("mild_faults", dict(severity=0.7)), ("noise_x1.5", dict(noise_mult=1.5))]:
            for s in SEEDS:
                r, _ = run_seed(s, chance=False, light=True, **kw)
                for k, v in r["res"].items(): rows.append(dict(cond=cond, seed=s, model=k, **v))
            print(cond, flush=True)
        pd.DataFrame(rows).to_csv(R + "robust.csv", index=False)
    elif mode == "containment":
        acc = {}
        for s in SEEDS:
            r, ctx = run_seed(s, chance=False, light=True)
            sub = ctx["sub"]; te = (sub.split == "test").values; idx = np.where(te)[0]
            for lab in [2, 4, 8]:
                for resp in [1, 2, 4]:
                    globals().update(LAB_DELAY=lab, RESPONSE=resp)
                    for k in ["SPC", "T2", "LR-DS", "XGB-DS-u", "Hybrid"]:
                        em = episode_metrics(sub, ctx["alarms"][k][te], ctx["eps"], idx)
                        acc.setdefault((lab, resp, k), []).append(1 - em.nc_model.sum() / max(1, em.nc_sq.sum()))
            globals().update(LAB_DELAY=4, RESPONSE=1); print("cont", s, flush=True)
        out = [dict(lab_delay=l, response=rp, model=k, reduction=100 * float(np.mean(v))) for (l, rp, k), v in acc.items()]
        pd.DataFrame(out).to_csv(R + "containment.csv", index=False)
