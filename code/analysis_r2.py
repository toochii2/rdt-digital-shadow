"""Summarise round-2 analyses: stronger comparators, per-type results, chance levels, cost sensitivity,
lot clustering null, lot-block threshold arm, unseen faults, nonlinear scenario 2, confirmatory tests."""
import json, numpy as np, pandas as pd
from scipy import stats
R = "../results/"
M = json.load(open(R + "r2_main.json")); V = json.load(open(R + "v2_main.json"))
CAUSES = ["nozzle", "humidity", "drying", "membrane", "reagent"]


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


d = pd.DataFrame([dict(seed=r["seed"], model=k, **v) for r in M for k, v in r["res"].items()])
order = ["SPC", "SPC-DS", "CCS", "GoodsIn", "T2", "LR-DS", "XGB-DS-u", "XGB-DS", "Hybrid"]
cols = ["pr_auc", "far_ok", "far_clean", "false_alarms", "ep_detect", "chance_total_ep", "chance_fa_ep", "early", "nc_model",
        "nc_sq", "precision", "recall"] + [f"net_{c}" for c in [0.0, 0.5, 1.0, 2.0, 4.0]]
rows = []
for m in order:
    g = d[d.model == m]; row = {"model": m}
    for c in cols:
        mm, h = ci(g[c]); row[c] = mm; row[c + "_ci"] = h
    row["far_clean_min"] = g.far_clean.min(); row["far_clean_max"] = g.far_clean.max()
    row["fa_min"] = g.false_alarms.min(); row["fa_max"] = g.false_alarms.max()
    for t in ["nozzle", "humidity", "drying", "membrane", "reagent"]:
        row[f"ep_{t}"] = g[f"ep_{t}"].mean()
        row[f"nc_{t}"] = g[f"nc_{t}"].sum(); row[f"ncsq_{t}"] = g[f"ncsq_{t}"].sum(); row[f"n_{t}"] = g[f"n_{t}"].sum()
    rows.append(row)
T = pd.DataFrame(rows).set_index("model"); T["reduction"] = 100 * (1 - T.nc_model / T.nc_sq)
T.to_csv(R + "r2_table.csv")
pd.set_option("display.width", 250)
print(T[["pr_auc", "far_ok", "far_clean", "false_alarms", "ep_detect", "chance_total_ep", "chance_fa_ep", "early", "nc_model", "reduction",
         "precision", "net_0.5", "net_2.0", "net_4.0"]].round(3).to_string())
print("\nper-type timely detection and NC line-h (sum over 10 reps)")
print(T[[f"ep_{t}" for t in CAUSES] + [f"nc_{t}" for t in CAUSES]].round(2).to_string())
print("episodes per type (sum):", T.loc["LR-DS", [f"n_{t}" for t in CAUSES]].to_dict(), "ncsq per type", T.loc["LR-DS", [f"ncsq_{t}" for t in CAUSES]].to_dict())
print("fault-free FAR range", T[["far_clean", "far_clean_min", "far_clean_max", "fa_min", "fa_max"]].round(3).to_string())
# reduction per seed vs CCS baseline (NC with CCS as the status quo)
red = {}
for m in order:
    a = d[d.model == m].sort_values("seed"); c = d[d.model == "CCS"].sort_values("seed")
    red[m] = ci(1 - a.nc_model.values / c.nc_model.values) if m != "CCS" else (0, 0)
print("reduction vs CCS", {k: (round(100 * v[0], 1), round(100 * v[1], 1)) for k, v in red.items()})
red_sq = {m: ci(1 - d[d.model == m].sort_values("seed").nc_model.values / d[d.model == m].sort_values("seed").nc_sq.values) for m in order}
print("reduction vs end-of-batch", {k: (round(100 * v[0], 1), round(100 * v[1], 1)) for k, v in red_sq.items()})
# attribution
A = []
for r in M:
    for k, v in r["att"].items():
        for kind in ["all", "alarmed"]:
            if v.get(kind): A.append(dict(seed=r["seed"], model=k, kind=kind, **{kk: vv for kk, vv in v[kind].items() if kk != "n"}, n=v[kind].get("n")))
A = pd.DataFrame(A)
att = A.groupby(["kind", "model"])[["overall"] + CAUSES].mean()
att["macro"] = att[CAUSES].mean(axis=1)
print("\nattribution\n", att.round(3).to_string())
# lot clustering null
C = pd.DataFrame([dict(seed=r["seed"], model=k, **v) for r in M for k, v in r["clus"].items()])
cg = C.groupby("model").agg(top=("top_share", "mean"), null=("null_share", "mean"), sig=("p", lambda p: int((p < 0.05).sum())),
                            n=("p", lambda p: int(p.notna().sum())))
print("\nlot clustering\n", cg.round(3).to_string())
# lot-block threshold arm
L = pd.DataFrame([dict(seed=r["seed"], model=k, **{kk: vv for kk, vv in v.items() if kk in ("far_ok", "far_ok_std", "ep_detect", "nc_model", "nc_sq", "quantile", "false_alarms")}) for r in M for k, v in r["lotblock"].items()])
L["cover"] = L.far_ok <= 0.02; L["cover_std"] = L.far_ok_std <= 0.02
lb = L.groupby("model")[["far_ok", "far_ok_std", "cover", "cover_std", "ep_detect", "nc_model", "quantile"]].mean()
print("\nlot-block threshold arm\n", lb.round(3).to_string())
ov = np.mean([r["overlap"] for r in M]); ovt = np.mean([r["overlap_te"] for r in M]); lv = pd.DataFrame([r["lots_val"] for r in M]).mean().to_dict()
print("overlap sub-lots per rep", ov, "test", ovt, "lots in validation", lv)
# unseen faults
U = pd.read_csv(R + "r2_lofo.csv")
ug = U.groupby("model")[["ep_detect", "chance_total_ep", "chance_fa_ep"]].mean()
print("\nunseen faults (mean over types and seeds)\n", ug.round(3).to_string())
ut = U.groupby(["held_out", "model"]).agg(ep=("ep_detect", "mean"), ch_tot=("chance_total_ep", "mean"), ch_fa=("chance_fa_ep", "mean"),
                                          n_ep=("n_ep", "sum"), n_det=("n_det", "sum"), attr=("attr", "mean"))
print(ut.round(3).to_string())
# significance of LR vs chance on unseen
lr = U[U.model == "LR-DS"].groupby("seed")[["ep_detect", "chance_total_ep", "chance_fa_ep"]].mean()
p_tot = stats.wilcoxon(lr.ep_detect, lr.chance_total_ep).pvalue; p_fa = stats.wilcoxon(lr.ep_detect, lr.chance_fa_ep).pvalue
print("LR unseen vs chance total p", round(p_tot, 3), "vs chance fa p", round(p_fa, 3))
# mixture break-even (known from r2 main, unseen from r2 lofo)
known = T.ep_detect; unseen = ug.ep_detect
mix = {}
for m in ["LR-DS", "Hybrid", "SPC-DS", "CCS", "T2"]:
    mix[m] = (float(known[m]), float(unseen[m]))
s_star = (mix["LR-DS"][0] - mix["Hybrid"][0]) / ((mix["LR-DS"][0] - mix["Hybrid"][0]) + (mix["Hybrid"][1] - mix["LR-DS"][1]))
print("\nmixture (known, unseen)", {k: tuple(round(x, 3) for x in v) for k, v in mix.items()}, "hybrid break-even novel share", round(s_star, 3))
# nonlinear scenario 2
N = pd.read_csv(R + "r2_nonlinear2.csv")
ng = N.groupby("model")[["pr_auc", "ep_detect", "nc_model", "false_alarms", "far_ok"]].mean()
print("\nnonlinear scenario 2\n", ng.round(3).to_string(), "\nchanged NC share", N.changed_share.mean().round(3))
x = N[N.model == "LR-DS"].sort_values("seed").pr_auc.values; y = N[N.model == "XGB-DS-u"].sort_values("seed").pr_auc.values
print("NL2 LR minus XGB-u", round((x - y).mean(), 3), "ci", round(ci(x - y)[1], 3), "p", round(stats.wilcoxon(x, y).pvalue, 3))
# confirmatory family (declared): Holm over six tests
dv = pd.DataFrame([dict(seed=r["seed"], model=k, **v) for r in V for k, v in r["res"].items()])
def pair(df, a, b, c):
    x = df[df.model == a].sort_values("seed")[c].values; y = df[df.model == b].sort_values("seed")[c].values
    return x - y, stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
lo = U.groupby(["seed", "model"]).ep_detect.mean().unstack()
tests = []
for lab, (diff, p) in [("LR-DS vs LR-snap, PR-AUC", pair(dv, "LR-DS", "LR-snap", "pr_auc")),
                       ("LR-DS vs XGB-DS-u, PR-AUC", pair(dv, "LR-DS", "XGB-DS-u", "pr_auc")),
                       ("LR-DS vs XGB-DS-u, false alarms", pair(dv, "LR-DS", "XGB-DS-u", "false_alarms")),
                       ("LR-DS vs SPC-DS, NC line-h", pair(d, "LR-DS", "SPC-DS", "nc_model")),
                       ("LR-DS vs CCS, NC line-h", pair(d, "LR-DS", "CCS", "nc_model")),
                       ("Hybrid vs LR-DS, unseen timely detection", ((lo.Hybrid - lo["LR-DS"]).values, stats.wilcoxon(lo.Hybrid, lo["LR-DS"]).pvalue))]:
    tests.append(dict(test=lab, mean_diff=float(np.mean(diff)), ci=ci(diff)[1], p=float(p)))
P = pd.DataFrame(tests).sort_values("p").reset_index(drop=True); k = len(P)
P["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(P.p)]))
print("\nconfirmatory family\n", P.round(4).to_string())
P.to_csv(R + "r2_confirmatory.csv", index=False)
# descriptive extra tests
for a_, b_, c_ in [("LR-DS", "SPC-DS", "pr_auc"), ("LR-DS", "SPC-DS", "ep_detect"), ("LR-DS", "CCS", "ep_detect"), ("LR-DS", "CCS", "false_alarms"),
                   ("Hybrid", "CCS", "nc_model"), ("SPC-DS", "CCS", "nc_model"), ("LR-DS", "SPC", "nc_model")]:
    diff, p = pair(d, a_, b_, c_); print("desc", a_, b_, c_, round(np.mean(diff), 3), "±", round(ci(diff)[1], 3), "p", round(p, 4))
key = dict(T=T.round(4).to_dict(orient="index"), att=att.round(4).reset_index().to_dict(orient="records"), clus=cg.to_dict(orient="index"),
           lotblock=lb.to_dict(orient="index"), unseen=ug.to_dict(orient="index"), unseen_type=ut.reset_index().to_dict(orient="records"),
           mix=mix, s_star=s_star, nl2=ng.to_dict(orient="index"), nl2_changed=float(N.changed_share.mean()),
           overlap=ov, overlap_te=ovt, lots_val=lv, red_ccs={k: v for k, v in red.items()}, red_sq=red_sq, confirm=P.to_dict(orient="records"))
json.dump(key, open(R + "r2_key.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer, np.bool_)) else str(o))
# stress scenarios with rule-based and DS-chart comparators
st = {}
for cond in ["mild", "shared", "nonlinear2"]:
    try:
        Sd = pd.read_csv(R + f"r2_stress_{cond}.csv")
        g = Sd.groupby("model")[["pr_auc", "ep_detect", "nc_model", "nc_sq", "false_alarms", "far_ok", "net_0.5", "net_2.0"]].mean()
        g["reduction"] = Sd.assign(red=1 - Sd.nc_model / Sd.nc_sq).groupby("model").red.mean() * 100
        print(f"\nstress {cond}\n", g.round(3).to_string()); st[cond] = g.round(4).to_dict(orient="index")
    except FileNotFoundError:
        print("missing stress", cond)
key["stress"] = st
json.dump(key, open(R + "r2_key.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer, np.bool_)) else str(o))
