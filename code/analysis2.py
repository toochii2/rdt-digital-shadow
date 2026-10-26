"""v2 analysis: summary table, Holm-corrected paired tests, key numbers JSON for the manuscript."""
import json, numpy as np, pandas as pd
from scipy import stats
RES = "../results/"
R = json.load(open(RES + "v2_main.json"))
d = pd.DataFrame([dict(seed=r["seed"], model=k, **v) for r in R for k, v in r["res"].items()])
A = pd.DataFrame([dict(seed=r["seed"], model=k, **{kk: vv for kk, vv in v.items() if kk != "n"}) for r in R for k, v in r["att"].items()])
order = ["SPC", "LR-snap", "XGB-snap", "T2", "LR-DS", "XGB-DS-u", "XGB-DS", "Hybrid"]


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


rows = []
for m in order:
    g = d[d.model == m]; row = {"model": m}
    for c in ["pr_auc", "recall", "precision", "far", "far_ok", "alarm_rate", "brier", "false_alarms", "ep_detect", "chance_ep_detect",
              "ep_delay", "lead_vs_release", "nc_model", "chance_nc", "nc_sq", "net_saving", "fa_top_lot_share"]:
        mm, h = ci(g[c]); row[c] = mm; row[c + "_ci"] = h
    a = A[A.model == m]
    if len(a): mm, h = ci(a.overall); row["attr"] = mm; row["attr_ci"] = h
    rows.append(row)
T = pd.DataFrame(rows).set_index("model")
T["nc_reduction_pct"] = 100 * (1 - T.nc_model / T.nc_sq)
T.to_csv(RES + "v2_table_main.csv")
pd.set_option("display.width", 260)
print(T[["pr_auc", "pr_auc_ci", "recall", "far_ok", "far", "ep_detect", "chance_ep_detect", "ep_delay", "lead_vs_release",
         "nc_model", "chance_nc", "nc_sq", "nc_reduction_pct", "false_alarms", "net_saving", "attr", "fa_top_lot_share", "brier"]].round(3).to_string())

# reductions per seed with CI
red = {}
for m in order:
    x = (1 - d[d.model == m].sort_values("seed").nc_model.values / d[d.model == m].sort_values("seed").nc_sq.values)
    red[m] = ci(x)
print("reductions", {k: (round(100 * v[0], 1), round(100 * v[1], 1)) for k, v in red.items()})
hq = np.array([r["hourly_qc_nc"] for r in R]); print("hourly QC nc", hq.mean())

# paired tests with Holm correction
tests = [("LR-DS", "LR-snap", "pr_auc"), ("XGB-DS-u", "XGB-snap", "pr_auc"), ("LR-DS", "XGB-DS-u", "pr_auc"),
         ("LR-DS", "XGB-DS", "pr_auc"), ("LR-DS", "T2", "pr_auc"), ("Hybrid", "LR-DS", "ep_detect"),
         ("Hybrid", "T2", "ep_detect"), ("Hybrid", "SPC", "nc_model"), ("LR-DS", "SPC", "nc_model"), ("Hybrid", "LR-DS", "nc_model"),
         ("Hybrid", "LR-DS", "fa_top_lot_share")]
out = []
for a_, b_, c in tests:
    x = d[d.model == a_].sort_values("seed")[c].values; y = d[d.model == b_].sort_values("seed")[c].values
    diff = x - y; p = stats.wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0
    out.append(dict(a=a_, b=b_, metric=c, mean_diff=diff.mean(), ci=ci(diff)[1], p=p))
P = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
k = len(P); P["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(P.p)]))
print(P.round(4).to_string())
P.to_csv(RES + "v2_tests.csv", index=False)

print("\nAttribution by cause (mean)")
print(A.groupby("model").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
abl = pd.DataFrame([r["abl"] for r in R]); print("ablations", abl.mean().round(3).to_dict(), "LR-DS", round(T.loc['LR-DS', 'pr_auc'], 3))
fc = pd.DataFrame([dict(seed=r["seed"], model=m, **v) for r in R for m, v in r["farchk"].items()])
fc["cov_sub"] = fc.test <= fc.ub_sublot; fc["cov_lot"] = fc.test <= fc.ub_lot
print(fc.groupby("model")[["test", "ub_sublot", "ub_lot", "cov_sub", "cov_lot"]].mean().round(3))
print("tuned", [r["tuned"] for r in R])
print("hybrid LR share", np.nanmean([r["hybrid_lr_share"] for r in R]))
print("prev", np.mean([r["prevalence"] for r in R]), "test", np.mean([r["test_prev"] for r in R]))
print("n nc by cause (test, summed)", pd.DataFrame([r["n_nc_by_cause"] for r in R]).sum().to_dict())
print("episodes/test", d[d.model == "Hybrid"].n_episodes.describe()[["mean", "min", "max"]].to_dict())
print("completeness", np.mean([r["dq"]["sample_completeness"] for r in R]))
key = dict(T=T.round(4).to_dict(orient="index"), red={k: v for k, v in red.items()}, hourly_qc=float(hq.mean()),
           abl=abl.mean().to_dict(), farchk=fc.groupby("model")[["test", "ub_sublot", "ub_lot", "cov_sub", "cov_lot"]].mean().to_dict(orient="index"),
           att=A.groupby("model").mean(numeric_only=True).drop(columns="seed").to_dict(orient="index"))
for name in ["nonlinear", "shared", "lowprev"]:
    try:
        RR = json.load(open(RES + f"v2_{name}.json"))
        dd = pd.DataFrame([dict(seed=r["seed"], model=k, **v) for r in RR for k, v in r["res"].items()])
        AA = pd.DataFrame([dict(seed=r["seed"], model=k, overall=v.get("overall", np.nan)) for r in RR for k, v in r["att"].items()])
        g = dd.groupby("model")[["pr_auc", "precision", "recall", "ep_detect", "nc_model", "nc_sq", "far_ok"]].mean()
        g["attr"] = AA.groupby("model").overall.mean()
        print(f"\n== {name}\n", g.round(3).to_string())
        x = dd[dd.model == "LR-DS"].sort_values("seed").pr_auc.values; y = dd[dd.model == "XGB-DS-u"].sort_values("seed").pr_auc.values
        print("LR-DS minus XGB-DS-u", round((x - y).mean(), 3), stats.wilcoxon(x, y).pvalue)
        print("prevalence", np.mean([r["test_prev"] for r in RR]))
        key[name] = dict(g=g.round(4).to_dict(orient="index"), lr_minus_xgbu=float((x - y).mean()), p=float(stats.wilcoxon(x, y).pvalue),
                         prev=float(np.mean([r["test_prev"] for r in RR])))
    except FileNotFoundError:
        print("missing", name)
try:
    L = pd.read_csv(RES + "v2_lofo.csv")
    print("\nLOFO\n", L.groupby(["held_out", "model"])[["ep_detect", "chance"]].mean().unstack().round(2).to_string())
    print(L.groupby("model")[["recall", "ep_detect", "chance"]].mean().round(3))
    h = L[L.model == "Hybrid"]; print("hybrid unseen attr", h.attr.mean(), "t2 share", h.t2_share.mean())
    key["lofo"] = dict(by_model=L.groupby("model")[["recall", "ep_detect", "chance"]].mean().to_dict(orient="index"),
                       by_type=L.groupby(["held_out", "model"]).ep_detect.mean().unstack().to_dict(orient="index"),
                       hybrid_attr=float(h.attr.mean()))
except FileNotFoundError: print("missing lofo")
try:
    S = json.load(open(RES + "v2_stability.json"))
    s = pd.DataFrame([dict(seed=k, model=m, **v) for k, dd_ in S.items() for m, v in dd_.items()])
    print(s.groupby("model")[["kendall_tau", "local_agreement"]].mean().round(3)); key["stab"] = s.groupby("model")[["kendall_tau", "local_agreement"]].mean().to_dict(orient="index")
except FileNotFoundError: print("missing stab")
try:
    Rb = pd.read_csv(RES + "v2_robust.csv")
    rb = Rb.groupby(["cond", "model"])[["pr_auc", "ep_detect"]].mean().unstack("model")
    print(rb.round(3).to_string()); key["robust"] = {c: {f"{k[0]}|{k[1]}": v for k, v in rb.loc[c].to_dict().items()} for c in rb.index}
except FileNotFoundError: print("missing robust")
try:
    C = pd.read_csv(RES + "v2_containment.csv"); print(C.to_string()); key["containment"] = C.to_dict(orient="records")
except FileNotFoundError: print("missing containment")
json.dump(key, open(RES + "v2_key.json", "w"), indent=1, default=lambda o: str(o))
