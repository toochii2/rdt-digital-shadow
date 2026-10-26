"""Round-2 figures: drawn at final print width (122 mm) with lettering of at least 6.5 pt, one vocabulary and one
palette across figures. Detector hues: LR green, Hybrid blue, T2 orange, rule-based control violet, others grey."""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import experiments as E, sim

plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 800, "figure.dpi": 150, "pdf.fonttype": 42, "hatch.linewidth": 0.5,
})
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, GREY, MGREY, LGREY = "#0b0b0b", "#6b6a66", "#9a9994", "#c9c8c3"
W = 122 / 25.4
OUT = "../figures/"
CAUSE = {"nozzle": "Nozzle", "humidity": "Humidity", "drying": "Oven", "membrane": "Membrane lot", "reagent": "Antibody lot"}
NAME = {"SPC": "SPC (snapshot)", "LR-snap": "LR (snapshot)", "XGB-snap": "XGB (snapshot)", "SPC-DS": "SPC (DS)",
        "CCS": "Rules (DS)", "T2": "T² (DS)", "LR-DS": "LR (DS)", "XGB-DS-u": "XGB (DS)", "XGB-DS": "XGB-mono (DS)",
        "Hybrid": "Hybrid (DS)", "Naive-z": "Model-free z"}
COL = {"SPC": LGREY, "LR-snap": MGREY, "XGB-snap": MGREY, "SPC-DS": GREY, "CCS": VIOLET, "T2": ORANGE, "LR-DS": AQUA,
       "XGB-DS-u": INK, "XGB-DS": INK, "Hybrid": BLUE}


def save(fig, name):
    fig.savefig(OUT + name + ".png", dpi=800, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------- Fig. 1 architecture with evidence-card mock-up ----------------
def fig_architecture():
    fig = plt.figure(figsize=(W, 2.9))
    ax = fig.add_axes([0, 0, 0.6, 1]); ax.set_xlim(-8, 100); ax.set_ylim(0, 100); ax.axis("off")
    def box(x, y, w, h, title, body, fc="#ffffff", ec=INK, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text(x + 1.4, y + h - 1.4, title, ha="left", va="top", fontsize=7, weight="bold")
        ax.text(x + 1.4, y + h - 6.4, body, ha="left", va="top", fontsize=6.5, color="#1a1a1a", linespacing=1.1)
    layers = [
        ("L1  Operation and records", "Line sensors, lot genealogy, maintenance,\nincoming inspection, in-process QC"),
        ("L2  Integration", "5-min signals to 1-h sub-lots; lot linkage;\ncompleteness and provenance stamps"),
        ("L3  Digital shadow (one-way sync)", "18-feature state: speed-normalised back-\npressure, trends, humidity exposure, lot values"),
        ("L4  Analytics (tiered)", "Rules and SPC on the shadow; logistic risk;\nT² novelty gate; pre-set false-alarm rate"),
        ("L5  Evidence card", "Risk, cause groups, z cross-check, novelty\nflag, reliability, sources, versions"),
        ("L6  Human QA disposition", "Operator check, QA decision, reasons for\noverrides; no autonomous release"),
    ]
    h, gap = 12.6, 2.2; y = 100 - h; ys = []
    for t, b_ in layers:
        hl = t.startswith(("L3", "L4"))
        box(0, y, 99, h, t, b_, fc="#f2f7fd" if hl else "#ffffff", ec=BLUE if hl else INK, lw=1.0 if hl else 0.7)
        ys.append(y); y -= h + gap
    for yy in ys[:-1]:
        ax.add_patch(FancyArrowPatch((50, yy + 0.1), (50, yy - gap - 0.1), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=INK))
    yt = ys[0] + h / 2; yb = ys[-1] + h / 2
    ax.plot([-0.6, -4.2, -4.2], [yb, yb, yt], color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax.add_patch(FancyArrowPatch((-4.2, yt), (-0.7, yt), arrowstyle="-|>", mutation_scale=5, lw=0.8, color=GREY))
    ax.text(-6.4, (yt + yb) / 2, "human-closed feedback", rotation=90, ha="center", va="center", fontsize=6.5, color=GREY)
    GH = 12.6; gy = ys[-1] - gap - GH
    ax.add_patch(FancyBboxPatch((0, gy), 99, GH, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#f2f1ee", ec=GREY, lw=0.7))
    ax.text(1.4, gy + GH - 1.4, "Governance (all layers)", ha="left", va="top", fontsize=7, weight="bold")
    ax.text(1.4, gy + GH - 6.4, "Access control, audit trail, versioning, change control,\nlot-blocked threshold validation, clustering monitor",
            ha="left", va="top", fontsize=6.5, linespacing=1.1)
    ax.set_ylim(gy - 1, 100.5)
    # evidence card mock-up
    cx = fig.add_axes([0.62, 0.0, 0.38, 1]); cx.set_xlim(0, 100); cx.set_ylim(0, 100); cx.axis("off")
    cx.add_patch(FancyBboxPatch((1, 1), 98, 97, boxstyle="round,pad=0.4,rounding_size=2", fc="#ffffff", ec=BLUE, lw=1.0))
    cx.add_patch(Rectangle((1, 89), 98, 9, fc="#f2f7fd", ec="none"))
    cx.text(4, 96.5, "Evidence card (illustrative)", fontsize=7, weight="bold", va="top")
    cx.text(4, 92.0, "Line A, sub-lot 14:00–15:00, batch 2", fontsize=6.5, va="top", color=GREY)
    cx.text(4, 85.5, "Priority  HIGH   (episode open 2 h)", fontsize=6.8, va="top", weight="bold", color=INK)
    cx.text(4, 80.0, "More unusual than 99% of validated\nnormal hours (logistic component)", fontsize=6.5, va="top", linespacing=1.1)
    cx.text(4, 69.5, "Cause groups", fontsize=6.8, va="top", weight="bold")
    for i, (lab, v) in enumerate([("Nozzle", 0.86), ("Humidity", 0.08), ("Oven", 0.04), ("Lots", 0.02)]):
        yy = 64.5 - i * 4.6
        cx.text(4, yy, lab, fontsize=6.5, va="center")
        cx.add_patch(Rectangle((33, yy - 1.4), 50 * v, 2.8, fc=AQUA if i == 0 else LGREY, ec="none"))
        cx.text(35 + 50 * v, yy, f"{v:.2f}", fontsize=6.5, va="center")
    cx.text(4, 44.5, "z cross-check: agrees (Nozzle)", fontsize=6.5, va="top")
    cx.text(4, 39.8, "Novelty gate: within known patterns", fontsize=6.5, va="top")
    cx.text(4, 35.1, "Past reliability, Nozzle: 100% (sim.)", fontsize=6.5, va="top")
    cx.text(4, 30.4, "Sources: back-pressure trend; last\ncleaning 26 h ago; lots M-07, R-02", fontsize=6.5, va="top", linespacing=1.1)
    cx.text(4, 20.3, "Data 12/12 samples; model v2.0", fontsize=6.5, va="top", color=GREY)
    for j, b in enumerate(["Investigate", "Contain", "Override"]):
        cx.add_patch(FancyBboxPatch((4 + j * 31.5, 5.5), 28, 7.5, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#ffffff", ec=INK, lw=0.6))
        cx.text(18 + j * 31.5, 9.3, b, fontsize=6.5, ha="center", va="center")
    cx.text(50, 2.6, "override requires a recorded reason", fontsize=6, ha="center", va="center", color=GREY)
    save(fig, "fig1_architecture")


# ---------------- Fig. 2 detection and attribution ----------------
def fig_results():
    T = pd.read_csv("../results/v2_table_main.csv").set_index("model")
    T2r = pd.read_csv("../results/r2_table.csv").set_index("model")
    R = json.load(open("../results/v2_main.json")); R2 = json.load(open("../results/r2_main.json"))
    A = pd.DataFrame([dict(seed=r["seed"], model=k, **{kk: vv for kk, vv in v.items() if kk != "n"}) for r in R for k, v in r["att"].items()])
    A2 = pd.DataFrame([dict(seed=r["seed"], model="SPC-DS", **{kk: vv for kk, vv in r["att"]["SPC-DS"]["all"].items() if kk != "n"}) for r in R2])
    A = pd.concat([A, A2])
    fig = plt.figure(figsize=(W, 2.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    order = ["SPC", "LR-snap", "XGB-snap", "SPC-DS", "T2", "XGB-DS", "XGB-DS-u", "LR-DS", "Hybrid"]
    yy = np.arange(len(order))[::-1]
    for y, m in zip(yy, order):
        src = T2r if m == "SPC-DS" else T
        mu, h = src.loc[m, "pr_auc"], src.loc[m, "pr_auc_ci"]
        ds_ = m not in ("SPC", "LR-snap", "XGB-snap")
        c = COL[m] if m in ("LR-DS", "Hybrid", "T2") else (INK if ds_ else MGREY)
        mk = "D" if m == "Hybrid" else ("o" if ds_ else "s")
        ax.errorbar(mu, y, xerr=h, fmt=mk, color=c, ms=3.6, mfc=c if ds_ else "white", lw=0.8, capsize=1.8)
        ax.text(mu + h + 0.01, y, f"{mu:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(yy); ax.set_yticklabels([NAME[m] for m in order])
    ax.axhspan(5.5, 8.5, color="#f2f1ee", zorder=0, lw=0)
    ax.set_xlim(0.45, 0.97); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); ax.set_xlabel("PR-AUC (mean, 95% CI)")
    ax.text(-0.62, 1.04, "(a)", transform=ax.transAxes, weight="bold", fontsize=8)
    ax = fig.add_subplot(gs[1])
    morder = ["SPC", "Naive-z", "SPC-DS", "T2", "XGB-DS-u", "XGB-DS", "LR-DS", "Hybrid"]
    M = A.groupby("model").mean(numeric_only=True).reindex(morder)[sim.FAULT_TYPES]
    ax.imshow(np.where(M.values < 0.05, np.nan, M.values), cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]; ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.82 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels([CAUSE[c] for c in sim.FAULT_TYPES], rotation=35, ha="right")
    ax.set_yticks(range(len(morder))); ax.set_yticklabels([NAME[m] for m in morder])
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax.set_title("Correct cause group (%)", fontsize=7.5, loc="left")
    ax.text(-0.62, 1.04, "(b)", transform=ax.transAxes, weight="bold", fontsize=8)
    save(fig, "fig2_detection_attribution")


# ---------------- Fig. 3 episode ----------------
def fig_story(seed=10):
    r, ctx = E.run_seed(seed, chance=False)
    sub = ctx["sub"]; ds = ctx["feats"]["ds"]; eps = ctx["eps"]
    te = (sub.split == "test").values; t0 = np.where(te)[0][0]
    al = ctx["alarms"]["Hybrid"]; sc = ctx["scores"]["Hybrid"]; thr = ctx["thr"]["Hybrid"]
    best = None
    for e in eps:
        if e["type"] != "nozzle" or e["start"] < t0 + 12: continue
        pre = np.arange(e["start"] - 10, e["start"]); win = np.arange(e["start"], e["start"] + e["dur"])
        post = np.arange(e["start"] + e["dur"], e["start"] + e["dur"] + 6)
        if (sub.nc.values[pre] == 1).any() or (sub.cause.values[np.concatenate([pre, post])] != "").any(): continue
        if (sub.cause.values[win] != "nozzle").any(): continue
        ncw = win[sub.nc.values[win] == 1]
        if len(ncw) < 4: continue
        onset = ncw[0]; a = win[al[win]]
        if len(a) == 0: continue
        tsq = (onset // 8 + 1) * 8 - 1 + E.LAB_DELAY
        if best is None or tsq - a[0] > best[0]: best = (tsq - a[0], e, onset, a[0], tsq)
    gain, e, onset, t_al, tsq = best
    lo, hi = e["start"] - 10, e["start"] + e["dur"] + 6
    idx = np.arange(lo, hi); x = idx - onset
    A = ctx["models"]["LR-DS"][0].attrib(ds.iloc[idx]).reset_index(drop=True)
    G = pd.DataFrame({g: A[[c for c in cols if c in A.columns]].clip(lower=0).sum(1) for g, cols in sim.FEATURE_GROUPS.items()})
    fig, axs = plt.subplots(3, 1, figsize=(W, 2.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.15], hspace=0.2))
    ax = axs[0]; ax.plot(x, ds.bp_norm.values[idx], color=INK, lw=1.0, marker="o", ms=2.2)
    ax.set_ylabel("Back-pressure at\n50 mm/s (kPa)")
    ax2 = axs[1]; ax2.plot(x, sc[idx], color=BLUE, lw=1.0, marker="s", ms=2.2)
    ax2.axhline(thr, color=GREY, lw=0.7, ls="--"); ax2.text(x[0] + 0.2, thr + 0.015, "alarm threshold (2% false-alarm rate)", fontsize=6.5, color=GREY, va="bottom")
    ax2.set_ylabel("Hybrid score\n(percentile)"); ax2.set_ylim(min(0.9, np.nanmin(sc[idx]) - 0.03), 1.15)
    ax3 = axs[2]; bottom = np.zeros(len(idx))
    sty = {"nozzle": (INK, ""), "humidity": ("#ffffff", "////"), "drying": (LGREY, ""), "membrane": ("#ffffff", "...."), "reagent": (MGREY, "")}
    for g in sim.FEATURE_GROUPS:
        v = G[g].values; fc, hh = sty[g]
        ax3.bar(x, v, bottom=bottom, width=0.72, color=fc, edgecolor=INK if fc == "#ffffff" else "white", lw=0.4, hatch=hh, label=CAUSE[g])
        bottom += v
    ax3.set_ylabel("Positive contribution\n(LR component)"); ax3.set_ylim(0, bottom.max() * 1.45)
    ax3.legend(ncol=5, loc="upper left", frameon=False, handlelength=1.1, columnspacing=0.8, fontsize=6.5, bbox_to_anchor=(0, 1.08))
    ncx = x[sub.nc.values[idx] == 1]
    for a in axs:
        a.axvspan(e["start"] - onset - 0.5, e["start"] + e["dur"] - onset - 0.5, color="#eef3fb", zorder=0)
        a.axvline(0, color=INK, lw=0.7, ls=":"); a.axvline(t_al - onset, color=BLUE, lw=0.9); a.axvline(tsq - onset, color=INK, lw=0.9, ls="-.")
    y0 = axs[0].get_ylim()[1]
    axs[0].plot(ncx, np.full(len(ncx), y0), "v", color=INK, ms=3.2, clip_on=False, zorder=5, label="QC nonconforming sub-lot")
    axs[0].legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.0, borderaxespad=0.1)
    for xv, lab, c, ha, dx in [(0, "first NC", INK, "left", 0.25), (t_al - onset, "first alarm", BLUE, "right", -0.25),
                               (tsq - onset, "QC result", INK, "right", -0.25)]:
        axs[1].text(xv + dx, 0.06, lab, transform=axs[1].get_xaxis_transform(), rotation=90, fontsize=6.5, color=c, ha=ha, va="bottom")
    ax3.set_xlabel("Hours relative to the first nonconforming sub-lot")
    for a, lab in zip(axs, "abc"):
        a.text(-0.205, 1.02, f"({lab})", transform=a.transAxes, fontsize=8, weight="bold", va="top")
    save(fig, "fig3_episode")
    return dict(seed=seed, onset=int(onset), alarm=int(t_al), tsq=int(tsq), lead_vs_qc=int(tsq - t_al), alarm_rel_onset=int(t_al - onset))


# ---------------- Fig. 4 per-type detection, seen and unseen ----------------
def fig_types():
    T = pd.read_csv("../results/r2_table.csv").set_index("model")
    U = pd.read_csv("../results/r2_lofo.csv")
    rows = ["CCS", "SPC-DS", "T2", "LR-DS", "XGB-DS", "Hybrid"]
    seen = T.loc[rows, [f"ep_{c}" for c in sim.FAULT_TYPES]].values
    seen = np.column_stack([seen, T.loc[rows, "ep_detect"].values])
    un = U.groupby(["model", "held_out"]).ep_detect.mean().unstack().reindex(rows)[sim.FAULT_TYPES]
    pooled = U.groupby(["model", "seed"])[["n_det", "n_ep"]].sum(); pooled = (pooled.n_det / pooled.n_ep).groupby("model").mean()
    un["all"] = pooled.reindex(rows).values; unseen = un.values   # per-replication pooled over episodes, as in panel (a)
    chs = [T.loc[rows, "chance_total_ep"].mean(), None]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.8), gridspec_kw=dict(wspace=0.08))
    labs = [CAUSE[c] for c in sim.FAULT_TYPES] + ["All"]
    for k, (ax, Mv, ttl) in enumerate(zip(axs, [seen, unseen], ["(a) Fault type seen in training", "(b) Fault type withheld from training"])):
        ax.imshow(Mv, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(Mv.shape[0]):
            for j in range(Mv.shape[1]):
                v = Mv[i, j]; ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else INK,
                                      weight="bold" if j == Mv.shape[1] - 1 else None)
        ax.axvline(4.5, color="white", lw=1.5)
        ax.set_xticks(range(6)); ax.set_xticklabels(labs, rotation=35, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[m] for m in rows] if k == 0 else [])
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
        ax.set_title(ttl, fontsize=7.5, loc="left")
    save(fig, "fig4_fault_types")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["arch", "results", "story", "types"]
    if "arch" in which: fig_architecture()
    if "results" in which: fig_results()
    if "types" in which: fig_types()
    if "story" in which: print(fig_story(10))
