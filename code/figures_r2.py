"""Figures as published: drawn at the final print width (122 mm) with lettering of at least 8 pt (the template asks for
about 2 mm), one vocabulary and one palette. Detector hues: LR green, Hybrid blue, T2 orange, others grey or black."""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D
import experiments as E, sim

FS = 8.0          # minimum lettering at print size
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": FS, "axes.titlesize": 8.5, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 880, "figure.dpi": 150, "pdf.fonttype": 42, "hatch.linewidth": 0.5,
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
    fig.savefig(OUT + name + ".png", dpi=880, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------- Fig. 1 architecture with evidence-card mock-up ----------------
def fig_architecture():
    fig = plt.figure(figsize=(W, 3.1))
    ax = fig.add_axes([0, 0, 0.59, 1]); ax.set_xlim(-9, 100); ax.set_ylim(0, 100); ax.axis("off")
    def box(x, y, w, h, title, body, fc="#ffffff", ec=INK, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text(x + 1.6, y + h - 1.3, title, ha="left", va="top", fontsize=8.5, weight="bold")
        ax.text(x + 1.6, y + h - 5.4, body, ha="left", va="top", fontsize=FS, color="#1a1a1a", linespacing=1.15)
    layers = [
        ("L1  Operation and records", "Line sensors, lot genealogy, inspection, QC"),
        ("L2  Integration", "Hourly sub-lots, lot linkage, provenance stamps"),
        ("L3  Digital shadow (one-way)", "18 features: normalised back-pressure, lot values"),
        ("L4  Analytics (tiered)", "Rules, SPC, logistic risk, T\u00b2 gate, pre-set FAR"),
        ("L5  Evidence card", "Risk, cause groups, cross-check, novelty, sources"),
        ("L6  Human QA disposition", "Operator check, QA decision; no autonomous release"),
    ]
    h, gap = 9.6, 3.9; y = 100 - h; ys = []
    for t, b_ in layers:
        hl = t.startswith(("L3", "L4"))
        box(0, y, 99, h, t, b_, fc="#f2f7fd" if hl else "#ffffff", ec=BLUE if hl else INK, lw=1.0 if hl else 0.7)
        ys.append(y); y -= h + gap
    for yy in ys[:-1]:
        ax.add_patch(FancyArrowPatch((50, yy - 0.3), (50, yy - gap + 0.3), arrowstyle="-|>", mutation_scale=9, lw=1.0,
                                     color=INK, shrinkA=0, shrinkB=0))
    yt = ys[0] + h / 2; yb = ys[-1] + h / 2
    ax.plot([-0.6, -4.6, -4.6], [yb, yb, yt], color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax.add_patch(FancyArrowPatch((-4.6, yt), (-0.7, yt), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=GREY))
    ax.text(-7.2, (yt + yb) / 2, "human-closed feedback", rotation=90, ha="center", va="center", fontsize=FS, color=GREY)
    GH = 9.6; gy = ys[-1] - gap - GH
    ax.add_patch(FancyBboxPatch((0, gy), 99, GH, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#f2f1ee", ec=GREY, lw=0.7))
    ax.text(1.6, gy + GH - 1.3, "Governance (all layers)", ha="left", va="top", fontsize=8.5, weight="bold")
    ax.text(1.6, gy + GH - 5.4, "Audit trail, change control, lot-blocked thresholds",
            ha="left", va="top", fontsize=FS, linespacing=1.15)
    ax.set_ylim(gy - 1, 100.6)
    cx = fig.add_axes([0.615, 0.0, 0.38, 1]); cx.set_xlim(0, 101.5); cx.set_ylim(0, 100); cx.axis("off")
    cx.add_patch(FancyBboxPatch((1, 1), 98, 97.5, boxstyle="round,pad=0.4,rounding_size=2", fc="#ffffff", ec=BLUE, lw=1.0))
    cx.add_patch(Rectangle((1, 89.5), 98, 9, fc="#f2f7fd", ec="none"))
    cx.text(4, 97.2, "Evidence card (illustrative)", fontsize=8.5, weight="bold", va="top")
    cx.text(4, 92.8, "Line A, sub-lot 14:00–15:00", fontsize=FS, va="top", color=GREY)
    cx.text(4, 86.6, "Priority HIGH (episode 2 h)", fontsize=8.5, va="top", weight="bold", color=INK)
    cx.text(4, 81.8, "More unusual than 99% of\nvalidated normal hours", fontsize=FS, va="top", linespacing=1.15)
    cx.text(4, 71.4, "Cause groups", fontsize=8.5, va="top", weight="bold")
    for i, (lab, v) in enumerate([("Nozzle", 0.86), ("Humidity", 0.08), ("Oven", 0.04), ("Lots", 0.02)]):
        yy = 65.6 - i * 4.9
        cx.text(4, yy, lab, fontsize=FS, va="center")
        cx.add_patch(Rectangle((37, yy - 1.5), 44 * v, 3.0, fc=AQUA if i == 0 else LGREY, ec="none"))
        cx.text(39 + 44 * v, yy, f"{v:.2f}", fontsize=FS, va="center")
    cx.text(4, 45.0, "z cross-check: agrees", fontsize=FS, va="top")
    cx.text(4, 40.0, "Novelty gate: known pattern", fontsize=FS, va="top")
    cx.text(4, 35.0, "Past reliability: 100% (sim.)", fontsize=FS, va="top")
    cx.text(4, 30.0, "Sources: back-pressure trend,\ncleaning log, lots M-07, R-02", fontsize=FS, va="top", linespacing=1.15)
    cx.text(4, 20.6, "Data 12/12; model v2.0", fontsize=FS, va="top", color=GREY)
    for j, b in enumerate(["Investigate", "Contain", "Override"]):
        cx.add_patch(FancyBboxPatch((3 + j * 32, 7.0), 29.5, 7.0, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#ffffff", ec=INK, lw=0.6))
        cx.text(17.75 + j * 32, 10.5, b, fontsize=FS, ha="center", va="center")
    cx.text(50, 3.4, "override needs a recorded reason", fontsize=FS, ha="center", va="center", color=GREY)
    save(fig, "fig1_architecture")


# ---------------- Fig. 2 detection and attribution ----------------
def fig_results():
    T = pd.read_csv("../results/v2_table_main.csv").set_index("model")
    T2r = pd.read_csv("../results/r2_table.csv").set_index("model")
    R = json.load(open("../results/v2_main.json")); R2 = json.load(open("../results/r2_main.json"))
    A = pd.DataFrame([dict(seed=r["seed"], model=k, **{kk: vv for kk, vv in v.items() if kk != "n"}) for r in R for k, v in r["att"].items()])
    A2 = pd.DataFrame([dict(seed=r["seed"], model="SPC-DS", **{kk: vv for kk, vv in r["att"]["SPC-DS"]["all"].items() if kk != "n"}) for r in R2])
    A = pd.concat([A, A2])
    fig = plt.figure(figsize=(W, 2.05)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.72)
    ax = fig.add_subplot(gs[0])
    order = ["SPC", "LR-snap", "XGB-snap", "SPC-DS", "T2", "XGB-DS", "XGB-DS-u", "LR-DS", "Hybrid"]
    yy = np.arange(len(order))[::-1]
    for y, m in zip(yy, order):
        src = T2r if m == "SPC-DS" else T
        mu, h = src.loc[m, "pr_auc"], src.loc[m, "pr_auc_ci"]
        ds_ = m not in ("SPC", "LR-snap", "XGB-snap")
        c = COL[m] if m in ("LR-DS", "Hybrid", "T2") else (INK if ds_ else MGREY)
        mk = "D" if m == "Hybrid" else ("o" if ds_ else "s")
        ax.errorbar(mu, y, xerr=h, fmt=mk, color=c, ms=4, mfc=c if ds_ else "white", lw=0.9, capsize=2)
        ax.text(mu + h + 0.012, y, f"{mu:.2f}", va="center", fontsize=FS)
    ax.set_yticks(yy); ax.set_yticklabels([NAME[m] for m in order])
    ax.axhspan(5.5, 8.5, color="#f2f1ee", zorder=0, lw=0)
    ax.set_xlim(0.45, 0.99); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); ax.set_xlabel("PR-AUC (mean, 95% CI)")
    ax.text(-0.66, 1.03, "(a)", transform=ax.transAxes, weight="bold", fontsize=9)
    ax = fig.add_subplot(gs[1])
    morder = ["SPC", "Naive-z", "SPC-DS", "T2", "XGB-DS-u", "XGB-DS", "LR-DS", "Hybrid"]
    M = A.groupby("model").mean(numeric_only=True).reindex(morder)[sim.FAULT_TYPES]
    ax.imshow(np.where(M.values < 0.05, np.nan, M.values), cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]; ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=FS, color="white" if v > 0.82 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels([CAUSE[c] for c in sim.FAULT_TYPES], rotation=40, ha="right")
    ax.set_yticks(range(len(morder))); ax.set_yticklabels([NAME[m] for m in morder])
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax.set_title("Correct cause group (%)", fontsize=8.5, loc="left")
    ax.text(-0.66, 1.03, "(b)", transform=ax.transAxes, weight="bold", fontsize=9)
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
    fig, axs = plt.subplots(3, 1, figsize=(W, 2.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.1], hspace=0.42))
    bp = ds.bp_norm.values[idx]
    ax = axs[0]; ax.plot(x, bp, color=INK, lw=1.0, marker="o", ms=2.6)
    # bridge each sensor-dropout run with a thin dashed segment between the readings either side; no value is drawn
    # at the missing hour, so nothing is imputed
    miss = ~np.isfinite(bp); j = 0
    while j < len(bp):
        if miss[j]:
            k = j
            while k < len(bp) and miss[k]: k += 1
            if j > 0 and k < len(bp):
                ax.plot([x[j - 1], x[k]], [bp[j - 1], bp[k]], color=GREY, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
            j = k
        else: j += 1
    ax.set_ylabel("Back-pressure\n(kPa)")
    ax2 = axs[1]; ax2.plot(x, sc[idx], color=BLUE, lw=1.0, marker="s", ms=2.6)
    ax2.axhline(thr, color=GREY, lw=0.8, ls="--")
    ax2.set_ylabel("Hybrid score\n(percentile)"); ax2.set_ylim(min(0.9, np.nanmin(sc[idx]) - 0.03), 1.06)
    ax3 = axs[2]; bottom = np.zeros(len(idx))
    sty = {"nozzle": (INK, ""), "humidity": ("#ffffff", "////"), "drying": (LGREY, ""), "membrane": ("#ffffff", "...."), "reagent": (MGREY, "")}
    for g in sim.FEATURE_GROUPS:
        v = G[g].values; fc, hh = sty[g]
        ax3.bar(x, v, bottom=bottom, width=0.72, color=fc, edgecolor=INK if fc == "#ffffff" else "white", lw=0.4, hatch=hh)
        bottom += v
    ax3.set_ylabel("LR cause\ncontribution"); ax3.set_ylim(0, bottom.max() * 1.1)
    ncx = x[sub.nc.values[idx] == 1]
    for a in axs:
        a.axvspan(e["start"] - onset - 0.5, e["start"] + e["dur"] - onset - 0.5, color="#eef3fb", zorder=0)
        a.axvline(0, color=INK, lw=0.8, ls=":"); a.axvline(t_al - onset, color=BLUE, lw=1.0); a.axvline(tsq - onset, color=INK, lw=0.9, ls="-.")
    y0 = axs[0].get_ylim()[1]
    axs[0].plot(ncx, np.full(len(ncx), y0), "v", color=INK, ms=3.6, clip_on=False, zorder=5)
    ax3.set_xlabel("Hours relative to the first nonconforming sub-lot"); fig.align_ylabels(axs)
    for a, lab in zip(axs, "abc"):
        a.text(-0.2, 1.0, f"({lab})", transform=a.transAxes, fontsize=9, weight="bold", va="top")
    # all legends sit below the figure: event markers first, then the cause groups of panel (c)
    ev = [Line2D([], [], marker="v", ls="", color=INK, ms=4, label="QC nonconforming"),
          Line2D([], [], color=BLUE, lw=1.0, label="first alarm"), Line2D([], [], color=INK, lw=0.8, ls=":", label="first NC"),
          Line2D([], [], color=INK, lw=0.9, ls="-.", label="QC result"), Line2D([], [], color=GREY, lw=0.8, ls="--", label="2% FAR threshold"),
          Patch(fc="#eef3fb", ec="none", label="injected episode"),
          Line2D([], [], color=GREY, lw=0.8, ls=(0, (2, 1.5)), label="dropout (no reading)")]
    cg = [Patch(fc=sty[g][0], ec=INK if sty[g][0] == "#ffffff" else "none", hatch=sty[g][1], lw=0.4, label=CAUSE[g]) for g in sim.FEATURE_GROUPS]
    l1 = fig.legend(handles=ev, loc="upper center", bbox_to_anchor=(0.53, -0.03), ncol=4, frameon=False, handlelength=1.2, handletextpad=0.4,
                    columnspacing=0.7, fontsize=FS, borderaxespad=0.2, labelspacing=0.3)
    fig.legend(handles=cg, loc="upper center", bbox_to_anchor=(0.53, -0.13), ncol=5, frameon=False, handlelength=1.3,
               handletextpad=0.5, columnspacing=1.0, fontsize=FS)
    fig.add_artist(l1)
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
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.8), gridspec_kw=dict(wspace=0.08))
    labs = [CAUSE[c] for c in sim.FAULT_TYPES] + ["All"]
    for k, (ax, Mv, ttl) in enumerate(zip(axs, [seen, unseen], ["(a) Seen in training", "(b) Withheld from training"])):
        ax.imshow(Mv, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(Mv.shape[0]):
            for j in range(Mv.shape[1]):
                v = Mv[i, j]; ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=FS, color="white" if v > 0.6 else INK,
                                      weight="bold" if j == Mv.shape[1] - 1 else None)
        ax.axvline(4.5, color="white", lw=1.5)
        ax.set_xticks(range(6)); ax.set_xticklabels(labs, rotation=40, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[m] for m in rows] if k == 0 else [])
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
        ax.set_title(ttl, fontsize=8.5, loc="left")
    save(fig, "fig4_fault_types")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["arch", "results", "story", "types"]
    if "arch" in which: fig_architecture()
    if "results" in which: fig_results()
    if "types" in which: fig_types()
    if "story" in which: print(fig_story(10))
