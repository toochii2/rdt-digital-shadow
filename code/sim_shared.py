"""
Semi-synthetic simulator of a lateral-flow RDT card-build operation
(test/control-line dispensing onto nitrocellulose, drying, card lamination).

Unit of analysis: a 1-hour sub-lot. Sensors sampled every 5 min (12 per sub-lot).
Batch (release lot) = one 8-hour shift = 8 sub-lots. Two shifts per production day.

All parameter values are illustrative and chosen to be physically plausible;
they do not describe any specific manufacturer, product or process.
"""
import numpy as np
import pandas as pd

SUBLOTS_PER_DAY = 16
SAMPLES_PER_SUBLOT = 12
BATCH_SIZE = 8

FAULT_TYPES = ["nozzle", "humidity", "drying", "membrane", "reagent"]

# expected episodes per 4000 sub-lots (approx. 250 production days)
BASE_RATES = {"nozzle": 14, "humidity": 12, "drying": 8, "membrane": 5, "reagent": 4}

# QC specification (illustrative)
SPEC = dict(int_min=0.82, width_min=0.75, width_max=1.30, flow_max=150.0)
N_STRIPS = 5           # strips sampled per sub-lot for in-process QC
FAIL_STRIPS = 2        # sub-lot nonconforming if >= 2 strips fail any CQA


def _schedule(n_sub, rng, rates, severity, segments):
    """Place fault episodes so that every segment (train/val/test) contains each type."""
    episodes = []
    occupied = {k: np.zeros(n_sub, bool) for k in FAULT_TYPES}
    for ftype in FAULT_TYPES:
        total = rates[ftype] * n_sub / 4000.0
        for (a, b) in segments:
            frac = (b - a) / n_sub
            n_ep = max(1, rng.poisson(total * frac))
            tries = 0
            placed = 0
            while placed < n_ep and tries < 500:
                tries += 1
                if ftype == "humidity":
                    ramp = rng.integers(1, 3); hold = rng.integers(2, 7); rec = rng.integers(1, 3)
                    dur = ramp + hold + rec
                elif ftype in ("nozzle", "drying"):
                    dur = 12
                else:
                    dur = 16
                start = int(rng.integers(a + 4, max(a + 5, b - dur - 2)))
                if occupied[ftype][max(0, start - 6):start + dur + 6].any():
                    continue
                occupied[ftype][start:start + dur] = True
                ep = dict(type=ftype, start=start, dur=dur)
                if ftype == "humidity":
                    ep.update(ramp=int(ramp), hold=int(hold), rec=int(rec),
                              peak=float(38 + severity * rng.uniform(8, 20)))
                elif ftype == "nozzle":
                    ep.update(ramp=int(rng.integers(4, 11)),
                              fmin=float(1 - severity * rng.uniform(0.15, 0.30)))
                elif ftype == "drying":
                    ep.update(ramp=int(rng.integers(4, 9)),
                              tmin=float(42 - severity * rng.uniform(5, 9)))
                elif ftype == "membrane":
                    ep.update(coa=float(135 + severity * rng.uniform(15, 27)))
                elif ftype == "reagent":
                    ep.update(act=float(1 - severity * rng.uniform(0.10, 0.20)))
                episodes.append(ep)
                placed += 1
    episodes.sort(key=lambda e: e["start"])
    for i, e in enumerate(episodes):
        e["id"] = i
    return episodes


def simulate(seed=0, n_days=250, severity=1.0, noise_mult=1.0, miss_rate=0.02,
             rates=None, split=(0.5, 0.2)):
    rng = np.random.default_rng(seed)
    rates = rates or BASE_RATES
    n_sub = n_days * SUBLOTS_PER_DAY
    n1 = int(split[0] * n_sub); n2 = int((split[0] + split[1]) * n_sub)
    segments = [(0, n1), (n1, n2), (n2, n_sub)]
    episodes = _schedule(n_sub, rng, rates, severity, segments)

    # ---------------- latent per-sub-lot state ----------------
    f_ratio = 1.0 + rng.normal(0, 0.015, n_sub)                # deposition ratio (pump noise)
    rh_base = 35 + 1.5 * np.sin(np.arange(n_sub) * 2 * np.pi / SUBLOTS_PER_DAY) \
        + np.cumsum(rng.normal(0, 0.15, n_sub)) * 0.0         # diurnal pattern
    rh_base = rh_base + rng.normal(0, 0.6, n_sub)
    oven = 42 + rng.normal(0, 0.5, n_sub)
    # line speed set-point changes (benign; dispensing rate scales with speed)
    speed = np.empty(n_sub); s = 50.0; i = 0
    while i < n_sub:
        L = int(rng.integers(10, 30)); speed[i:i + L] = s; i += L
        s = float(rng.choice([40.0, 50.0, 60.0]))

    # membrane and reagent lot genealogy
    mem_lot = np.zeros(n_sub, int); mem_coa = np.zeros(n_sub)
    rea_lot = np.zeros(n_sub, int); rea_act = np.zeros(n_sub)
    forced_mem = {e["start"]: e for e in episodes if e["type"] == "membrane"}
    forced_rea = {e["start"]: e for e in episodes if e["type"] == "reagent"}
    lot = 0; t = 0
    while t < n_sub:
        if t in forced_mem:
            e = forced_mem[t]; L = e["dur"]; val = e["coa"]
        else:
            nxt = [k for k in forced_mem if k > t]
            L = int(rng.integers(32, 57)); val = float(np.clip(rng.normal(135, 5), 122, 148))
            if nxt: L = min(L, min(nxt) - t)
        mem_lot[t:t + L] = lot; mem_coa[t:t + L] = val; lot += 1; t += L
    lot = 0; t = 0
    while t < n_sub:
        if t in forced_rea:
            e = forced_rea[t]; L = e["dur"]; val = e["act"]
        else:
            nxt = [k for k in forced_rea if k > t]
            L = int(rng.integers(90, 160)); val = float(np.clip(rng.normal(1.0, 0.03), 0.93, 1.07))
            if nxt: L = min(L, min(nxt) - t)
        rea_lot[t:t + L] = lot; rea_act[t:t + L] = val; lot += 1; t += L

    cause = np.array([""] * n_sub, dtype=object)
    ep_id = -np.ones(n_sub, int)
    clean_events = set(range(0, n_sub, 2 * SUBLOTS_PER_DAY))   # preventive nozzle cleaning every 2 days
    rh = rh_base.copy()
    for e in episodes:
        a, d = e["start"], e["dur"]
        idx = np.arange(a, min(a + d, n_sub))
        k = np.arange(len(idx))
        if e["type"] == "nozzle":
            prog = np.minimum(1.0, (k + 1) / e["ramp"])
            f_ratio[idx] *= 1 - (1 - e["fmin"]) * prog
            clean_events.add(min(a + d, n_sub - 1))
        elif e["type"] == "humidity":
            prof = np.concatenate([np.linspace(0, 1, e["ramp"] + 1)[1:], np.ones(e["hold"]),
                                   np.linspace(1, 0, e["rec"] + 1)[:-1] + 0.0])[:len(idx)]
            rh[idx] = rh_base[idx] + (e["peak"] - 35) * prof
        elif e["type"] == "drying":
            prog = np.minimum(1.0, (k + 1) / e["ramp"])
            oven[idx] = 42 - (42 - e["tmin"]) * prog + rng.normal(0, 0.4, len(idx))
        for j in idx:
            cause[j] = e["type"] if cause[j] == "" else cause[j] + "+" + e["type"]
            if ep_id[j] < 0: ep_id[j] = e["id"]
    for e in episodes:
        if e["type"] in ("membrane", "reagent"):
            pass  # effects carried by lot genealogy arrays

    hours_since_clean = np.zeros(n_sub); last = 0
    for t in range(n_sub):
        if t in clean_events: last = t
        hours_since_clean[t] = t - last
    # slow nozzle wear between cleanings (benign-to-mild)
    f_ratio *= 1 - 0.0015 * hours_since_clean

    # ---------------- QC outcome (strip-level) ----------------
    spread = 1 + 0.035 * np.maximum(0, rh - 38)
    dry = 1 - 0.06 * np.maximum(0, 40 - oven) - 0.05 * np.maximum(0, oven - 47)
    int_true = f_ratio * rea_act * dry / spread ** 0.8 * (mem_coa / 135) ** 0.35
    wid_true = np.sqrt(np.maximum(f_ratio, 0.05)) * spread
    flw_true = 0.97 * mem_coa
    nm = noise_mult
    I = int_true[:, None] * (1 + rng.normal(0, 0.045 * nm, (n_sub, N_STRIPS)))
    W = wid_true[:, None] + rng.normal(0, 0.05 * nm, (n_sub, N_STRIPS))
    F = flw_true[:, None] + rng.normal(0, 5.0 * nm, (n_sub, N_STRIPS))
    fail = (I < SPEC["int_min"]) | (W < SPEC["width_min"]) | (W > SPEC["width_max"]) | (F > SPEC["flow_max"])
    nc = (fail.sum(1) >= FAIL_STRIPS).astype(int)
    # sporadic nonconformances with no observable process cause (e.g. handling, particulates)
    sporadic = (rng.random(n_sub) < 0.01) & (nc == 0)
    nc[sporadic] = 1

    # ---------------- high-frequency sensors (5-min) ----------------
    S = SAMPLES_PER_SUBLOT
    def hf(level, sd, smooth=True):
        """Causal: samples within sub-lot t move linearly from level[t-1] to level[t] (no look-ahead)."""
        base = np.repeat(level, S).astype(float)
        if smooth:
            prv = np.insert(level[:-1], 0, level[0])
            w = np.tile((np.arange(S) + 1) / S, n_sub)
            base = np.repeat(prv, S) * (1 - w) + np.repeat(level, S) * w
        return base + rng.normal(0, sd * nm, n_sub * S)

    # SHARED-CHANNEL VARIANT: humidity raises back-pressure (viscosity/evaporation at the nozzle) and a hot
    # oven drift also lifts room temperature, so causes no longer map one-to-one onto feature groups
    bp_level = 80 * (speed / 50) * (1 + 2.2 * (1 - f_ratio)) * (1 + 0.025 * np.maximum(0, rh - 38))
    sensors = pd.DataFrame({
        "rh": hf(rh + 0.35 * np.maximum(0, 42 - oven), 0.5),   # oven fault perturbs local humidity reading
        "room_T": hf(22 + 0.6 * np.sin(np.arange(n_sub) * 2 * np.pi / SUBLOTS_PER_DAY), 0.2),
        "bp": hf(bp_level, 1.5),
        "line_speed": np.repeat(speed, S) + rng.normal(0, 0.3, n_sub * S),
        "oven_T": hf(oven, 0.3),
        "oven_air": hf(np.full(n_sub, 1.2), 0.05),
    })
    # benign sensor glitches
    g = rng.random(n_sub * S) < 0.002
    sensors.loc[g, "bp"] += rng.normal(0, 25, g.sum())
    # missing data: random dropouts + block outages
    for c in sensors.columns:
        m = rng.random(n_sub * S) < miss_rate
        n_blocks = rng.poisson(miss_rate * n_sub / 10)
        for st in rng.integers(0, n_sub, n_blocks):
            m[st * S:(st + 1) * S] = True
        sensors.loc[m, c] = np.nan
    sensors["sublot"] = np.repeat(np.arange(n_sub), S)

    # incoming-QC context (noisy measurements of lot properties)
    lot_noise_m = {l: rng.normal(0, 3.0) for l in np.unique(mem_lot)}
    lot_noise_r = {l: rng.normal(0, 0.035) for l in np.unique(rea_lot)}
    mem_coa_obs = mem_coa + np.array([lot_noise_m[l] for l in mem_lot])
    rea_act_obs = rea_act + np.array([lot_noise_r[l] for l in rea_lot])

    sub = pd.DataFrame(dict(
        sublot=np.arange(n_sub), day=np.arange(n_sub) // SUBLOTS_PER_DAY,
        batch=np.arange(n_sub) // BATCH_SIZE,
        nc=nc, cause=np.where(sporadic, "sporadic", cause), episode=ep_id,
        mem_lot=mem_lot, rea_lot=rea_lot,
        mem_coa_obs=mem_coa_obs, rea_act_obs=rea_act_obs,
        hours_since_clean=hours_since_clean,
        # latent truth (never given to models)
        _f=f_ratio, _rh=rh, _oven=oven, _int=int_true,
    ))
    sub["split"] = np.where(sub.sublot < n1, "train", np.where(sub.sublot < n2, "val", "test"))
    return sub, sensors, episodes


# ---------------- digital-shadow integration layer ----------------
def _slope(y):
    x = np.arange(len(y), dtype=float); ok = ~np.isnan(y)
    if ok.sum() < 3: return np.nan
    x, y = x[ok], y[ok]; x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def build_features(sub, sensors):
    """Aggregate HF signals to sub-lot records and join batch/lot genealogy.
    Returns (snapshot feature frame, digital-shadow feature frame, data-quality report)."""
    g = sensors.groupby("sublot")
    means = g[["rh", "room_T", "bp", "line_speed", "oven_T", "oven_air"]].mean()
    snap = means.add_suffix("_mean")
    ds = snap.copy()
    ds["rh_max"] = g["rh"].max()
    ds["rh_exposure_min"] = g["rh"].apply(lambda s: 5.0 * (s > 45).sum())
    ds["bp_std"] = g["bp"].std()
    ds["bp_slope"] = g["bp"].apply(lambda s: _slope(s.values))
    ds["oven_T_min"] = g["oven_T"].min()
    ds["oven_T_slope"] = g["oven_T"].apply(lambda s: _slope(s.values))
    # physics-informed normalisation: back-pressure per unit line speed
    ds["bp_norm"] = ds["bp_mean"] / ds["line_speed_mean"] * 50.0
    # short-horizon trends across consecutive sub-lots (state memory)
    ds["bp_norm_trend3"] = ds["bp_norm"].rolling(3, min_periods=2).apply(lambda v: _slope(v), raw=True)
    ds["oven_T_trend3"] = ds["oven_T_mean"].rolling(3, min_periods=2).apply(lambda v: _slope(v), raw=True)
    # genealogy and maintenance context
    ds["hours_since_clean"] = sub["hours_since_clean"].values
    ds["mem_coa_obs"] = sub["mem_coa_obs"].values
    ds["rea_act_obs"] = sub["rea_act_obs"].values
    # data-quality flags
    completeness = g[["rh", "room_T", "bp", "line_speed", "oven_T", "oven_air"]].apply(lambda d: d.notna().mean().mean())
    dq = dict(sample_completeness=float(sensors.drop(columns="sublot").notna().mean().mean()),
              sublot_feature_completeness=float(ds.notna().mean().mean()),
              batch_linkage=float((sub.batch.values == sub.sublot.values // BATCH_SIZE).mean()),
              lot_linkage=float((sub.mem_lot.notna() & sub.rea_lot.notna()).mean()))
    return snap.reset_index(drop=True), ds.reset_index(drop=True), dq, completeness.values


FEATURE_GROUPS = {
    "nozzle": ["bp_mean", "bp_std", "bp_slope", "bp_norm", "bp_norm_trend3", "hours_since_clean"],
    "humidity": ["rh_mean", "rh_max", "rh_exposure_min"],
    "drying": ["oven_T_mean", "oven_T_min", "oven_T_slope", "oven_T_trend3"],
    "membrane": ["mem_coa_obs"],
    "reagent": ["rea_act_obs"],
}


def group_of(feat):
    for k, v in FEATURE_GROUPS.items():
        if feat in v: return k
    return "other"
