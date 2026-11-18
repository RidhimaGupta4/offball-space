"""Build off-ball run features with nested, match-level cross-fitting.

For every SkillCorner-tagged off-ball run (start frame t0, end frame t1) we compute, in a frame where the
runner's team attacks towards +x:
  PC_full(t)  - attacking-team pitch control on a 50x32 grid (scaled to that match's pitch size)
  PC_noR(t)   - the same with the runner removed from his team
and summarise each grid as the share of 'pitch value' controlled:  S = sum(V * PC) / sum(V),
where V(z) = P(possessing team shoots within 10 s | ball in cell z), estimated from tracking frames.

Features per run:
  team_space_t0/t1    = S(PC_full)           valuable space the team controls
  mates_space_t0/t1   = S(PC_noR)            ... without the runner (teammates' space)
  teammate_space_gain = mates_space_t1 - mates_space_t0
  runner_space_t0/t1  = team_space - mates_space   (runner's own marginal space)

Leakage control (no feature ever uses its own match's outcomes, no test match ever informs a fold):
  * fold f (held-out test match f):
      - pitch-control parameters theta_{-f}: fitted on pass contests from all matches except f
      - test match f features use the value surface V_{-f}
      - every training match m uses V_{-f,-m} (excludes the test match AND m itself), so training
        features are as 'out-of-sample' as test features (no in-sample target encoding)
  * 'all' (descriptive analyses only, never scored): theta from all matches, and each match m uses V_{-m}.
Velocities are causal (backward differences), see core.velocities.

Outputs: data/runs_base.parquet, data/runs_feat_{fold}.parquet, data/run_grids/{match}.npz (all-data theta),
         data/value_samples.parquet, data/value_surface_all.npz (for the figure only)
"""
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from core import (PITCH_L, PITCH_W, ROOT, attack_sign, frame_index, load_events, load_meta, load_tracking,
                  make_grid, match_ids, pitch_control)
from fit_pitch_control import fit_pc, load as load_passes

_, SHAPE = make_grid()
HORIZONS = {"shot5_after": 50, "shot10_after": 100, "shot15_after": 150}  # frames
VALUE_HORIZON = 100
_cache = {}


def _trk(mid):
    if mid not in _cache:
        _cache.clear()
        _cache[mid] = load_tracking(mid)
    return _cache[mid]


# ----------------------------------------------------------------------------- value surface
def value_samples(mid, step=5):
    """Ball locations (normalised: possessing team attacks +x, pitch rescaled to 105x68) and
    'possessing team shoots within 10 s' labels."""
    trk, ev, meta = _trk(mid), load_events(mid), load_meta(mid)
    L, W = meta["pitch_length"], meta["pitch_width"]
    home_id = meta["home_team"]["id"]
    pp = ev[ev.event_type == "player_possession"]
    shots = pp[pp.end_type == "shot"]
    shot_frames = {0: np.sort(shots[shots.team_id == home_id].frame_end.values),
                   1: np.sort(shots[shots.team_id != home_id].frame_end.values)}
    idx = np.arange(0, len(trk["frame"]), step)
    g, b, per, fr = trk["poss_group"][idx], trk["ball"][idx], trk["period"][idx], trk["frame"][idx]
    ok = (g >= 0) & np.isfinite(b).all(1)
    g, b, per, fr = g[ok], b[ok], per[ok], fr[ok]
    sign = np.array([attack_sign(meta, gi == 0, int(p)) for gi, p in zip(g, per, strict=True)])
    x = b[:, 0] * sign * PITCH_L / L
    y = b[:, 1] * sign * PITCH_W / W
    lab = np.zeros(len(g), dtype=int)
    for team in (0, 1):
        sf, sel = shot_frames[team], g == team
        if len(sf) == 0 or not sel.any():
            continue
        nxt = np.searchsorted(sf, fr[sel], side="right")
        has = nxt < len(sf)
        team_lab = np.zeros(sel.sum(), dtype=int)
        team_lab[has] = (sf[nxt[has]] - fr[sel][has] <= VALUE_HORIZON).astype(int)
        lab[sel] = team_lab
    return pd.DataFrame(dict(match_id=mid, x=x, y=y, shot10=lab))


def fit_value_surface(samples, sigma=2.0, prior_k=20.0):
    """Smoothed P(shot within 10 s | ball cell) on the normalised 105x68 pitch, beta prior at the base rate."""
    ny, nx = SHAPE
    ix = np.clip(((samples.x + PITCH_L / 2) / PITCH_L * nx).astype(int), 0, nx - 1)
    iy = np.clip(((samples.y + PITCH_W / 2) / PITCH_W * ny).astype(int), 0, ny - 1)
    n, s = np.zeros(SHAPE), np.zeros(SHAPE)
    np.add.at(n, (iy, ix), 1)
    np.add.at(s, (iy, ix), samples.shot10.values)
    n, s = gaussian_filter(n, sigma), gaussian_filter(s, sigma)
    base = samples.shot10.mean()
    return ((s + prior_k * base) / (n + prior_k)).ravel()


# ----------------------------------------------------------------------------- runs
def base_runs(mid):
    ev, meta = load_events(mid), load_meta(mid)
    home_id = meta["home_team"]["id"]
    pp = ev[ev.event_type == "player_possession"]
    shots = pp[pp.end_type == "shot"][["frame_end", "team_id"]]
    rows = []
    for _, r in ev[ev.event_type == "off_ball_run"].iterrows():
        t0, t1 = int(r.frame_start), int(r.frame_end)
        ts = shots[shots.team_id == r.team_id].frame_end.values
        row = dict(
            match_id=mid, event_id=r.event_id, player_id=int(r.player_id), player_name=r.player_name,
            player_position=r.player_position, team_id=int(r.team_id), team=r.team_shortname,
            is_home=int(r.team_id) == home_id, period=int(r.period), t0=t0, t1=t1,
            minute=float(r.minute_start), subtype=r.event_subtype, game_state=r.game_state,
            shot_during=bool(((ts >= t0) & (ts <= t1)).any()),
            # SkillCorner post-run outcomes: kept ONLY for reference, never used as model inputs
            sc_targeted=r.targeted, sc_received=r.received, sc_lead_to_shot=r.lead_to_shot,
        )
        for k, h in HORIZONS.items():
            row[k] = int(((ts > t1) & (ts <= t1 + h)).any())
        rows.append(row)
    return pd.DataFrame(rows)


CTX_COLS = ["ball_x0", "ball_absy0", "ball_x1", "ball_absy1", "runner_x0", "runner_absy0",
            "runner_x1", "runner_absy1", "duration_s", "det_frac"]


def run_grids(mid, runs, params):
    """Per-run pitch-control grids [n,4,cells] (full_t0, noR_t0, full_t1, noR_t1) + context (metres,
    normalised so the runner's team attacks +x). Runs whose runner or <20 players are missing -> NaN."""
    trk, meta = _trk(mid), load_meta(mid)
    grid, _ = make_grid(length=meta["pitch_length"], width=meta["pitch_width"])
    col = {p: i for i, p in enumerate(trk["pid"])}
    out = np.full((len(runs), 4, len(grid)), np.nan, dtype=np.float32)
    ctx = np.full((len(runs), len(CTX_COLS)), np.nan)
    for k, r in enumerate(runs.itertuples()):
        i0, i1 = frame_index(trk, [r.t0, r.t1])
        j = col.get(r.player_id)
        if i0 < 0 or i1 < 0 or j is None:
            continue
        side = 0 if r.is_home else 1
        s = attack_sign(meta, r.is_home, r.period)
        ok, g = True, np.empty((4, len(grid)), dtype=np.float32)
        for slot, i in enumerate((i0, i1)):
            pos, vel = trk["pos"][i] * s, trk["vel"][i] * s
            present = np.isfinite(pos[:, 0])
            if present.sum() < 20 or not present[j]:
                ok = False
                break
            att = present & (trk["team"] == side)
            dfd = present & (trk["team"] != side)
            g[2 * slot] = pitch_control(pos[att], vel[att], pos[dfd], vel[dfd], grid, params)
            att[j] = False
            g[2 * slot + 1] = pitch_control(pos[att], vel[att], pos[dfd], vel[dfd], grid, params)
        if not ok:
            continue
        out[k] = g
        b0, b1 = trk["ball"][i0] * s, trk["ball"][i1] * s
        p0, p1 = trk["pos"][i0, j] * s, trk["pos"][i1, j] * s
        det = np.r_[trk["det"][i0][np.isfinite(trk["pos"][i0, :, 0])],
                    trk["det"][i1][np.isfinite(trk["pos"][i1, :, 0])]].mean()
        ctx[k] = [b0[0], abs(b0[1]), b1[0], abs(b1[1]), p0[0], abs(p0[1]), p1[0], abs(p1[1]),
                  (r.t1 - r.t0) / 10, det]
    return out, ctx


def summarise(grids, V):
    w = V / V.sum()
    S = np.einsum("nkc,c->nk", grids.astype(np.float64), w)
    return pd.DataFrame(dict(
        team_space_t0=S[:, 0], mates_space_t0=S[:, 1], team_space_t1=S[:, 2], mates_space_t1=S[:, 3],
        teammate_space_gain=S[:, 3] - S[:, 1], runner_space_t1=S[:, 2] - S[:, 3],
        runner_space_t0=S[:, 0] - S[:, 1],
    ))


# Sensitivity: ALPHA0=1 rebuilds every fold's features with the receiver-advantage term set to 0
# (alpha is estimated on passes, where the receiver knows where the ball is going; for open 'space'
# with no pass in flight, a neutral alpha is arguably the more appropriate choice).
ALPHA0 = os.environ.get("ALPHA0") == "1"
SUFFIX = "_alpha0" if ALPHA0 else ""

if __name__ == "__main__":
    mids = match_ids()
    (ROOT / "data" / "run_grids").mkdir(exist_ok=True)
    base = pd.concat([base_runs(m) for m in mids], ignore_index=True)
    vs = pd.concat([value_samples(m) for m in mids], ignore_index=True)
    vs.to_parquet(ROOT / "data" / "value_samples.parquet")
    if not ALPHA0:
        np.savez(ROOT / "data" / "value_surface_all.npz", V=fit_value_surface(vs), shape=SHAPE)
    print("runs", len(base), "value samples", len(vs), "shot10 rate", round(vs.shot10.mean(), 4), flush=True)

    pdata = load_passes()
    folds = mids + ["all"]
    theta = {f: fit_pc(pdata, np.where(pdata["mid"] != f)[0] if f != "all" else np.arange(len(pdata["y"])))[0]
             for f in folds}
    if ALPHA0:
        theta = {f: {**p, "alpha": 0.0} for f, p in theta.items()}

    def V_excl(excl):
        return fit_value_surface(vs[~vs.match_id.isin(excl)])

    V_cache = {}

    def V_for(f, m):
        excl = frozenset({m} if f == "all" else ({f} if m == f else {f, m}))
        if excl not in V_cache:
            V_cache[excl] = V_excl(excl)
        return V_cache[excl]

    feats = {f: [] for f in folds}
    ctx_all = []
    for m in mids:
        rm = base[base.match_id == m]
        for f in folds:
            g, ctx = run_grids(m, rm, theta[f])
            if f == "all" and not ALPHA0:
                ctx_all.append(pd.DataFrame(ctx, columns=CTX_COLS, index=rm.index))
                np.savez_compressed(ROOT / "data" / "run_grids" / f"{m}.npz", grids=g.astype(np.float16))
            feats[f].append(summarise(g, V_for(f, m)).set_index(rm.index))
        print("match", m, "done", flush=True)
    if not ALPHA0:
        base = base.join(pd.concat(ctx_all))
        base.to_parquet(ROOT / "data" / "runs_base.parquet")
    for f, lst in feats.items():
        pd.concat(lst).to_parquet(ROOT / "data" / f"runs_feat{SUFFIX}_{f}.parquet")
    pd.DataFrame([dict(fold=str(f), **p) for f, p in theta.items()]).to_csv(
        ROOT / "results" / f"run_feature_fold_params{SUFFIX}.csv", index=False)
    print("saved")
