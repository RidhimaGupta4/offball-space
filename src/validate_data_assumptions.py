"""Checks of two inputs every later result depends on.

1. Pass timing: SkillCorner's possession `frame_end` is taken as the moment a pass is played. If that were
   late or early, 'positions at the pass' would be wrong. Check: median ball speed in the frames around
   frame_end should jump from dribbling speed to pass speed right after frame 0.
2. Value surface: V(z) = P(shot within 10 s | ball cell) is a model too. Each match is scored by a surface
   fitted on the other 19 (log loss, AUC vs the base rate). The smoothing bandwidth used everywhere
   (sigma = 2 cells, fixed in advance) is compared with alternatives as a sensitivity check only; it is
   not re-selected.
Outputs: results/pass_timing_check.csv, results/value_surface_validation.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from build_runs import SHAPE, fit_value_surface
from core import PITCH_L, PITCH_W, ROOT, frame_index, load_events, load_tracking, match_ids

RES = ROOT / "results"


def pass_timing():
    rows = []
    for mid in match_ids():
        trk, ev = load_tracking(mid), load_events(mid)
        pp = ev[(ev.event_type == "player_possession") & (ev.end_type == "pass")]
        b = trk["ball"]
        for i in frame_index(trk, pp.frame_end.astype(int).values):
            if i < 6 or i + 6 >= len(b):
                continue
            seg = b[i - 5:i + 6]
            if np.isfinite(seg).all() and len(set(trk["period"][i - 5:i + 6])) == 1:
                rows.append(np.linalg.norm(np.diff(seg, axis=0), axis=1) * 10)
    S = np.array(rows)
    out = pd.DataFrame(dict(step=[f"{k:+d} to {k + 1:+d}" for k in range(-5, 5)],
                            median_ball_speed_ms=np.median(S, 0), n_passes=len(S)))
    out.to_csv(RES / "pass_timing_check.csv", index=False)
    print(out.round(1).to_string(index=False))


def value_surface():
    vs = pd.read_parquet(ROOT / "data" / "value_samples.parquet")
    ny, nx = SHAPE
    ix = np.clip(((vs.x + PITCH_L / 2) / PITCH_L * nx).astype(int), 0, nx - 1)
    iy = np.clip(((vs.y + PITCH_W / 2) / PITCH_W * ny).astype(int), 0, ny - 1)
    cell = (iy * nx + ix).values
    rows = []
    for sigma in (0.5, 1.0, 2.0, 3.0, 4.0):
        p, base = np.empty(len(vs)), np.empty(len(vs))
        for m in vs.match_id.unique():
            te = (vs.match_id == m).values
            p[te] = fit_value_surface(vs[~te], sigma=sigma)[cell[te]]
            base[te] = vs.shot10.values[~te].mean()
        rows.append(dict(sigma_cells=sigma, used_in_analysis=sigma == 2.0, heldout_log_loss=log_loss(vs.shot10, p),
                         base_rate_log_loss=log_loss(vs.shot10, base), heldout_auc=roc_auc_score(vs.shot10, p)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "value_surface_validation.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    pass_timing()
    value_surface()
