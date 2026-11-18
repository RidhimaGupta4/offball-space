"""Build the pass dataset (passes with a recorded receiver or interceptor) used to fit and validate the pitch-control model.

For each completed or failed pass we take the player positions/velocities at the moment the pass is
played, and the location where the next player possession starts (reception or interception).
Label = 1 if the passing team won the ball there.

Only next possessions that start with 'pass_reception' or 'pass_interception' within 5 s are kept, so
the label is a genuine contest outcome (balls out of play, fouls, long loose-ball scrambles are excluded).
The passer is removed from the attacking set (he cannot receive his own pass); passes whose passer is
missing from the match metadata are skipped. Velocities are causal (see core.velocities).

Note: using the realised reception location (as in Spearman 2018) makes absolute accuracy optimistic;
we only use it to compare models on held-out matches, where all models see the same information.
"""
import pickle

import numpy as np

from core import ROOT, frame_index, load_events, load_meta, load_tracking, match_ids

MAX_GAP = 50  # frames = 5 s


def build_match(mid):
    trk = load_tracking(mid)
    home_id = load_meta(mid)["home_team"]["id"]
    ev = load_events(mid)
    pp = ev[ev.event_type == "player_possession"].sort_values("frame_start").reset_index(drop=True)
    col = {p: i for i, p in enumerate(trk["pid"])}
    out = []
    starts = pp.frame_start.values
    for _, row in pp.iterrows():
        if row.end_type != "pass" or row.pass_outcome not in ("successful", "unsuccessful"):
            continue
        f_pass = int(row.frame_end)
        j = np.searchsorted(starts, f_pass, side="right")
        if j >= len(pp):
            continue
        nxt = pp.iloc[j]
        if nxt.start_type not in ("pass_reception", "pass_interception"):
            continue
        if nxt.frame_start - f_pass > MAX_GAP or nxt.frame_start <= f_pass:
            continue
        i0, i1 = frame_index(trk, [f_pass, int(nxt.frame_start)])
        if i0 < 0 or i1 < 0:
            continue
        pos0, vel0 = trk["pos"][i0], trk["vel"][i0]
        present = np.isfinite(pos0[:, 0])
        if present.sum() < 20:
            continue
        passer = col.get(int(row.player_id))
        if passer is None:  # passer absent from match metadata -> cannot exclude him from receivers; skip
            continue
        team_home = trk["team"] == (0 if int(row.team_id) == home_id else 1)
        att = present & team_home
        att[passer] = False
        dfd = present & ~team_home
        target = trk["ball"][i1]
        if not np.isfinite(target).all():
            rc = col.get(int(nxt.player_id))
            if rc is None or not np.isfinite(trk["pos"][i1, rc]).all():
                continue
            target = trk["pos"][i1, rc]
        label = int(nxt.team_id == row.team_id)
        det_frac = trk["det"][i0][present].mean()
        out.append(dict(
            match_id=mid, frame=f_pass, label=label, outcome=row.pass_outcome,
            target=target.astype(np.float64), ball=trk["ball"][i0].astype(np.float64),
            att_pos=pos0[att].astype(np.float64), att_vel=vel0[att].astype(np.float64),
            def_pos=pos0[dfd].astype(np.float64), def_vel=vel0[dfd].astype(np.float64),
            det_frac=det_frac,
        ))
    return out


if __name__ == "__main__":
    allp = []
    for mid in match_ids():
        ps = build_match(mid)
        allp += ps
        lab = np.array([p["label"] for p in ps])
        print(mid, len(ps), "passes, won by passing team:", lab.mean().round(3))
    # sanity: label should agree with SkillCorner's pass_outcome almost always
    agree = np.mean([(p["label"] == 1) == (p["outcome"] == "successful") for p in allp])
    print("total", len(allp), "label/outcome agreement", round(agree, 4))
    with open(ROOT / "data" / "passes.pkl", "wb") as f:
        pickle.dump(allp, f)
