"""Export real tracking moments + the fitted model for the interactive explorer (explorer/).

Scenario rule (fixed in advance, not chosen by outcome): for each of the 10 SkillCorner run types, the
eligible run with the highest share of players detected on camera (ties -> longer run), requiring >= 2 s
duration and all 22 players present; both its start frame and end frame are exported.
Coordinates are in metres on that match's pitch, rotated so the runner's team attacks towards +x.

For each scenario the script also stores reference values computed in Python with the same model
(all-data parameters, all-data value surface), used by tests/test_explorer_parity.mjs to check that the
JavaScript implementation reproduces the Python numbers.
Output: explorer/data.js  (window.EXPLORER_DATA = {...})
"""
import json

import numpy as np
import pandas as pd

from build_runs import summarise
from core import ROOT, attack_sign, frame_index, load_meta, load_tracking, make_grid, pitch_control
from fit_pitch_control import fit_pc, load as load_passes


def frame_payload(trk, meta, pid_names, i, side, s, runner_col):
    pos, vel = trk["pos"][i] * s, trk["vel"][i] * s
    players = []
    for c in np.where(np.isfinite(pos[:, 0]))[0]:
        info = pid_names[int(trk["pid"][c])]
        players.append(dict(id=int(trk["pid"][c]), name=info["name"], num=info["num"], pos=info["pos"],
                            team="att" if trk["team"][c] == side else "def",
                            x=round(float(pos[c, 0]), 2), y=round(float(pos[c, 1]), 2),
                            vx=round(float(vel[c, 0]), 2), vy=round(float(vel[c, 1]), 2),
                            runner=bool(c == runner_col), detected=bool(trk["det"][i, c])))
    b = trk["ball"][i] * s
    ball = [round(float(b[0]), 2), round(float(b[1]), 2)] if np.isfinite(b).all() else None
    return players, ball


def reference_values(players, L, W, params, V):
    """Python-side team share and runner marginal share, computed exactly as in the analysis."""
    grid, _ = make_grid(length=L, width=W)
    P = np.array([[p["x"], p["y"]] for p in players])
    U = np.array([[p["vx"], p["vy"]] for p in players])
    att = np.array([p["team"] == "att" for p in players])
    run = np.array([p["runner"] for p in players])
    full = pitch_control(P[att], U[att], P[~att], U[~att], grid, params)
    nor = pitch_control(P[att & ~run], U[att & ~run], P[~att], U[~att], grid, params)
    g = np.stack([full, nor, full, nor])[None].astype(np.float32)
    S = summarise(g, V).iloc[0]
    return dict(team_share=float(S.team_space_t0), mates_share=float(S.mates_space_t0),
                runner_share=float(S.runner_space_t0), pc_at_centre=float(
                    pitch_control(P[att], U[att], P[~att], U[~att], np.array([[0.0, 0.0]]), params)[0]))


def main():
    df = pd.read_parquet(ROOT / "data" / "runs_analysis.parquet")
    pdata = load_passes()
    params, _ = fit_pc(pdata, np.arange(len(pdata["y"])))
    V = np.load(ROOT / "data" / "value_surface_all.npz")["V"]
    cand = df[df.duration_s >= 2.0].sort_values(["det_frac", "duration_s"], ascending=[False, False])
    scenarios = []
    for st in sorted(df.subtype.unique()):
        for _, r in cand[cand.subtype == st].iterrows():
            mid = int(r.match_id)
            trk, meta = load_tracking(mid), load_meta(mid)
            i0, i1 = frame_index(trk, [int(r.t0), int(r.t1)])
            if (np.isfinite(trk["pos"][i0, :, 0]).sum() != 22) or (np.isfinite(trk["pos"][i1, :, 0]).sum() != 22):
                continue
            names = {p["id"]: dict(name=p["short_name"], num=p.get("number"),
                                   pos=p["player_role"]["acronym"]) for p in meta["players"]}
            col = {p: k for k, p in enumerate(trk["pid"])}
            side = 0 if bool(r.is_home) else 1
            s = attack_sign(meta, bool(r.is_home), int(r.period))
            L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
            opp = meta["away_team"]["short_name"] if r.is_home else meta["home_team"]["short_name"]
            for tag, i in (("start", i0), ("end", i1)):
                players, ball = frame_payload(trk, meta, names, i, side, s, col[int(r.player_id)])
                scenarios.append(dict(
                    id=f"{mid}-{r.event_id}-{tag}", run_type=st, moment=tag, match_id=mid,
                    attacking_team=r.team, opponent=opp, minute=int(r.minute), runner=r.player_name,
                    det_frac=round(float(trk["det"][i][np.isfinite(trk["pos"][i, :, 0])].mean()), 3),
                    duration_s=float(r.duration_s), pitch=[L, W], players=players, ball=ball,
                    reference=reference_values(players, L, W, params, V)))
            break
    out = dict(params={k: float(v) for k, v in params.items()}, grid=[50, 32], V=[float(v) for v in V],
               v_max=float(V.max()), n_passes=int(len(pdata["y"])), scenarios=scenarios)
    (ROOT / "explorer").mkdir(exist_ok=True)
    (ROOT / "explorer" / "data.js").write_text("window.EXPLORER_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(len(scenarios), "scenarios;", {k: round(v, 3) for k, v in out["params"].items()})


if __name__ == "__main__":
    main()
