"""Parse SkillCorner open tracking data (JSONL, 10 fps) into compact per-match arrays.

Output per match: data/tracking/{match_id}.npz with
  frame [F], period [F], ball [F,2], poss_group [F] (0 home, 1 away, -1 none),
  pid [P], team [P] (0 home, 1 away), pos [F,P,2] (NaN where absent), det [F,P] (detected on camera)

Usage: python src/ingest.py <skillcorner_data_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import orjson


def ingest_match(match_dir: Path, out_dir: Path) -> None:
    mid = match_dir.name
    meta = json.loads((match_dir / f"{mid}_match.json").read_text())
    home_id = meta["home_team"]["id"]
    players = meta["players"]
    pid = np.array([p["id"] for p in players], dtype=np.int64)
    team = np.array([0 if p["team_id"] == home_id else 1 for p in players], dtype=np.int8)
    col = {p: i for i, p in enumerate(pid)}

    rows = []
    with open(match_dir / f"{mid}_tracking_extrapolated.jsonl", "rb") as f:
        for line in f:
            fr = orjson.loads(line)
            if fr["period"] is None:
                continue
            rows.append(fr)

    F, P = len(rows), len(pid)
    frame = np.empty(F, dtype=np.int64)
    period = np.empty(F, dtype=np.int8)
    ball = np.full((F, 2), np.nan, dtype=np.float32)
    poss = np.full(F, -1, dtype=np.int8)
    pos = np.full((F, P, 2), np.nan, dtype=np.float32)
    det = np.zeros((F, P), dtype=bool)
    for i, fr in enumerate(rows):
        frame[i] = fr["frame"]
        period[i] = fr["period"]
        b = fr["ball_data"]
        if b and b["x"] is not None:
            ball[i] = (b["x"], b["y"])
        g = fr["possession"]["group"]
        poss[i] = 0 if g == "home team" else (1 if g == "away team" else -1)
        for pdx in fr["player_data"]:
            j = col.get(pdx["player_id"])
            if j is None or pdx["x"] is None:
                continue
            pos[i, j] = (pdx["x"], pdx["y"])
            det[i, j] = bool(pdx["is_detected"])

    np.savez_compressed(
        out_dir / f"{mid}.npz", frame=frame, period=period, ball=ball, poss_group=poss,
        pid=pid, team=team, pos=pos, det=det,
        pitch=np.array([meta["pitch_length"], meta["pitch_width"]], dtype=np.float32),
    )
    print(mid, F, "frames", P, "players", "possession groups seen:", np.unique(poss))


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted((src / "matches").iterdir()):
        ingest_match(d, out)  # always rebuild: never reuse stale outputs
