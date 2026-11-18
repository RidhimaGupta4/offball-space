"""Core utilities: loading, direction normalisation, velocities and a fittable pitch-control model.

Pitch-control model (a smooth, fittable variant of Spearman 2018 "Beyond Expected Goals"):
  time-to-intercept for player i to reach location z:
      tau_i(z) = r + || p_i + v_i * r - z || / vmax
  (player keeps current velocity during reaction time r, then runs straight at vmax)
  attacking-team control of z:
      PC(z) = sigmoid( alpha + logsumexp_{i in att}(-tau_i/beta) - logsumexp_{j in def}(-tau_j/beta) )
  r, vmax, beta, alpha are fitted by maximum likelihood on held-out-safe pass outcomes (see fit_pitch_control.py).
  alpha > 0 captures the receiver's informational advantage (he knows where the pass is going).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

FPS = 10
PITCH_L, PITCH_W = 105.0, 68.0
ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SKILLCORNER_DATA", ROOT / "data_raw" / "opendata" / "data"))


# ----------------------------------------------------------------------------- loading
def load_tracking(mid: int | str):
    z = np.load(ROOT / "data" / "tracking" / f"{mid}.npz")
    d = {k: z[k] for k in z.files}
    d["vel"] = velocities(d["pos"], d["period"])
    return d


def load_events(mid: int | str, raw: Path = RAW) -> pd.DataFrame:
    return pd.read_csv(raw / "matches" / str(mid) / f"{mid}_dynamic_events.csv", low_memory=False)


def load_meta(mid: int | str, raw: Path = RAW) -> dict:
    return json.loads((raw / "matches" / str(mid) / f"{mid}_match.json").read_text())


def match_ids(raw: Path = RAW) -> list[int]:
    return sorted(int(p.name) for p in (raw / "matches").iterdir())


def frame_index(trk, frames):
    """Map SkillCorner frame numbers to row indices (-1 if absent)."""
    frames = np.asarray(frames, dtype=np.int64)
    idx = np.searchsorted(trk["frame"], frames)
    idx = np.clip(idx, 0, len(trk["frame"]) - 1)
    return np.where(trk["frame"][idx] == frames, idx, -1)


# ----------------------------------------------------------------------------- kinematics
def velocities(pos, period, lag=5, vmax_clip=12.0):
    """CAUSAL velocity: backward difference over `lag` frames (0.5 s at 10 fps).

    Only positions at or before frame t are used, so a velocity "at the moment of a pass" or "at the end of
    a run" never contains information from later frames. Never computed across a period boundary; frames
    without enough history get zero velocity.
    """
    vel = np.full_like(pos, np.nan)
    v = (pos[lag:] - pos[:-lag]) / (lag / FPS)
    v[period[lag:] != period[:-lag]] = np.nan
    vel[lag:] = v
    sp = np.linalg.norm(vel, axis=-1, keepdims=True)
    scale = np.where(sp > vmax_clip, vmax_clip / np.maximum(sp, 1e-9), 1.0)
    return np.nan_to_num(vel * scale, nan=0.0)


def attack_sign(meta: dict, team_is_home: bool, period: int) -> float:
    """+1 if the team attacks towards +x in this period, else -1."""
    home_side = meta["home_team_side"][period - 1]
    home_ltr = home_side == "left_to_right"
    ltr = home_ltr if team_is_home else not home_ltr
    return 1.0 if ltr else -1.0


# ----------------------------------------------------------------------------- pitch control
DEFAULT_PARAMS = dict(r=0.7, vmax=5.0, beta=0.45, alpha=0.0)


def make_grid(nx=50, ny=32, length=PITCH_L, width=PITCH_W):
    """Cell centres of an nx-by-ny grid covering a pitch of the given size (metres, centred at 0).

    Cell (i, j) always refers to the same *relative* pitch location, so a value surface estimated on
    normalised coordinates can be applied to pitches of different sizes (104-106 m in this data)."""
    xs = (np.arange(nx) + 0.5) / nx * length - length / 2
    ys = (np.arange(ny) + 0.5) / ny * width - width / 2
    gx, gy = np.meshgrid(xs, ys, indexing="xy")  # [ny, nx]
    return np.stack([gx.ravel(), gy.ravel()], axis=1), (ny, nx)


def tti(pos, vel, targets, r, vmax):
    """pos, vel: [n,2]; targets: [m,2] -> [n,m] time to intercept."""
    p_r = pos + vel * r
    d = np.linalg.norm(p_r[:, None, :] - targets[None, :, :], axis=-1)
    return r + d / vmax


def pitch_control(att_pos, att_vel, def_pos, def_vel, targets, params):
    r, vmax, beta, alpha = params["r"], params["vmax"], params["beta"], params["alpha"]
    ta = tti(att_pos, att_vel, targets, r, vmax)
    td = tti(def_pos, def_vel, targets, r, vmax)
    logit = alpha + logsumexp(-ta / beta, axis=0) - logsumexp(-td / beta, axis=0)
    return 1.0 / (1.0 + np.exp(-logit))
