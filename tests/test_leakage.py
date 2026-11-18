"""Guards against the leakage and timing errors that most often invalidate tracking-data analyses.
Run: pytest -q tests   (requires the pipeline outputs in data/)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_runs import HORIZONS, fit_value_surface, summarise  # noqa: E402
from core import ROOT, attack_sign, frame_index, load_events, load_meta, load_tracking, match_ids, velocities  # noqa: E402
from validate_runs import eligible, load, model_specs  # noqa: E402

HINDSIGHT = {"sc_targeted", "sc_received", "sc_lead_to_shot", "lead_to_shot", "lead_to_goal", "targeted",
             "received", "received_in_space", "team_in_possession_phase_type", "n_simultaneous_runs",
             "shot5_after", "shot10_after", "shot15_after", "shot_during", "det_frac"}


def test_no_hindsight_features_in_any_model():
    for use_subtype in (True, False):
        for num, cat in model_specs(use_subtype).values():
            assert not HINDSIGHT & set(num + cat)


def test_velocity_is_causal():
    """Changing positions after frame t must not change the velocity at frame t."""
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(size=(60, 4, 2)), 0).astype(np.float32)
    per = np.ones(60, dtype=np.int8)
    fut = pos.copy()
    fut[30:] += 50.0
    assert np.allclose(velocities(pos, per)[:30], velocities(fut, per)[:30])


def test_velocity_never_crosses_period_boundary():
    pos = np.zeros((40, 1, 2), dtype=np.float32)
    pos[20:, 0, 0] = 100.0
    per = np.r_[np.ones(20), 2 * np.ones(20)].astype(np.int8)
    assert np.abs(velocities(pos, per)).max() == 0.0


def test_direction_normalisation_puts_shots_in_attacking_half():
    xs = []
    for mid in match_ids()[:5]:
        trk, ev, meta = load_tracking(mid), load_events(mid), load_meta(mid)
        sh = ev[(ev.event_type == "player_possession") & (ev.end_type == "shot")]
        for (_, s), i in zip(sh.iterrows(), frame_index(trk, sh.frame_end.astype(int).values), strict=True):
            if i >= 0 and np.isfinite(trk["ball"][i]).all():
                xs.append(trk["ball"][i, 0] * attack_sign(meta, s.team_id == meta["home_team"]["id"], int(s.period)))
    assert len(xs) > 50 and np.mean(np.array(xs) > 0) > 0.99


def test_fold_features_are_test_blind_and_training_features_exclude_own_match():
    """Recompute features for a training match m inside fold f from the saved grids: they must match
    V_{-f,-m}, not V_{-f} (no in-sample target encoding). Grids saved on disk use all-data parameters, so
    this checks the value-surface nesting with a tolerance reflecting the (pass-fitted) parameter differences."""
    base = pd.read_parquet(ROOT / "data" / "runs_base.parquet")
    vs = pd.read_parquet(ROOT / "data" / "value_samples.parquet")
    mids = match_ids()
    f, m = mids[0], mids[1]
    feat_f = pd.read_parquet(ROOT / "data" / f"runs_feat_{f}.parquet").loc[base.match_id == m]
    g = np.load(ROOT / "data" / "run_grids" / f"{m}.npz")["grids"]
    nested = summarise(g, fit_value_surface(vs[~vs.match_id.isin({f, m})])).set_index(feat_f.index)
    leaky = summarise(g, fit_value_surface(vs[vs.match_id != f])).set_index(feat_f.index)
    ok = feat_f.team_space_t1.notna()
    err_nested = np.abs(feat_f.team_space_t1[ok] - nested.team_space_t1[ok]).mean()
    err_leaky = np.abs(feat_f.team_space_t1[ok] - leaky.team_space_t1[ok]).mean()
    assert err_nested < err_leaky


@pytest.mark.parametrize("mid", [1925299, 2017461])
def test_outcome_window_strictly_after_inputs(mid):
    df = eligible(load("all"))
    df = df[df.match_id == mid]
    ev = load_events(mid)
    shots = ev[(ev.event_type == "player_possession") & (ev.end_type == "shot")]
    for r in df.itertuples():
        ts = shots[shots.team_id == r.team_id].frame_end.values
        assert not ((ts >= r.t0) & (ts <= r.t1)).any()
        for k, h in HORIZONS.items():
            assert getattr(r, k) == int(((ts > r.t1) & (ts <= r.t1 + h)).any())


def test_lomo_folds_are_disjoint():
    import validate_runs
    src = Path(validate_runs.__file__).read_text()
    assert "assert f not in set(tr.match_id)" in src


def test_is_home_stays_boolean_after_loading():
    """Regression test: a string 'False' is truthy and once flipped attacking direction in a figure."""
    df = load("all")
    assert df.is_home.dtype == bool
    assert set(df.venue.unique()) <= {"home", "away"}


def test_explorer_matches_python_model():
    """The interactive explorer's JavaScript model must reproduce the Python numbers (needs node)."""
    import shutil
    import subprocess
    if shutil.which("node") is None or not (ROOT / "explorer" / "data.js").exists():
        pytest.skip("node or explorer/data.js not available")
    r = subprocess.run(["node", str(ROOT / "tests" / "test_explorer_parity.mjs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_pass_frame_is_the_moment_the_ball_is_played():
    """Ball speed must jump right after SkillCorner's possession frame_end (otherwise 'positions at the pass'
    would be measured at the wrong time)."""
    t = pd.read_csv(ROOT / "results" / "pass_timing_check.csv")
    before = t.median_ball_speed_ms.iloc[:4].mean()   # steps -5..-1
    after = t.median_ball_speed_ms.iloc[7:].mean()    # steps +2..+5
    assert after > 2.5 * before


def test_value_surface_beats_base_rate_out_of_sample():
    v = pd.read_csv(ROOT / "results" / "value_surface_validation.csv")
    used = v[v.used_in_analysis].iloc[0]
    assert used.heldout_log_loss < used.base_rate_log_loss and used.heldout_auc > 0.7


def test_sandbox_page_is_self_contained():
    """docs/index.html must load nothing from other sites (no fonts, scripts or images from third parties)."""
    import re
    html = (ROOT / "docs" / "index.html").read_text()
    loads = re.findall(r'<(?:script|link|img|iframe)[^>]+(?:src|href)="(https?://[^"]+)"', html)
    assert loads == [], loads
    assert "url(http" not in html


def test_explorer_edge_cases():
    """Sandbox model stays finite and within [0, 1] under extreme user actions (needs node)."""
    import shutil
    import subprocess
    if shutil.which("node") is None or not (ROOT / "explorer" / "data.js").exists():
        pytest.skip("node or explorer/data.js not available")
    r = subprocess.run(["node", str(ROOT / "tests" / "test_explorer_edge_cases.mjs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
