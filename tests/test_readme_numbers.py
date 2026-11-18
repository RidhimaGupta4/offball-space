"""Every headline number in README.md must match the results files (guards against stale or mistyped text)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
README = (ROOT / "README.md").read_text()
R = ROOT / "results"
ANALYSES = ["primary", "high_detection", "no_run_type_labels", "horizon_5s", "horizon_15s", "gradient_boosting",
            "alpha0"]


def present(text):
    assert text in README, f"README does not contain the value computed from results: {text!r}"


def signed(v, scale=1000, nd=2):
    return ("+" if v > 0.5 * 10 ** (-nd) / scale else "−") + f"{abs(v) * scale:.{nd}f}"


def test_pitch_control_numbers():
    pc = pd.read_csv(R / "pc_cv_scores.csv").set_index("model")
    for m in ["const", "pc_default", "dist_logit", "pc_fit"]:
        present(f"{pc.log_loss[m]:.3f}")
        present(f"{pc.brier[m]:.3f}")
    fp = pd.read_csv(R / "pc_fold_params.csv")
    for c in ["r", "vmax", "beta"]:
        present(f"{fp[c].min():.2f}–{fp[c].max():.2f}")
    present(f"α ≈ {fp.alpha.mean():.2f}")
    present(f"{np.exp(fp.alpha.mean()):.1f}×")
    bd = pd.read_csv(R / "pc_by_detection.csv")
    for t in ["highest third", "lowest third"]:
        x = bd[bd.detection == t].set_index("model").log_loss
        present(f"{x['pc_fit']:.3f} vs {x['dist_logit']:.3f}")


def test_model_comparison_table():
    for t in ANALYSES:
        c = pd.read_csv(R / f"runs_cv_comparisons_{t}.csv").set_index("comparison")
        for comp in ["B1_team_space vs B0_context", "B2_runner_vs_mates vs B0_context"]:
            r = c.loc[comp]
            present(f"{signed(r.logloss_gain)} [{signed(r.ci_lo)}, {signed(r.ci_hi)}]")
    prim = pd.read_csv(R / "runs_cv_comparisons_primary.csv").set_index("comparison")
    present(f"{prim.loc['B2_runner_vs_mates vs B0_context'].ci_hi_pct_of_baseline:.2f}% of the baseline log loss")
    worst = max(pd.read_csv(R / f"runs_cv_comparisons_{t}.csv").set_index("comparison")
                .loc["B2_runner_vs_mates vs B0_context"].ci_hi_pct_of_baseline for t in ANALYSES)
    present(f"at most {worst:.1f}%")
    sc = pd.read_csv(R / "runs_cv_scores_primary.csv")
    present(f"primary AUC {sc.auc.iloc[0]:.3f}")


def test_run_type_table():
    rt = pd.read_csv(R / "run_types.csv")

    def g(v):
        return ("+" if v > 0 else "−") + f"{abs(v):.2f}"

    for lvl in ["pulling_wide", "dropping_off", "coming_short", "behind", "overlap", "underlap"]:
        for an in ["adj_full | all runs", "adj_pre | all runs"]:
            r = rt[(rt.analysis == an) & (rt.level == lvl)].iloc[0]
            present(f"**{g(r['mean'])}** [{g(r.lo)}, {g(r.hi)}]")
            assert r.p_holm < 0.05


def test_reliability_glm_and_inputs():
    rel = pd.read_csv(R / "player_reliability.csv").set_index("adjustment")
    present(f"r = {rel.r_half['adj_full']:.2f} (95% CI {rel.r_lo['adj_full']:.2f} to {rel.r_hi['adj_full']:.2f})")
    present(f"(r = {rel.r_half['adj_pre']:.2f})")
    glm = pd.read_csv(R / "runs_glm_effects.csv").set_index("term").loc["teammate_space_gain_z"]
    present(f"{glm.odds_ratio_per_sd:.2f} per SD (95% CI {glm.or_lo:.2f} to {glm.or_hi:.2f}; "
            f"Holm-adjusted p = {glm.p_holm:.2f})")
    v = pd.read_csv(R / "value_surface_validation.csv")
    u = v[v.used_in_analysis].iloc[0]
    present(f"log loss {u.heldout_log_loss:.3f} vs {u.base_rate_log_loss:.3f}, AUC {u.heldout_auc:.2f}")
    pt = pd.read_csv(R / "pass_timing_check.csv")
    present(f"over {pt.n_passes.iloc[0]:,} passes")


def test_run_type_reversal_claims():
    rev = pd.read_csv(R / "run_type_reversals.csv").set_index("specification")
    lo, hi = rev.n_significant_reversals.min(), rev.n_significant_reversals.max()
    present(f"{lo} to {hi} of ten run types")
    present(f"ranges from {lo} to {hi}")
    present(f"from {rev.rank_correlation.max():.2f} to {rev.rank_correlation.min():.2f}".replace("-", "−"))
    present(f"correlate at {rev.rank_correlation['all runs']:.2f}".replace("-", "−"))
    always = set.intersection(*[set(s.split(", ")) for s in rev.reversed_types])
    assert always == {"coming_short", "pulling_wide"}, always
    assert len(rev) == 5
    present("Coming short and pulling wide reverse in all five")


def test_positive_control_claims():
    pc = pd.read_csv(R / "positive_control.csv").iloc[0]
    present(f"improves held-out log loss by {pc.logloss_gain:.4f} (95% CI {pc.ci_lo:.4f} to {pc.ci_hi:.4f}), "
            f"in {pc.matches_improved} of {pc.n_matches} matches")
    present(f"about {pc.gain_relative_to_space_bound:.0f} times the largest gain")
    assert pc.ci_lo > 0
