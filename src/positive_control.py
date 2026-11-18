"""Positive control: can the validation pipeline detect a real effect when one exists?

The main result is a null (pitch-control space adds no out-of-sample predictive value). A null is only
informative if the same pipeline - leave-one-match-out CV, nested penalty tuning, match-level bootstrap -
reliably detects variables that genuinely matter. Here it is run on real data, comparing:
  C0: B0 context WITHOUT where the ball and runner ended up (start-of-run state, duration, run type, minute,
      game state, venue)
  C1: the full B0 context (adds ball and runner location at the end of the run)
Run after validate_runs.py. End location is known to matter for a shot in the next 10 s, so the pipeline
should report a clear gain.
Output: results/positive_control.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from core import ROOT
from validate_runs import CAT0, NUM0, eligible, fit_predict, load

END = ["ball_x1", "ball_absy1", "runner_x1", "runner_absy1"]


def main():
    mids = sorted(pd.read_parquet(ROOT / "data" / "runs_base.parquet").match_id.unique())
    specs = {"C0_without_end_location": ([c for c in NUM0 if c not in END], CAT0), "C1_full_context": (NUM0, CAT0)}
    preds = []
    for f in mids:
        df = eligible(load(f))
        tr, te = df[df.match_id != f], df[df.match_id == f]
        assert f not in set(tr.match_id)
        out = te[["match_id", "shot10_after"]].rename(columns={"shot10_after": "y"}).copy()
        for name, (num, cat) in specs.items():
            out[name], _ = fit_predict("logit", num, cat, tr, te, "shot10_after")
        preds.append(out)
    p = pd.concat(preds)
    pm = p.groupby("match_id").apply(lambda g: pd.Series(
        {k: log_loss(g.y, g[k], labels=[0, 1]) for k in specs} | {"n": len(g)}), include_groups=False)
    d, w = (pm.C0_without_end_location - pm.C1_full_context).values, pm.n.values
    rng = np.random.default_rng(1)
    bs = np.array([np.average(d[i], weights=w[i]) for i in rng.integers(0, len(d), (5000, len(d)))])
    gain = np.average(d, weights=w)
    # largest predictive gain for space that the primary analysis does not rule out (wider of the two intervals)
    prim = pd.read_csv(ROOT / "results" / "runs_cv_comparisons_primary.csv").set_index("comparison")
    space_bound = max(prim.loc["B2_runner_vs_mates vs B0_context", ["ci_hi", "t_ci_hi"]])
    res = pd.DataFrame([dict(comparison="full context vs without end location", logloss_gain=gain,
                             ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                             matches_improved=int((d > 0).sum()), n_matches=len(d),
                             gain_relative_to_space_bound=gain / space_bound)])
    res.to_csv(ROOT / "results" / "positive_control.csv", index=False)
    print(res.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
