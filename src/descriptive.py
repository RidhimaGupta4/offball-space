"""Descriptive analysis of 'teammates' space gain' by run type, camera coverage, position, and player.

Adjusted gain ("above expected"): residual of teammate_space_gain from an additive cubic-spline ridge model
fitted on the OTHER matches (cross-fitted by match; run type is deliberately NOT in the adjustment model).
Two adjustment sets, because one of them conditions on a mediator:
  adj_pre : only pre-run state (ball and runner location at run start, team space at start, minute)
  adj_full: adj_pre + where the ball and runner ended up + run duration. Ball/runner end positions are
            partly CAUSED by the run, so this removes part of any genuine run effect (a 'bad control'
            for causal claims); it is reported because it isolates space beyond ball progression.
Inference: OLS with CR1 match-clustered SEs and a t(19) reference (20 clusters); Holm correction across
the 10 run types within each analysis. Features: cross-fitted 'all' set from build_runs.py.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from statsmodels.stats.multitest import multipletests

from core import ROOT
from validate_runs import NUM0, SPACE2, eligible, load

RES = ROOT / "results"
SCALE = 100  # report in percentage points (pp) of pitch value
ADJ = {"adj_pre": ["ball_x0", "ball_absy0", "runner_x0", "runner_absy0", "team_space_t0", "minute"],
       "adj_full": ["ball_x0", "ball_absy0", "runner_x0", "runner_absy0", "team_space_t0", "minute",
                    "ball_x1", "ball_absy1", "runner_x1", "runner_absy1", "duration_s"]}
POS = {"CB": ["LCB", "RCB", "CB"], "FB/WB": ["LB", "RB", "LWB", "RWB"],
       "DM/CM": ["LDM", "RDM", "DM", "LCM", "RCM", "CM"], "AM/W": ["AM", "LM", "RM", "LW", "RW", "LAM", "RAM"],
       "CF": ["CF", "LF", "RF"]}


def crossfit_residual(df, cols, target="teammate_space_gain"):
    out = pd.Series(np.nan, index=df.index)
    for f in df.match_id.unique():
        tr, te = df.match_id != f, df.match_id == f
        lr = make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3, knots="quantile"),
                           Ridge(alpha=1.0)).fit(df.loc[tr, cols], df.loc[tr, target])
        out[te] = df.loc[te, target] - lr.predict(df.loc[te, cols])
    return out * SCALE


def group_means(df, col, by, label):
    """Mean of `col` per level of `by`, CR1 match-clustered t(19) CI, raw p and Holm-adjusted p."""
    d = df[[col, by, "match_id"]].dropna().rename(columns={col: "v", by: "g"})
    fit = smf.ols("v ~ 0 + C(g)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.match_id})
    G = d.match_id.nunique()
    q = stats.t.ppf(0.975, G - 1)
    lv = [n.split("[")[1].rstrip("]").replace("T.", "") for n in fit.params.index]
    p = 2 * stats.t.sf(np.abs(fit.params / fit.bse), G - 1)
    holm = multipletests(p, method="holm")[1]
    return pd.DataFrame(dict(analysis=label, level=lv, n=d.groupby("g").size().loc[lv].values,
                             mean=fit.params.values, lo=(fit.params - q * fit.bse).values,
                             hi=(fit.params + q * fit.bse).values, p=p, p_holm=holm))


def overlap_weights(df):
    """1 / number of runs by the same team whose time window overlaps this run (including itself).
    63% of runs overlap another run of the same team; weighting by this counts each moment once."""
    w = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["match_id", "team_id"]):
        t0, t1 = g.t0.values, g.t1.values
        w[g.index] = 1.0 / ((t0[:, None] <= t1[None, :]) & (t1[:, None] >= t0[None, :])).sum(1)
    return w


def weighted_group_means(df, col, by, w, label):
    """As group_means, but weighted least squares (CR1 match-clustered SEs, t(19), Holm)."""
    d = df[[col, by, "match_id"]].assign(w=w).dropna().rename(columns={col: "v", by: "g"})
    fit = smf.wls("v ~ 0 + C(g)", data=d, weights=d.w).fit(cov_type="cluster", cov_kwds={"groups": d.match_id})
    G = d.match_id.nunique()
    q = stats.t.ppf(0.975, G - 1)
    lv = [n.split("[")[1].rstrip("]").replace("T.", "") for n in fit.params.index]
    p = 2 * stats.t.sf(np.abs(fit.params / fit.bse), G - 1)
    return pd.DataFrame(dict(analysis=label, level=lv, n=d.groupby("g").size().loc[lv].values,
                             mean=fit.params.values, lo=(fit.params - q * fit.bse).values,
                             hi=(fit.params + q * fit.bse).values, p=p, p_holm=multipletests(p, method="holm")[1]))


def selection_sensitivities(df):
    """(a) each moment counted once (overlap weights); (b) runs with a shot during them included
    (that exclusion is only needed for the predictive analysis)."""
    out = []
    w = overlap_weights(df)
    for k in ADJ:
        out.append(weighted_group_means(df, k, "subtype", w, f"{k} | moment-weighted"))
    allr = load("all")
    allr = allr[allr[SPACE2].notna().all(axis=1) & allr[NUM0].notna().all(axis=1)].copy()
    for k, cols in ADJ.items():
        allr[k] = crossfit_residual(allr, cols)
        out.append(group_means(allr, k, "subtype", f"{k} | incl. runs with a shot during"))
    return out


def alpha0_run_types():
    """Sensitivity: the same run-type analysis with features built with no receiver advantage (alpha = 0)."""
    path = ROOT / "data" / "runs_feat_alpha0_all.parquet"
    if not path.exists():
        return []
    a0 = eligible(load("all", "_alpha0")).copy()
    out = []
    for k, cols in ADJ.items():
        a0[k] = crossfit_residual(a0, cols)
        out.append(group_means(a0, k, "subtype", f"{k} | alpha0"))
    return out


def main():
    df = eligible(load("all")).copy()
    inv = {p: g for g, ps in POS.items() for p in ps}
    df["pos_group"] = df.player_position.map(inv).fillna("GK/other")
    df["raw_pp"] = df.teammate_space_gain * SCALE
    df["runner_space_t1_pp"] = df.runner_space_t1 * SCALE
    for k, cols in ADJ.items():
        df[k] = crossfit_residual(df, cols)
    det_med = pd.read_csv(RES / "det_frac_median.csv").det_frac_median.iloc[0]
    hi = df[df.det_frac >= det_med]

    rt = pd.concat([
        group_means(df, "raw_pp", "subtype", "raw | all runs"),
        group_means(df, "adj_full", "subtype", "adj_full | all runs"),
        group_means(df, "adj_pre", "subtype", "adj_pre | all runs"),
        group_means(hi, "adj_full", "subtype", "adj_full | high detection"),
        group_means(hi, "adj_pre", "subtype", "adj_pre | high detection"),
        group_means(df, "runner_space_t1_pp", "subtype", "runner own space t1 | all runs")
        .assign(p=np.nan, p_holm=np.nan),  # non-negative by construction: a test against zero is meaningless
    ] + alpha0_run_types() + selection_sensitivities(df))
    rt.to_csv(RES / "run_types.csv", index=False)
    rows = []
    for spec in ["all runs", "high detection", "alpha0", "moment-weighted", "incl. runs with a shot during"]:
        a = rt[rt.analysis == f"adj_full | {spec}"].set_index("level")
        b = rt[rt.analysis == f"adj_pre | {spec}"].set_index("level").loc[a.index]
        flips = [lv for lv in a.index if a.p_holm[lv] < 0.05 and b.p_holm[lv] < 0.05 and a["mean"][lv] * b["mean"][lv] < 0]
        rho = stats.spearmanr(a["mean"], b["mean"]).statistic
        rows.append(dict(specification=spec, n_significant_reversals=len(flips), reversed_types=", ".join(flips),
                         rank_correlation=rho))
    rev = pd.DataFrame(rows)
    rev.to_csv(RES / "run_type_reversals.csv", index=False)
    print("\n=== reversals across specifications\n", rev.round(2).to_string(index=False))
    pd.set_option("display.width", 200)
    for a, g in rt.groupby("analysis", sort=False):
        print(f"\n=== run types: {a}")
        print(g.drop(columns="analysis").sort_values("mean").round(3).to_string(index=False))

    # camera coverage: does adjusted gain depend on the share of extrapolated players? (all 10 types, Holm)
    df["undetected_10pts"] = (1 - df.det_frac) * 10  # slope per +10 percentage points off-camera
    G = df.match_id.nunique()
    rows = []
    for st, g in df.groupby("subtype"):
        for adj in ADJ:
            fit = smf.ols(f"{adj} ~ undetected_10pts", data=g).fit(cov_type="cluster", cov_kwds={"groups": g.match_id})
            b, se = fit.params["undetected_10pts"], fit.bse["undetected_10pts"]
            rows.append(dict(subtype=st, adjustment=adj, n=len(g), slope_per10=b, se=se,
                             p=2 * stats.t.sf(abs(b / se), G - 1)))
    sl = pd.DataFrame(rows)
    sl["p_holm"] = sl.groupby("adjustment").p.transform(lambda p: multipletests(p, method="holm")[1])
    sl.to_csv(RES / "detection_slopes.csv", index=False)
    print("\n=== slope of adjusted gain per +10 pts of players off-camera (Holm within adjustment)")
    print(sl.sort_values(["adjustment", "p"]).round(4).to_string(index=False))

    # positions
    pos = pd.concat([group_means(df[df.pos_group != "GK/other"], a, "pos_group", f"{a} | all runs")
                     for a in ADJ])
    cover = df.groupby("pos_group").det_frac.mean().rename("det_frac_mean")
    pos = pos.merge(cover, left_on="level", right_index=True)
    pos.to_csv(RES / "positions.csv", index=False)
    print("\n=== position groups\n", pos.round(3).to_string(index=False))

    # player-level reliability: split each player's matches into two halves (alternating by match order)
    rel = []
    for adj in ADJ:
        df["half"] = df.groupby("player_id").match_id.transform(lambda s: s.rank(method="dense") % 2)
        h = df.groupby(["player_id", "half"])[adj].agg(["mean", "count"]).unstack()
        h = h[(h[("count", 0)] >= 15) & (h[("count", 1)] >= 15)]
        a, b = h[("mean", 0)].values, h[("mean", 1)].values
        r = np.corrcoef(a, b)[0, 1]
        rng = np.random.default_rng(4)
        bs = [np.corrcoef(a[i], b[i])[0, 1] for i in rng.integers(0, len(a), (5000, len(a)))]
        rel.append(dict(adjustment=adj, n_players=len(h), r_half=r, r_lo=np.percentile(bs, 2.5),
                        r_hi=np.percentile(bs, 97.5), r_full_spearman_brown=2 * r / (1 + r)))
        if adj == "adj_full":
            h.columns = ["mean_A", "mean_B", "n_A", "n_B"]
            h.to_csv(RES / "player_split_half.csv")
    rel = pd.DataFrame(rel)
    rel.to_csv(RES / "player_reliability.csv", index=False)
    print("\n=== player split-half reliability\n", rel.round(3).to_string(index=False))
    pd.DataFrame([dict(det_frac_overall=df.det_frac.mean())]).to_csv(RES / "det_frac_overall.csv", index=False)
    df.to_parquet(ROOT / "data" / "runs_analysis.parquet")


if __name__ == "__main__":
    main()
