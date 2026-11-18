"""Does pitch-control 'space' around off-ball runs carry predictive signal? Leakage-free validation.

Outcome: the runner's team takes a shot within H seconds AFTER the run ends (H = 10 s primary; 5 s and
15 s as sensitivity). Runs with a team shot during the run are dropped, so every input is measured
strictly before the outcome window opens.

Nested models, scored by leave-one-match-out (LOMO) CV. Fold f uses features built with pitch-control
parameters and value surfaces that never saw match f (and training matches never see their own shots;
see build_runs.py):
  B0 context     : ball location at run start and end, runner location at start and end, run duration,
                   run type, minute, game state, home/away  (no pitch control)
  B1 + team space: + attacking team's value-weighted pitch control at start and end
  B2 + split     : + the same quantity without the runner (=> runner's own space vs teammates' space)
Model: logistic regression on cubic B-splines (quantile knots) + one-hot categories; the L2 penalty is
chosen INSIDE each training fold by grouped (by match) 4-fold CV, so the test match never informs it.
Model-class check: gradient boosting with fixed, pre-specified hyperparameters.

Never used as inputs: SkillCorner's targeted / received / lead_to_shot / phase-of-play labels /
simultaneous-run counts (hindsight or overlapping the outcome window).

Uncertainty: per-match score differences with a 95% match-level percentile bootstrap (runs within a match
are dependent; the match is the resampling unit), plus a cluster t(19) interval, which is wider with only 20
matches. The "largest effect not ruled out" uses the more conservative of the two upper bounds.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from core import ROOT

RES = ROOT / "results"
NUM0 = ["ball_x0", "ball_absy0", "ball_x1", "ball_absy1", "runner_x0", "runner_absy0", "runner_x1",
        "runner_absy1", "duration_s", "minute"]
CAT0 = ["subtype", "game_state", "venue"]
SPACE1 = ["team_space_t0", "team_space_t1"]
SPACE2 = SPACE1 + ["mates_space_t0", "mates_space_t1"]
MODEL_NAMES = ["B0_context", "B1_team_space", "B2_runner_vs_mates"]
C_GRID = [0.01, 0.1, 1.0]


def model_specs(use_subtype=True):
    cat = CAT0 if use_subtype else [c for c in CAT0 if c != "subtype"]
    return {"B0_context": (NUM0, cat), "B1_team_space": (NUM0 + SPACE1, cat),
            "B2_runner_vs_mates": (NUM0 + SPACE2, cat)}


def spline_block():
    return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3, knots="quantile"))


def make_logit(num, cat):
    pre = ColumnTransformer([("num", spline_block(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(pre, LogisticRegression(max_iter=5000))


def make_hgb(num, cat):
    pre = ColumnTransformer([("num", "passthrough", num),
                             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)])
    cat_mask = [False] * len(num) + [True] * len(cat)
    return make_pipeline(pre, HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0,
        categorical_features=cat_mask, early_stopping=False, random_state=0))


def fit_predict(kind, num, cat, tr, te, y):
    X_tr, X_te = tr[num + cat], te[num + cat]
    if kind == "hgb":
        return make_hgb(num, cat).fit(X_tr, tr[y]).predict_proba(X_te)[:, 1], np.nan
    gs = GridSearchCV(make_logit(num, cat), {"logisticregression__C": C_GRID}, scoring="neg_log_loss",
                      cv=GroupKFold(n_splits=4))
    gs.fit(X_tr, tr[y], groups=tr.match_id)
    return gs.predict_proba(X_te)[:, 1], gs.best_params_["logisticregression__C"]


def load(fold, suffix=""):
    base = pd.read_parquet(ROOT / "data" / "runs_base.parquet")
    feat = pd.read_parquet(ROOT / "data" / f"runs_feat{suffix}_{fold}.parquet")
    df = base.join(feat)
    assert df.is_home.dtype == bool
    df["venue"] = np.where(df.is_home, "home", "away")  # categorical copy for models; is_home stays boolean
    return df


def eligible(df):
    return df[(~df.shot_during) & df[SPACE2].notna().all(axis=1) & df[NUM0].notna().all(axis=1)]


def lomo(y="shot10_after", subset=None, kind="logit", use_subtype=True, suffix=""):
    mids = sorted(pd.read_parquet(ROOT / "data" / "runs_base.parquet").match_id.unique())
    specs = model_specs(use_subtype)
    preds, chosen = [], []
    for f in mids:
        df = eligible(load(f, suffix))
        if subset is not None:
            df = df[subset(df)]
        tr, te = df[df.match_id != f], df[df.match_id == f]
        assert f not in set(tr.match_id) and set(te.match_id) <= {f}
        out = te[["match_id", "event_id", y]].rename(columns={y: "y"}).copy()
        for name, (num, cat) in specs.items():
            out[name], c = fit_predict(kind, num, cat, tr, te, y)
            chosen.append(dict(fold=f, model=name, C=c))
        preds.append(out)
    return pd.concat(preds), pd.DataFrame(chosen)


def report(p, tag):
    y = p.y.values
    sc = pd.DataFrame([dict(model=k, log_loss=log_loss(y, p[k]), brier=brier_score_loss(y, p[k]),
                            auc=roc_auc_score(y, p[k])) for k in MODEL_NAMES])
    pm = p.groupby("match_id").apply(lambda g: pd.Series(
        {k: log_loss(g.y, g[k], labels=[0, 1]) for k in MODEL_NAMES} | {"n": len(g)}), include_groups=False)
    rng = np.random.default_rng(1)
    boot_idx = rng.integers(0, len(pm), (5000, len(pm)))
    rows = []
    for a, b in [("B0_context", "B1_team_space"), ("B1_team_space", "B2_runner_vs_mates"),
                 ("B0_context", "B2_runner_vs_mates")]:
        d, w = (pm[a] - pm[b]).values, pm.n.values
        bs = np.array([np.average(d[i], weights=w[i]) for i in boot_idx])
        base_ll = log_loss(y, p[a])
        # small-sample alternative to the percentile bootstrap: cluster (match) t-interval, t(G-1)
        G, ww = len(d), w / w.sum()
        m = np.sum(ww * d)
        se = np.sqrt(G / (G - 1) * np.sum(ww ** 2 * (d - m) ** 2))
        q = stats.t.ppf(0.975, G - 1)
        hi_conservative = max(np.percentile(bs, 97.5), m + q * se)
        rows.append(dict(comparison=f"{b} vs {a}", logloss_gain=np.average(d, weights=w),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         t_ci_lo=m - q * se, t_ci_hi=m + q * se,
                         ci_hi_pct_of_baseline=100 * hi_conservative / base_ll,
                         matches_improved=int((d > 0).sum()), n_matches=len(d)))
    cr = pd.DataFrame(rows)
    sc = sc.merge(calibration(p), on="model")
    print(f"\n=== {tag}: n_runs={len(p)}, outcome rate={y.mean():.3f}")
    print(sc.round(4).to_string(index=False))
    print(cr.round(5).to_string(index=False), flush=True)
    sc.assign(analysis=tag, n_runs=len(p), outcome_rate=y.mean()).to_csv(RES / f"runs_cv_scores_{tag}.csv", index=False)
    cr.assign(analysis=tag).to_csv(RES / f"runs_cv_comparisons_{tag}.csv", index=False)


def calibration(p):
    """Calibration intercept (calibration-in-the-large) and slope of out-of-fold predictions:
    logit P(y=1) = a + b * logit(p_hat); a well-calibrated model has a ~ 0 (with b fixed at 1) and b ~ 1."""
    y = np.asarray(p.y, dtype=float)
    rows = []
    for k in MODEL_NAMES:
        q = np.clip(np.asarray(p[k], dtype=float), 1e-6, 1 - 1e-6)
        lp = np.log(q / (1 - q))
        slope = np.asarray(sm.GLM(y, np.column_stack([np.ones_like(lp), lp]), family=sm.families.Binomial()).fit().params)[1]
        icpt = np.asarray(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params)[0]
        rows.append(dict(model=k, calib_intercept=float(icpt), calib_slope=float(slope)))
    return pd.DataFrame(rows)


def cluster_t(fit, groups):
    """Small-sample cluster inference: CR1 SEs with t(G-1) reference distribution."""
    G = len(np.unique(groups))
    t = fit.params / fit.bse
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    q = stats.t.ppf(0.975, G - 1)
    return p, fit.params - q * fit.bse, fit.params + q * fit.bse


def inference():
    """Descriptive context-adjusted logistic GLM on all matches (cross-fitted 'all' features): direction and
    size of the runner-space and teammate-space associations. Context enters as centred natural cubic
    regression splines (patsy cr, df=4, sum-to-zero constraint -> identifiable intercept); match-clustered
    CR1 SEs with a t(19) reference distribution."""
    import statsmodels.formula.api as smf
    df = eligible(load("all")).copy()
    terms = ["teammate_space_gain", "runner_space_t1", "team_space_t0", "runner_space_t0"]
    for t in terms:
        df[t + "_z"] = (df[t] - df[t].mean()) / df[t].std()
    rhs = " + ".join([t + "_z" for t in terms] + [f"cr({c}, df=4, constraints='center')" for c in NUM0] +
                     ["C(subtype)", "C(game_state)", "C(venue)"])
    fit = smf.glm(f"shot10_after ~ {rhs}", data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df.match_id.values})
    p, lo, hi = cluster_t(fit, df.match_id.values)
    names = ["Intercept"] + [t + "_z" for t in terms]
    tab = pd.DataFrame({"term": names, "coef_per_sd": fit.params[names].values, "se_cluster": fit.bse[names].values,
                        "odds_ratio_per_sd": np.exp(fit.params[names].values), "or_lo": np.exp(lo[names].values),
                        "or_hi": np.exp(hi[names].values), "p_t19": p[fit.params.index.get_indexer(names)]})
    X = fit.model.exog
    print(f"\n=== GLM (context-adjusted, match-clustered SE, t(19)), per-SD effects; rank {np.linalg.matrix_rank(X)}"
          f"/{X.shape[1]}, condition number {np.linalg.cond(X):.0f}, converged {fit.converged}")
    from statsmodels.stats.multitest import multipletests
    tab["p_holm"] = np.r_[np.nan, multipletests(tab.p_t19.values[1:], method="holm")[1]]
    print(tab.round(4).to_string(index=False))
    tab.iloc[1:].to_csv(RES / "runs_glm_effects.csv", index=False)


if __name__ == "__main__":
    analyses = [
        ("primary", dict()),
        ("high_detection", dict(subset=lambda d: d.det_frac >= DET_MED)),
        ("no_run_type_labels", dict(use_subtype=False)),
        ("horizon_5s", dict(y="shot5_after")),
        ("horizon_15s", dict(y="shot15_after")),
        ("gradient_boosting", dict(kind="hgb")),
        ("alpha0", dict(suffix="_alpha0")),
    ]
    DET_MED = eligible(load("all")).det_frac.median()
    pd.DataFrame([dict(det_frac_median=DET_MED)]).to_csv(RES / "det_frac_median.csv", index=False)
    for tag, kw in analyses:
        p, ch = lomo(**kw)
        p.to_csv(RES / ("runs_oof_predictions.csv" if tag == "primary" else f"runs_oof_predictions_{tag}.csv"), index=False)
        if tag == "primary":
            ch.to_csv(RES / "runs_chosen_C.csv", index=False)
            print("chosen C per model:\n", ch.groupby("model").C.value_counts().to_string())
        report(p, tag)
    inference()
