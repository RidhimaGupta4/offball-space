"""Fit and validate the pitch-control model on pass contests with leave-one-match-out (LOMO) CV.

Every model is fitted on 19 matches and scored on the held-out 20th: no frame, pass or parameter from
the test match is ever seen in training. Compared models:
  const      - training base rate
  dist_logit - logistic regression on (nearest defender dist - nearest attacker dist) to the target
  pc_default - pitch control with unfitted literature-style values (r=0.7 s and vmax=5 m/s as in
               Spearman 2018; beta=0.45 and a neutral alpha=0 for this functional form)
  pc_fit     - pitch control with (r, vmax, beta, alpha) fitted by maximum likelihood on training matches
Outputs: results/pc_fold_params.csv, results/pc_cv_scores.csv, results/pc_oof_predictions.csv
"""
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from core import DEFAULT_PARAMS, ROOT

FAR = 1e4


def pad(arrs, n=11):
    out = np.full((len(arrs), n, 2), FAR)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return out


def load():
    P = pickle.load(open(ROOT / "data" / "passes.pkl", "rb"))
    d = dict(
        mid=np.array([p["match_id"] for p in P]),
        y=np.array([p["label"] for p in P]),
        tgt=np.stack([p["target"] for p in P]),
        ap=pad([p["att_pos"] for p in P]), av=pad([p["att_vel"] for p in P]) * 0,
        dp=pad([p["def_pos"] for p in P]), dv=pad([p["def_vel"] for p in P]) * 0,
        det=np.array([p["det_frac"] for p in P]),
    )
    # velocities: padded rows must have zero velocity, real rows their velocity
    for key, src in (("av", "att_vel"), ("dv", "def_vel")):
        for i, p in enumerate(P):
            d[key][i, : len(p[src])] = p[src]
    return d


def pc_prob(d, idx, r, vmax, beta, alpha):
    t = d["tgt"][idx][:, None, :]
    ta = r + np.linalg.norm(d["ap"][idx] + d["av"][idx] * r - t, axis=-1) / vmax
    td = r + np.linalg.norm(d["dp"][idx] + d["dv"][idx] * r - t, axis=-1) / vmax
    return expit(alpha + logsumexp(-ta / beta, axis=1) - logsumexp(-td / beta, axis=1))


def unpack(th):
    return dict(r=np.exp(th[0]), vmax=np.exp(th[1]), beta=np.exp(th[2]), alpha=th[3])


def fit_pc(d, idx):
    y = d["y"][idx]

    def nll(th):
        p = np.clip(pc_prob(d, idx, **unpack(th)), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    th0 = np.array([np.log(0.7), np.log(5.0), np.log(0.45), 0.0])
    bounds = [(np.log(0.05), np.log(2.0)), (np.log(2.0), np.log(10.0)), (np.log(0.05), np.log(3.0)), (-3, 5)]
    res = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
    return unpack(res.x), res


def dist_feature(d, idx):
    t = d["tgt"][idx][:, None, :]
    da = np.linalg.norm(d["ap"][idx] - t, axis=-1).min(1)
    dd = np.linalg.norm(d["dp"][idx] - t, axis=-1).min(1)
    return (dd - da)[:, None]


def scores(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return dict(log_loss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p),
                auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan)


if __name__ == "__main__":
    d = load()
    mids = np.unique(d["mid"])
    oof = {k: np.full(len(d["y"]), np.nan) for k in ("const", "dist_logit", "pc_default", "pc_fit")}
    fold_params = []
    for m in mids:
        tr, te = np.where(d["mid"] != m)[0], np.where(d["mid"] == m)[0]
        oof["const"][te] = d["y"][tr].mean()
        lr = LogisticRegression(C=1e6).fit(dist_feature(d, tr), d["y"][tr])
        oof["dist_logit"][te] = lr.predict_proba(dist_feature(d, te))[:, 1]
        oof["pc_default"][te] = pc_prob(d, te, **DEFAULT_PARAMS)
        prm, res = fit_pc(d, tr)
        oof["pc_fit"][te] = pc_prob(d, te, **prm)
        fold_params.append(dict(held_out_match=m, **prm, train_nll=res.fun, converged=res.success))
        print(m, {k: round(v, 3) for k, v in prm.items()}, "conv", res.success)

    fp = pd.DataFrame(fold_params)
    fp.to_csv(ROOT / "results" / "pc_fold_params.csv", index=False)

    # pooled OOF scores + per-match scores (for match-level bootstrap of differences)
    rows = []
    for k, p in oof.items():
        rows.append(dict(model=k, scope="pooled", **scores(d["y"], p)))
    per_match = []
    for m in mids:
        te = d["mid"] == m
        for k, p in oof.items():
            per_match.append(dict(match=m, model=k, n=te.sum(), **scores(d["y"][te], p[te])))
    pm = pd.DataFrame(per_match)
    # match-level bootstrap CI for log-loss improvement of pc_fit over dist_logit
    rng = np.random.default_rng(0)
    piv = pm.pivot(index="match", columns="model", values="log_loss")
    w = pm.groupby("match").n.first().loc[piv.index].values
    diffs = []
    for _ in range(5000):
        b = rng.integers(0, len(piv), len(piv))
        diffs.append(np.average(piv["dist_logit"].values[b] - piv["pc_fit"].values[b], weights=w[b]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    wins = (piv["pc_fit"] < piv["dist_logit"]).sum()
    sc = pd.DataFrame(rows)
    print(sc.round(4).to_string(index=False))
    print(f"log-loss gain pc_fit vs dist_logit: 95% match-bootstrap CI [{lo:.4f}, {hi:.4f}]; "
          f"pc_fit better in {wins}/{len(piv)} held-out matches")
    sc.to_csv(ROOT / "results" / "pc_cv_scores.csv", index=False)
    pm.to_csv(ROOT / "results" / "pc_cv_per_match.csv", index=False)
    pd.DataFrame(dict(match=d["mid"], y=d["y"], det_frac=d["det"], **oof)).to_csv(
        ROOT / "results" / "pc_oof_predictions.csv", index=False)
    # sensitivity: does the advantage over the distance baseline hold where most players are on camera?
    terc = pd.qcut(d["det"], 3, labels=["lowest third", "middle third", "highest third"])
    rows = []
    for t in terc.categories:
        sel = np.asarray(terc == t)
        for k in ("dist_logit", "pc_fit"):
            rows.append(dict(detection=t, model=k, n=sel.sum(), det_lo=d["det"][sel].min(),
                             det_hi=d["det"][sel].max(), **scores(d["y"][sel], oof[k][sel])))
    bt = pd.DataFrame(rows)
    bt.to_csv(ROOT / "results" / "pc_by_detection.csv", index=False)
    print(bt.round(4).to_string(index=False))
    print(fp[["r", "vmax", "beta", "alpha"]].describe().round(3))
