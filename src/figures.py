"""All figures for the README/report. Palette: blue = attacking team, red = defending team,
neutral grey midpoint (diverging pair), single-hue blue for magnitude."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, Rectangle

from core import ROOT, attack_sign, frame_index, load_meta, load_tracking, make_grid, pitch_control
from fit_pitch_control import fit_pc, load as load_passes

FIG, RES = ROOT / "figures", ROOT / "results"
SURF, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"
BLUE, RED, GREY_MID = "#2a78d6", "#e34948", "#f0efec"
ORANGE = "#eb6834"
DIV = LinearSegmentedColormap.from_list("div", ["#a8322f", RED, "#f2a3a0", GREY_MID, "#9ec5f4", BLUE, "#1c5cab"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
GRID, SHAPE = make_grid()
EXT = [-52.5, 52.5, -34, 34]

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": LINE, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})


def credit(fig):
    """Data credit on every figure (figures are often shared on their own)."""
    fig.text(0.995, 0.995, "Data: SkillCorner Open Data (A-League 2024/25)", ha="right", va="top",
             fontsize=7.5, color=MUTED)


def draw_pitch(ax, color="#7d7c77", lw=0.9, L=105.0, W=68.0):
    """Pitch markings for an L x W pitch centred at the origin (standard 16.5 m / 5.5 m boxes, 9.15 m circle)."""
    hl, hw = L / 2, W / 2
    ax.add_patch(Rectangle((-hl, -hw), L, W, fill=False, ec=color, lw=lw))
    ax.plot([0, 0], [-hw, hw], c=color, lw=lw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, ec=color, lw=lw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * hl - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, fill=False, ec=color, lw=lw))
        ax.add_patch(Rectangle((s * hl - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, fill=False, ec=color, lw=lw))
        ax.add_patch(Arc((s * (hl - 11), 0), 18.3, 18.3, theta1=(127 if s > 0 else -53), theta2=(233 if s > 0 else 53),
                         ec=color, lw=lw))
    ax.set_xlim(-hl - 1.5, hl + 1.5)
    ax.set_ylim(-hw - 1.5, hw + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def extent(L, W):
    return [-L / 2, L / 2, -W / 2, W / 2]


# ----------------------------------------------------------------------------- fig 1
def fig_pitch_control_validation():
    oof = pd.read_csv(RES / "pc_oof_predictions.csv")
    pm = pd.read_csv(RES / "pc_cv_per_match.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], c=LINE, lw=1, zorder=0)
    labels = {"pc_fit": ("Pitch control (fitted)", BLUE, "o"), "dist_logit": ("Nearest-player distance", ORANGE, "s"),
              "pc_default": ("Pitch control (unfitted literature values)", MUTED, "^")}
    bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001])
    for k, (lab, c, mk) in labels.items():
        b = np.digitize(oof[k], bins) - 1
        g = oof.groupby(b).agg(p=(k, "mean"), y=("y", "mean"), n=("y", "size"))
        g = g[g.n >= 30]
        ax.plot(g.p, g.y, marker=mk, ms=6, lw=2, c=c, label=lab)
    ax.set_xlabel("Predicted P(passing team wins the ball)")
    ax.set_ylabel("Observed share won")
    ax.set_title("Calibration on held-out matches")
    ax.legend(frameon=False, loc="upper left")
    ax = axs[1]
    piv = pm.pivot(index="match", columns="model", values="log_loss").sort_values("pc_fit")
    yy = np.arange(len(piv))
    ax.hlines(yy, piv.pc_fit, piv.dist_logit, color=LINE, lw=2)
    ax.plot(piv.dist_logit, yy, "s", c=ORANGE, ms=6, label="Nearest-player distance")
    ax.plot(piv.pc_fit, yy, "o", c=BLUE, ms=7, label="Pitch control (fitted)")
    ax.set_yticks(yy, [str(m) for m in piv.index], fontsize=7)
    ax.set_xlabel("Log loss on the held-out match (lower is better)")
    n_better = int((piv.pc_fit < piv.dist_logit).sum())
    ax.set_title(f"Leave-one-match-out: better in {n_better} of {len(piv)} matches")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.suptitle(f"Pitch-control model validated on {len(oof):,} passes (who wins the ball at the reception point)",
                 x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    credit(fig)
    fig.savefig(FIG / "fig1_pitch_control_validation.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 2
def fig_value_surface():
    V = np.load(ROOT / "data" / "value_surface_all.npz")["V"].reshape(SHAPE)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    im = ax.imshow(V * 100, extent=EXT, origin="lower", cmap=SEQ, interpolation="bilinear")
    draw_pitch(ax, color="#ffffff", lw=1.0)
    ax.annotate("", xy=(50, 31.5), xytext=(28, 31.5), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.3))
    ax.text(27, 31.5, "attacking direction ", ha="right", va="center", color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("P(shot within 10 s | ball here), %")
    cb.outline.set_visible(False)
    ax.set_title("Value surface: where possession turns into shots", loc="left")
    fig.tight_layout()
    credit(fig)
    fig.savefig(FIG / "fig2_value_surface.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- example run (fig 3 + gif)
def _home(row) -> bool:
    v = row.is_home
    if not isinstance(v, (bool, np.bool_)):
        raise TypeError(f"is_home must be boolean, got {type(v).__name__}: {v!r}")
    return bool(v)


def pick_example():
    df = pd.read_parquet(ROOT / "data" / "runs_analysis.parquet")
    c = df[(df.shot10_after == 1) & (df.det_frac >= 0.75) & (df.duration_s >= 2.5)]
    return c.sort_values("adj_full", ascending=False).iloc[0]


def frame_state(trk, meta, row, i, side, j, params):
    grid, _ = make_grid(length=meta["pitch_length"], width=meta["pitch_width"])
    s = attack_sign(meta, _home(row), int(row.period))
    pos, vel = trk["pos"][i] * s, trk["vel"][i] * s
    pr = np.isfinite(pos[:, 0])
    att, dfd = pr & (trk["team"] == side), pr & (trk["team"] != side)
    full = pitch_control(pos[att], vel[att], pos[dfd], vel[dfd], grid, params)
    attn = att.copy()
    attn[j] = False
    nor = pitch_control(pos[attn], vel[attn], pos[dfd], vel[dfd], grid, params)
    return pos, att, dfd, full.reshape(SHAPE), nor.reshape(SHAPE), trk["ball"][i] * s


def plot_players(ax, pos, att, dfd, j, ball, trail=None):
    ax.scatter(pos[dfd, 0], pos[dfd, 1], s=46, c=RED, ec=SURF, lw=1.2, zorder=5)
    ax.scatter(pos[att, 0], pos[att, 1], s=46, c=BLUE, ec=SURF, lw=1.2, zorder=5)
    ax.scatter(pos[j, 0], pos[j, 1], s=120, c=BLUE, ec=INK, lw=1.6, zorder=6)
    if np.isfinite(ball).all():
        ax.scatter(*ball, s=26, c="#ffffff", ec=INK, lw=1.2, zorder=7)
    if trail is not None:
        ax.plot(trail[:, 0], trail[:, 1], c=INK, lw=1.4, ls="--", zorder=4)


def fig_example_run(params):
    row = pick_example()
    trk, meta = load_tracking(int(row.match_id)), load_meta(int(row.match_id))
    col = {p: k for k, p in enumerate(trk["pid"])}
    j, side = col[int(row.player_id)], 0 if _home(row) else 1
    i0, i1 = frame_index(trk, [int(row.t0), int(row.t1)])
    s = attack_sign(meta, _home(row), int(row.period))
    trail = trk["pos"][i0:i1 + 1, j] * s
    st0 = frame_state(trk, meta, row, i0, side, j, params)
    st1 = frame_state(trk, meta, row, i1, side, j, params)
    V = np.load(ROOT / "data" / "value_surface_all.npz")["V"].reshape(SHAPE)
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, st, ttl in ((axs[0], st0, "Run starts"), (axs[1], st1, "Run ends")):
        ax.imshow(st[3], extent=extent(L, W), origin="lower", cmap=DIV, vmin=0, vmax=1, interpolation="bilinear")
        draw_pitch(ax, L=L, W=W)
        plot_players(ax, st[0], st[1], st[2], j, st[5], trail if st is st1 else None)
        ax.set_title(ttl, loc="left")
    d = (st1[4] - st0[4]) * V / V.max()
    lim = np.abs(d).max()
    ax = axs[2]
    ax.imshow(d, extent=extent(L, W), origin="lower", cmap=DIV, vmin=-lim, vmax=lim, interpolation="bilinear")
    draw_pitch(ax, L=L, W=W)
    plot_players(ax, st1[0], st1[1], st1[2], j, st1[5], trail)
    ax.set_title("Teammates' valuable space: gained (blue) / lost (red)", loc="left")
    tsg = row.teammate_space_gain * 100
    run_name = row.subtype.replace("_", " ")
    fig.suptitle(f"Illustration: {row.player_name} ({row.team}), {run_name} run, minute {int(row.minute)}. "
                 f"Teammates' controlled pitch value {tsg:+.1f} pp raw, {row.adj_full:+.1f} pp above expected; "
                 "shot within 10 s",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.01, "Shading = attacking-team pitch control (blue = attackers, red = defenders). "
             "Ringed dot = runner, dashed line = his path, white dot = ball. Attack is left to right.\n"
             "Roughly half of the raw gain is explained by where the ball and runner went (the team breaking forward); "
             "all analyses use the adjusted value.",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    credit(fig)
    fig.savefig(FIG / "fig3_example_run.png", dpi=200)
    plt.close(fig)

    # animated GIF of the same run, 2 s before to 2 s after
    frames = []
    a, b = max(i0 - 20, 0), min(i1 + 20, len(trk["frame"]) - 1)
    for i in range(a, b + 1, 2):
        if trk["period"][i] != row.period:
            continue
        st = frame_state(trk, meta, row, i, side, j, params)
        fg, ax = plt.subplots(figsize=(8, 5.4))
        ax.imshow(st[3], extent=extent(L, W), origin="lower", cmap=DIV, vmin=0, vmax=1, interpolation="bilinear")
        draw_pitch(ax, L=L, W=W)
        tr = trk["pos"][i0:min(i, i1) + 1, j] * s if i > i0 else None
        plot_players(ax, st[0], st[1], st[2], j, st[5], tr)
        status = "before run" if i < i0 else ("RUN" if i <= i1 else "after run")
        ax.set_title(f"{row.player_name}: {row.subtype.replace('_', ' ')}   [{status}]", loc="left")
        fg.tight_layout()
        credit(fg)
        fg.canvas.draw()
        frames.append(np.asarray(fg.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fg)
    from PIL import Image
    ims = [Image.fromarray(f).resize((640, int(640 * f.shape[0] / f.shape[1])), Image.LANCZOS)
           .quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    ims[0].save(FIG / "example_run.gif", save_all=True, append_images=ims[1:], duration=200, loop=0, optimize=True)
    return row


# ----------------------------------------------------------------------------- fig 4
def fig_run_types():
    rt = pd.read_csv(RES / "run_types.csv")
    panels = [("adj_full", "Adjusted for where the ball and runner ENDED UP\n(space beyond ball progression)"),
              ("adj_pre", "Adjusted for the PRE-RUN state only\n(includes the team's movement during the run)")]
    order = rt[rt.analysis == "adj_full | all runs"].sort_values("mean").level.tolist()
    n_all = rt[rt.analysis == "adj_full | all runs"].set_index("level").n
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    yy = np.arange(len(order))
    for ax, (adj, ttl) in zip(axs, panels, strict=True):
        ax.axvline(0, c=LINE, lw=1)
        for k, (smp, lab, col, mk) in enumerate([("all runs", "All runs", BLUE, "o"),
                                                 ("high detection", "Runs with most players on camera", ORANGE, "s")]):
            g = rt[rt.analysis == f"{adj} | {smp}"].set_index("level").loc[order]
            y = yy + (0.5 - k) * 0.3
            ax.hlines(y, g.lo, g.hi, color=col, lw=2, alpha=0.55)
            sig = g.p_holm.values < 0.05
            ax.scatter(g["mean"][sig], y[sig], marker=mk, s=48, c=col, zorder=3, label=f"{lab} (Holm p < 0.05)")
            ax.scatter(g["mean"][~sig], y[~sig], marker=mk, s=48, facecolors=SURF, edgecolors=col, lw=1.6, zorder=3,
                       label=f"{lab} (not significant)")
        ax.set_title(ttl, loc="left", fontsize=10)
        ax.set_xlabel("Teammates' space gain above expected (pp of pitch value; 95% CI, t(19))")
    axs[0].set_yticks(yy, [f"{s.replace('_', ' ')} (n={n_all[s]})" for s in order])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Which runs open space for teammates? The answer reverses with the adjustment choice",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    credit(fig)
    fig.savefig(FIG / "fig4_run_types.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 5
def fig_model_comparison():
    analyses = [("primary", "Primary (10 s, logistic)"), ("high_detection", "Most players on camera"),
                ("no_run_type_labels", "Without run-type labels"), ("horizon_5s", "Shot within 5 s"),
                ("horizon_15s", "Shot within 15 s"), ("gradient_boosting", "Gradient boosting"),
                ("alpha0", "No receiver advantage (α = 0)")]
    comps = [("B1_team_space vs B0_context", "+ team pitch control", BLUE, "o"),
             ("B2_runner_vs_mates vs B0_context", "+ team control + runner/teammate split", ORANGE, "s")]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axvline(0, c=LINE, lw=1)
    for i, (tag, _label) in enumerate(analyses):
        c = pd.read_csv(RES / f"runs_cv_comparisons_{tag}.csv").set_index("comparison")
        for k, (comp, clab, col, mk) in enumerate(comps):
            r = c.loc[comp]
            y = i + (k - 0.5) * 0.3
            ax.hlines(y, r.ci_lo * 1000, r.ci_hi * 1000, color=col, lw=2.2, alpha=0.7)
            ax.plot(r.logloss_gain * 1000, y, mk, c=col, ms=7.5, label=clab if i == 0 else None)
            ax.text(r.ci_hi * 1000 + 0.08, y, f"{int(r.matches_improved)}/{int(r.n_matches)}", va="center",
                    fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(analyses)), [label for _, label in analyses])
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-sample log-loss improvement over the context-only model, x1000 "
                  "(95% match-bootstrap CI; label = matches improved)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("Does pitch-control space predict a shot beyond ball and runner location? Not detectably.",
                 loc="left")
    fig.tight_layout()
    credit(fig)
    fig.savefig(FIG / "fig5_model_comparison.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 6
def fig_players():
    h = pd.read_csv(RES / "player_split_half.csv")
    rel = pd.read_csv(RES / "player_reliability.csv").set_index("adjustment").loc["adj_full"]
    posall = pd.read_csv(RES / "positions.csv")
    pos = posall[posall.analysis == "adj_full | all runs"].sort_values("mean")
    pre = posall[posall.analysis == "adj_pre | all runs"].set_index("level").loc[pos.level]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    lim = np.abs(h[["mean_A", "mean_B"]].values).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], c=LINE, lw=1)
    ax.scatter(h.mean_A, h.mean_B, s=34, c=BLUE, ec=SURF, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Player's mean in match set A (pp)")
    ax.set_ylabel("Same player, match set B (pp)")
    ax.set_title(f"Split-half reliability r = {rel.r_half:.2f} [{rel.r_lo:.2f}, {rel.r_hi:.2f}], "
                 f"{int(rel.n_players)} players", loc="left", fontsize=10)
    ax = axs[1]
    yy = np.arange(len(pos))
    ax.axvline(0, c=LINE, lw=1)
    ax.hlines(yy + 0.15, pos.lo, pos.hi, color=BLUE, lw=2, alpha=0.55)
    ax.plot(pos["mean"], yy + 0.15, "o", c=BLUE, ms=7, label="end-state adjustment")
    ax.hlines(yy - 0.15, pre.lo, pre.hi, color=MUTED, lw=2, alpha=0.55)
    ax.plot(pre["mean"], yy - 0.15, "D", c=MUTED, ms=6, label="pre-run adjustment")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_yticks(yy, [f"{lvl} (n={n}, {d * 100:.0f}% on camera)"
                       for lvl, n, d in zip(pos.level, pos.n, pos.det_frac_mean, strict=True)])
    ax.set_xlabel("Teammates' space gain above expected (pp; 95% CI, t(19))")
    ax.set_title("By position group: conclusions depend on the adjustment", loc="left", fontsize=10)
    fig.suptitle("Player-level signal is too unstable to rank individuals from 20 matches", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    credit(fig)
    fig.savefig(FIG / "fig6_players_reliability.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    pdata = load_passes()
    params, _ = fit_pc(pdata, np.arange(len(pdata["y"])))
    fig_pitch_control_validation()
    fig_value_surface()
    ex = fig_example_run(params)
    print("example:", ex.player_name, ex.team, ex.match_id, ex.event_id, ex.subtype)
    fig_run_types()
    fig_model_comparison()
    fig_players()
    print("figures done")
