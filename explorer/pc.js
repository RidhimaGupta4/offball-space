/* Pitch-control model: JavaScript port of src/core.py (pitch_control) and src/build_runs.py (summarise).
 *
 *   tau_i(z) = r + || p_i + v_i * r - z || / vmax
 *   PC(z)    = sigmoid( alpha + logsumexp_att(-tau/beta) - logsumexp_def(-tau/beta) )
 *   share    = sum_z V(z) * PC(z) / sum_z V(z)   on a 50 x 32 grid scaled to the pitch
 *
 * Used by explorer/index.html and verified against Python by tests/test_explorer_parity.mjs.
 */
(function (root) {
  "use strict";

  /** Players -> projected positions after the reaction time (Float64Arrays), split by team. */
  function prepare(players, prm, useVelocity) {
    const r = prm.r;
    const ax = [], ay = [], dx = [], dy = [];
    for (const p of players) {
      if (p.removed) continue;
      const vx = useVelocity ? p.vx : 0, vy = useVelocity ? p.vy : 0;
      const x = p.x + vx * r, y = p.y + vy * r;
      if (p.team === "att") { ax.push(x); ay.push(y); } else { dx.push(x); dy.push(y); }
    }
    return { ax: Float64Array.from(ax), ay: Float64Array.from(ay),
             dx: Float64Array.from(dx), dy: Float64Array.from(dy) };
  }

  /** Stable logsumexp of -tau/beta for one team at target (tx, ty). */
  function lse(xs, ys, tx, ty, prm) {
    const n = xs.length;
    if (n === 0) return -Infinity;
    const z = new Float64Array(n);
    let m = -Infinity;
    for (let i = 0; i < n; i++) {
      const tau = prm.r + Math.hypot(xs[i] - tx, ys[i] - ty) / prm.vmax;
      z[i] = -tau / prm.beta;
      if (z[i] > m) m = z[i];
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.exp(z[i] - m);
    return m + Math.log(s);
  }

  /** Probability that the attacking team controls location (tx, ty). */
  function control(P, tx, ty, prm) {
    const la = lse(P.ax, P.ay, tx, ty, prm), ld = lse(P.dx, P.dy, tx, ty, prm);
    if (la === -Infinity) return 0;
    if (ld === -Infinity) return 1;
    return 1 / (1 + Math.exp(-(prm.alpha + la - ld)));
  }

  /** Cell centres of the nx x ny grid on an L x W pitch (row-major, same order as numpy ravel). */
  function grid(nx, ny, L, W) {
    const xs = new Float64Array(nx * ny), ys = new Float64Array(nx * ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      xs[j * nx + i] = ((i + 0.5) / nx) * L - L / 2;
      ys[j * nx + i] = ((j + 0.5) / ny) * W - W / 2;
    }
    return { xs, ys, nx, ny };
  }

  /** Value-weighted share of the pitch controlled by the attacking team. */
  function share(players, prm, V, G, useVelocity) {
    const P = prepare(players, prm, useVelocity);
    let num = 0, den = 0;
    for (let k = 0; k < V.length; k++) {
      num += V[k] * control(P, G.xs[k], G.ys[k], prm);
      den += V[k];
    }
    return num / den;
  }

  const api = { prepare, control, grid, share };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PCModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
