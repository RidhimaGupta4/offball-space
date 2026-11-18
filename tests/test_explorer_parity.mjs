// Parity test: the explorer's JavaScript model must reproduce the Python analysis numbers.
// Run: node tests/test_explorer_parity.mjs   (after src/export_explorer.py has written explorer/data.js)
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ctx = { globalThis: {} };
ctx.window = ctx.globalThis;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(root, "explorer/pc.js"), "utf8"), ctx);
vm.runInContext(fs.readFileSync(path.join(root, "explorer/data.js"), "utf8"), ctx);
const M = ctx.globalThis.PCModel, D = ctx.window.EXPLORER_DATA;

let worst = 0, n = 0;
for (const sc of D.scenarios) {
  const G = M.grid(D.grid[0], D.grid[1], sc.pitch[0], sc.pitch[1]);
  const team = M.share(sc.players, D.params, D.V, G, true);
  const mates = M.share(sc.players.map(p => ({ ...p, removed: p.runner })), D.params, D.V, G, true);
  const centre = M.control(M.prepare(sc.players, D.params, true), 0, 0, D.params);
  for (const [js, py] of [[team, sc.reference.team_share], [mates, sc.reference.mates_share],
                          [team - mates, sc.reference.runner_share], [centre, sc.reference.pc_at_centre]]) {
    worst = Math.max(worst, Math.abs(js - py));
    n++;
  }
}
// Python stores the player inputs rounded to 0.01 m before computing references, so agreement should be
// at float32-summation level.
const TOL = 1e-5;
console.log(`explorer parity: ${n} comparisons over ${D.scenarios.length} scenarios, max |JS - Python| = ${worst.toExponential(2)}`);
if (!(worst < TOL)) { console.error(`FAIL: exceeds tolerance ${TOL}`); process.exit(1); }
console.log("PASS");
