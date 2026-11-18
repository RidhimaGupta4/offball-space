// Stress test: the sandbox model must return finite probabilities in [0, 1] for any user action
// (whole teams removed, extreme sliders, players stacked or off the pitch). Run: node tests/test_explorer_edge_cases.mjs
import fs from "node:fs"; import vm from "node:vm"; import path from "node:path"; import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ctx={globalThis:{}}; ctx.window=ctx.globalThis; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(root,"explorer/pc.js"),"utf8"),ctx); vm.runInContext(fs.readFileSync(path.join(root,"explorer/data.js"),"utf8"),ctx);
const M=ctx.globalThis.PCModel, D=ctx.window.EXPLORER_DATA, sc=D.scenarios[0];
const G=M.grid(50,32,sc.pitch[0],sc.pitch[1]); const P=D.params; const bad=[];
const check=(name,v,lo=0,hi=1)=>{ if(!(Number.isFinite(v)&&v>=lo-1e-12&&v<=hi+1e-12)) bad.push(`${name}: ${v}`); else console.log(`ok  ${name}: ${v.toFixed(4)}`); };
check("all defenders removed", M.share(sc.players.map(p=>({...p,removed:p.team==="def"})),P,D.V,G,true));
check("all attackers removed", M.share(sc.players.map(p=>({...p,removed:p.team==="att"})),P,D.V,G,true));
check("everyone removed", M.share(sc.players.map(p=>({...p,removed:true})),P,D.V,G,true));
for (const [r,vmax,beta] of [[0.3,2.5,0.1],[1.5,8,1.0],[0.3,8,0.1],[1.5,2.5,1.0]]) for (const alpha of [0,P.alpha])
  check(`sliders r=${r} vmax=${vmax} beta=${beta} alpha=${alpha.toFixed(2)}`, M.share(sc.players,{r,vmax,beta,alpha},D.V,G,true));
check("all players stacked on one spot", M.share(sc.players.map(p=>({...p,x:0,y:0,vx:0,vy:0})),P,D.V,G,true));
check("max speed 10 m/s every player", M.share(sc.players.map(p=>({...p,vx:10,vy:0})),P,D.V,G,true));
check("player dragged 2 m outside pitch", M.share(sc.players.map((p,i)=>i===0?{...p,x:sc.pitch[0]/2+2,y:sc.pitch[1]/2+2}:p),P,D.V,G,true));
check("control at far corner, extreme beta", M.control(M.prepare(sc.players,{...P,beta:0.1},true),-60,40,{...P,beta:0.1}));
if (bad.length) { console.error("FAILURES:\n"+bad.join("\n")); process.exit(1); }
console.log("all edge cases finite and within [0, 1]");
