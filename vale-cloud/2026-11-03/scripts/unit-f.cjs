// Vale unit 1103CL-F: NBA game-level shared volatility. Public data only; writes aggregates only.
// Usage: node unit-f.cjs <data dir with nba-box, nba-tonight, dk-results, nba-dk-*.json> <out.json>
const fs = require('fs'), zlib = require('zlib'), path = require('path');
const D = process.argv[2], OUT = process.argv[3];
const NIGHTS = []; for (let d = new Date('2026-10-20T12:00Z'); d <= new Date('2026-11-02T12:00Z'); d = new Date(+d + 864e5)) NIGHTS.push(d.toISOString().slice(0, 10));
const ESPN2DK = { GS: 'GSW', NO: 'NOP', SA: 'SAS', NY: 'NYK', UTAH: 'UTA', WSH: 'WAS' }, tm = a => ESPN2DK[a] || a;
const norm = s => String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const etNight = iso => new Date(Date.parse(iso) - 5 * 3600e3).toISOString().slice(0, 10);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
// Inverse normal CDF (Acklam).
function qnorm(p) { p = Math.min(Math.max(p, 1e-6), 1 - 1e-6); const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239], b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572], c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783], d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416], pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - pl) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  const q = p - 0.5, r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
let seed = 20261103; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;

// ---- Box scores: DK points per player-game.
const box = new Map(); // night|teamDK -> {gameKey, players: Map(norm -> {dk, min, dnp})}
const games = new Map(); // gameKey -> {night, home, away, pts:{}, tip}
for (const f of fs.readdirSync(path.join(D, 'nba-box/2026-27'))) {
  const B = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(D, 'nba-box/2026-27', f))));
  for (const g of B.games) {
    const c = g.s.header.competitions[0], night = etNight(c.date); if (!NIGHTS.includes(night)) continue;
    const home = tm(c.competitors.find(x => x.homeAway === 'home').team.abbreviation), away = tm(c.competitors.find(x => x.homeAway === 'away').team.abbreviation);
    const gk = night + '|' + [away, home].sort().join('-'); const G = { night, home, away, tip: c.date, pts: {} }; games.set(gk, G);
    for (const tp of g.s.boxscore.players || []) {
      const team = tm(tp.team.abbreviation), st = tp.statistics[0], L = st.labels, ix = k => L.indexOf(k); const pm = new Map(); let tpts = 0;
      for (const a of st.athletes) {
        const s = a.stats || []; const dnp = a.didNotPlay || !s.length || !(parseFloat(s[ix('MIN')]) > 0);
        let dk = 0; if (!dnp) { const n = k => parseFloat(s[ix(k)]) || 0, PTS = n('PTS'), TPM = parseFloat(String(s[ix('3PT')]).split('-')[0]) || 0, REB = n('REB'), AST = n('AST'), STL = n('STL'), BLK = n('BLK'), TO = n('TO');
          const ten = [PTS, REB, AST, STL, BLK].filter(x => x >= 10).length; dk = PTS + 0.5 * TPM + 1.25 * REB + 1.5 * AST + 2 * STL + 2 * BLK - 0.5 * TO + (ten >= 2 ? 1.5 : 0) + (ten >= 3 ? 3 : 0); tpts += PTS; }
        pm.set(norm(a.athlete.displayName), { dk, dnp });
      }
      G.pts[team] = tpts; box.set(night + '|' + team, { gk, players: pm });
    }
  }
}
// ---- Pre-lock lines from nba-tonight (read the morning of each night).
const lines = new Map(), linesAny = new Map();
for (const night of NIGHTS) { const f = path.join(D, 'nba-tonight', night + '.json'); if (!fs.existsSync(f)) continue; const T = JSON.parse(fs.readFileSync(f));
  for (const g of T.games || []) { if (!g.line || g.line.total == null) continue; const gk = night + '|' + [tm(g.away.abbr), tm(g.home.abbr)].sort().join('-'); const L = { total: g.line.total, spread: g.line.spread, home: tm(g.home.abbr) }; linesAny.set(gk, L); if (Date.parse(T.readAt) < Date.parse(g.tip)) lines.set(gk, L); } }
// ---- Pre-lock sim runs (Showdown, Turbo, Early, Late Night; Main/Night on main are 11/4). Stokastic fields are never read.
const parseET = (date, label) => { const m = /(\d+)\/(\d+) (\d+):(\d+)(AM|PM) Eastern/.exec(label); if (!m) return null; let h = +m[3] % 12 + (m[5] === 'PM' ? 12 : 0); return date + 'T' + String(h).padStart(2, '0') + ':' + m[4]; };
const parseUpd = s => { const m = /(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+) (AM|PM)/.exec(s || ''); if (!m) return null; const h = +m[4] % 12 + (m[7] === 'PM' ? 12 : 0); return m[3] + '-' + m[1].padStart(2, '0') + '-' + m[2].padStart(2, '0') + 'T' + String(h).padStart(2, '0') + ':' + m[5]; };
const runs = [], runLog = { inWindow: 0, preLock: 0, postLock: [], outOfWindow: 0 };
for (const f of fs.readdirSync(D).filter(f => /^nba-dk-.*\.json$/.test(f))) {
  const j = JSON.parse(fs.readFileSync(path.join(D, f))); if (!NIGHTS.includes(j.slateDate)) { runLog.outOfWindow++; continue; } runLog.inWindow++;
  const lock = parseET(j.slateDate, j.slateLabel || ''), upd = parseUpd(j.simUpdated);
  if (!lock || !upd || upd >= lock) { runLog.postLock.push({ file: f, slate: j.slateLabel, simUpdated: j.simUpdated }); continue; } runLog.preLock++;
  runs.push({ file: f, night: j.slateDate, upd, kind: j.isShowdown ? 'Showdown' : (f.match(/nba-dk-([a-z-]+?)(-\d+)?\.json/) || [])[1], players: j.playerResults.map(p => ({ name: p.name, team: p.team, price: p.price, proj: p.proj, floor: p.floor, median: p.median, ceiling: p.ceiling, hit: [30, 40, 50, 60].map(t => [t, p['hit' + t]]) })) });
}
// One run per game: the latest pre-lock run that carries both teams.
const pick = new Map();
for (const r of runs) { const teams = new Set(r.players.map(p => p.team)); for (const t of teams) { const b = box.get(r.night + '|' + t); if (!b) continue; const G = games.get(b.gk); if (!teams.has(G.home) || !teams.has(G.away)) continue; const cur = pick.get(b.gk); if (!cur || r.upd > cur.upd) pick.set(b.gk, r); } }
// ---- Player-game rows.
const rows = [], drop = { dnp: 0, unmatched: 0, noProj: 0 };
for (const [gk, r] of pick) { const G = games.get(gk);
  for (const p of r.players) { if (p.team !== G.home && p.team !== G.away) continue; if (!(p.proj > 0) || !(p.ceiling > p.floor)) { drop.noProj++; continue; }
    const b = box.get(G.night + '|' + p.team), a = b && b.players.get(norm(p.name)); if (!a) { drop.unmatched++; continue; } if (a.dnp) { drop.dnp++; continue; }
    const sdq = (p.ceiling - p.floor) / (1.6448536 + 1.2815516);
    // PIT z: interpolate the sim's CDF points in normal-score space.
    const pts = [[p.floor, 0.10], [p.median, 0.50], [p.ceiling, 0.95], ...p.hit.filter(([t, h]) => h > 0.002 && h < 0.998).map(([t, h]) => [t, 1 - h])].sort((x, y) => x[0] - y[0]).map(([x, q]) => [x, qnorm(q)]);
    const mono = []; for (const q of pts) if (!mono.length || (q[0] > mono[mono.length - 1][0] + 1e-9 && q[1] > mono[mono.length - 1][1])) mono.push(q);
    const x = a.dk; let i = 1; while (i < mono.length - 1 && x > mono[i][0]) i++; const [x0, z0] = mono[i - 1], [x1, z1] = mono[i];
    const zpit = Math.max(-4, Math.min(4, z0 + (z1 - z0) * (x - x0) / (x1 - x0)));
    rows.push({ night: G.night, gk, tk: gk + '|' + p.team, team: p.team, kind: r.kind, price: p.price, proj: p.proj, actual: x, zq: (x - p.proj) / sdq, zpit });
  } }
// Which quantiles are floor / median / ceiling? Hit rates of players whose quantile sits within 1 point of a hit threshold.
const allP = runs.flatMap(r => r.players).filter(p => p.proj > 3), qcheck = {};
for (const key of ['floor', 'median', 'ceiling']) { qcheck[key] = {}; for (const T of [30, 40, 50, 60]) { const near = allP.filter(p => Math.abs(p[key] - T) < 1); if (near.length >= 5) qcheck[key][T] = { n: near.length, hitRate: mean(near.map(p => p.hit.find(h => h[0] === T)[1])) }; } }
qcheck.skewRatio = mean(allP.filter(p => p.median > p.floor).map(p => (p.ceiling - p.median) / (p.median - p.floor)));
// Salary-band residual sd ($1,000 bands of the run's price; in-sample, as Vale's fallback).
const band = r => Math.min(11, Math.floor(r.price / 1000)), bandSd = {};
for (const b of new Set(rows.map(band))) { const res = rows.filter(r => band(r) === b).map(r => r.actual - r.proj); bandSd[b] = Math.sqrt(mean(res.map(v => v * v))); }
for (const r of rows) r.zband = (r.actual - r.proj) / bandSd[band(r)];

// ---- Step 2: variance components by pairwise moments, cluster-equal weights.
function groupBy(a, k) { const m = new Map(); for (const x of a) { const v = x[k]; if (!m.has(v)) m.set(v, []); m.get(v).push(x); } return m; }
function decomp(R, key) {
  const mu = mean(R.map(r => r[key])), z = r => r[key] - mu; const byG = groupBy(R, 'gk'), byN = groupBy(R, 'night');
  const V = mean([...byG.values()].map(g => mean(g.map(r => z(r) ** 2))));
  const team = [], opp = [], night = [];
  for (const g of byG.values()) { const byT = [...groupBy(g, 'team').values()];
    for (const t of byT) { if (t.length < 2) continue; const S = t.reduce((s, r) => s + z(r), 0), Q = t.reduce((s, r) => s + z(r) ** 2, 0); team.push((S * S - Q) / (t.length * (t.length - 1))); }
    if (byT.length === 2) { const SA = byT[0].reduce((s, r) => s + z(r), 0), SB = byT[1].reduce((s, r) => s + z(r), 0); opp.push(SA * SB / (byT[0].length * byT[1].length)); } }
  for (const n of byN.values()) { const gs = [...groupBy(n, 'gk').values()]; if (gs.length < 2) continue; const Sg = gs.map(g => g.reduce((s, r) => s + z(r), 0)), ng = gs.map(g => g.length); const S = Sg.reduce((a, b) => a + b, 0), N = ng.reduce((a, b) => a + b, 0);
    night.push((S * S - Sg.reduce((a, b) => a + b * b, 0)) / (N * N - ng.reduce((a, b) => a + b * b, 0))); }
  const Ct = mean(team), Co = mean(opp), Cn = night.length ? mean(night) : 0;
  return { total: V, meanZ: mu, night: Cn, game: Co - Cn, teamInGame: Ct - Co, player: V - Ct, nightsWithTwoPlusGames: night.length };
}
// ---- Step 3: lognormal shared scale from cross-team (within game) and cross-game (within night) products of squared, team-demeaned z.
function scale(R, key) {
  const byT = groupBy(R, 'tk'); const e = new Map();
  for (const t of byT.values()) { const m = mean(t.map(r => r[key])); for (const r of t) e.set(r, t.length >= 2 ? (r[key] - m) * Math.sqrt(t.length / (t.length - 1)) : null); }
  const gm = [], P = [], A = [], Bm = [];
  for (const [gk, g] of groupBy(R, 'gk')) { const byTeam = [...groupBy(g.filter(r => e.get(r) != null), 'team').values()]; if (byTeam.length !== 2) continue;
    const mA = mean(byTeam[0].map(r => e.get(r) ** 2)), mB = mean(byTeam[1].map(r => e.get(r) ** 2)); P.push(mA * mB); A.push(mA); Bm.push(mB); gm.push({ gk, night: g[0].night, m: (mA + mB) / 2, logScale: 0.5 * Math.log(mA * mB) }); }
  const Rg = mean(P) / (mean(A) * mean(Bm)), gn = groupBy(gm, 'night'); const cross = []; let all = mean(gm.map(x => x.m));
  for (const n of gn.values()) { if (n.length < 2) continue; const s = n.reduce((a, x) => a + x.m, 0), q = n.reduce((a, x) => a + x.m * x.m, 0); cross.push((s * s - q) / (n.length * (n.length - 1))); }
  const Rn = cross.length ? mean(cross) / (all * all) : 1, s2gn = Math.log(Rg) / 4, s2n = Math.log(Rn) / 4, s2g = s2gn - s2n;
  return { sigmaGamePlusNight: Math.sqrt(Math.max(0, s2gn)), sigmaNight: Math.sqrt(Math.max(0, s2n)), sigmaGame: Math.sqrt(Math.max(0, s2g)), var: { gamePlusNight: s2gn, night: s2n, game: s2g }, games: gm.length, perGame: gm };
}
function bootNights(R, f, B = 1000) { const ns = [...new Set(R.map(r => r.night))], byN = groupBy(R, 'night'), out = [];
  for (let b = 0; b < B; b++) { const S = []; for (let i = 0; i < ns.length; i++) { const n = ns[Math.floor(rnd() * ns.length)]; for (const r of byN.get(n)) S.push({ ...r, night: n + '#' + i, gk: r.gk + '#' + i, tk: r.tk + '#' + i }); } out.push(f(S)); } return out; }
const seOf = (vals) => sd(vals.filter(v => Number.isFinite(v)));
const result = { inputs: { nights: [...new Set(rows.map(r => r.night))].sort(), games: pick.size, playerGames: rows.length, dropped: drop, runs: runLog, runsUsedByKind: Object.fromEntries([...groupBy([...pick.values()], 'kind')].map(([k, v]) => [k, v.length])), gamesWithLines: [...pick.keys()].filter(k => lines.has(k)).length, bandSd, quantileCheck: qcheck } };
// Check DK points from box against DK-published actuals where names match.
{ let n = 0, ad = 0, exact = 0;
  for (const night of NIGHTS) { const f = path.join(D, 'dk-results', night + '.json'); if (!fs.existsSync(f)) continue; const R = JSON.parse(fs.readFileSync(f)); const dkpts = new Map();
    for (const c of R.contests) if (!c.isShowdown) for (const v of Object.values(c.playerActuals || {})) dkpts.set(norm(v.name), v.fpts);
    for (const [gk, r] of pick) { if (games.get(gk).night !== night) continue; for (const p of r.players) { const b = box.get(night + '|' + p.team), a = b && b.players.get(norm(p.name)); const v = dkpts.get(norm(p.name)); if (a && !a.dnp && v != null) { n++; ad += Math.abs(v - a.dk); if (Math.abs(v - a.dk) < 0.01) exact++; } } } }
  result.inputs.boxVsDkCheck = { matched: n, exactShare: n ? exact / n : null, meanAbsDiff: n ? ad / n : null }; }
const summ = (est, boots) => ({ mean: est, se: seOf(boots) });
const zs = (R, k) => ({ mean: mean(R.map(r => r[k])), var: mean(R.map(r => r[k] ** 2)) - mean(R.map(r => r[k])) ** 2, absOver2: mean(R.map(r => Math.abs(r[k]) > 2 ? 1 : 0)) });
result.zSummary = Object.fromEntries(['zq', 'zpit', 'zband'].map(k => { const e = zs(rows, k), bs = bootNights(rows, S => zs(S, k), 600); return [k, Object.fromEntries(Object.keys(e).map(m => [m, { mean: e[m], se: seOf(bs.map(b => b[m])) }]))]; }));
result.decomposition = {};
for (const k of ['zq', 'zpit', 'zband']) { const d = decomp(rows, k), bs = bootNights(rows, S => decomp(S, k), 600); const o = { nightsWithTwoPlusGames: d.nightsWithTwoPlusGames, meanZ: summ(d.meanZ, bs.map(b => b.meanZ)), total: summ(d.total, bs.map(b => b.total)) };
  for (const c of ['player', 'teamInGame', 'game', 'night']) { o[c] = summ(d[c], bs.map(b => b[c])); o[c + 'Share'] = summ(d[c] / d.total, bs.map(b => b[c] / b.total)); } result.decomposition[k] = o; }
// Step 3: scale, its bootstrap se, leave-one-night-out folds, and pre-lock predictors.
result.scale = {};
for (const k of ['zq', 'zpit']) { const s = scale(rows, k), bs = bootNights(rows, S => scale(S, k), 600); const ns = [...new Set(rows.map(r => r.night))].sort();
  const folds = ns.map(n => { const f = scale(rows.filter(r => r.night !== n), k); return { leftOut: n, sigmaGame: f.sigmaGame, sigmaNight: f.sigmaNight, varGame: f.var.game }; });
  const fv = folds.map(f => f.varGame), K = fv.length, fm = mean(fv), jse = Math.sqrt((K - 1) / K * fv.reduce((a, v) => a + (v - fm) ** 2, 0));
  result.scale[k] = { sigmaGame: summ(s.sigmaGame, bs.map(b => b.sigmaGame)), sigmaNight: summ(s.sigmaNight, bs.map(b => b.sigmaNight)), sigmaGamePlusNight: summ(s.sigmaGamePlusNight, bs.map(b => b.sigmaGamePlusNight)), varGame: summ(s.var.game, bs.map(b => b.var.game)), varNight: summ(s.var.night, bs.map(b => b.var.night)), games: s.games,
    lono: { folds, sigmaGameMin: Math.min(...folds.map(f => f.sigmaGame)), sigmaGameMax: Math.max(...folds.map(f => f.sigmaGame)), varGameJackknifeSe: jse, foldsWithZeroGameScale: folds.filter(f => f.varGame <= 0).length } };
  // Pre-lock predictors of the per-game realized scale (and of the game's mean z), leave-one-night-out.
  const predictorsFor = LM => { const per = s.perGame.filter(g => LM.has(g.gk)).map(g => { const L = LM.get(g.gk), z = rows.filter(r => r.gk === g.gk).map(r => r[k]); return { ...g, total: L.total, absSpread: Math.abs(L.spread), meanZ: mean(z) }; });
  const fitLono = (yk, xs) => { const out = [], slopes = []; for (const n of [...new Set(per.map(g => g.night))]) { const tr = per.filter(g => g.night !== n), te = per.filter(g => g.night === n); const X = tr.map(g => [1, ...xs.map(x => g[x])]), y = tr.map(g => g[yk]);
      const XtX = X[0].map((_, i) => X[0].map((__, j) => X.reduce((a, r) => a + r[i] * r[j], 0))), Xty = X[0].map((_, i) => X.reduce((a, r, t) => a + r[i] * y[t], 0)); const b = solve(XtX, Xty); slopes.push(b.slice(1));
      const ym = mean(y); for (const g of te) out.push({ night: n, y: g[yk], yhat: b[0] + xs.reduce((a, x, i) => a + b[i + 1] * g[x], 0), base: ym }); }
    const sse = out.reduce((a, o) => a + (o.y - o.yhat) ** 2, 0), sst = out.reduce((a, o) => a + (o.y - o.base) ** 2, 0); const full = (() => { const X = per.map(g => [1, ...xs.map(x => g[x])]), y = per.map(g => g[yk]); const XtX = X[0].map((_, i) => X[0].map((__, j) => X.reduce((a, r) => a + r[i] * r[j], 0))), Xty = X[0].map((_, i) => X.reduce((a, r, t) => a + r[i] * y[t], 0)); return solve(XtX, Xty); })();
    const corr = (() => { const a = out.map(o => o.y), b = out.map(o => o.yhat), ma = mean(a), mb = mean(b); return a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0) * b.reduce((s, v) => s + (v - mb) ** 2, 0)); })();
    const r2 = O => 1 - O.reduce((a, o) => a + (o.y - o.yhat) ** 2, 0) / O.reduce((a, o) => a + (o.y - o.base) ** 2, 0), on = [...new Set(out.map(o => o.night))], bo = []; for (let t = 0; t < 600; t++) { const O = []; for (let i = 0; i < on.length; i++) { const n = on[Math.floor(rnd() * on.length)]; for (const o of out) if (o.night === n) O.push(o); } bo.push(r2(O)); }
    return { games: per.length, heldOutR2Se: seOf(bo), heldOutR2: 1 - sse / sst, heldOutCorr: corr, fullSampleSlopes: Object.fromEntries(xs.map((x, i) => [x, full[i + 1]])), foldSignAgreement: Object.fromEntries(xs.map((x, i) => [x, mean(slopes.map(s => Math.sign(s[i]) === Math.sign(full[i + 1]) ? 1 : 0))])) }; };
  return { logScale: { total: fitLono('logScale', ['total']), absSpread: fitLono('logScale', ['absSpread']), both: fitLono('logScale', ['total', 'absSpread']) }, gameMeanZ: { total: fitLono('meanZ', ['total']), absSpread: fitLono('meanZ', ['absSpread']) } }; };
  result.scale[k].predictors = predictorsFor(lines); result.scale[k].predictorsIncludingLinesReadAfterTip = predictorsFor(linesAny);
  // Bootstrap percentiles, and a power check: inject a known lognormal game scale on these same rows.
  const sgB = bs.map(b => b.sigmaGame).sort((x, y) => x - y); result.scale[k].sigmaGameBootP95 = sgB[Math.floor(0.95 * sgB.length)]; result.scale[k].sigmaGameBootShareZero = mean(sgB.map(v => v === 0 ? 1 : 0));
  result.scale[k].injection = [0.12, 0.24, 0.36].map(sig => { const est = []; for (let rep = 0; rep < 200; rep++) { const sc = new Map(); for (const gk of new Set(rows.map(r => r.gk))) { const u = Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); sc.set(gk, Math.exp(sig * u - sig * sig)); }
      est.push(scale(rows.map(r => ({ ...r, [k]: r[k] * sc.get(r.gk) })), k).sigmaGame); } return { injectedSigma: sig, meanEstimate: mean(est), sdEstimate: sd(est), shareEstimatedAbove0_10: mean(est.map(v => v > 0.10 ? 1 : 0)) }; });
}
function solve(A, b) { const n = b.length, M = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < n; i++) { let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k; [M[i], M[p]] = [M[p], M[i]]; for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; } } const x = Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } return x; }
// Step 4 inputs: is the sim's projected top-1% line published anywhere in the runs?
result.step4 = { simLineFieldFound: false, note: 'nba-dk-*.json carry playerResults, contestScenario (field size and payout) and settlementCheck, but no projected top-1% score.' };
fs.writeFileSync(OUT, JSON.stringify(result, null, 1)); console.log(JSON.stringify(result.inputs, null, 1).slice(0, 1500));
