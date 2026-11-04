// Vale unit 1103CL-I: what sets the height of the real NBA top-1% line. Public data only; writes aggregates only.
// Usage: node unit-i.cjs <data dir with dk-results/, nba-tonight/> <out.json>
const fs = require('fs'), path = require('path');
const D = process.argv[2], OUT = process.argv[3];
const norm = s => String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const mean = a => a.reduce((x, y) => x + y, 0) / a.length, sd = a => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
let seed = 20261104; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'];
const BANDS = [['under 5%', 0, 5], ['5–15%', 5, 15], ['15–30%', 15, 30], ['30%+', 30, 1e9]], bandOf = o => BANDS.find(([, lo, hi]) => o >= lo && o < hi)[0];
// Salaries by draft group from nba-tonight slate rows (only some nights carry them).
const sal = new Map(), games = new Map();
for (const f of fs.readdirSync(path.join(D, 'nba-tonight'))) { const T = JSON.parse(fs.readFileSync(path.join(D, 'nba-tonight', f))); const night = f.slice(0, 10); games.set(night, (T.games || []).length);
  for (const r of (T.slate && T.slate.rows) || []) { if (!r.draftGroup || !(r.salary > 0)) continue; if (!sal.has(r.draftGroup)) sal.set(r.draftGroup, new Map()); sal.get(r.draftGroup).set(norm(r.name), r.salary); } }
// Assignable to the 8 classic slots? (bipartite matching)
function assignable(ps) { const owner = new Array(SLOTS.length).fill(-1);
  const tryP = (i, seen) => { for (let s = 0; s < SLOTS.length; s++) if (!seen[s] && ps[i].slots.has(SLOTS[s])) { seen[s] = true; if (owner[s] < 0 || tryP(owner[s], seen)) { owner[s] = i; return true; } } return false; };
  for (let i = 0; i < ps.length; i++) if (!tryP(i, new Array(SLOTS.length).fill(false))) return false; return true; }
// Best 8-player lineup by actual points from a pool, optionally under the $50,000 cap (depth-first with an upper bound).
function best(pool, cap) { const P = pool.slice().sort((a, b) => b.fpts - a.fpts); let bestPts = -1; const cur = [];
  const minSal = P.map(p => p.salary || 0).sort((a, b) => a - b);
  const rec = (i, pts, salUsed) => { if (cur.length === 8) { if (pts > bestPts) bestPts = pts; return; } if (i >= P.length) return;
    let ub = pts; for (let k = i, n = cur.length; k < P.length && n < 8; k++, n++) ub += P[k].fpts; if (ub <= bestPts) return;
    const p = P[i]; if (!cap || salUsed + p.salary <= 50000) { cur.push(p); if (assignable(cur)) rec(i + 1, pts + p.fpts, salUsed + (p.salary || 0)); cur.pop(); }
    rec(i + 1, pts, salUsed); };
  rec(0, 0, 0); return bestPts > 0 ? bestPts : null; }
const contests = [];
for (const f of fs.readdirSync(path.join(D, 'dk-results')).filter(f => f >= '2026-10-20').sort()) {
  const night = f.slice(0, 10), R = JSON.parse(fs.readFileSync(path.join(D, 'dk-results', f)));
  const gcNight = R.contests.map(c => c.gameCount).find(g => g > 0) || games.get(night) || null;
  for (const c of R.contests) { if (c.isShowdown) continue; const S = c.summary || {}, line = S.topThresholds && S.topThresholds.top1pct; if (line == null) continue;
    const act = new Map(Object.values(c.playerActuals || {}).map(v => [norm(v.name), v])), SM = sal.get(String(c.draftGroupId));
    const own = Object.entries(S.fieldOwnership || {}).map(([k, v]) => ({ nk: norm(k), own: +v })).filter(x => x.own > 0);
    const pl = own.map(x => { const a = act.get(x.nk); return a ? { nk: x.nk, own: x.own, fpts: +a.fpts, slots: new Set(a.positions || []), salary: SM ? SM.get(x.nk) : undefined } : null; }).filter(Boolean);
    const sumOwn = pl.reduce((s, p) => s + p.own, 0);
    // Chalk score (a): best lineup from players owned 20%+, on actuals; with the cap where salaries are public, and without.
    const chalkPool = pl.filter(p => p.own >= 20), capOK = SM && chalkPool.every(p => p.salary > 0);
    const chalkA = capOK ? best(chalkPool, true) : null, chalkAnoCap = best(chalkPool, false);
    // Chalk score (b): ownership-weighted actual points × 8 (= the average field lineup when ownership sums to 800%).
    const chalkB = pl.reduce((s, p) => s + p.own * p.fpts, 0) / sumOwn * 8;
    // Field features from fieldOwnership.
    const sorted = own.map(x => x.own).sort((a, b) => b - a), nTop = Math.max(1, Math.ceil(sorted.length / 10)), tot = sorted.reduce((a, b) => a + b, 0);
    const salKnown = SM ? pl.filter(p => p.salary > 0) : [];
    const feat = { topDecileShare: sorted.slice(0, nTop).reduce((a, b) => a + b, 0) / tot, hhi: sorted.reduce((a, o) => a + (o / tot) ** 2, 0), n30: sorted.filter(o => o >= 30).length,
      salaryWeightedOwn: SM && salKnown.length / pl.length > 0.95 ? salKnown.reduce((s, p) => s + p.own * p.salary, 0) / salKnown.reduce((s, p) => s + p.own, 0) : null,
      games: c.gameCount > 0 ? c.gameCount : gcNight, logEntries: Math.log10(S.totalEntries), entryFee: c.structure ? c.structure.entryFee : null };
    // Step 4: ownership-band share of points, top-1% lineups vs the ownership-weighted field.
    const tops = (c.topFieldLineups || []).filter(l => l.players && l.players.length === 8 && l.points >= line - 1e-6); const topBand = Object.fromEntries(BANDS.map(b => [b[0], 0])); let topPts = 0, used = 0;
    const ownMap = new Map(own.map(x => [x.nk, x.own]));
    for (const l of tops) { const ps = l.players.map(p => ({ nk: norm(p.name) })); if (ps.some(p => !act.has(p.nk))) continue; used++; for (const p of ps) { const v = +act.get(p.nk).fpts, b = bandOf(ownMap.get(p.nk) || 0); topBand[b] += v; topPts += v; } }
    const fieldBand = Object.fromEntries(BANDS.map(b => [b[0], 0])); let fieldPts = 0; for (const p of pl) { fieldBand[bandOf(p.own)] += p.own * p.fpts; fieldPts += p.own * p.fpts; }
    contests.push({ night, dg: String(c.draftGroupId), entries: S.totalEntries, line, top1: S.topThresholds.top1, median: S.topThresholds.median, chalkA, chalkAnoCap, chalkB, chalkPoolSize: chalkPool.length, sumOwn, feat,
      topLineupsUsed: used, topLineupsAvailable: tops.length, topShare: used ? Object.fromEntries(BANDS.map(b => [b[0], topBand[b[0]] / topPts])) : null, topPtsPerLineup: used ? Object.fromEntries(BANDS.map(b => [b[0], topBand[b[0]] / used])) : null, fieldPtsPerLineup: Object.fromEntries(BANDS.map(b => [b[0], fieldBand[b[0]] / sumOwn * 8])), fieldShare: Object.fromEntries(BANDS.map(b => [b[0], fieldBand[b[0]] / fieldPts])) });
  }
}
// ---- bootstrap helpers (nights are the clusters)
const nights = [...new Set(contests.map(c => c.night))].sort(), byNight = n => contests.filter(c => c.night === n);
const bootN = (rows, stat, B = 1000) => { const ns = [...new Set(rows.map(r => r.night))], out = []; for (let b = 0; b < B; b++) { const S = []; for (let i = 0; i < ns.length; i++) { const n = ns[Math.floor(rnd() * ns.length)]; for (const r of rows) if (r.night === n) S.push(r); } const v = stat(S); if (v != null && Number.isFinite(v)) out.push(v); } return out.length > 2 ? sd(out) : null; };
const est = (rows, stat) => ({ mean: stat(rows), se: bootN(rows, stat) });
function solve(A, b) { const n = b.length, M = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < n; i++) { let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k; [M[i], M[p]] = [M[p], M[i]]; for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; } } const x = Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } return x; }
const ols = (X, y) => { const k = X[0].length, A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? 1e-9 : 0))), b = Array.from({ length: k }, (_, i) => X.reduce((s, r, t) => s + r[i] * y[t], 0)); return solve(A, b); };
// Step 1 summary.
const rowsA = contests.filter(c => c.chalkA != null);
const step1 = { contests: contests.length, nights: nights.length, contestsWithCapChalk: rowsA.length, nightsWithCapChalk: new Set(rowsA.map(c => c.night)).size,
  line: est(contests, R => mean(R.map(c => c.line))), chalkA: est(rowsA, R => mean(R.map(c => c.chalkA))), lineMinusChalkA: est(rowsA, R => mean(R.map(c => c.line - c.chalkA))),
  chalkAnoCap: est(contests.filter(c => c.chalkAnoCap != null), R => mean(R.map(c => c.chalkAnoCap))), lineMinusChalkAnoCap: est(contests.filter(c => c.chalkAnoCap != null), R => mean(R.map(c => c.line - c.chalkAnoCap))),
  chalkB: est(contests, R => mean(R.map(c => c.chalkB))), lineMinusChalkB: est(contests, R => mean(R.map(c => c.line - c.chalkB))), chalkBMinusFieldMedian: est(contests, R => mean(R.map(c => c.chalkB - c.median))),
  capCostOnSalaryNights: est(rowsA.filter(c => c.chalkAnoCap != null), R => mean(R.map(c => c.chalkAnoCap - c.chalkA))), chalkPoolSize: est(contests, R => mean(R.map(c => c.chalkPoolSize))), sumOwn: est(contests, R => mean(R.map(c => c.sumOwn))),
  perNight: nights.map(n => { const C = byNight(n), A = C.filter(c => c.chalkA != null); return { night: n, contests: C.length, games: C[0].feat.games, line: mean(C.map(c => c.line)), lineSd: C.length > 1 ? sd(C.map(c => c.line)) : null, chalkA: A.length ? mean(A.map(c => c.chalkA)) : null, chalkAnoCap: mean(C.filter(c => c.chalkAnoCap != null).map(c => c.chalkAnoCap)), chalkB: mean(C.map(c => c.chalkB)), chalkPool: mean(C.map(c => c.chalkPoolSize)), n30: mean(C.map(c => c.feat.n30)), topDecile: mean(C.map(c => c.feat.topDecileShare)) }; }) };
// Step 3: leave-one-night-out OLS for each target and feature set; held-out R² with a night bootstrap of held-out predictions.
const FEATS = ['topDecileShare', 'hhi', 'n30', 'salaryWeightedOwn', 'games', 'logEntries', 'entryFee'];
const TARGETS = { 'top-1% line': c => c.line, 'line − chalk (a, cap)': c => c.chalkA == null ? null : c.line - c.chalkA, 'line − chalk (a, no cap)': c => c.chalkAnoCap == null ? null : c.line - c.chalkAnoCap, 'line − chalk (b)': c => c.line - c.chalkB };
const SETS = { ...Object.fromEntries(FEATS.map(f => [f, [f]])), 'games + logEntries + entryFee': ['games', 'logEntries', 'entryFee'], 'ownership trio (topDecile, HHI, n30)': ['topDecileShare', 'hhi', 'n30'], 'all but salary-weighted': ['topDecileShare', 'hhi', 'n30', 'games', 'logEntries', 'entryFee'], 'chalk (b) only': ['chalkB'], 'chalk (b) + games': ['chalkB', 'games'] };
const fval = (c, f) => f === 'chalkB' ? c.chalkB : c.feat[f];
const step3 = {};
for (const [tn, tf] of Object.entries(TARGETS)) { step3[tn] = {};
  for (const [sn, fs_] of Object.entries(SETS)) { if (tn !== 'top-1% line' && fs_.includes('chalkB')) continue;
    const rows = contests.filter(c => tf(c) != null && fs_.every(f => fval(c, f) != null)); const ns = [...new Set(rows.map(c => c.night))]; if (ns.length < 5) continue; const out = [], slopes = [];
    for (const n of ns) { const tr = rows.filter(c => c.night !== n), te = rows.filter(c => c.night === n); const b = ols(tr.map(c => [1, ...fs_.map(f => fval(c, f))]), tr.map(tf)), ym = mean(tr.map(tf)); slopes.push(b.slice(1));
      for (const c of te) out.push({ night: n, y: tf(c), yhat: b[0] + fs_.reduce((s, f, i) => s + b[i + 1] * fval(c, f), 0), base: ym }); }
    const r2 = O => 1 - O.reduce((s, o) => s + (o.y - o.yhat) ** 2, 0) / O.reduce((s, o) => s + (o.y - o.base) ** 2, 0);
    const full = ols(rows.map(c => [1, ...fs_.map(f => fval(c, f))]), rows.map(tf));
    step3[tn][sn] = { contests: rows.length, nights: ns.length, heldOutR2: r2(out), heldOutR2Se: bootN(out, r2), heldOutRmse: Math.sqrt(mean(out.map(o => (o.y - o.yhat) ** 2))), baselineRmse: Math.sqrt(mean(out.map(o => (o.y - o.base) ** 2))),
      fullSampleSlopes: Object.fromEntries(fs_.map((f, i) => [f, full[i + 1]])), foldSignAgreement: Object.fromEntries(fs_.map((f, i) => [f, mean(slopes.map(s => Math.sign(s[i]) === Math.sign(full[i + 1]) ? 1 : 0))])) }; } }
// Step 4: ownership-band share of points.
const s4rows = contests.filter(c => c.topShare);
const step4 = { contests: s4rows.length, topLineups: s4rows.reduce((s, c) => s + c.topLineupsUsed, 0), pooled: {}, perNight: [] };
step4.points = {}; for (const [b] of BANDS) step4.points[b] = { top: est(s4rows, R => mean(R.map(c => c.topPtsPerLineup[b]))), field: est(s4rows, R => mean(R.map(c => c.fieldPtsPerLineup[b]))), excess: est(s4rows, R => mean(R.map(c => c.topPtsPerLineup[b] - c.fieldPtsPerLineup[b]))) };
step4.points.total = { excess: est(s4rows, R => mean(R.map(c => Object.values(c.topPtsPerLineup).reduce((x, y) => x + y, 0) - Object.values(c.fieldPtsPerLineup).reduce((x, y) => x + y, 0)))) };
for (const [b] of BANDS) step4.pooled[b] = { top: est(s4rows, R => mean(R.map(c => c.topShare[b]))), field: est(s4rows, R => mean(R.map(c => c.fieldShare[b]))), topMinusField: est(s4rows, R => mean(R.map(c => c.topShare[b] - c.fieldShare[b]))) };
for (const n of nights) { const C = s4rows.filter(c => c.night === n); if (!C.length) continue; const o = { night: n, contests: C.length, topLineups: C.reduce((s, c) => s + c.topLineupsUsed, 0) };
  o.excessPoints = Object.fromEntries(BANDS.map(([b]) => [b, mean(C.map(c => c.topPtsPerLineup[b] - c.fieldPtsPerLineup[b]))]));
  for (const [b] of BANDS) { const t = C.map(c => c.topShare[b]), f = C.map(c => c.fieldShare[b]); o[b] = { top: mean(t), field: mean(f), topMinusField: mean(t.map((v, i) => v - f[i])), topMinusFieldSe: C.length > 1 ? sd(t.map((v, i) => v - f[i])) / Math.sqrt(C.length) : null }; }
  step4.perNight.push(o); }
step4.unmatchedTopLineups = contests.reduce((s, c) => s + c.topLineupsAvailable - c.topLineupsUsed, 0);
const out = { builtAt: new Date().toISOString(), nights, salaryNights: [...new Set(rowsA.map(c => c.night))], step1, step3, step4,
  featureSummary: Object.fromEntries(FEATS.map(f => { const v = contests.map(c => c.feat[f]).filter(x => x != null); return [f, { n: v.length, mean: mean(v), sd: sd(v), min: Math.min(...v), max: Math.max(...v) }]; })) };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1)); console.log(JSON.stringify({ contests: contests.length, nights: nights.length, capChalk: rowsA.length, salaryNights: out.salaryNights }));
