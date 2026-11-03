// Vale unit 1103CL-B: tail calibration on the public run summaries, weeks 1-8 of 2026.
// For every published DK run (results.playerResults: avgPts, floor = p10, ceiling = p95) and every player who took part in that
// game, compare the real DK points (nfl-dk-results playerActuals fpts, same draft group; Showdown rows = FLEX points) with the
// run's own p95 and p10. Standard errors: player-game Poisson bootstrap (each team-week-player resampled as one unit, with all of
// its slate rows). A per-position spread multiplier m on mean + m*(q - mean) is fitted on weeks 1-6 and checked on weeks 7-8.
// Usage: node calib-b.cjs <dir with the repo's nfl-dk-*.json, nfl-recap/, nfl-dk-results/> <out.json>. Aggregates only.
const fs = require('fs'), path = require('path');
const D = process.argv[2], OUT = process.argv[3];
const J = f => JSON.parse(fs.readFileSync(path.join(D, f), 'utf8'));
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
  .replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TEAM = { JAC: 'JAX', LA: 'LAR', WSH: 'WAS' }, T = t => { const u = String(t || '').toUpperCase(); return TEAM[u] || u; };

// ---- recap: who took part, kickoff time (ET), DK points from the box for a scoring cross-check ----
const SCHED = J('nfl-schedule.json');
const tw = {}; // team|week -> {kickoffET, players: Map nk -> {pos, snaps, box}}
for (let w = 1; w <= 9; w++) {
  const f = 'nfl-recap/2026-w' + String(w).padStart(2, '0') + '.json'; if (!fs.existsSync(path.join(D, f))) continue;
  for (const g of J(f).games) {
    if (g.status !== 'final') continue;
    // Kickoff times come from nfl-schedule.json (the recap stores a bare date for some games). ET = UTC-4 until 11/1 06:00 UTC, then UTC-5.
    for (const side of [g.home, g.away]) { const sk = SCHED.teams[T(side)] && SCHED.teams[T(side)][w], ko = sk && sk.kickoff ? new Date(sk.kickoff) : null;
      const et = ko ? new Date(ko.getTime() + (ko < new Date('2026-11-01T06:00:00Z') ? -4 : -5) * 3600e3) : null;
      tw[T(side) + '|' + w] = { kickoffET: et ? et.toISOString().slice(0, 16).replace('T', ' ') : null, etHour: et ? et.getUTCHours() : null, etDay: et ? et.getUTCDay() : null, players: new Map() }; }
    for (const p of g.players || []) { const k = T(p.team) + '|' + w; if (!tw[k]) continue;
      const py = +p.passYds || 0, ry = +p.rushYds || 0, cy = +p.recYds || 0;
      const box = py * .04 + (+p.passTD || 0) * 4 - (+p.ints || 0) + (py >= 300 ? 3 : 0) + ry * .1 + (+p.rushTD || 0) * 6 + (ry >= 100 ? 3 : 0) + (+p.rec || 0) + cy * .1 + (+p.recTD || 0) * 6 + (cy >= 100 ? 3 : 0);
      tw[k].players.set(norm(p.name), { pos: p.offensePosition || p.pos, snaps: +p.snaps || 0, box: Math.round(box * 100) / 100 }); }
  }
}
// ---- DK actuals: by date and draft group ----
const act = {}; // date -> {dg -> Map nk -> fpts, classic: Map}
for (const f of fs.readdirSync(path.join(D, 'nfl-dk-results'))) {
  const r = J('nfl-dk-results/' + f), o = act[r.date] = act[r.date] || { classic: new Map() };
  for (const c of r.contests) { const m = o[c.draftGroupId] = o[c.draftGroupId] || new Map();
    for (const [k, v] of Object.entries(c.playerActuals || {})) { if (!Number.isFinite(+v.fpts) || v.ambiguous) continue; const nk = norm(v.name || k);
      if (!m.has(nk)) m.set(nk, +v.fpts); if (!c.isShowdown && !o.classic.has(nk)) o.classic.set(nk, +v.fpts); } }
}
// ---- runs ----
const rows = [], runs = [], cross = { sdVsClassic: { n: 0, equal: 0, ratio15: 0 }, actualVsBox: { n: 0, equal: 0 } };
for (const f of fs.readdirSync(D).filter(f => /^nfl-dk-(main|early-only|afternoon-only|showdown)-\d+\.json$/.test(f))) {
  const w = J(f), id = w.identity, R = w.results; if (!id || !R || !R.playerResults) continue;
  const week = +String(id.weekKey).replace(/.*W/, ''), type = /showdown/.test(f) ? 'SD' : /early/.test(f) ? 'early' : /afternoon/.test(f) ? 'afternoon' : 'main';
  const dgAct = (act[id.date] || {})[id.rawSlateId], clAct = (act[id.date] || {}).classic;
  let used = 0, src = { sameDG: 0, sameDateClassic: 0 }, pt = null;
  for (const p of Object.values(R.playerResults)) {
    if (!(p.proj > 0) || p.floor == null || p.ceiling == null || p.avgPts == null) continue;
    const g = tw[T(p.team) + '|' + week]; if (!g) continue;
    const nk = norm(p.name), box = g.players.get(nk), pos = p.pos;
    const played = pos === 'DST' ? true : pos === 'K' ? !!box : !!(box && box.snaps > 0); if (!played) continue;
    let a = dgAct && dgAct.has(nk) ? dgAct.get(nk) : null, s = 'sameDG';
    if (a == null && pos !== 'K' && clAct && clAct.has(nk)) { a = clAct.get(nk); s = 'sameDateClassic'; }
    if (a == null) continue;
    if (type === 'SD' && s === 'sameDG' && pos !== 'K' && pos !== 'DST' && clAct && clAct.has(nk)) { cross.sdVsClassic.n++; const c = clAct.get(nk);
      if (Math.abs(c - a) < 0.011) cross.sdVsClassic.equal++; else if (Math.abs(a - 1.5 * c) < 0.02) cross.sdVsClassic.ratio15++; }
    if (pos !== 'K' && pos !== 'DST' && box) { cross.actualVsBox.n++; if (Math.abs(box.box - a) < 0.051) cross.actualVsBox.equal++;
      if (type === 'SD' && s === 'sameDG') { const c = cross.sdFlexVsBox || (cross.sdFlexVsBox = { n: 0, equalBox: 0, equal15xBox: 0 }); c.n++;
        if (Math.abs(box.box - a) < 0.051) c.equalBox++; else if (Math.abs(1.5 * box.box - a) < 0.08) c.equal15xBox++; } }
    if (type === 'SD') pt = g.etHour == null ? 'unknown' : (g.etHour >= 19 || g.etDay !== 0) ? 'primetime' : 'sunday-day';
    const sal = +p.salary, band = sal < 4000 ? 'under $4,000' : sal < 7000 ? '$4,000-$6,999' : '$7,000+';
    rows.push({ unit: week + '|' + T(p.team) + '|' + nk, run: f, week, pop: type === 'SD' ? (pt === 'primetime' ? 'Showdown prime-time' : 'Showdown Sunday day') : 'Classic',
      pos, band, mean: +p.avgPts, p10: +p.floor, p95: +p.ceiling, a, lowOK: p.floor > 0 || pos === 'DST' });
    used++; src[s]++;
  }
  runs.push({ file: f, type, dg: id.rawSlateId, week, date: id.date, primetime: type === 'SD' ? pt : null, rowsUsed: used, actualSource: src, hadSameDGResults: !!dgAct });
}
// ---- bootstrap machinery (Poisson weights per player-game unit) ----
const units = [...new Set(rows.map(r => r.unit))], uIx = new Map(units.map((u, i) => [u, i])); rows.forEach(r => r.ui = uIx.get(r.unit));
let seed = 20261103; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const pois1 = () => { let L = Math.exp(-1), k = 0, p = 1; do { k++; p *= rnd(); } while (p > L); return k - 1; };
const B = 1000, Wts = Array.from({ length: B }, () => Float32Array.from(units, () => pois1()));
const above = (r, m) => r.a > r.mean + m * (r.p95 - r.mean) ? 1 : 0, below = (r, m) => r.a < r.mean + m * (r.p10 - r.mean) ? 1 : 0;
function share(set, f, wt) { let s = 0, n = 0; for (const r of set) { const x = wt ? wt[r.ui] : 1; if (!x) continue; s += x * f(r); n += x; } return n ? s / n : null; }
function cell(set, m = 1) {
  const lo = set.filter(r => r.lowOK), sd = a => { const v = a.filter(x => x != null), mu = v.reduce((s, x) => s + x, 0) / v.length; return Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / (v.length - 1)); };
  const a95 = share(set, r => above(r, m)), b10 = share(lo, r => below(r, m)), bias = share(set, r => r.a - r.mean);
  return { rows: set.length, playerGames: new Set(set.map(r => r.unit)).size, above95: a95, above95se: sd(Wts.map(w => share(set, r => above(r, m), w))),
    lowRows: lo.length, below10: b10, below10se: lo.length ? sd(Wts.map(w => share(lo, r => below(r, m), w))) : null,
    zeroFloorShare: share(set, r => r.lowOK ? 0 : 1), meanBias: bias, meanBiasSe: sd(Wts.map(w => share(set, r => r.a - r.mean, w))) };
}
const POS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'], BANDS = ['under $4,000', '$4,000-$6,999', '$7,000+'], POPS = ['Classic', 'Showdown prime-time', 'Showdown Sunday day'];
const tables = { byPopulation: {}, byPosition: {}, byBand: {}, byPositionPopulation: {}, byBandPopulation: {} };
for (const pop of ['all', ...POPS]) tables.byPopulation[pop] = cell(rows.filter(r => pop === 'all' || r.pop === pop));
for (const p of POS) tables.byPosition[p] = cell(rows.filter(r => r.pos === p));
for (const b of BANDS) tables.byBand[b] = cell(rows.filter(r => r.band === b));
for (const pop of POPS) { tables.byPositionPopulation[pop] = {}; tables.byBandPopulation[pop] = {};
  for (const p of POS) { const s = rows.filter(r => r.pop === pop && r.pos === p); if (s.length) tables.byPositionPopulation[pop][p] = cell(s); }
  for (const b of BANDS) { const s = rows.filter(r => r.pop === pop && r.band === b); if (s.length) tables.byBandPopulation[pop][b] = cell(s); } }
tables.weeks2to8 = { byPopulation: {}, byPosition: {}, byPositionShowdownPrime: {} };
for (const pop of ['all', ...POPS]) tables.weeks2to8.byPopulation[pop] = cell(rows.filter(r => r.week >= 2 && (pop === 'all' || r.pop === pop)));
for (const p of POS) { tables.weeks2to8.byPosition[p] = cell(rows.filter(r => r.week >= 2 && r.pos === p));
  const sp = rows.filter(r => r.week >= 2 && r.pos === p && r.pop === 'Showdown prime-time'); if (sp.length) tables.weeks2to8.byPositionShowdownPrime[p] = cell(sp); }
tables.byWeek = {}; tables.byWeekSkill = {};
for (let w = 1; w <= 8; w++) { const s = rows.filter(r => r.week === w); if (s.length) tables.byWeek[w] = cell(s);
  const k = s.filter(r => r.pos === 'RB' || r.pos === 'WR' || r.pos === 'TE'); if (k.length) tables.byWeekSkill[w] = cell(k); }
// ---- spread multiplier: one m per position, fitted on weeks 1-6 so both tails hit par as closely as possible ----
const grid = []; for (let m = 0.40; m <= 3.0001; m += 0.01) grid.push(+m.toFixed(2));
// A row is above the scaled p95 exactly when m < t95 = (a - mean)/(p95 - mean), and below the scaled p10 exactly when
// m < t10 = (mean - a)/(mean - p10); so each tail share is a weighted survival curve of fixed thresholds.
const thr = (num, den) => den > 0 ? num / den : (num > 0 ? Infinity : -Infinity);
for (const r of rows) { r.t95 = thr(r.a - r.mean, r.p95 - r.mean); r.t10 = r.lowOK ? thr(r.mean - r.a, r.mean - r.p10) : null; }
function curve(set, key, wt) { // share with threshold > m, for every m in grid
  const s = set.filter(r => r[key] != null).sort((x, y) => y[key] - x[key]), out = new Float64Array(grid.length);
  let tot = 0; for (const r of s) tot += wt ? wt[r.ui] : 1; if (!tot) return null;
  let i = 0, acc = 0; for (let g = grid.length - 1; g >= 0; g--) { while (i < s.length && s[i][key] > grid[g]) { acc += wt ? wt[s[i].ui] : 1; i++; } out[g] = acc / tot; }
  return out;
}
const fitM = (set, wt) => { const a = curve(set, 't95', wt), b = curve(set, 't10', wt); let best = null;
  for (let g = 0; g < grid.length; g++) { const l = (a[g] - 0.05) ** 2 / (0.05 * 0.95) + (b ? (b[g] - 0.10) ** 2 / (0.10 * 0.90) : 0); if (best == null || l < best.l) best = { m: grid[g], l }; }
  return best.m; };
function multiplier(sel, label, firstTrainWeek = 1) {
  const tr = rows.filter(r => sel(r) && r.week >= firstTrainWeek && r.week <= 6), ho = rows.filter(r => sel(r) && r.week >= 7); if (tr.length < 30) return null;
  const m = fitM(tr), boots = []; for (let b = 0; b < 200; b++) boots.push(fitM(tr, Wts[b]));
  const mu = boots.reduce((s, x) => s + x, 0) / boots.length, se = Math.sqrt(boots.reduce((s, x) => s + (x - mu) ** 2, 0) / (boots.length - 1));
  const h1 = cell(ho, 1), hm = cell(ho, m), t1 = cell(tr, 1), tm = cell(tr, m);
  const toward = (x1, xm, par) => x1 == null || xm == null ? null : Math.abs(xm - par) < Math.abs(x1 - par) ? 'toward par' : Math.abs(xm - par) > Math.abs(x1 - par) ? 'away from par' : 'unchanged';
  return { label, trainWeeks: firstTrainWeek + '-6', m, mSe: +se.toFixed(3), train: { rows: tr.length, playerGames: t1.playerGames, above95_m1: t1.above95, above95_m: tm.above95, below10_m1: t1.below10, below10_m: tm.below10 },
    holdout: { rows: ho.length, playerGames: h1.playerGames, above95_m1: h1.above95, above95_m1se: h1.above95se, above95_m: hm.above95, above95_mse: hm.above95se,
      below10_m1: h1.below10, below10_m1se: h1.below10se, below10_m: hm.below10, below10_mse: hm.below10se,
      p95move: toward(h1.above95, hm.above95, 0.05), p10move: toward(h1.below10, hm.below10, 0.10) } };
}
const mult = { byPosition: {}, showdownOnly: {}, classicOnly: {}, byPositionTrain2to6: {}, showdownOnlyTrain2to6: {} };
for (const p of POS) { mult.byPosition[p] = multiplier(r => r.pos === p, p + ' (all runs)');
  mult.showdownOnly[p] = multiplier(r => r.pos === p && r.pop !== 'Classic', p + ' (Showdown)');
  mult.classicOnly[p] = multiplier(r => r.pos === p && r.pop === 'Classic', p + ' (Classic)');
  mult.byPositionTrain2to6[p] = multiplier(r => r.pos === p, p + ' (all runs, wk2-6)', 2);
  mult.showdownOnlyTrain2to6[p] = multiplier(r => r.pos === p && r.pop !== 'Classic', p + ' (Showdown, wk2-6)', 2); }
const out = { builtAt: new Date().toISOString(), rows: rows.length, playerGames: units.length, runs, crossChecks: cross, tables, multipliers: mult };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(JSON.stringify({ rows: rows.length, playerGames: units.length, runsWithRows: runs.filter(r => r.rowsUsed).length, runsWithoutSameDG: runs.filter(r => !r.hadSameDGResults).length, cross }));
