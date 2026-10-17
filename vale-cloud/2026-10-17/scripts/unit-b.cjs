// Unit B (Vale 10/17): are the Classic sim's tails too thin? Coverage of each published Classic run's own per-player
// p10 (floor) and p95 (ceiling) against real DK points. p90 and p5 are not published; they are read off a normal line
// through the run's own p10 and p95 (an approximation, stated as such). se clustered by slate. Aggregates only.
const fs = require('fs');
const X = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), OUT = process.argv[3];
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
  .replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const tg = {}; for (const g of X.games) for (const t of Object.values(g.teams)) tg[t.team + '|' + g.week] = { ...t, map: new Map(t.players.map(p => [p.nk, p])) };
const Z10 = 1.2815516, Z95 = 1.6448536;
const rows = [], slates = [], excluded = { noTeamGame: 0, didNotPlay: 0, noActual: 0, noQuantiles: 0, zeroProj: 0 };
for (const r of X.runs) {
  if (r.type === 'SD' || !(r.week >= 1 && r.week <= 5)) continue;
  const totals = {}; for (const q of r.rows) if (q.implied != null && q.oppImplied != null) totals[q.team] = +q.implied + +q.oppImplied;
  let used = 0;
  for (const p of r.pr) {
    if (!(p.proj > 0)) { excluded.zeroProj++; continue; }
    if (p.floor == null || p.ceiling == null) { excluded.noQuantiles++; continue; }
    const g = tg[p.team + '|' + r.week]; if (!g) { excluded.noTeamGame++; continue; }
    const nk = norm(p.name), box = g.map.get(nk);
    const played = p.pos === 'DST' ? true : !!(box && box.snaps > 0);
    if (!played) { excluded.didNotPlay++; continue; }
    const off = (X.actualByDate[r.date] || {})[nk], act = off != null ? off : (p.pos !== 'DST' && box ? box.dkBox : null);
    if (act == null) { excluded.noActual++; continue; }
    const b = (p.ceiling - p.floor) / (Z10 + Z95), a = p.floor + Z10 * b, p90 = a + Z10 * b, p5 = a - Z95 * b;
    const sal = +p.salary, tier = sal < 4000 ? 'under $4,000' : sal < 7000 ? '$4,000-$6,999' : '$7,000+';
    const tot = totals[p.team];
    rows.push({ slate: r.dg, type: r.type, week: r.week, date: r.date, period: r.date < '2026-10-01' ? 'before 10/1' : 'since 10/1',
      pos: p.pos, tier, total: tot == null ? 'unknown' : tot < 44 ? 'under 44' : '44+',
      // Lower-tail tests only where p10 > 0: with p10 = 0 the run puts a point mass at zero, a strict 'below' cannot fire and the
      // normal line would put p5 below zero. Those players are reported separately (share finishing at or below their p10).
      // DST can score below zero, so its lower tail is tested whatever the sign of its p10.
      above95: act > p.ceiling ? 1 : 0, above90: act > p90 ? 1 : 0, below10: (p.floor > 0 || p.pos === 'DST') ? (act < p.floor ? 1 : 0) : null,
      below5: (p.floor > 0 || p.pos === 'DST') ? (act < p5 ? 1 : 0) : null, zeroFloor: (p.floor > 0 || p.pos === 'DST') ? 0 : 1,
      zeroFloorAtOrBelow: (p.floor > 0 || p.pos === 'DST') ? null : (act <= p.floor ? 1 : 0),
      tie10: act === p.floor ? 1 : 0, tie95: act === p.ceiling ? 1 : 0, resid: act - p.proj });
    used++;
  }
  slates.push({ dg: r.dg, type: r.type, week: r.week, date: r.date, runId: r.runId, file: r.file, players: used });
}
function cm(items) { items = items.filter(x => x.v != null); const n = items.length; if (!n) return { n: 0, slates: 0, share: null, se: null };
  const m = items.reduce((s, x) => s + x.v, 0) / n, by = new Map(); for (const x of items) by.set(x.c, (by.get(x.c) || 0) + x.v - m);
  const G = by.size; let s = 0; for (const v of by.values()) s += v * v; const se = G > 1 ? Math.sqrt(s * G / (G - 1)) / n : null;
  return { n, slates: G, share: +m.toFixed(4), se: se == null ? null : +se.toFixed(4) }; }
const METRICS = [['above95', 'above p95', 0.05], ['above90', 'above p90 (approx.)', 0.10], ['below10', 'below p10', 0.10], ['below5', 'below p5 (approx.)', 0.05]];
const cell = set => Object.fromEntries(METRICS.map(([k, , par]) => { const o = cm(set.map(x => ({ c: x.slate, v: x[k] }))); o.par = par;
  o.z = o.se ? +((o.share - par) / o.se).toFixed(2) : null; return [k, o]; }));
const groups = { all: [['all', () => true]],
  salary: ['under $4,000', '$4,000-$6,999', '$7,000+'].map(t => [t, x => x.tier === t]),
  position: ['QB', 'RB', 'WR', 'TE', 'DST'].map(p => [p, x => x.pos === p]),
  gameTotal: ['under 44', '44+'].map(t => [t, x => x.total === t]),
  slateType: ['main', 'early', 'afternoon'].map(t => [t, x => x.type === t]),
  period: ['before 10/1', 'since 10/1'].map(t => [t, x => x.period === t]) };
const table = {}; for (const [g, cells] of Object.entries(groups)) { table[g] = {}; for (const [name, sel] of cells) { const s = rows.filter(sel); table[g][name] = { players: s.length, ...cell(s) }; } }
const offPar = [];
for (const [g, cells] of Object.entries(groups)) { if (g === 'period' || g === 'slateType') continue; for (const [name, sel] of cells) for (const [k, label, par] of METRICS) {
  const c = table[g][name][k]; if (c.z == null || Math.abs(c.z) <= 2) continue;
  const s = rows.filter(sel), split = Object.fromEntries(['before 10/1', 'since 10/1'].map(p => { const q = cm(s.filter(x => x.period === p).map(x => ({ c: x.slate, v: x[k] }))); return [p, q]; }));
  offPar.push({ group: g, cell: name, metric: label, share: c.share, se: c.se, par, z: c.z, n: c.n, slates: c.slates, byPeriod: split }); } }
const tie = { atFloor: cm(rows.map(x => ({ c: x.slate, v: x.tie10 }))), atCeiling: cm(rows.map(x => ({ c: x.slate, v: x.tie95 }))),
  zeroFloorPlayers: cm(rows.map(x => ({ c: x.slate, v: x.zeroFloor }))), zeroFloorFinishedAtOrBelowIt: cm(rows.map(x => ({ c: x.slate, v: x.zeroFloorAtOrBelow }))),
  zeroFloorBySalary: Object.fromEntries(['under $4,000', '$4,000-$6,999', '$7,000+'].map(t => [t, cm(rows.filter(x => x.tier === t).map(x => ({ c: x.slate, v: x.zeroFloor })))])) };
const meanResid = cm(rows.map(x => ({ c: x.slate, v: x.resid })));
fs.writeFileSync(OUT, JSON.stringify({ slates, excluded, players: rows.length, ties: tie, actualMinusProj: meanResid, table, offPar }, null, 1));
const f = o => o.share == null ? '-' : (100 * o.share).toFixed(1) + '% (' + (100 * o.se).toFixed(1) + ', n' + o.n + ')' + (Math.abs(o.z) > 2 ? '*' : '');
console.log('slates', slates.length, 'players', rows.length, 'excluded', JSON.stringify(excluded), 'ties', JSON.stringify(tie), 'resid', JSON.stringify(meanResid));
for (const [g, cells] of Object.entries(table)) for (const [n, c] of Object.entries(cells)) console.log(g.padEnd(9), n.padEnd(14), String(c.players).padStart(5), METRICS.map(([k]) => k + ' ' + f(c[k])).join(' | '));
console.log('offPar', JSON.stringify(offPar, null, 1));
