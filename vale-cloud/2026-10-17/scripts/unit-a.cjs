// Unit A (Vale 10/17): snap ceiling as a share of team snaps, and the target step on teams with absences.
// Every prior is built from recap weeks before the game. Rules and constants are chosen on weeks 2-4 and scored on weeks 5-6.
// Standard errors are clustered by team-game (a team-game can appear on several slates). Output: aggregates only.
const fs = require('fs');
const X = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), OUT = process.argv[3];
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
  .replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TRAIN = w => w >= 2 && w <= 4, HOLD = w => w >= 5 && w <= 6, SKILL = new Set(['RB', 'WR', 'TE']);

// ---------- team-game index and pre-game histories ----------
const tg = {}; // key team|week -> {team, week, teamSnaps, teamTargets, players: Map nk->row}
for (const g of X.games) for (const t of Object.values(g.teams)) {
  tg[t.team + '|' + g.week] = { team: t.team, week: g.week, date: g.date, teamSnaps: t.teamSnaps, teamTargets: t.teamTargets,
    players: new Map(t.players.map(p => [p.nk, p])) };
}
const teamWeeks = {}; for (const k of Object.keys(tg)) { const [t, w] = k.split('|'); (teamWeeks[t] = teamWeeks[t] || []).push(+w); }
for (const t in teamWeeks) teamWeeks[t].sort((a, b) => a - b);
const priorGames = (team, w) => (teamWeeks[team] || []).filter(x => x < w).map(x => tg[team + '|' + x]);
function history(team, nk, w) { // games the player appeared in with snaps > 0, before week w
  const out = []; for (const g of priorGames(team, w)) { const p = g.players.get(nk);
    if (p && p.snaps > 0) out.push({ week: g.week, snaps: p.snaps, share: g.teamSnaps ? p.snaps / g.teamSnaps : p.snapPct,
      tshare: g.teamTargets ? p.targets / g.teamTargets : 0, targets: p.targets, pos: p.pos }); }
  return out;
}
// Regulars: >= 2 prior games played, mean snap share >= 50%, last game played within the team's last two games.
function absences(team, w) {
  const g = tg[team + '|' + w]; if (!g) return [];
  const pg = priorGames(team, w), seen = new Map();
  for (const x of pg) for (const p of x.players.values()) if (SKILL.has(p.pos)) seen.set(p.nk, p.pos);
  const lastTwo = new Set(pg.slice(-2).map(x => x.week)), out = [];
  for (const [nk, pos] of seen) {
    const h = history(team, nk, w); if (h.length < 2) continue;
    const ms = h.reduce((a, x) => a + x.share, 0) / h.length; if (ms < 0.5) continue;
    if (!h.some(x => lastTwo.has(x.week))) continue;
    const now = g.players.get(nk); if (now && now.snaps > 0) continue;
    out.push({ nk, pos, meanShare: ms, tshare: h.reduce((a, x) => a + x.tshare, 0) / h.length, games: h.length, weeksPlayed: new Set(h.map(x => x.week)) });
  }
  return out;
}

// ---------- stats helpers ----------
function clusterMean(items) { // items: [{c, v}] -> mean, clustered se, n, clusters
  const n = items.length; if (!n) return { n: 0, clusters: 0, mean: null, se: null };
  const m = items.reduce((a, x) => a + x.v, 0) / n, by = new Map();
  for (const x of items) by.set(x.c, (by.get(x.c) || 0) + (x.v - m));
  const G = by.size; let s = 0; for (const v of by.values()) s += v * v;
  const se = G > 1 ? Math.sqrt(s * G / (G - 1)) / n : null;
  return { n, clusters: G, mean: +m.toFixed(4), se: se == null ? null : +se.toFixed(4) };
}
const median = a => { const b = [...a].sort((x, y) => x - y), n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : 0; };

// ---------- slate-player rows on team-games with an absence ----------
const absCache = {}; const absOf = (t, w) => absCache[t + '|' + w] || (absCache[t + '|' + w] = absences(t, w));
const rows = [];
const runsUsed = [];
for (const r of X.runs) {
  if (!(r.week >= 2 && r.week <= 6)) continue;
  let used = 0;
  for (const p of r.rows) {
    if (!SKILL.has(p.pos) || p.out || !(p.expSnaps > 0)) continue;
    const g = tg[p.team + '|' + r.week]; if (!g) continue;
    const nk = norm(p.name), act = g.players.get(nk); if (!act || !(act.snaps > 0)) continue; // scored only if he played
    const dkOff = (X.actualByDate[r.date] || {})[nk], dkPts = dkOff != null ? dkOff : act.dkBox;
    const h = history(p.team, nk, r.week), ab = absOf(p.team, r.week);
    // engine knew the absence: the absent regular is not an active row of this run
    const knew = ab.filter(a => !r.rows.some(q => norm(q.name) === a.nk && q.team === p.team && q.expSnaps > 0 && !q.out));
    rows.push({ slate: r.dg, type: r.type, week: r.week, date: r.date, team: p.team, nk, pos: p.pos, c: p.team + '|' + r.week,
      E: p.expSnaps, plays: p.plays, projDK: p.projDK, tEng: p.targets, actSnaps: act.snaps, actTargets: act.targets, dk: dkPts,
      teamSnaps: g.teamSnaps, teamTargets: g.teamTargets, h, abs: ab, knew });
    used++;
  }
  if (used) runsUsed.push({ file: r.file, type: r.type, week: r.week, date: r.date, runId: r.runId, rowsUsed: used });
}

// ================= A1: snap ceiling =================
// Binding: the engine's expectation sits at the player's prior raw-snap high (within 1 snap), his group has a known
// absence, and his best prior share x the engine's expected team plays is at least 1 snap higher.
// All teams (the ceiling is not specific to absences); the absence-group subset is reported separately.
const a1All = [];
for (const x of rows) {
  if (x.h.length < 1 || !(x.plays > 0)) continue;
  const raw = Math.max(...x.h.map(q => q.snaps)), shares = x.h.map(q => q.share).sort((a, b) => b - a);
  const cBest = Math.min(x.plays, shares[0] * x.plays), cTop2 = Math.min(x.plays, (shares.length > 1 ? (shares[0] + shares[1]) / 2 : shares[0]) * x.plays);
  a1All.push({ ...x, raw, cBest, cTop2, absenceGroup: x.knew.some(a => a.pos === x.pos), binding: Math.abs(x.E - raw) <= 1 && cBest > raw + 1 });
}
let a1 = a1All;
const variants = {
  engine: x => x.E,
  shareBestFull: x => x.binding ? Math.max(x.E, x.cBest) : x.E,
  shareBestHalf: x => x.binding ? x.E + 0.5 * (Math.max(x.E, x.cBest) - x.E) : x.E,
  shareTop2Full: x => x.binding ? Math.max(x.E, x.cTop2) : x.E,
};
const ptsOf = (x, s) => x.projDK * (x.E > 0 ? s / x.E : 1);
function scoreA1(set, f, shift = 0) {
  return { snaps: clusterMean(set.map(x => ({ c: x.c, v: Math.abs(f(x) - x.actSnaps) }))),
    pts: clusterMean(set.map(x => ({ c: x.c, v: Math.abs(ptsOf(x, f(x)) + shift - x.dk) }))) };
}
function diffA1(set, f, g, sf = 0, sg = 0) { // mean of |err f| - |err g|, clustered
  return { snaps: clusterMean(set.map(x => ({ c: x.c, v: Math.abs(f(x) - x.actSnaps) - Math.abs(g(x) - x.actSnaps) }))),
    pts: clusterMean(set.map(x => ({ c: x.c, v: Math.abs(ptsOf(x, f(x)) + sf - x.dk) - Math.abs(ptsOf(x, g(x)) + sg - x.dk) }))) };
}
const A1 = { definition: 'rows = every engine row (RB/WR/TE, expected snaps > 0, at least one prior game, played); binding = expected snaps within 1 snap of the prior raw-snap high (the raw ceiling) while best prior snap share x the engine expected team plays is at least 1 snap higher; non-binding rows are unchanged under every rule', splits: {} };
for (const [name, sel] of [['train', x => TRAIN(x.week)], ['holdout', x => HOLD(x.week)]]) for (const sd of ['all', 'classic', 'showdown', 'absenceGroup']) {
  const set = a1.filter(x => sel(x) && (sd === 'all' || (sd === 'absenceGroup' ? x.absenceGroup : (sd === 'showdown') === (x.type === 'SD'))));
  const bind = set.filter(x => x.binding), out = { rows: set.length, teamGames: new Set(set.map(x => x.c)).size, bindingRows: bind.length,
    bindingTeamGames: new Set(bind.map(x => x.c)).size, slates: [...new Set(set.map(x => x.slate))].length, mae: {}, vsEngine: {} };
  for (const [v, f] of Object.entries(variants)) { out.mae[v] = scoreA1(set, f); if (v !== 'engine') out.vsEngine[v] = diffA1(set, f, variants.engine); }
  if (bind.length) out.bindingOnly = { actualMinusEngineSnaps: clusterMean(bind.map(x => ({ c: x.c, v: x.actSnaps - x.E }))),
    actualMinusShareCeiling: clusterMean(bind.map(x => ({ c: x.c, v: x.actSnaps - x.cBest }))),
    rawCeilingSnaps: clusterMean(bind.map(x => ({ c: x.c, v: x.raw }))), shareCeilingSnaps: clusterMean(bind.map(x => ({ c: x.c, v: x.cBest }))) };
  A1.splits[name + ':' + sd] = out;
}
// choose on training (all slates) by points MAE; level-shift bar fitted on training absence-group rows
const trainAll = a1.filter(x => TRAIN(x.week)), holdAll = a1.filter(x => HOLD(x.week));
const pick = Object.keys(variants).filter(v => v !== 'engine').map(v => ({ v, d: diffA1(trainAll, variants[v], variants.engine).pts.mean }))
  .sort((a, b) => a.d - b.d)[0];
const shiftA1 = median(trainAll.map(x => x.dk - x.projDK));
A1.choice = { chosenOnTrain: pick.v, trainPointsGainVsEngine: pick.d, levelShiftFittedOnTrain: +shiftA1.toFixed(3) };
A1.holdout = { chosen: { vsEngine: diffA1(holdAll, variants[pick.v], variants.engine) },
  levelShift: { vsEngine: diffA1(holdAll, variants.engine, variants.engine, shiftA1, 0) },
  chosenVsLevelShift: diffA1(holdAll, variants[pick.v], variants.engine, 0, shiftA1),
  chosenPlusShiftVsLevelShift: diffA1(holdAll, variants[pick.v], variants.engine, shiftA1, shiftA1) };

// ================= A2: target step on teams with absences =================
// Real redistribution (recap only, every team-game with an absent regular receiver or back).
const receiverAbs = (t, w) => absOf(t, w).filter(a => a.tshare >= (a.pos === 'RB' ? 0.10 : 0.12));
const red = [];
for (const k of Object.keys(tg)) {
  const g = tg[k]; if (!(g.week >= 2 && g.week <= 6) || !(g.teamTargets > 0)) continue;
  const ab = receiverAbs(g.team, g.week); if (!ab.length) continue;
  const X0 = ab.reduce((a, b) => b.tshare > a.tshare ? b : a), V = ab.reduce((s, a) => s + a.tshare, 0);
  const withX = new Set(ab.reduce((acc, a) => acc ? new Set([...acc].filter(w => a.weeksPlayed.has(w))) : a.weeksPlayed, null));
  const rem = [];
  for (const p of g.players.values()) {
    if (!SKILL.has(p.pos) || !(p.snaps > 0)) continue;
    const h = history(g.team, p.nk, g.week), hw = h.filter(q => withX.has(q.week)), base = hw.length ? hw : h;
    const s = base.length ? base.reduce((a, q) => a + q.tshare, 0) / base.length : 0;
    const ss = base.length ? base.reduce((a, q) => a + q.share, 0) / base.length : 0;
    rem.push({ nk: p.nk, pos: p.pos, s, a: p.targets / g.teamTargets, dSnap: p.snaps / g.teamSnaps - ss });
  }
  const same = rem.filter(p => p.pos === X0.pos && p.dSnap > 0);
  const repl = same.length ? same.reduce((a, b) => b.dSnap > a.dSnap ? b : a).nk : null;
  const bucket = p => p.nk === repl ? 'replacement' : p.pos === 'WR' ? 'otherReceivers' : 'rbTe';
  const sumS = rem.reduce((a, p) => a + p.s, 0), f = {}, fp = {};
  for (const b of ['replacement', 'otherReceivers', 'rbTe']) {
    const mem = rem.filter(p => bucket(p) === b);
    f[b] = mem.reduce((a, p) => a + (p.a - p.s), 0) / V;
    fp[b] = sumS > 0 ? mem.reduce((a, p) => a + p.s, 0) / sumS : 0;
  }
  red.push({ c: k, week: g.week, absentPos: X0.pos, V, f, fp, unaccounted: 1 - (f.replacement + f.otherReceivers + f.rbTe), hasRepl: !!repl });
}
const sumRed = sel => { const s = red.filter(sel), o = { teamGames: s.length, meanVacatedShare: clusterMean(s.map(x => ({ c: x.c, v: x.V }))) };
  for (const b of ['replacement', 'otherReceivers', 'rbTe']) { o[b] = { actual: clusterMean(s.map(x => ({ c: x.c, v: x.f[b] }))),
    proportionalWouldGive: clusterMean(s.map(x => ({ c: x.c, v: x.fp[b] }))), actualMinusProportional: clusterMean(s.map(x => ({ c: x.c, v: x.f[b] - x.fp[b] }))) }; }
  o.notRecovered = clusterMean(s.map(x => ({ c: x.c, v: x.unaccounted }))); return o; };
const A2real = { definition: 'share of the absent regulars\' prior target share (V) gained by each remaining group, actual target share minus the player\'s own prior share (prior games with the absent player active); replacement = same-position player with the largest snap-share rise', all: sumRed(() => true), train: sumRed(x => TRAIN(x.week)), holdout: sumRed(x => HOLD(x.week)),
  byAbsentPos: Object.fromEntries(['WR', 'TE', 'RB'].map(p => [p, sumRed(x => x.absentPos === p)])) };
const fTrain = { replacement: A2real.train.replacement.actual.mean, otherReceivers: A2real.train.otherReceivers.actual.mean, rbTe: A2real.train.rbTe.actual.mean };

// Engine scoring: rows on team-games where the run knew a receiver/back absence.
const pptTrain = {}; { const acc = {}; for (const k of Object.keys(tg)) { const g = tg[k]; if (!TRAIN(g.week)) continue;
  for (const p of g.players.values()) { if (!SKILL.has(p.pos) || !(p.targets > 0)) continue; const o = acc[p.pos] || (acc[p.pos] = [0, 0]);
    o[0] += p.rec + 0.1 * p.recYds + 6 * p.recTD + (p.recYds >= 100 ? 3 : 0); o[1] += p.targets; } }
  for (const [k, v] of Object.entries(acc)) pptTrain[k] = +(v[0] / v[1]).toFixed(3); }
const a2 = []; // per slate-team-game: rows of remaining players with prior shares, engine targets, rule targets
const byST = new Map(); for (const x of rows) { const key = x.slate + '|' + x.c; (byST.get(key) || byST.set(key, []).get(key)).push(x); }
for (const [key, grp] of byST) {
  const x0 = grp[0], ab = x0.knew.filter(a => a.tshare >= (a.pos === 'RB' ? 0.10 : 0.12)); if (!ab.length) continue;
  if (grp.some(x => x.tEng == null)) continue;
  const X0 = ab.reduce((a, b) => b.tshare > a.tshare ? b : a), V = ab.reduce((s, a) => s + a.tshare, 0);
  const withX = new Set(ab.reduce((acc, a) => acc ? new Set([...acc].filter(w => a.weeksPlayed.has(w))) : a.weeksPlayed, null));
  const TT = grp.reduce((s, x) => s + x.tEng, 0); if (!(TT > 0)) continue;
  for (const x of grp) { const hw = x.h.filter(q => withX.has(q.week)), base = hw.length ? hw : x.h; x.s = base.length ? base.reduce((a, q) => a + q.tshare, 0) / base.length : null; }
  const known = grp.filter(x => x.s != null), sumS = known.reduce((s, x) => s + x.s, 0), engKnown = known.reduce((s, x) => s + x.tEng, 0);
  if (!(sumS > 0)) continue;
  // replacement = same-position remaining player with the largest expected-snap rise over his prior share (engine's own expectation)
  const same = known.filter(x => x.pos === X0.pos);
  const repl = same.length ? same.reduce((a, b) => (b.E / (b.plays || 1) - avgShare(b)) > (a.E / (a.plays || 1) - avgShare(a)) ? b : a) : null;
  function avgShare(z) { return z.h.length ? z.h.reduce((s, q) => s + q.share, 0) / z.h.length : 0; }
  const bucket = z => z === repl ? 'replacement' : z.pos === 'WR' ? 'otherReceivers' : 'rbTe';
  const bs = {}; for (const z of known) bs[bucket(z)] = (bs[bucket(z)] || 0) + z.s;
  const fsum = (bs.replacement ? fTrain.replacement : 0) + (bs.otherReceivers ? fTrain.otherReceivers : 0) + (bs.rbTe ? fTrain.rbTe : 0);
  for (const z of known) {
    const prop = engKnown * z.s / sumS; // prior shares renormalized over the remaining players, engine's own team total kept
    const b = bucket(z), add = bs[b] > 0 ? V * fTrain[b] * z.s / bs[b] : 0;
    const measuredShare = z.s + add, measuredTotal = known.reduce((s, q) => s + q.s, 0) + V * fsum;
    const meas = engKnown * measuredShare / measuredTotal;
    a2.push({ c: z.c, slate: z.slate, week: z.week, type: z.type, pos: z.pos, bucket: b, dk: z.dk, projDK: z.projDK, tEng: z.tEng, tProp: prop, tMeas: meas, act: z.actTargets, ppt: pptTrain[z.pos] || 1.8 });
  }
}
const pts = (z, t, sh = 0) => z.projDK + (t - z.tEng) * z.ppt + sh;
const shiftA2 = median(a2.filter(z => TRAIN(z.week)).map(z => z.dk - z.projDK));
const rules = { engine: z => z.tEng, proportional: z => z.tProp, measuredSplit: z => z.tMeas };
function scoreA2(set) {
  const o = { rows: set.length, teamGames: new Set(set.map(z => z.c)).size, slates: new Set(set.map(z => z.slate)).size, targetsMAE: {}, pointsMAE: {}, vsEngine: {}, vsLevelShift: {} };
  for (const [k, f] of Object.entries(rules)) { o.targetsMAE[k] = clusterMean(set.map(z => ({ c: z.c, v: Math.abs(f(z) - z.act) }))); o.pointsMAE[k] = clusterMean(set.map(z => ({ c: z.c, v: Math.abs(pts(z, f(z)) - z.dk) }))); }
  o.pointsMAE.levelShift = clusterMean(set.map(z => ({ c: z.c, v: Math.abs(pts(z, z.tEng, shiftA2) - z.dk) })));
  for (const k of ['proportional', 'measuredSplit']) {
    o.vsEngine[k] = clusterMean(set.map(z => ({ c: z.c, v: Math.abs(pts(z, rules[k](z)) - z.dk) - Math.abs(pts(z, z.tEng) - z.dk) })));
    o.vsLevelShift[k] = clusterMean(set.map(z => ({ c: z.c, v: Math.abs(pts(z, rules[k](z)) - z.dk) - Math.abs(pts(z, z.tEng, shiftA2) - z.dk) })));
    o.vsLevelShift[k + '+shift'] = clusterMean(set.map(z => ({ c: z.c, v: Math.abs(pts(z, rules[k](z), shiftA2) - z.dk) - Math.abs(pts(z, z.tEng, shiftA2) - z.dk) })));
  }
  o.levelShiftVsEngine = clusterMean(set.map(z => ({ c: z.c, v: Math.abs(pts(z, z.tEng, shiftA2) - z.dk) - Math.abs(pts(z, z.tEng) - z.dk) })));
  o.engineTargetsMinusActual = clusterMean(set.map(z => ({ c: z.c, v: z.tEng - z.act })));
  o.byBucketEngineMinusActual = Object.fromEntries(['replacement', 'otherReceivers', 'rbTe'].map(b => [b, clusterMean(set.filter(z => z.bucket === b).map(z => ({ c: z.c, v: z.tEng - z.act })))]));
  return o;
}
const A2 = { real: A2real, ruleFitOnTrain: { measuredSplit: fTrain, levelShift: +shiftA2.toFixed(3), dkPointsPerTargetByPos: pptTrain }, scoring: {} };
for (const [name, sel] of [['train', z => TRAIN(z.week)], ['holdout', z => HOLD(z.week)]]) for (const sd of ['all', 'classic', 'showdown'])
  A2.scoring[name + ':' + sd] = scoreA2(a2.filter(z => sel(z) && (sd === 'all' || (sd === 'showdown') === (z.type === 'SD'))));

const absTG = new Set(Object.keys(tg).filter(k => { const g = tg[k]; return g.week >= 2 && g.week <= 6 && absOf(g.team, g.week).length; }));
const absByWeek = {}; for (const k of absTG) { const w = k.split('|')[1]; absByWeek[w] = (absByWeek[w] || 0) + 1; }
const result = { teamGamesWithAbsence: { total: absTG.size, byWeek: absByWeek, withReceiverOrBackAbsence: red.length },
  runsUsed, A1, A2 };
fs.writeFileSync(OUT, JSON.stringify(result, null, 1));
console.log(JSON.stringify({ absTG: result.teamGamesWithAbsence, A1choice: A1.choice, A1hold: A1.holdout, A1trainAll: A1.splits['train:all'] && { rows: A1.splits['train:all'].rows, binding: A1.splits['train:all'].bindingRows }, A1holdAll: A1.splits['holdout:all'] && { rows: A1.splits['holdout:all'].rows, binding: A1.splits['holdout:all'].bindingRows } }, null, 1));
