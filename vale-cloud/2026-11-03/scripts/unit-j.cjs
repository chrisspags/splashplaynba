// Vale unit 1103CL-J: TNF 11/5 Showdown (JAX @ BAL). Outcome cells from nflverse 2021–25 closing lines, and K/DST/RB shifts per cell
// measured on the 25 public prime-time Showdowns (same set as units C and G). Public data only; writes aggregates only.
// Usage: node unit-j.cjs <splashplaynba data dir> <nflverse games.csv> <out.json>
const fs = require('fs'), path = require('path');
const D = process.argv[2], CSV = process.argv[3], OUT = process.argv[4];
const J = f => JSON.parse(fs.readFileSync(path.join(D, f), 'utf8'));
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TEAM = { JAC: 'JAX', LA: 'LAR', WSH: 'WAS' }, T = t => { const u = String(t || '').toUpperCase(); return TEAM[u] || u; };
const avg = a => a.reduce((s, x) => s + x, 0) / a.length, sdv = a => { const m = avg(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
let seed = 1105; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const band = m => m <= 3 ? '1–3' : m <= 8 ? '4–8' : '9+';
const CELLS = ['fav 9+', 'fav 4–8', 'fav 1–3', 'dog 1–3', 'dog 4–8', 'dog 9+'];
// ---- The TNF game: schedule line and the published Showdown run.
const SCHED = J('nfl-schedule.json'); let tnf = null;
for (const [t, w] of Object.entries(SCHED.teams)) for (const [wk, g] of Object.entries(w)) if (g && g.date === '2026-11-05' && g.home) tnf = { week: +wk, home: T(t), away: T(g.opp), total: g.pregame ? g.pregame.total : g.total, homeSpread: g.pregame ? g.pregame.homeSpread : g.spread, kickoff: g.kickoff, lineObservedAt: g.oddsObservedAt, lineSource: g.oddsSource };
const runFile = fs.readdirSync(D).filter(f => /^nfl-dk-showdown-\d+\.json$/.test(f)).find(f => { const id = J(f).identity; return id && id.date === '2026-11-05'; });
const run = runFile ? J(runFile) : null; const runLine = run ? (run.results.publicForecast.lines || []).find(l => T(l.home) === tnf.home && T(l.away) === tnf.away) : null;
tnf.runFile = runFile || null; tnf.runAt = run ? run.results.at : null; tnf.runLine = runLine || null;
tnf.favorite = tnf.homeSpread < 0 ? tnf.home : tnf.away; tnf.underdog = tnf.favorite === tnf.home ? tnf.away : tnf.home; tnf.favBy = Math.abs(tnf.homeSpread);
// Sim's pre-lock winner exposure for K, DST, RB by team (any slot; RB summed over the team's RBs; counts per lineup).
if (run) { tnf.sim = {}; for (const side of ['favorite', 'underdog']) { const t = tnf[side], ps = Object.values(run.results.playerResults).filter(p => T(p.team) === t);
  tnf.sim[side] = { team: t, K: ps.filter(p => p.pos === 'K').reduce((s, p) => s + p.exposure / 100, 0), DST: ps.filter(p => p.pos === 'DST').reduce((s, p) => s + p.exposure / 100, 0), RB: ps.filter(p => p.pos === 'RB').reduce((s, p) => s + p.exposure / 100, 0), RBcpt: ps.filter(p => p.pos === 'RB').reduce((s, p) => s + (p.cptWinnerExposure || 0) / 100, 0) }; } }
// ---- nflverse closing lines, 2021–25 (regular season + playoffs).
const lines = fs.readFileSync(CSV, 'utf8').trim().split('\n'), H = lines[0].split(','), ix = k => H.indexOf(k);
const G = lines.slice(1).map(l => l.split(',')).filter(r => +r[ix('season')] >= 2021 && +r[ix('season')] <= 2025 && r[ix('result')] !== '' && r[ix('spread_line')] !== '' && r[ix('total_line')] !== '')
  .map(r => { const spread = +r[ix('spread_line')], res = +r[ix('result')], tot = +r[ix('total')], tl = +r[ix('total_line')]; // spread_line > 0: home favored
    const favMargin = spread > 0 ? res : -res; return { season: +r[ix('season')], favBy: Math.abs(spread), favMargin, total: tot, totalLine: tl, overUnder: tot > tl ? 'high' : tot < tl ? 'low' : 'push', pick: spread === 0 }; }).filter(g => !g.pick);
const cellOf = m => m > 0 ? 'fav ' + band(m) : m < 0 ? 'dog ' + band(-m) : 'tie';
const nflv2026 = lines.slice(1).map(l => l.split(',')).find(r => r[ix('season')] === '2026' && T(r[ix('home_team')]) === tnf.home && T(r[ix('away_team')]) === tnf.away);
tnf.nflverse2026Row = nflv2026 ? { spread_line: nflv2026[ix('spread_line')], total_line: nflv2026[ix('total_line')], gameday: nflv2026[ix('gameday')] } : null;
function cellProbs(rows, label) { const n = rows.length, out = { label, games: n, cells: {}, ties: rows.filter(g => g.favMargin === 0).length, pushes: rows.filter(g => g.overUnder === 'push').length };
  const pr = sel => { const k = rows.filter(sel).length, p = k / n; return { mean: p, se: Math.sqrt(p * (1 - p) / n), count: k }; };
  for (const c of CELLS) { out.cells[c] = pr(g => cellOf(g.favMargin) === c); for (const ou of ['high', 'low']) out.cells[c + ' / ' + ou] = pr(g => cellOf(g.favMargin) === c && g.overUnder === ou); }
  out.favWins = pr(g => g.favMargin > 0); out.favCovers = pr(g => g.favMargin > g.favBy); out.high = pr(g => g.overUnder === 'high'); out.meanFavMargin = { mean: avg(rows.map(g => g.favMargin)), se: sdv(rows.map(g => g.favMargin)) / Math.sqrt(n) }; return out; }
const nflverse = {
  spreadWindow: cellProbs(G.filter(g => g.favBy >= 8 && g.favBy <= 11), 'favorite by 8 to 11 (all totals)'),
  spreadAndTotalWindow: cellProbs(G.filter(g => g.favBy >= 8 && g.favBy <= 11 && g.totalLine >= 38 && g.totalLine <= 45), 'favorite by 8 to 11, total 38 to 45'),
  exactishSpread: cellProbs(G.filter(g => g.favBy >= 9 && g.favBy <= 10), 'favorite by 9 to 10 (all totals)') };
// ---- K/DST/RB shifts per outcome cell on the 25 public prime-time Showdowns.
const contestsByDG = {}; for (const f of fs.readdirSync(path.join(D, 'nfl-dk-results'))) for (const c of J('nfl-dk-results/' + f).contests) (contestsByDG[c.draftGroupId] = contestsByDG[c.draftGroupId] || []).push(c);
const games = [];
for (const f of fs.readdirSync(D).filter(f => /^nfl-dk-showdown-\d+\.json$/.test(f))) {
  const w = J(f), id = w.identity, R = w.results, cs = contestsByDG[id.rawSlateId]; if (!cs) continue;
  const week = +String(id.weekKey).replace(/.*W/, ''), P = new Map(); for (const p of Object.values(R.playerResults)) P.set(norm(p.name), { ...p, team: T(p.team) });
  const teams = [...new Set([...P.values()].map(p => p.team))]; if (teams.length !== 2) continue;
  const sc = teams.map(t => (SCHED.teams[t] || {})[week] || {}); if (!(sc[0].completed && sc[1].completed) && (sc[0].score == null || (sc[0].score === 0 && sc[1].score === 0))) continue;
  const homeRow = (R.publicForecast.rows || []).find(r => r.home === true), home = homeRow ? T(homeRow.team) : null; if (!home) continue; const away = teams.find(t => t !== home);
  const line = (R.publicForecast.lines || []).find(l => T(l.home) === home && T(l.away) === away); if (!line || +line.homeSpread === 0) continue;
  const fav = +line.homeSpread < 0 ? home : away, dog = fav === home ? away : home, s = Object.fromEntries(teams.map((t, i) => [t, sc[i].score])), favMargin = s[fav] - s[dog];
  const g = { dg: id.rawSlateId, week, fav, dog, favBy: Math.abs(+line.homeSpread), favMargin, cell: cellOf(favMargin), totalHigh: s[fav] + s[dog] > +line.total, metrics: [] };
  for (const c of cs) { const S = c.summary || {}, l1 = S.topThresholds && S.topThresholds.top1pct; const L = [];
    for (const l of (c.topFieldLineups || []).filter(l => l.players && l.players.length === 6 && (l1 == null || l.points >= l1 - 1e-6))) { const cpt = l.players.find(x => x.pos === 'CPT'), flex = l.players.filter(x => x.pos !== 'CPT'); const ids = [cpt, ...flex].map(x => x && norm(x.name)); if (!cpt || flex.length !== 5 || ids.some(nk => !P.has(nk))) continue; L.push(ids.map(nk => P.get(nk))); }
    if (!L.length) continue; const own = new Map(Object.entries(S.fieldOwnership || {}).map(([k, v]) => [norm(k), +v / 100])), cap = new Map(Object.values(S.captainOwnership || {}).map(v => [norm(v.name), +v.capOwn / 100]));
    const m = {}; for (const [side, t] of [['favorite', fav], ['underdog', dog]]) for (const pos of ['K', 'DST', 'RB']) { const sel = p => p.team === t && p.pos === pos;
      m[side + ' ' + pos] = { top: L.reduce((a, x) => a + x.filter(sel).length, 0) / L.length, field: [...P.values()].filter(sel).reduce((a, p) => a + (own.get(norm(p.name)) || 0), 0) };
      if (pos === 'RB') m[side + ' RB as CPT'] = { top: L.reduce((a, x) => a + (sel(x[0]) ? 1 : 0), 0) / L.length, field: [...P.values()].filter(sel).reduce((a, p) => a + (cap.get(norm(p.name)) || 0), 0) }; }
    g.metrics.push(m); }
  if (!g.metrics.length) continue; g.m = {}; for (const k of Object.keys(g.metrics[0])) g.m[k] = { top: avg(g.metrics.map(x => x[k].top)), field: avg(g.metrics.map(x => x[k].field)) }; games.push(g);
}
const boot = (Gs, stat, B = 2000) => { const e = stat(Gs); if (Gs.length < 2) return { mean: e, se: null, games: Gs.length }; const bs = []; for (let b = 0; b < B; b++) { const S = Array.from({ length: Gs.length }, () => Gs[Math.floor(rnd() * Gs.length)]); bs.push(stat(S)); } return { mean: e, se: sdv(bs), games: Gs.length }; };
const KEYS = ['favorite K', 'favorite DST', 'favorite RB', 'favorite RB as CPT', 'underdog K', 'underdog DST', 'underdog RB', 'underdog RB as CPT'];
const summarize = Gs => Object.fromEntries(KEYS.map(k => [k, Gs.length ? { top: boot(Gs, S => avg(S.map(g => g.m[k].top))), field: boot(Gs, S => avg(S.map(g => g.m[k].field))), shift: boot(Gs, S => avg(S.map(g => g.m[k].top - g.m[k].field))) } : null]));
const shifts = { all: { games: games.length, ...summarize(games) }, byCell: {}, byCellAndTotal: {}, byResult: {} };
for (const c of CELLS) { const Gs = games.filter(g => g.cell === c); shifts.byCell[c] = { games: Gs.length, ...summarize(Gs) }; for (const hi of [true, false]) { const H2 = Gs.filter(g => g.totalHigh === hi); shifts.byCellAndTotal[c + ' / ' + (hi ? 'high' : 'low')] = { games: H2.length }; } }
for (const [k, sel] of [['favorite wins', g => g.favMargin > 0], ['underdog wins', g => g.favMargin < 0], ['high total', g => g.totalHigh], ['low total', g => !g.totalHigh]]) { const Gs = games.filter(sel); shifts.byResult[k] = { games: Gs.length, ...summarize(Gs) }; }
// ---- Probability-weighted expectation for TNF: each outcome cell's measured top share (or the favorite-wins / underdog-wins pool when a cell has fewer than 2 games).
const P0 = nflverse.spreadWindow.cells, expected = {};
for (const k of KEYS) { let top = 0, field = 0, w = 0; const used = {};
  for (const c of CELLS) { const p = P0[c].mean; const src = shifts.byCell[c].games >= 2 ? shifts.byCell[c] : shifts.byResult[c.startsWith('fav') ? 'favorite wins' : 'underdog wins']; used[c] = shifts.byCell[c].games >= 2 ? 'cell' : 'pooled side'; top += p * src[k].top.mean; field += p * src[k].field.mean; w += p; }
  expected[k] = { top: top / w, field: field / w, shift: (top - field) / w, source: used }; }
// Bootstrap se for the expectation: resample the nflverse window games (cell probabilities) and the Showdown games within each cell.
{ const W = G.filter(g => g.favBy >= 8 && g.favBy <= 11), byC = Object.fromEntries(CELLS.map(c => [c, games.filter(g => g.cell === c)])), bs = Object.fromEntries(KEYS.map(k => [k, []]));
  for (let b = 0; b < 1000; b++) { const Wb = Array.from({ length: W.length }, () => W[Math.floor(rnd() * W.length)]), pb = Object.fromEntries(CELLS.map(c => [c, Wb.filter(g => cellOf(g.favMargin) === c).length / Wb.length]));
    const gb = Object.fromEntries(CELLS.map(c => [c, Array.from({ length: byC[c].length }, () => byC[c][Math.floor(rnd() * byC[c].length)])]));
    for (const k of KEYS) { let v = 0, w = 0; for (const c of CELLS) { if (!gb[c].length) continue; v += pb[c] * avg(gb[c].map(g => g.m[k].top - g.m[k].field)); w += pb[c]; } bs[k].push(v / w); } }
  for (const k of KEYS) expected[k].shiftSe = sdv(bs[k]); }
const out = { builtAt: new Date().toISOString(), tnf, nflverse, showdownGames: games.map(g => ({ dg: g.dg, week: g.week, cell: g.cell, totalHigh: g.totalHigh, favBy: g.favBy })), shifts, expected, probabilityWeight: 'nflverse spreadWindow (favorite by 8 to 11)' };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1)); console.log(JSON.stringify({ tnf: { home: tnf.home, away: tnf.away, total: tnf.total, homeSpread: tnf.homeSpread, runFile: tnf.runFile, nflverse2026: tnf.nflverse2026Row }, nflverseGames: nflverse.spreadWindow.games, showdowns: games.length }));
