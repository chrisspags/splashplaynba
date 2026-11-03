// Vale unit 1103CL-C: what wins prime-time DK NFL Showdowns, and how the pre-lock sim's winner rates compare.
// Inputs (public): nfl-dk-showdown-<dg>.json (results.playerResults, results.publicForecast), nfl-dk-results/<date>.json
// (contests: topFieldLineups, summary.topThresholds/fieldOwnership/captainOwnership/dupes), nfl-schedule.json (scores).
// Usage: node unit-c.cjs <data dir> <out.json>. Output is aggregates only (no lineups, users or entries).
const fs = require('fs'), path = require('path');
const D = process.argv[2], OUT = process.argv[3];
const J = f => JSON.parse(fs.readFileSync(path.join(D, f), 'utf8'));
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TEAM = { JAC: 'JAX', LA: 'LAR', WSH: 'WAS' }, T = t => { const u = String(t || '').toUpperCase(); return TEAM[u] || u; };
const SCHED = J('nfl-schedule.json');
const contestsByDG = {}; for (const f of fs.readdirSync(path.join(D, 'nfl-dk-results'))) for (const c of J('nfl-dk-results/' + f).contests) (contestsByDG[c.draftGroupId] = contestsByDG[c.draftGroupId] || []).push(c);
const POS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'], tierOf = s => s < 4000 ? 'under $4,000' : s < 8000 ? '$4,000-$7,999' : '$8,000+', TIERS = ['under $4,000', '$4,000-$7,999', '$8,000+'];
const games = [], players = [], coverage = { lineups: 0, lineupsUsed: 0, unmatchedLineups: 0, contests: 0 };
for (const f of fs.readdirSync(D).filter(f => /^nfl-dk-showdown-\d+\.json$/.test(f))) {
  const w = J(f), id = w.identity, R = w.results, cs = contestsByDG[id.rawSlateId]; if (!cs) continue;
  const week = +String(id.weekKey).replace(/.*W/, ''), P = new Map(); for (const p of Object.values(R.playerResults)) P.set(norm(p.name), { ...p, team: T(p.team) });
  const fr = new Map(); for (const r of (R.publicForecast && R.publicForecast.rows) || []) fr.set(norm(r.name), r);
  const teams = [...new Set([...P.values()].map(p => p.team))]; if (teams.length !== 2) continue;
  const sc = teams.map(t => (SCHED.teams[t] || {})[week] || {}); if (sc[0].score == null || sc[1].score == null) continue;
  const winner = sc[0].score > sc[1].score ? teams[0] : sc[1].score > sc[0].score ? teams[1] : null, total = sc[0].score + sc[1].score;
  const g = { dg: id.rawSlateId, week, date: id.date, teams, winner, total, contests: [] };
  const pAgg = new Map(); // nk -> {cpt:[], flex:[], capOwn:[], flexOwn:[]}
  for (const c of cs) {
    const S = c.summary || {}, line = S.topThresholds && S.topThresholds.top1pct;
    const tops = (c.topFieldLineups || []).filter(l => l.players && l.players.length === 6 && (line == null || l.points >= line - 1e-6));
    if (!tops.length) continue; coverage.contests++;
    const own = new Map(Object.entries(S.fieldOwnership || {}).map(([k, v]) => [norm(k), +v])), cap = new Map(Object.values(S.captainOwnership || {}).map(v => [norm(v.name), +v.capOwn]));
    const flexOwn = nk => Math.max(0, (own.get(nk) || 0) - (cap.get(nk) || 0)), capOwn = nk => cap.get(nk) || 0;
    const capRank = new Map([...P.keys()].sort((a, b) => capOwn(b) - capOwn(a)).map((nk, i) => [nk, i + 1]));
    const rk = r => r === 1 ? 'rank 1' : r <= 3 ? 'rank 2-3' : r <= 6 ? 'rank 4-6' : 'rank 7+';
    const L = []; const seen = new Map();
    for (const l of tops) { coverage.lineups++;
      const cpt = l.players.find(x => x.pos === 'CPT'), flex = l.players.filter(x => x.pos !== 'CPT');
      const ids = [cpt, ...flex].map(x => x && norm(x.name)); if (!cpt || flex.length !== 5 || ids.some(nk => !P.has(nk))) { coverage.unmatchedLineups++; continue; }
      coverage.lineupsUsed++; const key = ids[0] + '|' + ids.slice(1).sort().join(','); seen.set(key, (seen.get(key) || 0) + 1); L.push({ ids, key });
    }
    if (!L.length) continue;
    const m = { contestEntries: S.totalEntries, topLineups: L.length, top: {}, field: {} };
    const share = f => L.reduce((s, x) => s + f(x), 0) / L.length;
    const pc = nk => P.get(nk);
    for (const p of POS) { m.top['cptPos ' + p] = share(x => pc(x.ids[0]).pos === p ? 1 : 0); }
    m.top['cpt on game winner'] = winner ? share(x => pc(x.ids[0]).team === winner ? 1 : 0) : null;
    for (const t of TIERS) m.top['cpt tier ' + t] = share(x => tierOf(pc(x.ids[0]).salary) === t ? 1 : 0);
    for (const b of ['rank 1', 'rank 2-3', 'rank 4-6', 'rank 7+']) m.top['cpt own ' + b] = share(x => rk(capRank.get(x.ids[0])) === b ? 1 : 0);
    m.top['K in FLEX (count)'] = share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'K').length);
    m.top['DST in FLEX (count)'] = share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'DST').length);
    m.top['K or DST anywhere (count)'] = share(x => x.ids.filter(nk => pc(nk).pos === 'K' || pc(nk).pos === 'DST').length);
    m.top['K in FLEX on captain team (count)'] = share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'K' && pc(nk).team === pc(x.ids[0]).team).length);
    m.top['K in FLEX on other team (count)'] = share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'K' && pc(nk).team !== pc(x.ids[0]).team).length);
    m.top['DST in FLEX on captain team (count)'] = share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'DST' && pc(nk).team === pc(x.ids[0]).team).length);
    m.top['DST in FLEX on other team (count)'] = share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'DST' && pc(nk).team !== pc(x.ids[0]).team).length);
    m.top['K in FLEX on game winner (count)'] = winner ? share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'K' && pc(nk).team === winner).length) : null;
    m.top['DST in FLEX on game winner (count)'] = winner ? share(x => x.ids.slice(1).filter(nk => pc(nk).pos === 'DST' && pc(nk).team === winner).length) : null;
    m.top['share of slots on game winner'] = winner ? share(x => x.ids.filter(nk => pc(nk).team === winner).length / 6) : null;
    for (const s of ['5-1', '4-2', '3-3']) m.top['split ' + s] = share(x => { const a = x.ids.filter(nk => pc(nk).team === teams[0]).length, k = Math.max(a, 6 - a) + '-' + Math.min(a, 6 - a); return k === s ? 1 : 0; });
    m.top['split 5-1 toward captain team'] = share(x => x.ids.filter(nk => pc(nk).team === pc(x.ids[0]).team).length === 5 ? 1 : 0);
    m.top['salary left ($)'] = share(x => 50000 - (pc(x.ids[0]).cptSalary || 1.5 * pc(x.ids[0]).salary) - x.ids.slice(1).reduce((s, nk) => s + pc(nk).salary, 0));
    m.top['salary left <= $500'] = share(x => (50000 - (pc(x.ids[0]).cptSalary || 1.5 * pc(x.ids[0]).salary) - x.ids.slice(1).reduce((s, nk) => s + pc(nk).salary, 0)) <= 500 ? 1 : 0);
    m.top['sum of field own (%)'] = share(x => capOwn(x.ids[0]) + x.ids.slice(1).reduce((s, nk) => s + flexOwn(nk), 0));
    m.top['entries in a roster repeated within the top set'] = L.filter(x => seen.get(x.key) > 1).length / L.length;
    // field-implied (from the contest's own ownership marginals)
    const capTot = [...P.keys()].reduce((s, nk) => s + capOwn(nk), 0) || 100, ownTot = [...P.keys()].reduce((s, nk) => s + (own.get(nk) || 0), 0) || 600;
    for (const p of POS) m.field['cptPos ' + p] = [...P].filter(([, v]) => v.pos === p).reduce((s, [nk]) => s + capOwn(nk), 0) / capTot;
    m.field['cpt on game winner'] = winner ? [...P].filter(([, v]) => v.team === winner).reduce((s, [nk]) => s + capOwn(nk), 0) / capTot : null;
    for (const t of TIERS) m.field['cpt tier ' + t] = [...P].filter(([, v]) => tierOf(v.salary) === t).reduce((s, [nk]) => s + capOwn(nk), 0) / capTot;
    for (const b of ['rank 1', 'rank 2-3', 'rank 4-6', 'rank 7+']) m.field['cpt own ' + b] = [...P.keys()].filter(nk => rk(capRank.get(nk)) === b).reduce((s, nk) => s + capOwn(nk), 0) / capTot;
    m.field['K in FLEX (count)'] = [...P].filter(([, v]) => v.pos === 'K').reduce((s, [nk]) => s + flexOwn(nk), 0) / 100;
    m.field['DST in FLEX (count)'] = [...P].filter(([, v]) => v.pos === 'DST').reduce((s, [nk]) => s + flexOwn(nk), 0) / 100;
    m.field['K or DST anywhere (count)'] = [...P].filter(([, v]) => v.pos === 'K' || v.pos === 'DST').reduce((s, [nk]) => s + (own.get(nk) || 0), 0) / 100;
    m.field['K in FLEX on game winner (count)'] = winner ? [...P].filter(([, v]) => v.pos === 'K' && v.team === winner).reduce((s, [nk]) => s + flexOwn(nk), 0) / 100 : null;
    m.field['DST in FLEX on game winner (count)'] = winner ? [...P].filter(([, v]) => v.pos === 'DST' && v.team === winner).reduce((s, [nk]) => s + flexOwn(nk), 0) / 100 : null;
    m.field['share of slots on game winner'] = winner ? [...P].filter(([, v]) => v.team === winner).reduce((s, [nk]) => s + (own.get(nk) || 0), 0) / ownTot : null;
    m.field['salary left ($)'] = 50000 - [...P].reduce((s, [nk, v]) => s + flexOwn(nk) / 100 * v.salary + capOwn(nk) / 100 * (v.cptSalary || 1.5 * v.salary), 0);
    m.field['sum of field own (%)'] = [...P.keys()].reduce((s, nk) => s + flexOwn(nk) ** 2 / 100 + capOwn(nk) ** 2 / 100, 0);
    if (S.dupes && S.dupes.totalEntriesHashed) m.field['entries in a roster repeated within the top set'] = 1 - Math.max(0, S.dupes.uniqueLineups - S.dupes.lineupsRepeated) / S.dupes.totalEntriesHashed;
    g.contests.push(m);
    for (const nk of P.keys()) { const a = pAgg.get(nk) || pAgg.set(nk, { cpt: [], flex: [], capOwn: [], flexOwn: [] }).get(nk);
      a.cpt.push(share(x => x.ids[0] === nk ? 1 : 0)); a.flex.push(share(x => x.ids.slice(1).includes(nk) ? 1 : 0)); a.capOwn.push(capOwn(nk) / 100); a.flexOwn.push(flexOwn(nk) / 100); }
  }
  if (!g.contests.length) continue;
  const avg = a => a.reduce((s, x) => s + x, 0) / a.length;
  g.top = {}; g.field = {}; for (const k of Object.keys(g.contests[0].top)) { const v = g.contests.map(c => c.top[k]).filter(x => x != null); g.top[k] = v.length ? avg(v) : null;
    const u = g.contests.map(c => c.field[k]).filter(x => x != null); g.field[k] = u.length ? avg(u) : null; }
  games.push(g);
  for (const [nk, a] of pAgg) { const p = P.get(nk), r = fr.get(nk) || {};
    players.push({ game: g.dg, week, pos: p.pos, team: p.team, onWinner: winner ? p.team === winner : null, total, salary: p.salary, cptSalary: p.cptSalary, proj: p.proj, ceiling: p.ceiling, stdDev: p.stdDev,
      exposure: p.exposure / 100, simCpt: p.cptWinnerExposure != null ? p.cptWinnerExposure / 100 : null, simFlex: p.flexWinnerExposure != null ? p.flexWinnerExposure / 100 : null, candCpt: p.cptExposure != null ? p.cptExposure / 100 : null,
      implied: r.implied != null ? +r.implied : null, gameTotalLine: r.total != null ? +r.total : null, spread: r.spread != null ? +r.spread : null,
      realCpt: avg(a.cpt), realFlex: avg(a.flex), fieldCpt: avg(a.capOwn), fieldFlex: avg(a.flexOwn) }); }
}
// ---- game bootstrap ----
const G = games.map(g => g.dg), gIx = new Map(G.map((d, i) => [d, i]));
let seed = 1103; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const BOOT = Array.from({ length: 2000 }, () => { const c = new Float64Array(G.length); for (let i = 0; i < G.length; i++) c[Math.floor(rnd() * G.length)]++; return c; });
const sdv = a => { const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
function gmean(vals) { // vals: [{g, v}] game-level values -> mean and game-bootstrap se
  const v = vals.filter(x => x.v != null && Number.isFinite(x.v)); if (!v.length) return null;
  const est = w => { let s = 0, n = 0; for (const x of v) { const k = w ? w[gIx.get(x.g)] : 1; s += k * x.v; n += k; } return n ? s / n : null; };
  return { games: v.length, mean: est(), se: sdv(BOOT.map(est).filter(x => x != null)) };
}
const anatomy = {}; for (const k of Object.keys(games[0].top)) {
  anatomy[k] = { top: gmean(games.map(g => ({ g: g.dg, v: g.top[k] }))), field: gmean(games.map(g => ({ g: g.dg, v: g.field[k] }))),
    topMinusField: gmean(games.map(g => ({ g: g.dg, v: g.top[k] != null && g.field[k] != null ? g.top[k] - g.field[k] : null }))) }; }
// ---- part 2: sim winner rates vs real top-lineup rates, per player-game (game-cluster bootstrap) ----
function pmean(rows, f) { const v = rows.map(r => ({ g: r.game, v: f(r) })).filter(x => x.v != null && Number.isFinite(x.v)); if (!v.length) return null;
  const est = w => { let s = 0, n = 0; for (const x of v) { const k = w ? w[gIx.get(x.g)] : 1; s += k * x.v; n += k; } return n ? s / n : null; };
  return { n: v.length, games: new Set(v.map(x => x.g)).size, mean: est(), se: sdv(BOOT.slice(0, 1000).map(est).filter(x => x != null)) }; }
function ols(rows, yf, xf) { const v = rows.filter(r => yf(r) != null && xf(r) != null);
  const fit = w => { let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0; for (const r of v) { const k = w ? w[gIx.get(r.game)] : 1; if (!k) continue; const x = xf(r), y = yf(r); sw += k; sx += k * x; sy += k * y; sxx += k * x * x; sxy += k * x * y; }
    const b = (sxy - sx * sy / sw) / (sxx - sx * sx / sw); return { a: sy / sw - b * sx / sw, b }; };
  const f0 = fit(), bs = BOOT.slice(0, 1000).map(fit); return { n: v.length, intercept: f0.a, interceptSe: sdv(bs.map(x => x.a)), slope: f0.b, slopeSe: sdv(bs.map(x => x.b)) }; }
const terc = [...new Set(games.map(g => g.total))].sort((a, b) => a - b), q1 = games.map(g => g.total).sort((a, b) => a - b)[Math.floor(games.length / 3)], q2 = games.map(g => g.total).sort((a, b) => a - b)[Math.floor(2 * games.length / 3)];
const totT = t => t < q1 ? 'low total' : t < q2 ? 'mid total' : 'high total';
const sim = { calibration: {}, bias: {}, totalTercileCuts: [q1, q2] };
for (const [role, y, x, fx] of [['CPT', 'realCpt', 'simCpt', 'fieldCpt'], ['FLEX', 'realFlex', 'simFlex', 'fieldFlex']]) {
  sim.calibration[role] = { simWinner: ols(players, r => r[y], r => r[x]), fieldOwn: ols(players, r => r[y], r => r[fx]) };
  const groups = { all: () => true, ...Object.fromEntries(POS.map(p => ['pos ' + p, r => r.pos === p])),
    ...Object.fromEntries(TIERS.map(t => ['salary ' + t, r => tierOf(r.salary) === t])),
    'game winner team': r => r.onWinner === true, 'game loser team': r => r.onWinner === false,
    'low total': r => totT(r.total) === 'low total', 'mid total': r => totT(r.total) === 'mid total', 'high total': r => totT(r.total) === 'high total' };
  sim.bias[role] = {}; for (const [k, sel] of Object.entries(groups)) { const s = players.filter(sel);
    sim.bias[role][k] = { real: pmean(s, r => r[y]), sim: pmean(s, r => r[x]), field: pmean(s, r => r[fx]), realMinusSim: pmean(s, r => r[x] == null ? null : r[y] - r[x]), realMinusField: pmean(s, r => r[y] - r[fx]) }; }
}
// Game script by position (within a game the shares sum to a constant, so script only means something inside a position).
sim.byPositionScript = {};
for (const [role, y, x, fx] of [['CPT', 'realCpt', 'simCpt', 'fieldCpt'], ['FLEX', 'realFlex', 'simFlex', 'fieldFlex']]) { sim.byPositionScript[role] = {};
  for (const p of POS) { const o = sim.byPositionScript[role][p] = {};
    for (const [k, sel] of [['game winner team', r => r.onWinner === true], ['game loser team', r => r.onWinner === false], ['low total', r => totT(r.total) === 'low total'], ['mid total', r => totT(r.total) === 'mid total'], ['high total', r => totT(r.total) === 'high total']]) {
      const s2 = players.filter(r => r.pos === p && sel(r)); o[k] = { real: pmean(s2, r => r[y]), sim: pmean(s2, r => r[x]), field: pmean(s2, r => r[fx]), realMinusSim: pmean(s2, r => r[x] == null ? null : r[y] - r[x]) }; } } }
// K and DST per game (game-level counts only): top-lineup K/DST FLEX count vs the field and the sim's FLEX winner rate.
sim.kdstByGame = games.map(g => ({ dg: g.dg, week: g.week, date: g.date, topK: g.top['K in FLEX (count)'], fieldK: g.field['K in FLEX (count)'], topDST: g.top['DST in FLEX (count)'], fieldDST: g.field['DST in FLEX (count)'],
  simK: players.filter(r => r.game === g.dg && r.pos === 'K').reduce((s, r) => s + (r.simFlex || 0), 0), simDST: players.filter(r => r.game === g.dg && r.pos === 'DST').reduce((s, r) => s + (r.simFlex || 0), 0) }));
// ---- part 3: pre-lock predictors of top-lineup inclusion (any slot), fit weeks 1-6, held out weeks 7-9 ----
function lsq(X, y) { const k = X[0].length, A = Array.from({ length: k }, () => new Float64Array(k)), b = new Float64Array(k);
  for (let i = 0; i < X.length; i++) for (let p = 0; p < k; p++) { b[p] += X[i][p] * y[i]; for (let q = 0; q < k; q++) A[p][q] += X[i][p] * X[i][q]; }
  for (let p = 0; p < k; p++) A[p][p] += 1e-6;
  for (let c = 0; c < k; c++) { let piv = c; for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r; [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
    for (let r = 0; r < k; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let q = c; q < k; q++) A[r][q] -= f * A[c][q]; b[r] -= f * b[c]; } }
  return Array.from(b, (v, i) => v / A[i][i]); }
const posD = r => POS.map(p => r.pos === p ? 1 : 0);
const MODELS = { 'position only': r => posD(r), 'position + proj + salary': r => [...posD(r), r.proj, r.salary / 1000],
  'position + sim winner exposure': r => [...posD(r), r.exposure], 'position + proj + salary + sim winner exposure': r => [...posD(r), r.proj, r.salary / 1000, r.exposure],
  'all pre-lock fields': r => [...posD(r), r.proj, r.salary / 1000, r.ceiling, r.stdDev, r.exposure, r.simCpt || 0, r.simFlex || 0, r.implied || 0, r.gameTotalLine || 0] };
const yAny = r => r.realCpt + r.realFlex;
const spearman = (a, b) => { const rk = v => { const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]), r = new Array(v.length); let i = 0; while (i < o.length) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2; i = j + 1; } return r; };
  const x = rk(a), y = rk(b), n = a.length, mx = (n - 1) / 2; let s = 0, sx = 0, sy = 0; for (let i = 0; i < n; i++) { s += (x[i] - mx) * (y[i] - mx); sx += (x[i] - mx) ** 2; sy += (y[i] - mx) ** 2; } return s / Math.sqrt(sx * sy); };
const tr = players.filter(r => r.week <= 6), ho = players.filter(r => r.week >= 7), pred = {};
for (const [name, f] of Object.entries(MODELS)) {
  const X = tr.map(r => [1, ...f(r)]), beta = lsq(X, tr.map(yAny)), yhat = r => [1, ...f(r)].reduce((s, x, i) => s + x * beta[i], 0);
  const mse = s => s.reduce((a, r) => a + (yhat(r) - yAny(r)) ** 2, 0) / s.length, varY = s => { const m = s.reduce((a, r) => a + yAny(r), 0) / s.length; return s.reduce((a, r) => a + (yAny(r) - m) ** 2, 0) / s.length; };
  const byGame = {}; for (const r of ho) (byGame[r.game] = byGame[r.game] || []).push(r);
  const rhos = Object.values(byGame).map(s => spearman(s.map(yhat), s.map(yAny)));
  const mae = s => s.reduce((a, r) => a + Math.abs(yhat(r) - yAny(r)), 0) / s.length;
  pred[name] = { trainRows: tr.length, holdoutRows: ho.length, holdoutGames: Object.keys(byGame).length, trainR2: 1 - mse(tr) / varY(tr), holdoutR2: 1 - mse(ho) / varY(ho), holdoutMAE: mae(ho),
    holdoutWithinGameSpearman: rhos.reduce((a, b) => a + b, 0) / rhos.length, holdoutSpearmanByGame: rhos };
}
// K / DST: which pre-lock fields track their real FLEX share (all weeks; small samples)
const kd = {}; for (const p of ['K', 'DST']) { const s = players.filter(r => r.pos === p); kd[p] = { rows: s.length, games: new Set(s.map(r => r.game)).size, spearmanWithRealFlex: {} };
  for (const k of ['proj', 'ceiling', 'stdDev', 'salary', 'exposure', 'simFlex', 'fieldFlex', 'implied', 'gameTotalLine', 'spread']) { const v = s.filter(r => r[k] != null); if (v.length > 10) kd[p].spearmanWithRealFlex[k] = { n: v.length, rho: spearman(v.map(r => r[k]), v.map(r => r.realFlex)) }; }
  kd[p].withinGameWinnerVsLoser = { onWinner: pmean(s.filter(r => r.onWinner === true), r => r.realFlex), onLoser: pmean(s.filter(r => r.onWinner === false), r => r.realFlex),
    simOnWinner: pmean(s.filter(r => r.onWinner === true), r => r.simFlex), simOnLoser: pmean(s.filter(r => r.onWinner === false), r => r.simFlex) }; }
const out = { builtAt: new Date().toISOString(), games: games.map(g => ({ dg: g.dg, week: g.week, date: g.date, contests: g.contests.length, topLineups: g.contests.reduce((s, c) => s + c.topLineups, 0), total: g.total })),
  coverage, playerGames: players.length, anatomy, sim, predictors: pred, kdst: kd };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(JSON.stringify({ games: games.length, playerGames: players.length, coverage }));
