// Vale unit 1103CL-R: bias and range calibration of the published pre-lock NBA projections. Public data only; aggregates only.
// Usage: node unit-r.cjs <data dir with nba-box, nba-tonight, dk-results, nba-dk-*.json> <out.json>
const fs = require('fs'), zlib = require('zlib'), path = require('path');
const D = process.argv[2], OUT = process.argv[3], SPLIT = '2026-10-27';
const NIGHTS = []; for (let d = new Date('2026-10-20T12:00Z'); d <= new Date('2026-11-03T12:00Z'); d = new Date(+d + 864e5)) NIGHTS.push(d.toISOString().slice(0, 10));
const ESPN2DK = { GS: 'GSW', NO: 'NOP', SA: 'SAS', NY: 'NYK', UTAH: 'UTA', WSH: 'WAS' }, tm = a => ESPN2DK[a] || a;
const norm = s => String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const etNight = iso => new Date(Date.parse(iso) - 5 * 3600e3).toISOString().slice(0, 10);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length, sd = a => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
let seed = 1109; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
// ---- box scores (DK points and minutes)
const box = new Map(), games = new Map();
for (const f of fs.readdirSync(path.join(D, 'nba-box/2026-27'))) { const B = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(D, 'nba-box/2026-27', f))));
  for (const g of B.games) { const c = g.s.header.competitions[0], night = etNight(c.date); if (!NIGHTS.includes(night)) continue;
    const home = tm(c.competitors.find(x => x.homeAway === 'home').team.abbreviation), away = tm(c.competitors.find(x => x.homeAway === 'away').team.abbreviation), gk = night + '|' + [away, home].sort().join('-'); games.set(gk, { night, home, away });
    for (const tp of g.s.boxscore.players || []) { const team = tm(tp.team.abbreviation), st = tp.statistics[0], Lb = st.labels, ix = k => Lb.indexOf(k), pm = new Map();
      for (const a of st.athletes) { const s = a.stats || [], mins = parseFloat(s[ix('MIN')]) || 0, dnp = a.didNotPlay || !s.length || !(mins > 0); let dk = 0;
        if (!dnp) { const n = k => parseFloat(s[ix(k)]) || 0, PTS = n('PTS'), TPM = parseFloat(String(s[ix('3PT')]).split('-')[0]) || 0, REB = n('REB'), AST = n('AST'), STL = n('STL'), BLK = n('BLK'), TO = n('TO'); const ten = [PTS, REB, AST, STL, BLK].filter(x => x >= 10).length; dk = PTS + 0.5 * TPM + 1.25 * REB + 1.5 * AST + 2 * STL + 2 * BLK - 0.5 * TO + (ten >= 2 ? 1.5 : 0) + (ten >= 3 ? 3 : 0); }
        pm.set(norm(a.athlete.displayName), { dk, dnp, mins }); }
      box.set(night + '|' + team, { gk, players: pm }); } } }
// ---- pre-lock lines (nba-tonight); 'any' keeps lines read after the first tip as a sensitivity
const lines = new Map(), linesAny = new Map();
for (const night of NIGHTS) { const f = path.join(D, 'nba-tonight', night + '.json'); if (!fs.existsSync(f)) continue; const T = JSON.parse(fs.readFileSync(f));
  for (const g of T.games || []) { if (!g.line || g.line.total == null) continue; const gk = night + '|' + [tm(g.away.abbr), tm(g.home.abbr)].sort().join('-'), L = { total: g.line.total, homeSpread: g.line.spread, home: tm(g.home.abbr) }; linesAny.set(gk, L); if (Date.parse(T.readAt) < Date.parse(g.tip)) lines.set(gk, L); } }
// ---- pre-lock runs (unit F rule); Stokastic fields are never read
const parseET = (date, label) => { const m = /(\d+)\/(\d+) (\d+):(\d+)(AM|PM) Eastern/.exec(label || ''); if (!m) return null; const h = +m[3] % 12 + (m[5] === 'PM' ? 12 : 0); return date + 'T' + String(h).padStart(2, '0') + ':' + m[4]; };
const parseUpd = s => { const m = /(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+) (AM|PM)/.exec(s || ''); if (!m) return null; const h = +m[4] % 12 + (m[7] === 'PM' ? 12 : 0); return m[3] + '-' + m[1].padStart(2, '0') + '-' + m[2].padStart(2, '0') + 'T' + String(h).padStart(2, '0') + ':' + m[5]; };
const runs = [], audit = { inWindow: 0, preLock: 0, postLock: 0, liveFields: 0 };
for (const f of fs.readdirSync(D).filter(f => /^nba-dk-.*\.json$/.test(f))) { const j = JSON.parse(fs.readFileSync(path.join(D, f))); if (!NIGHTS.includes(j.slateDate)) continue; audit.inWindow++;
  const lock = parseET(j.slateDate, j.slateLabel), upd = parseUpd(j.simUpdated), live = (j.playerResults || []).some(p => 'liveProj' in p || 'liveFinal' in p);
  if (!lock || !upd || upd >= lock) { audit.postLock++; continue; } if (live) { audit.liveFields++; continue; } audit.preLock++;
  runs.push({ f, night: j.slateDate, upd, kind: j.isShowdown ? 'Showdown' : 'Turbo/Classic', players: j.playerResults.map(p => ({ nk: norm(p.name), team: p.team, pos: String(p.pos || '').split('/')[0], price: p.price, proj: +p.proj, floor: +p.floor, median: +p.median, ceiling: +p.ceiling })) }); }
const pick = new Map();
for (const r of runs) { const teams = new Set(r.players.map(p => p.team)); for (const t of teams) { const b = box.get(r.night + '|' + t); if (!b) continue; const G = games.get(b.gk); if (!teams.has(G.home) || !teams.has(G.away)) continue; const cur = pick.get(b.gk); if (!cur || r.upd > cur.upd) pick.set(b.gk, r); } }
// ---- Main-contest ownership per night (mean over that night's Classic contests)
const ownN = new Map(); for (const night of NIGHTS) { const f = path.join(D, 'dk-results', night + '.json'); if (!fs.existsSync(f)) continue; const R = JSON.parse(fs.readFileSync(f)), acc = new Map(), cnt = R.contests.filter(c => !c.isShowdown).length;
  for (const c of R.contests) if (!c.isShowdown) for (const [k, v] of Object.entries(c.summary.fieldOwnership || {})) acc.set(norm(k), (acc.get(norm(k)) || 0) + (+v) / cnt); ownN.set(night, acc); }
// ---- rows
const rows = [], outs = [], drop = { unmatched: 0, zeroProj: 0 };
for (const [gk, r] of pick) { const G = games.get(gk), L = lines.get(gk), LA = linesAny.get(gk);
  for (const p of r.players) { if (p.team !== G.home && p.team !== G.away) continue; if (!(p.proj > 0)) { drop.zeroProj++; continue; } const b = box.get(G.night + '|' + p.team), a = b && b.players.get(p.nk);
    if (!a) { drop.unmatched++; continue; } const sp = L ? (p.team === L.home ? L.homeSpread : -L.homeSpread) : null, spA = LA ? (p.team === LA.home ? LA.homeSpread : -LA.homeSpread) : null, own = ownN.has(G.night) ? ownN.get(G.night).get(p.nk) : undefined;
    const row = { night: G.night, gk, kind: r.kind, pos: p.pos, price: p.price, proj: p.proj, floor: p.floor, ceiling: p.ceiling, median: p.median, teamSpread: sp, teamSpreadAny: spA, total: L ? L.total : null, totalAny: LA ? LA.total : null, own: own == null ? null : own, inMainPool: ownN.has(G.night) ? own != null : null };
    if (a.dnp) outs.push(row); else rows.push({ ...row, actual: a.dk, mins: a.mins, res: a.dk - p.proj }); } }
// tiers
const tert = (a, k) => { const v = a.map(x => x[k]).sort((x, y) => x - y); return [v[Math.floor(v.length / 3)], v[Math.floor(2 * v.length / 3)]]; };
const salCuts = {}; for (const kd of ['Showdown', 'Turbo/Classic']) salCuts[kd] = tert(rows.filter(r => r.kind === kd), 'price');
const projCuts = tert(rows, 'proj');
const salT = r => { const [a, b] = salCuts[r.kind]; return r.price < a ? 'low' : r.price < b ? 'mid' : 'high'; }, projT = r => r.proj < projCuts[0] ? 'low' : r.proj < projCuts[1] ? 'mid' : 'high';
const spreadB = s => s == null ? null : s <= -8 ? 'favorite by 8+' : s < 0 ? 'favorite by <8' : s < 8 ? 'underdog by <8' : 'underdog by 8+';
const totalB = t => t == null ? null : t < 225 ? 'total < 225' : t <= 235 ? 'total 225–235' : 'total > 235';
const ownB = o => o == null ? 'not in Main pool' : o < 5 ? 'under 5%' : o < 15 ? '5–15%' : o < 30 ? '15–30%' : '30%+';
// ---- game-cluster bootstrap
const boot = (R, stat, B = 800) => { const gs = [...new Set(R.map(r => r.gk))], by = new Map(gs.map(g => [g, R.filter(r => r.gk === g)])), o = []; for (let b = 0; b < B; b++) { const S = []; for (let i = 0; i < gs.length; i++) S.push(...by.get(gs[Math.floor(rnd() * gs.length)])); const v = stat(S); if (Number.isFinite(v)) o.push(v); } return o.length > 2 ? sd(o) : null; };
const est = (R, f) => R.length ? { n: R.length, games: new Set(R.map(r => r.gk)).size, mean: mean(R.map(f)), se: boot(R, S => mean(S.map(f))) } : null;
const groupStats = (sel, R = rows) => { const o = {}; for (const g of [...new Set(R.map(sel))].filter(x => x != null).sort()) { const S = R.filter(r => sel(r) === g); o[g] = { bias: est(S, r => r.res), meanProj: mean(S.map(r => r.proj)), aboveCeiling: est(S, r => r.actual > r.ceiling ? 1 : 0), belowFloor: est(S, r => r.actual < r.floor ? 1 : 0), belowMedian: est(S, r => r.actual < r.median ? 1 : 0) }; } return o; };
const step1 = { overall: groupStats(() => 'all'), byPosition: groupStats(r => r.pos), bySalaryTier: groupStats(salT), byProjTier: groupStats(projT), bySlateType: groupStats(r => r.kind), bySpread: groupStats(r => spreadB(r.teamSpread)), byTotal: groupStats(r => totalB(r.total)), bySpreadInclLateLines: groupStats(r => spreadB(r.teamSpreadAny)), byTotalInclLateLines: groupStats(r => totalB(r.totalAny)) };
// ---- Step 2: projected but out
const step2 = { played: rows.length, out: outs.length, outRate: outs.length / (outs.length + rows.length), outMeanProj: outs.length ? mean(outs.map(r => r.proj)) : null, outByProjTier: Object.fromEntries(['low', 'mid', 'high'].map(t => [t, outs.filter(r => projT(r) === t).length])),
  outsPerGame: outs.length / pick.size, biasIncludingOuts: est([...rows, ...outs.map(r => ({ ...r, actual: 0, res: -r.proj }))], r => r.res) };
// ---- Step 4: hold-out corrections (fit nights through 10/27, check 10/28 onward); MSE gain on held-out rows with game-cluster se
const tr = rows.filter(r => r.night <= SPLIT), ho = rows.filter(r => r.night > SPLIT);
const lin = (R, xs) => { const X = R.map(r => [1, ...xs.map(f => f(r))]), y = R.map(r => r.actual), k = X[0].length, A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0))), b = Array.from({ length: k }, (_, i) => X.reduce((s, r, t) => s + r[i] * y[t], 0));
  const M = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < k; i++) { let p = i; for (let q = i + 1; q < k; q++) if (Math.abs(M[q][i]) > Math.abs(M[p][i])) p = q; [M[i], M[p]] = [M[p], M[i]]; for (let q = i + 1; q < k; q++) { const fct = M[q][i] / M[i][i]; for (let j = i; j <= k; j++) M[q][j] -= fct * M[i][j]; } }
  const x = Array(k).fill(0); for (let i = k - 1; i >= 0; i--) { let s = M[i][k]; for (let j = i + 1; j < k; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } const fn = r => [1, ...xs.map(f => f(r))].reduce((s, v, i) => s + v * x[i], 0); fn.params = { intercept: x[0], slope: x[1] }; return fn; };
const groupShift = (R, sel) => { const m = new Map(); for (const g of new Set(R.map(sel))) m.set(g, mean(R.filter(r => sel(r) === g).map(r => r.res))); const fn = r => r.proj + (m.get(sel(r)) || 0); fn.params = Object.fromEntries([...m].sort()); return fn; };
const CORR = { 'overall shift': R => { const m = mean(R.map(r => r.res)); const fn = r => r.proj + m; fn.params = { shift: m }; return fn; }, 'shift by position': R => groupShift(R, r => r.pos), 'shift by salary tier': R => groupShift(R, salT), 'shift by projection tier': R => groupShift(R, projT), 'linear in proj (a + b·proj)': R => lin(R, [r => r.proj]), 'shift by slate type': R => groupShift(R, r => r.kind) };
const step4 = { trainNights: [...new Set(tr.map(r => r.night))].sort(), holdoutNights: [...new Set(ho.map(r => r.night))].sort(), trainRows: tr.length, holdoutRows: ho.length, corrections: {} };
for (const [k, mk] of Object.entries(CORR)) { const f = mk(tr), g = r => (r.actual - r.proj) ** 2 - (r.actual - f(r)) ** 2; step4.corrections[k] = { params: f.params || null, heldOutBaselineMse: mean(ho.map(r => (r.actual - r.proj) ** 2)), heldOutMseGain: est(ho, g), heldOutBiasBefore: est(ho, r => r.res), heldOutBiasAfter: est(ho, r => r.actual - f(r)), trainFit: k === 'overall shift' ? mean(tr.map(r => r.res)) : null }; }
// Range calibration hold-out: does widening (scale the floor/ceiling distance from the median) fitted on train hold?
{ const rate = (R, s) => ({ above: mean(R.map(r => r.actual > r.median + s * (r.ceiling - r.median) ? 1 : 0)), below: mean(R.map(r => r.actual < r.median - s * (r.median - r.floor) ? 1 : 0)) }); let best = 1, bestErr = 1e9;
  for (let s = 0.7; s <= 1.5; s += 0.01) { const q = rate(tr, s), e = (q.above - 0.05) ** 2 + (q.below - 0.10) ** 2; if (e < bestErr) { bestErr = e; best = s; } }
  step4.rangeScale = { fittedScale: best, train: { before: rate(tr, 1), after: rate(tr, best) }, holdout: { aboveCeilingBefore: est(ho, r => r.actual > r.ceiling ? 1 : 0), belowFloorBefore: est(ho, r => r.actual < r.floor ? 1 : 0), aboveAfter: est(ho, r => r.actual > r.median + best * (r.ceiling - r.median) ? 1 : 0), belowAfter: est(ho, r => r.actual < r.median - best * (r.median - r.floor) ? 1 : 0) } }; }
// ---- Step 5: bias by Main-contest ownership band
const step5 = groupStats(r => ownB(r.own), rows.filter(r => r.inMainPool !== null));
const out = { builtAt: new Date().toISOString(), audit, gamesWithRun: pick.size, nights: [...new Set(rows.map(r => r.night))].sort(), rowsPlayed: rows.length, rowsOut: outs.length, dropped: drop, runsByKind: Object.fromEntries(['Showdown', 'Turbo/Classic'].map(k => [k, [...pick.values()].filter(r => r.kind === k).length])),
  gamesWithLines: [...pick.keys()].filter(k => lines.has(k)).length, gamesWithAnyLines: [...pick.keys()].filter(k => linesAny.has(k)).length, salCuts, projCuts, minutesAvailable: false, step1, step2, step4, step5 };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1)); console.log(JSON.stringify({ audit, games: pick.size, nights: out.nights, played: rows.length, out: outs.length, dropped: drop, byKind: out.runsByKind, lines: out.gamesWithLines }));
