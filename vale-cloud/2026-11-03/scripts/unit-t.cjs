// Vale unit 1103CL-T: is unit R's post-10/28 over-projection a model shift or concentration (players, games, slate mix)?
// Public data only; aggregates only (no player names in the output). Usage: node unit-t.cjs <data dir> <out.json>
const fs = require('fs'), zlib = require('zlib'), path = require('path');
const D = process.argv[2], OUT = process.argv[3], SPLIT = '2026-10-28';
const NIGHTS = []; for (let d = new Date('2026-10-20T12:00Z'); d <= new Date('2026-11-03T12:00Z'); d = new Date(+d + 864e5)) NIGHTS.push(d.toISOString().slice(0, 10));
const ESPN2DK = { GS: 'GSW', NO: 'NOP', SA: 'SAS', NY: 'NYK', UTAH: 'UTA', WSH: 'WAS' }, tm = a => ESPN2DK[a] || a;
const norm = s => String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const etNight = iso => new Date(Date.parse(iso) - 5 * 3600e3).toISOString().slice(0, 10);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length, sd = a => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
let seed = 1110; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
// ---- box scores: per-game DK points, and every player's game log (for season-to-date means)
const box = new Map(), games = new Map(), log = new Map(); // log: team|nk -> [{night, dk}]
for (const f of fs.readdirSync(path.join(D, 'nba-box/2026-27'))) { const B = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(D, 'nba-box/2026-27', f))));
  for (const g of B.games) { const c = g.s.header.competitions[0], night = etNight(c.date); if (!NIGHTS.includes(night)) continue;
    const home = tm(c.competitors.find(x => x.homeAway === 'home').team.abbreviation), away = tm(c.competitors.find(x => x.homeAway === 'away').team.abbreviation), gk = night + '|' + [away, home].sort().join('-'); games.set(gk, { night, home, away, src: 'box' });
    for (const tp of g.s.boxscore.players || []) { const team = tm(tp.team.abbreviation), st = tp.statistics[0], Lb = st.labels, ix = k => Lb.indexOf(k), pm = new Map();
      for (const a of st.athletes) { const s = a.stats || [], mins = parseFloat(s[ix('MIN')]) || 0, dnp = a.didNotPlay || !s.length || !(mins > 0); let dk = 0;
        if (!dnp) { const n = k => parseFloat(s[ix(k)]) || 0, PTS = n('PTS'), TPM = parseFloat(String(s[ix('3PT')]).split('-')[0]) || 0, REB = n('REB'), AST = n('AST'), STL = n('STL'), BLK = n('BLK'), TO = n('TO'); const ten = [PTS, REB, AST, STL, BLK].filter(x => x >= 10).length; dk = PTS + 0.5 * TPM + 1.25 * REB + 1.5 * AST + 2 * STL + 2 * BLK - 0.5 * TO + (ten >= 2 ? 1.5 : 0) + (ten >= 3 ? 3 : 0); }
        const nk = norm(a.athlete.displayName); pm.set(nk, { dk, dnp }); if (!dnp) { const k = team + '|' + nk; if (!log.has(k)) log.set(k, []); log.get(k).push({ night, dk }); } }
      box.set(night + '|' + team, { gk, players: pm }); } } }
const boxNights = new Set([...games.values()].map(g => g.night));
// Nights after the box file: DK's own playerActuals from dk-results (Main contests). fpts 0 is treated as did-not-play.
const dkAct = new Map(); for (const night of NIGHTS) { if (boxNights.has(night)) continue; const f = path.join(D, 'dk-results', night + '.json'); if (!fs.existsSync(f)) continue; const R = JSON.parse(fs.readFileSync(f)), m = new Map();
  for (const c of R.contests) for (const v of Object.values(c.playerActuals || {})) m.set(norm(v.name), +v.fpts); dkAct.set(night, m); }
// ---- pre-lock runs (unit F rule), with structure fields
const parseET = (date, label) => { const m = /(\d+)\/(\d+) (\d+):(\d+)(AM|PM) Eastern/.exec(label || ''); if (!m) return null; const h = +m[3] % 12 + (m[5] === 'PM' ? 12 : 0); return date + 'T' + String(h).padStart(2, '0') + ':' + m[4]; };
const parseUpd = s => { const m = /(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+) (AM|PM)/.exec(s || ''); if (!m) return null; const h = +m[4] % 12 + (m[7] === 'PM' ? 12 : 0); return m[3] + '-' + m[1].padStart(2, '0') + '-' + m[2].padStart(2, '0') + 'T' + String(h).padStart(2, '0') + ':' + m[5]; };
const minsBefore = (lock, upd) => (Date.parse(lock + ':00Z') - Date.parse(upd + ':00Z')) / 60000;
const runs = [];
for (const f of fs.readdirSync(D).filter(f => /^nba-dk-.*\.json$/.test(f))) { const j = JSON.parse(fs.readFileSync(path.join(D, f))); if (!NIGHTS.includes(j.slateDate)) continue;
  const lock = parseET(j.slateDate, j.slateLabel), upd = parseUpd(j.simUpdated), live = (j.playerResults || []).some(p => 'liveProj' in p || 'liveFinal' in p); if (!lock || !upd || upd >= lock || live) continue;
  const k = j.simKernel || {}; runs.push({ f, night: j.slateDate, upd, lock, kind: j.isShowdown ? 'Showdown' : 'Turbo', minutesBeforeLock: minsBefore(lock, upd), numSim: j.numSim, metricBasis: j.metricBasis, keys: Object.keys(j).sort().join(','), playerKeys: Object.keys((j.playerResults || [])[0] || {}).filter(x => !/stok/i.test(x)).sort().join(','),
    kernel: { kernel: k.kernel, tables: k.tables, created: k.created, inputs: k.inputs, fit: (k.fit || []).join('+'), build: k.build, stepMinutesTable: k.stepMinutes && k.stepMinutes.table, stepMinutesPassed: k.stepMinutes && k.stepMinutes.passed, stepMinutesConverted: k.stepMinutes && k.stepMinutes.converted },
    lineupNewsChanged: j.lineupNews ? j.lineupNews.changed : null, lineupNewsOut: j.lineupNews ? j.lineupNews.out : null, players: j.playerResults.map(p => ({ nk: norm(p.name), team: p.team, price: p.price, proj: +p.proj, floor: +p.floor, median: +p.median, ceiling: +p.ceiling })) }); }
const pick = new Map();
for (const r of runs) { const teams = new Set(r.players.map(p => p.team));
  if (boxNights.has(r.night)) { for (const t of teams) { const b = box.get(r.night + '|' + t); if (!b) continue; const G = games.get(b.gk); if (!teams.has(G.home) || !teams.has(G.away)) continue; const cur = pick.get(b.gk); if (!cur || r.upd > cur.upd) pick.set(b.gk, r); } }
  else if (dkAct.has(r.night)) { // no box yet: a Showdown run is one game; a Turbo run is one unit (its games can't be split without a box score)
    const gk = r.night + '|' + (r.kind === 'Showdown' ? [...teams].sort().join('-') : 'turbo:' + r.f); if (!games.has(gk)) games.set(gk, { night: r.night, src: 'dk-results' }); const cur = pick.get(gk); if (!cur || r.upd > cur.upd) pick.set(gk, r); } }
// For dk-results nights, a player can sit in a Showdown and a Turbo run of the same game: keep one row per player-night (Showdown first, latest).
const rows = [], seenDk = new Set(), drop = { dnpOrZero: 0, unmatched: 0 };
const stdMean = (team, nk, night) => { const a = (log.get(team + '|' + nk) || []).filter(x => x.night < night); return a.length >= 2 ? { mean: mean(a.map(x => x.dk)), n: a.length } : null; };
for (const [gk, r] of [...pick].sort((a, b) => (a[1].kind === 'Showdown' ? 0 : 1) - (b[1].kind === 'Showdown' ? 0 : 1))) { const G = games.get(gk);
  for (const p of r.players) { if (!(p.proj > 0)) continue; let actual = null;
    if (G.src === 'box') { if (p.team !== G.home && p.team !== G.away) continue; const b = box.get(G.night + '|' + p.team), a = b && b.players.get(p.nk); if (!a) { drop.unmatched++; continue; } if (a.dnp) { drop.dnpOrZero++; continue; } actual = a.dk; }
    else { const k = G.night + '|' + p.team + '|' + p.nk; if (seenDk.has(k)) continue; const v = dkAct.get(G.night).get(p.nk); if (v == null) { drop.unmatched++; continue; } if (v === 0) { drop.dnpOrZero++; continue; } seenDk.add(k); actual = v; }
    const s = stdMean(p.team, p.nk, G.night); rows.push({ night: G.night, gk, src: G.src, kind: r.kind, run: r.f, newsChanged: r.lineupNewsChanged || 0, newsOut: r.lineupNewsOut || 0, pid: p.team + '|' + p.nk, price: p.price, proj: p.proj, floor: p.floor, ceiling: p.ceiling, actual, res: actual - p.proj, std: s ? s.mean : null, post: G.night >= SPLIT }); } }
// expensive = top salary tertile within slate type
const cuts = {}; for (const k of ['Showdown', 'Turbo']) { const v = rows.filter(r => r.kind === k).map(r => r.price).sort((a, b) => a - b); cuts[k] = v[Math.floor(2 * v.length / 3)]; } for (const r of rows) r.expensive = r.price >= cuts[r.kind];
const boot = (R, stat, key = 'gk', B = 800) => { const gs = [...new Set(R.map(r => r[key]))], by = new Map(gs.map(g => [g, R.filter(r => r[key] === g)])), o = []; for (let b = 0; b < B; b++) { const S = []; for (let i = 0; i < gs.length; i++) S.push(...by.get(gs[Math.floor(rnd() * gs.length)])); const v = stat(S); if (Number.isFinite(v)) o.push(v); } return o.length > 2 ? sd(o) : null; };
const est = (R, f, key) => R.length ? { n: R.length, games: new Set(R.map(r => r.gk)).size, mean: mean(R.map(f)), se: boot(R, S => mean(S.map(f)), key) } : null;
const PRE = rows.filter(r => !r.post), POST = rows.filter(r => r.post), POSTBOX = POST.filter(r => r.src === 'box');
// ---- Step 1: concentration of the post-10/28 miss (box nights only, matching unit R; dk-results nights reported separately)
const conc = (R, key) => { const m = new Map(); for (const r of R) m.set(r[key], (m.get(r[key]) || 0) + r.res); const net = R.reduce((s, r) => s + r.res, 0), v = [...m.values()].sort((a, b) => a - b); let c = 0, k50 = null, k80 = null; for (let i = 0; i < v.length; i++) { c += v[i]; if (k50 == null && c <= 0.5 * net) k50 = i + 1; if (k80 == null && c <= 0.8 * net) k80 = i + 1; }
  return { units: m.size, net, unitsTo50: k50, unitsTo80: k80, top1Share: v[0] / net, top2Share: (v[0] + v[1]) / net, top5Share: v.slice(0, 5).reduce((a, b) => a + b, 0) / net, positiveUnits: v.filter(x => x > 0).length }; };
// Benchmark: the same counts when residual signs are random (no shift; same magnitudes), to show what "concentrated" means here.
const benchConc = (R, key, reps = 300) => { const o50 = [], o80 = []; for (let i = 0; i < reps; i++) { const shift = mean(R.map(x => x.res)); const P = R.map((r, j) => ({ ...r, res: (rnd() < 0.5 ? -1 : 1) * (r.res - shift) + shift })); const c = conc(P, key); if (c.unitsTo50) o50.push(c.unitsTo50); if (c.unitsTo80) o80.push(c.unitsTo80); } return { unitsTo50: mean(o50), unitsTo80: mean(o80) }; };
const step1 = { postBox: { rows: POSTBOX.length, bias: est(POSTBOX, r => r.res), byPlayer: conc(POSTBOX, 'pid'), byGame: conc(POSTBOX, 'gk'), benchmarkSameShiftSpreadEvenly: { byPlayer: benchConc(POSTBOX, 'pid'), byGame: benchConc(POSTBOX, 'gk') },
  expensive: est(POSTBOX.filter(r => r.expensive), r => r.res), notExpensive: est(POSTBOX.filter(r => !r.expensive), r => r.res),
  withoutTop2Games: (() => { const m = new Map(); for (const r of POSTBOX) m.set(r.gk, (m.get(r.gk) || 0) + r.res); const worst = [...m].sort((a, b) => a[1] - b[1]).slice(0, 2).map(x => x[0]); return est(POSTBOX.filter(r => !worst.includes(r.gk)), r => r.res); })(),
  withoutTop10Players: (() => { const m = new Map(); for (const r of POSTBOX) m.set(r.pid, (m.get(r.pid) || 0) + r.res); const worst = new Set([...m].sort((a, b) => a[1] - b[1]).slice(0, 10).map(x => x[0])); return est(POSTBOX.filter(r => !worst.has(r.pid)), r => r.res); })(),
  repeatedPlayersShare: (() => { const cnt = new Map(); for (const r of POSTBOX) cnt.set(r.pid, (cnt.get(r.pid) || 0) + 1); const net = POSTBOX.reduce((s, r) => s + r.res, 0); return { playersWith2PlusGames: [...cnt.values()].filter(v => v >= 2).length, shareOfNetFromThem: POSTBOX.filter(r => cnt.get(r.pid) >= 2).reduce((s, r) => s + r.res, 0) / net }; })(),
  perNight: [...new Set(POSTBOX.map(r => r.night))].sort().map(n => { const R = POSTBOX.filter(r => r.night === n); return { night: n, games: new Set(R.map(r => r.gk)).size, rows: R.length, bias: est(R, r => r.res) }; }) },
  pre: { rows: PRE.length, bias: est(PRE, r => r.res), perNight: [...new Set(PRE.map(r => r.night))].sort().map(n => { const R = PRE.filter(r => r.night === n); return { night: n, games: new Set(R.map(r => r.gk)).size, rows: R.length, bias: est(R, r => r.res) }; }) },
  dkResultsNights: [...new Set(rows.filter(r => r.src === 'dk-results').map(r => r.night))].map(n => { const R = rows.filter(r => r.night === n); return { night: n, units: new Set(R.map(r => r.gk)).size, rows: R.length, bias: est(R, r => r.res), expensive: est(R.filter(r => r.expensive), r => r.res) }; }) };
// ---- Step 2: same players before and after: projection vs their own season-to-date mean
const both = [...new Set(rows.filter(r => r.std != null && !r.post).map(r => r.pid))].filter(pid => rows.some(r => r.pid === pid && r.post && r.std != null));
const pl = both.map(pid => { const a = rows.filter(r => r.pid === pid && !r.post && r.std != null), b = rows.filter(r => r.pid === pid && r.post && r.std != null); return { pid, exp: a.concat(b).some(r => r.expensive), gapPre: mean(a.map(r => r.proj - r.std)), gapPost: mean(b.map(r => r.proj - r.std)), prodPre: mean(a.map(r => r.actual - r.std)), prodPost: mean(b.map(r => r.actual - r.std)), resPre: mean(a.map(r => r.res)), resPost: mean(b.map(r => r.res)), projPre: mean(a.map(r => r.proj)), projPost: mean(b.map(r => r.proj)), stdPre: mean(a.map(r => r.std)), stdPost: mean(b.map(r => r.std)) }; });
const pboot = (P, f) => { const o = []; for (let b = 0; b < 1000; b++) { const S = Array.from({ length: P.length }, () => P[Math.floor(rnd() * P.length)]); o.push(mean(S.map(f))); } return sd(o); };
const pe = (P, f) => P.length ? { players: P.length, mean: mean(P.map(f)), se: P.length > 2 ? pboot(P, f) : null } : null;
const step2 = {}; for (const [lab, P] of [['all', pl], ['expensive', pl.filter(p => p.exp)], ['not expensive', pl.filter(p => !p.exp)]]) step2[lab] = { players: P.length,
  projMinusStdPre: pe(P, p => p.gapPre), projMinusStdPost: pe(P, p => p.gapPost), changeProjMinusStd: pe(P, p => p.gapPost - p.gapPre),
  actualMinusStdPre: pe(P, p => p.prodPre), actualMinusStdPost: pe(P, p => p.prodPost), changeActualMinusStd: pe(P, p => p.prodPost - p.prodPre), changeResidual: pe(P, p => p.resPost - p.resPre), changeProj: pe(P, p => p.projPost - p.projPre), changeStd: pe(P, p => p.stdPost - p.stdPre) };
// ---- Step 3: run structure before vs after
const used = [...new Set([...pick.values()])], grp = post => used.filter(r => (r.night >= SPLIT) === post);
const ratio = (R, f) => { const v = R.flatMap(r => r.players.filter(p => p.proj >= 10).map(f)); return { n: v.length, mean: mean(v), sd: v.length > 1 ? sd(v) : null }; };
const distinct = (R, f) => { const m = {}; for (const r of R) { const k = String(f(r)); m[k] = (m[k] || 0) + 1; } return m; };
const step3 = {}; for (const [lab, post] of [['before 10/28', false], ['from 10/28', true]]) { const R = grp(post); step3[lab] = { runs: R.length, byKind: distinct(R, r => r.kind),
  minutesBeforeLock: { mean: mean(R.map(r => r.minutesBeforeLock)), min: Math.min(...R.map(r => r.minutesBeforeLock)), max: Math.max(...R.map(r => r.minutesBeforeLock)) },
  floorOverProj: ratio(R, p => p.floor / p.proj), ceilingOverProj: ratio(R, p => p.ceiling / p.proj), medianOverProj: ratio(R, p => p.median / p.proj), spreadOverProj: ratio(R, p => (p.ceiling - p.floor) / p.proj),
  numSim: distinct(R, r => r.numSim), metricBasis: distinct(R, r => r.metricBasis), kernelTables: distinct(R, r => r.kernel.tables), kernelCreated: distinct(R, r => r.kernel.created), kernelInputs: distinct(R, r => r.kernel.inputs), kernelFit: distinct(R, r => r.kernel.fit), kernelBuild: distinct(R, r => r.kernel.build), stepMinutesTable: distinct(R, r => r.kernel.stepMinutesTable),
  stepMinutesPassedShare: mean(R.map(r => (r.kernel.stepMinutesConverted || 0) + (r.kernel.stepMinutesPassed || 0) > 0 ? (r.kernel.stepMinutesPassed || 0) / ((r.kernel.stepMinutesConverted || 0) + (r.kernel.stepMinutesPassed || 0)) : 0)),
  topLevelKeySets: Object.keys(distinct(R, r => r.keys)).length, playerKeySets: distinct(R, r => r.playerKeys), meanProjPlayers10plus: ratio(R, p => p.proj), lineupNewsChanged: distinct(R, r => r.lineupNewsChanged) }; }
// Same split, Showdown only (slate mix check)
const mix = {}; for (const kd of ['Showdown', 'Turbo']) mix[kd] = { pre: est(PRE.filter(r => r.kind === kd && r.src === 'box'), r => r.res), post: est(POSTBOX.filter(r => r.kind === kd), r => r.res) };
// ---- Step 4: nights needed. Per-night spread of the night bias, and how many post nights it takes for the pre/post gap to reach 2 and 3 se.
const nb = n => mean(rows.filter(r => r.night === n).map(r => r.res)), preN = [...new Set(PRE.map(r => r.night))], postN = [...new Set(POST.map(r => r.night))];
const sdNight = sd([...preN, ...postN].map(nb).map((v, i, a) => v - (i < preN.length ? mean(preN.map(nb)) : mean(postN.map(nb)))));
const gap = mean(postN.map(nb)) - mean(preN.map(nb)), postLevel = mean(postN.map(nb));
const seGap = k => Math.sqrt(sdNight ** 2 / preN.length + sdNight ** 2 / k);
const need = z => { for (let k = 1; k <= 200; k++) if (Math.abs(gap) / seGap(k) >= z) return k; return null; };
const needLevel = z => Math.ceil((z * sdNight / Math.abs(postLevel)) ** 2);
const step4 = { gap, postLevel, nightBiases: Object.fromEntries([...preN, ...postN].sort().map(n => [n, nb(n)])), nightSd: sdNight, preNights: preN.length, postNights: postN.length, zNow: Math.abs(gap) / seGap(postN.length), postNightsFor2se: need(2), postNightsFor3se: need(3), zLevelNow: Math.abs(postLevel) / (sdNight / Math.sqrt(postN.length)), postNightsForLevel2se: needLevel(2), postNightsForLevel3se: needLevel(3), postNightsNegative: postN.filter(n => nb(n) < 0).length, preNightsNegative: preN.filter(n => nb(n) < 0).length, note: 'nights needed assume the gap stays at its current size and night-level noise stays at its current sd; pre nights are fixed at the current count' };
// Lineup-news step: timeline, bias with and without it, and whether over-projection sits in players projected above their own recent output.
const newsTimeline = [...new Set(used.map(r => r.night))].sort().map(n => { const R = used.filter(r => r.night === n); return { night: n, runs: R.length, runsWithOut: R.filter(r => (r.lineupNewsOut || 0) > 0).length, meanOut: mean(R.map(r => r.lineupNewsOut || 0)), meanChanged: mean(R.map(r => r.lineupNewsChanged || 0)) }; });
const firstNewsNight = (newsTimeline.find(x => x.runsWithOut > 0) || {}).night || null;
const BOX = rows.filter(r => r.src === 'box'), news = r => r.newsOut > 0;
const gapB = g => g == null ? null : g < 0 ? 'proj below own season mean' : g < 5 ? 'proj 0–5 above' : 'proj 5+ above';
const newsStep = { firstNightWithOutMarks: firstNewsNight, timeline: newsTimeline,
  bias: { newsOff: est(BOX.filter(r => !news(r)), r => r.res), newsOn: est(BOX.filter(news), r => r.res), newsOnInclDkNights: est(rows.filter(news), r => r.res) },
  biasExpensive: { newsOff: est(BOX.filter(r => !news(r) && r.expensive), r => r.res), newsOn: est(BOX.filter(r => news(r) && r.expensive), r => r.res) },
  byGapNewsOff: Object.fromEntries(['proj below own season mean', 'proj 0–5 above', 'proj 5+ above'].map(g => [g, est(BOX.filter(r => !news(r) && gapB(r.std == null ? null : r.proj - r.std) === g), r => r.res)])),
  byGapNewsOn: Object.fromEntries(['proj below own season mean', 'proj 0–5 above', 'proj 5+ above'].map(g => [g, est(rows.filter(r => news(r) && gapB(r.std == null ? null : r.proj - r.std) === g), r => r.res)])),
  shareProj5PlusAbove: { newsOff: mean(BOX.filter(r => !news(r) && r.std != null).map(r => r.proj - r.std >= 5 ? 1 : 0)), newsOn: mean(rows.filter(r => news(r) && r.std != null).map(r => r.proj - r.std >= 5 ? 1 : 0)) } };
// The 10/27 games: news on, but before R's split.
newsStep.night1027 = est(BOX.filter(r => r.night === '2026-10-27'), r => r.res);
const out = { builtAt: new Date().toISOString(), split: SPLIT, rows: rows.length, rowsBySource: { box: rows.filter(r => r.src === 'box').length, dkResults: rows.filter(r => r.src === 'dk-results').length }, dropped: drop, runsUsed: used.length, boxLastNight: [...boxNights].sort().pop(), dkResultsNightsUsed: [...dkAct.keys()], expensiveCuts: cuts, slateMix: mix, step1, step2, step3, step4, newsStep };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1)); console.log(JSON.stringify({ rows: out.rows, bySrc: out.rowsBySource, drop, runs: used.length, boxLast: out.boxLastNight, dkNights: out.dkResultsNightsUsed }));
