// Vale unit 1103CL-G: prime-time NFL Showdown, the winning-team K/DST effect. Same 25 games as unit C.
// Inputs (public): nfl-dk-showdown-<dg>.json (results.playerResults, results.publicForecast), nfl-dk-results/<date>.json, nfl-schedule.json.
// Usage: node unit-g.cjs <data dir> <out.json>. Output is aggregates only (no lineups, users or entries).
const fs = require('fs'), path = require('path');
const D = process.argv[2], OUT = process.argv[3];
const J = f => JSON.parse(fs.readFileSync(path.join(D, f), 'utf8'));
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TEAM = { JAC: 'JAX', LA: 'LAR', WSH: 'WAS' }, T = t => { const u = String(t || '').toUpperCase(); return TEAM[u] || u; };
const SCHED = J('nfl-schedule.json');
const contestsByDG = {}; for (const f of fs.readdirSync(path.join(D, 'nfl-dk-results'))) for (const c of J('nfl-dk-results/' + f).contests) (contestsByDG[c.draftGroupId] = contestsByDG[c.draftGroupId] || []).push(c);
const KD = p => p.pos === 'K' || p.pos === 'DST', QR = p => p.pos === 'QB' || p.pos === 'RB';
const avg = a => a.reduce((s, x) => s + x, 0) / a.length;
const Phi = x => { const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989423 * Math.exp(-x * x / 2), p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return x > 0 ? 1 - p : p; };
const games = [];
for (const f of fs.readdirSync(D).filter(f => /^nfl-dk-showdown-\d+\.json$/.test(f))) {
  const w = J(f), id = w.identity, R = w.results, cs = contestsByDG[id.rawSlateId]; if (!cs) continue;
  const week = +String(id.weekKey).replace(/.*W/, ''), P = new Map(); for (const p of Object.values(R.playerResults)) P.set(norm(p.name), { ...p, team: T(p.team) });
  const teams = [...new Set([...P.values()].map(p => p.team))]; if (teams.length !== 2) continue;
  const sc = teams.map(t => (SCHED.teams[t] || {})[week] || {}); if (sc[0].score == null || sc[1].score == null) continue;
  const winner = sc[0].score > sc[1].score ? teams[0] : sc[1].score > sc[0].score ? teams[1] : null; if (!winner) continue; const loser = teams.find(t => t !== winner), margin = Math.abs(sc[0].score - sc[1].score);
  const rows = (R.publicForecast && R.publicForecast.rows) || [], homeRow = rows.find(r => r.home === true), home = homeRow ? T(homeRow.team) : teams[0], away = teams.find(t => t !== home);
  const line = ((R.publicForecast && R.publicForecast.lines) || []).find(l => T(l.home) === home && T(l.away) === away);
  const homeSpread = line ? +line.homeSpread : (homeRow && homeRow.spread != null ? +homeRow.spread : null);
  // Sim's own per-team shares (pre-lock): captain winner share, K/DST winner share (CPT + FLEX), all-slot winner share.
  const sumT = (t, f, sel = () => true) => [...P.values()].filter(p => p.team === t && sel(p)).reduce((s, p) => s + (f(p) || 0), 0);
  const shareHome = (f, sel) => { const a = sumT(home, f, sel), b = sumT(away, f, sel); return a + b > 0 ? a / (a + b) : null; };
  const sim = { cptTeam: shareHome(p => p.cptWinnerExposure), kdstTeam: shareHome(p => (p.cptWinnerExposure || 0) + (p.flexWinnerExposure || 0), KD), slotTeam: shareHome(p => p.exposure), spreadWinProb: homeSpread == null ? null : Phi(-homeSpread / 13.5) };
  const g = { dg: id.rawSlateId, week, date: id.date, home, away, winner, loser, margin, band: margin <= 3 ? '1–3' : margin <= 8 ? '4–8' : '9+', homeWon: winner === home ? 1 : 0, homeSpread, sim, contests: [] };
  for (const c of cs) {
    const S = c.summary || {}, line1 = S.topThresholds && S.topThresholds.top1pct;
    const tops = (c.topFieldLineups || []).filter(l => l.players && l.players.length === 6 && (line1 == null || l.points >= line1 - 1e-6)); const L = [];
    for (const l of tops) { const cpt = l.players.find(x => x.pos === 'CPT'), flex = l.players.filter(x => x.pos !== 'CPT'); const ids = [cpt, ...flex].map(x => x && norm(x.name)); if (!cpt || flex.length !== 5 || ids.some(nk => !P.has(nk))) continue; L.push(ids.map(nk => P.get(nk))); }
    if (!L.length) continue;
    const own = new Map(Object.entries(S.fieldOwnership || {}).map(([k, v]) => [norm(k), +v / 100])), cap = new Map(Object.values(S.captainOwnership || {}).map(v => [norm(v.name), +v.capOwn / 100]));
    const o = p => own.get(norm(p.name)) || 0, co = p => cap.get(norm(p.name)) || 0, fo = p => Math.max(0, o(p) - co(p));
    const sh = f => L.reduce((s, x) => s + f(x), 0) / L.length, has = (x, t, sel) => x.some(p => p.team === t && sel(p));
    const m = { top: {}, field: {} };
    for (const [lab, sel] of [['K/DST', KD], ['K', p => p.pos === 'K'], ['DST', p => p.pos === 'DST']]) {
      m.top['CPT ' + lab] = sh(x => sel(x[0]) ? 1 : 0); m.top['CPT ' + lab + ' on winner'] = sh(x => sel(x[0]) && x[0].team === winner ? 1 : 0);
      m.top['FLEX ' + lab] = sh(x => x.slice(1).filter(sel).length); m.top['FLEX ' + lab + ' on winner'] = sh(x => x.slice(1).filter(p => sel(p) && p.team === winner).length);
      const all = [...P.values()].filter(sel);
      m.field['CPT ' + lab] = all.reduce((s, p) => s + co(p), 0); m.field['CPT ' + lab + ' on winner'] = all.filter(p => p.team === winner).reduce((s, p) => s + co(p), 0);
      m.field['FLEX ' + lab] = all.reduce((s, p) => s + fo(p), 0); m.field['FLEX ' + lab + ' on winner'] = all.filter(p => p.team === winner).reduce((s, p) => s + fo(p), 0);
    }
    // K/DST picks (any slot) on the home side, for step 2.
    m.top.kdstPicks = sh(x => x.filter(KD).length); m.top.kdstPicksHome = sh(x => x.filter(p => KD(p) && p.team === home).length);
    const allKD = [...P.values()].filter(KD); m.field.kdstPicks = allKD.reduce((s, p) => s + o(p), 0); m.field.kdstPicksHome = allKD.filter(p => p.team === home).reduce((s, p) => s + o(p), 0);
    // Step 3: winning team's K or DST plus that team's QB or RB (any slots); same for the loser.
    for (const [lab, t] of [['winner', winner], ['loser', loser]]) {
      m.top[lab + ' K/DST'] = sh(x => has(x, t, KD) ? 1 : 0); m.top[lab + ' QB/RB'] = sh(x => has(x, t, QR) ? 1 : 0); m.top[lab + ' K/DST + QB/RB'] = sh(x => has(x, t, KD) && has(x, t, QR) ? 1 : 0);
      const pAny = sel => 1 - [...P.values()].filter(p => p.team === t && sel(p)).reduce((s, p) => s * (1 - Math.min(1, o(p))), 1);
      m.field[lab + ' K/DST'] = pAny(KD); m.field[lab + ' QB/RB'] = pAny(QR); m.field[lab + ' K/DST + QB/RB'] = pAny(KD) * pAny(QR);
    }
    g.contests.push(m);
  }
  if (!g.contests.length) continue;
  g.top = {}; g.field = {}; for (const k of Object.keys(g.contests[0].top)) g.top[k] = avg(g.contests.map(c => c.top[k])); for (const k of Object.keys(g.contests[0].field)) g.field[k] = avg(g.contests.map(c => c.field[k]));
  g.yHome = g.top.kdstPicks > 0 ? g.top.kdstPicksHome / g.top.kdstPicks : null; g.fieldHome = g.field.kdstPicks > 0 ? g.field.kdstPicksHome / g.field.kdstPicks : null;
  games.push(g);
}
// ---- game bootstrap ----
let seed = 1104; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const sdv = a => { const m = avg(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const boot = (G, stat, B = 2000) => { const est = stat(G), bs = []; for (let b = 0; b < B; b++) { const S = Array.from({ length: G.length }, () => G[Math.floor(rnd() * G.length)]); const v = stat(S); if (v != null && Number.isFinite(v)) bs.push(v); } return { mean: est, se: bs.length > 2 ? sdv(bs) : null, games: G.length }; };
const ratio = (G, num, den) => { const d = G.reduce((s, g) => s + den(g), 0); return d > 0 ? G.reduce((s, g) => s + num(g), 0) / d : null; };
// Step 1: share of K/DST picks that come from the winning team, by margin band, top vs field.
const step1 = {};
for (const band of ['all', '1–3', '4–8', '9+']) { const G = games.filter(g => band === 'all' || g.band === band); step1[band] = { games: G.length };
  for (const lab of ['K/DST', 'K', 'DST']) for (const slot of ['CPT', 'FLEX']) { const k = slot + ' ' + lab;
    step1[band][k] = { gamesWithAnyInTop: G.filter(g => g.top[k] > 0).length, top: boot(G, S => ratio(S, g => g.top[k + ' on winner'], g => g.top[k])), field: boot(G, S => ratio(S, g => g.field[k + ' on winner'], g => g.field[k])),
      topPerLineup: boot(G, S => avg(S.map(g => g.top[k]))), fieldPerLineup: boot(G, S => avg(S.map(g => g.field[k]))),
      topOnWinnerPerLineup: boot(G, S => avg(S.map(g => g.top[k + ' on winner']))), topOnLoserPerLineup: boot(G, S => avg(S.map(g => g.top[k] - g.top[k + ' on winner']))) }; } }
// Step 2: does a pre-lock signal pick the side whose K/DST lands in the top 1%? Target = home share of top-lineup K/DST picks.
const SIG = { 'spread (home win prob.)': g => g.sim.spreadWinProb, 'sim captain team share': g => g.sim.cptTeam, 'sim K/DST team share': g => g.sim.kdstTeam, 'sim all-slot team share': g => g.sim.slotTeam };
const REF = { 'field K/DST ownership split (post-lock)': g => g.fieldHome, 'actual winner (not pre-lock; ceiling)': g => g.homeWon };
const pear = (a, b) => { const ma = avg(a), mb = avg(b); const n = a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0); return n / Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0) * b.reduce((s, v) => s + (v - mb) ** 2, 0)); };
const tr = games.filter(g => g.week <= 6), ho = games.filter(g => g.week >= 7), step2 = { trainGames: tr.length, holdoutGames: ho.length, signals: {} };
for (const [name, f] of Object.entries({ ...SIG, ...REF })) {
  const ok = G => G.filter(g => f(g) != null && g.yHome != null), A = ok(games), TR = ok(tr), HO = ok(ho);
  const fit = G => { const x = G.map(f), y = G.map(g => g.yHome), mx = avg(x), my = avg(y), b = x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0) / x.reduce((s, v) => s + (v - mx) ** 2, 0); return { a: my - b * mx, b, my }; };
  const F = fit(TR), pred = g => F.a + F.b * f(g);
  const hoMse = avg(HO.map(g => (g.yHome - pred(g)) ** 2)), baseMse = avg(HO.map(g => (g.yHome - F.my) ** 2)), halfMse = avg(HO.map(g => (g.yHome - 0.5) ** 2));
  // "By how much": share of top-lineup K/DST picks on the side the signal favors (0.5 = no information).
  const fav = G => avg(G.map(g => f(g) > 0.5 ? g.yHome : f(g) < 0.5 ? 1 - g.yHome : 0.5));
  step2.signals[name] = { games: A.length, corrAll: boot(A, S => pear(S.map(f), S.map(g => g.yHome))), slopeTrain: F.b, heldOut: { games: HO.length, mse: hoMse, baselineMse: baseMse, mseVsHalf: halfMse, r2: 1 - hoMse / baseMse, r2Boot: boot(HO, S => { const b = avg(S.map(g => (g.yHome - F.my) ** 2)); return b > 0 ? 1 - avg(S.map(g => (g.yHome - pred(g)) ** 2)) / b : null; }), kdstShareOnFavoredSide: boot(HO, fav), favoredSideGotMajority: boot(HO, S => avg(S.map(g => (f(g) - 0.5) * (g.yHome - 0.5) > 0 ? 1 : 0))) },
    all: { kdstShareOnFavoredSide: boot(A, fav), favoredSideGotMajority: boot(A, S => avg(S.map(g => (f(g) - 0.5) * (g.yHome - 0.5) > 0 ? 1 : 0))) } };
}
step2.signalAgreement = { simCptVsSpreadCorr: boot(games.filter(g => g.sim.spreadWinProb != null), S => pear(S.map(g => g.sim.cptTeam), S.map(g => g.sim.spreadWinProb))), simKdstVsSpreadCorr: boot(games.filter(g => g.sim.spreadWinProb != null), S => pear(S.map(g => g.sim.kdstTeam), S.map(g => g.sim.spreadWinProb))),
  favoriteWon: boot(games.filter(g => g.homeSpread != null && g.homeSpread !== 0), S => avg(S.map(g => (g.homeSpread < 0) === (g.homeWon === 1) ? 1 : 0))), simCptFavoriteWon: boot(games, S => avg(S.map(g => (g.sim.cptTeam > 0.5) === (g.homeWon === 1) ? 1 : 0))) };
// Step 3: winner's K/DST + winner's QB/RB.
const step3 = {}; for (const lab of ['winner', 'loser']) { step3[lab] = {}; for (const k of ['K/DST', 'QB/RB', 'K/DST + QB/RB']) { const key = lab + ' ' + k;
  step3[lab][k] = { top: boot(games, S => avg(S.map(g => g.top[key]))), fieldIndependence: boot(games, S => avg(S.map(g => g.field[key]))), topMinusField: boot(games, S => avg(S.map(g => g.top[key] - g.field[key]))) }; }
  step3[lab]['top joint ÷ top product of marginals'] = boot(games, S => avg(S.map(g => g.top[lab + ' K/DST + QB/RB'])) / avg(S.map(g => g.top[lab + ' K/DST'] * g.top[lab + ' QB/RB']))); }
step2.gamesWithNoKdstInTop = games.filter(g => g.yHome == null).length;
const out = { games: games.map(g => ({ dg: g.dg, week: g.week, date: g.date, margin: g.margin, band: g.band, contests: g.contests.length, homeSpread: g.homeSpread })), bandCounts: Object.fromEntries(['1–3', '4–8', '9+'].map(b => [b, games.filter(g => g.band === b).length])), step1, step2, step3 };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1)); console.log(JSON.stringify({ games: games.length, bands: out.bandCounts, train: tr.length, holdout: ho.length }));
