// Vale 10/17 units A and B: join every published DK NFL run (weeks 1-6) to the recap box scores and the official DK points.
// Usage: node build-dataset.cjs <data repo> <out.json>. Reads only; writes one aggregate-free working file in scratch.
const fs = require('fs'), path = require('path'), zlib = require('zlib');
const D = process.argv[2], OUT = process.argv[3];
const J = f => JSON.parse(fs.readFileSync(path.join(D, f), 'utf8'));
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
  .replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TEAM = { ARI:'ARI',ATL:'ATL',BAL:'BAL',BUF:'BUF',CAR:'CAR',CHI:'CHI',CIN:'CIN',CLE:'CLE',DAL:'DAL',DEN:'DEN',DET:'DET',GB:'GB',HOU:'HOU',IND:'IND',JAX:'JAX',JAC:'JAX',KC:'KC',LV:'LV',LAC:'LAC',LAR:'LAR',LA:'LAR',MIA:'MIA',MIN:'MIN',NE:'NE',NO:'NO',NYG:'NYG',NYJ:'NYJ',PHI:'PHI',PIT:'PIT',SF:'SF',SEA:'SEA',TB:'TB',TEN:'TEN',WAS:'WAS',WSH:'WAS' };
const T = t => TEAM[String(t || '').toUpperCase()] || String(t || '').toUpperCase();

// ---- recap box scores: per team-game player rows, team offensive snaps ----
const dk = p => { // DK classic scoring from the box (fumbles lost and two-point conversions are not in the recap rows)
  const py = +p.passYds || 0, ry = +p.rushYds || 0, cy = +p.recYds || 0;
  return py * 0.04 + (+p.passTD || 0) * 4 - (+p.ints || 0) + (py >= 300 ? 3 : 0)
    + ry * 0.1 + (+p.rushTD || 0) * 6 + (ry >= 100 ? 3 : 0)
    + (+p.rec || 0) + cy * 0.1 + (+p.recTD || 0) * 6 + (cy >= 100 ? 3 : 0);
};
const games = []; // {week, gameId, date, teams:{T:{snaps, players:[...]}}}
for (let w = 1; w <= 6; w++) {
  const f = 'nfl-recap/2026-w' + String(w).padStart(2, '0') + '.json'; if (!fs.existsSync(path.join(D, f))) continue;
  const R = J(f);
  for (const g of R.games) {
    if (g.status !== 'final' || !g.snapsReady) continue;
    const teams = {};
    for (const t of [g.home, g.away]) teams[T(t)] = { team: T(t), opp: T(t === g.home ? g.away : g.home), players: [] };
    for (const p of g.players || []) {
      const tm = T(p.team); if (!teams[tm]) continue;
      teams[tm].players.push({ id: p.id, nk: norm(p.name), name: p.name, pos: p.offensePosition || p.pos, snaps: +p.snaps || 0, snapPct: +p.snapPct || 0,
        targets: +p.targets || 0, carries: +p.carries || 0, rec: +p.rec || 0, recYds: +p.recYds || 0, recTD: +p.recTD || 0, dkBox: +dk(p).toFixed(2) });
    }
    for (const tm of Object.values(teams)) {
      const est = tm.players.filter(p => p.snapPct >= 0.2 && p.snaps > 0).map(p => p.snaps / p.snapPct).sort((a, b) => a - b);
      tm.teamSnaps = est.length ? Math.round(est[Math.floor(est.length / 2)]) : null;
      tm.teamTargets = tm.players.reduce((a, p) => a + p.targets, 0);
    }
    games.push({ week: w, gameId: g.gameId || g.id, date: String(g.date || '').slice(0, 10), teams });
  }
}

// ---- official DK points per date (Classic contests first, then Showdown) ----
const actualByDate = {};
for (const f of fs.readdirSync(path.join(D, 'nfl-dk-results'))) {
  const r = J('nfl-dk-results/' + f), m = {};
  const cs = [...r.contests].sort((a, b) => (a.isShowdown ? 1 : 0) - (b.isShowdown ? 1 : 0));
  for (const c of cs) for (const [k, v] of Object.entries(c.playerActuals || {})) {
    const key = norm(v.name || k); if (m[key] == null && Number.isFinite(+v.fpts)) m[key] = +v.fpts;
  }
  actualByDate[r.date] = m;
}

// ---- published runs ----
const ptrs = fs.readdirSync(D).filter(f => /^nfl-dk-(main|early-only|afternoon-only|showdown)-\d+\.json$/.test(f));
const runs = [];
for (const f of ptrs) {
  const w = J(f); if (!w.identity) continue;
  let S = null, rows = null, edits = {};
  if (w.summary) {
    const s = w.summary;
    const buf = s.data ? Buffer.from(s.data, 'base64') : Buffer.concat([...s.parts].sort((a, b) => a.index - b.index)
      .map(x => Buffer.from(J(x.path).data, 'base64')));
    S = JSON.parse(zlib.gunzipSync(buf));
    const B = S.inputSnapshot && S.inputSnapshot.body; rows = (B && B.forecastContext && B.forecastContext.rows) || [];
    edits = (B && B.edits) || {};
    rows = rows.map(r => ({ nk: r.nk, name: r.name, team: T(r.team), opp: T(r.opp), pos: r.rolePosition || r.pos, salary: r.salary,
      expSnaps: +r.expSnaps || 0, plays: +r.plays || null, targets: r.line && r.line.targets != null ? +r.line.targets : (r.earningInfo && r.earningInfo.allocation ? +r.earningInfo.allocation.allocatedTargets : null),
      projDK: +r.projDK || 0, implied: r.implied, oppImplied: r.oppImplied, out: !!(edits[r.nk] && edits[r.nk].expSnaps === 0) }));
  } else if (w.results) {
    S = w.results; const F = S.publicForecast || {}, wl = {}; for (const x of F.workload || []) wl[x.nk] = x;
    rows = (F.rows || []).map(r => ({ nk: r.nk, name: r.name, team: T(r.team), opp: T(r.opp), pos: r.rolePosition || r.pos, salary: r.salary,
      expSnaps: +r.expSnaps || 0, plays: +r.plays || null, targets: wl[r.nk] ? +wl[r.nk].targets : null, projDK: +r.projDK || 0,
      implied: r.implied, oppImplied: r.oppImplied, out: r.status === 'O' || r.status === 'OUT' }));
  } else continue;
  const id = w.identity, type = /showdown/.test(f) ? 'SD' : /early/.test(f) ? 'early' : /afternoon/.test(f) ? 'afternoon' : 'main';
  const pr = Object.values(S.playerResults || {}).map(p => ({ nk: p.nk, name: p.name, team: T(p.team), pos: p.pos, salary: p.salary,
    proj: p.proj, floor: p.floor, ceiling: p.ceiling, avgPts: p.avgPts, stdDev: p.stdDev }));
  runs.push({ file: f, type, dg: id.rawSlateId, slate: id.slate, week: +String(id.weekKey).replace(/.*W/, ''), date: id.date,
    runId: id.runId, createdAt: new Date(id.createdAt).toISOString(), rows, pr });
}
runs.sort((a, b) => a.date.localeCompare(b.date) || a.dg.localeCompare(b.dg));
fs.writeFileSync(OUT, JSON.stringify({ builtAt: new Date().toISOString(), games, actualByDate, runs }));
console.log('games', games.length, 'by week', JSON.stringify(games.reduce((m, g) => (m[g.week] = (m[g.week] || 0) + 1, m), {})),
  '| result dates', Object.keys(actualByDate).length, '| runs', runs.length,
  JSON.stringify(runs.reduce((m, r) => (m[r.type] = (m[r.type] || 0) + 1, m), {})));
