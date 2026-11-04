// Vale unit 1103CL-P: what our published projections missed for Classic WRs in 2026 games with a closing total of 44.5–48.5.
// Inputs (public): splashplaynba nfl-dk-main-*.json / nfl-dk-afternoon-only-*.json (playerResults, publicForecast rows + workload),
// nfl-dk-results/<date>.json (playerActuals); nflverse stats_player_week_2025/2026.csv and games.csv (closing totals).
// Usage: node unit-p.cjs <splashplaynba data dir> <nflverse dir> <games.csv> <out.json>. Aggregates only.
const fs = require('fs'), path = require('path');
const [, , D, ND, GAMES, OUT] = process.argv, MIN_SAL = +(process.argv[6] || 4000);
const J = f => JSON.parse(fs.readFileSync(path.join(D, f), 'utf8'));
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TEAM = { JAC: 'JAX', LA: 'LAR', WSH: 'WAS' }, T = t => { const u = String(t || '').toUpperCase(); return TEAM[u] || u; };
const mean = a => a.reduce((s, x) => s + x, 0) / a.length, sdv = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const num = x => (x === '' || x == null || x === 'NA') ? 0 : +x;
function splitLine(l) { const o = []; let cur = '', q = false; for (let i = 0; i < l.length; i++) { const ch = l[i]; if (q) { if (ch === '"') { if (l[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += ch; } else if (ch === '"') q = true; else if (ch === ',') { o.push(cur); cur = ''; } else cur += ch; } o.push(cur); return o; }
const readCSV = (file, cols) => { const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/), H = splitLine(lines[0]), ix = cols.map(c => H.indexOf(c)); return lines.slice(1).filter(Boolean).map(l => { const f = splitLine(l), r = {}; cols.forEach((c, j) => r[c] = ix[j] >= 0 ? f[ix[j]] : undefined); return r; }); };
let seed = 1108; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
// ---- closing lines 2026 (nflverse)
const GL = new Map(); for (const r of readCSV(GAMES, ['game_id', 'season', 'game_type', 'week', 'home_team', 'away_team', 'total_line', 'spread_line'])) if (r.season === '2026' && r.game_type === 'REG') GL.set(+r.week + '|' + [T(r.home_team), T(r.away_team)].sort().join('-'), { gid: r.game_id, total: num(r.total_line), home: T(r.home_team), spread: num(r.spread_line) });
// ---- nflverse weekly stats: player rows and team pass volume
const SC = ['player_id', 'player_display_name', 'position', 'season', 'week', 'season_type', 'team', 'attempts', 'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'carries', 'rushing_yards', 'rushing_tds', 'receiving_fumbles_lost', 'rushing_fumbles_lost', 'receiving_2pt_conversions', 'rushing_2pt_conversions', 'special_teams_tds'];
const st = new Map(), teamWeek = new Map(), wrHist = new Map();
for (const y of [2025, 2026]) for (const r of readCSV(path.join(ND, 'stats_player_week_' + y + '.csv'), SC)) { if (r.season_type !== 'REG') continue; const tm = T(r.team), wk = +r.week, tk = y + '|' + wk + '|' + tm;
  const tw = teamWeek.get(tk) || { att: 0, tgt: 0 }; tw.att += num(r.attempts); tw.tgt += num(r.targets); teamWeek.set(tk, tw);
  const ry = num(r.receiving_yards), rec = num(r.receptions), nonTd = rec + 0.1 * ry + (ry >= 100 ? 3 : 0) - num(r.receiving_fumbles_lost) + 2 * num(r.receiving_2pt_conversions), rush = 0.1 * num(r.rushing_yards) + 6 * num(r.rushing_tds) + (num(r.rushing_yards) >= 100 ? 3 : 0) - num(r.rushing_fumbles_lost) + 2 * num(r.rushing_2pt_conversions) + 6 * num(r.special_teams_tds);
  const row = { y, wk, team: tm, pos: r.position, tgt: num(r.targets), recTd: num(r.receiving_tds), nonTd, rush };
  if (y === 2026) st.set(wk + '|' + tm + '|' + norm(r.player_display_name), row);
  if (r.position === 'WR') { const k = r.player_id; if (!wrHist.has(k)) wrHist.set(k, []); wrHist.get(k).push(row); } }
// Team pass attempts: this game vs the team's prior 2026 average (2025 average in week 1).
const teamPrior = (tm, wk) => { const cur = []; for (let w = 1; w < wk; w++) { const x = teamWeek.get('2026|' + w + '|' + tm); if (x) cur.push(x.att); } if (cur.length) return mean(cur); const last = []; for (let w = 1; w <= 18; w++) { const x = teamWeek.get('2025|' + w + '|' + tm); if (x) last.push(x.att); } return last.length ? mean(last) : null; };
// Season target share (2026, weeks 1–8) for WR roles.
const share26 = new Map(); { const tot = new Map(), byP = new Map(); for (const [k, r] of st) { if (r.wk > 8) continue; tot.set(r.team, (tot.get(r.team) || 0) + r.tgt); if (r.pos === 'WR') { const pk = r.team + '|' + k.split('|')[2]; byP.set(pk, (byP.get(pk) || 0) + r.tgt); } }
  const byTeam = new Map(); for (const [pk, t] of byP) { const tm = pk.split('|')[0]; if (!byTeam.has(tm)) byTeam.set(tm, []); byTeam.get(tm).push([pk, t]); }
  for (const [tm, a] of byTeam) a.sort((x, y) => y[1] - x[1]).forEach(([pk, t], i) => share26.set(pk, { rank: i + 1, share: t / tot.get(tm) })); }
// League WR TD per target (2025 + 2026 to date) for the projected TD split.
// ---- published Classic runs, weeks 1–8 (Main first; Afternoon fills players not in Main)
const contestsByDG = {}; for (const f of fs.readdirSync(path.join(D, 'nfl-dk-results'))) for (const c of J('nfl-dk-results/' + f).contests) (contestsByDG[c.draftGroupId] = contestsByDG[c.draftGroupId] || []).push(c);
const runs = fs.readdirSync(D).filter(f => /^nfl-dk-(main|afternoon-only)-\d+\.json$/.test(f)).map(f => ({ f, j: J(f) })).sort((a, b) => (a.f.includes('main') ? 0 : 1) - (b.f.includes('main') ? 0 : 1));
const seen = new Set(), rows = [], audit = { runs: runs.length, wrRows: 0, noActual: 0, noStats: 0, noWorkload: 0, dupSkipped: 0, noLine: 0 };
for (const { f, j } of runs) { const id = j.identity, wk = +String(id.weekKey).replace(/.*W/, ''), R = j.results; const cs = contestsByDG[id.rawSlateId] || []; if (!cs.length) continue;
  const act = new Map(); for (const c of cs) for (const v of Object.values(c.playerActuals || {})) act.set(norm(v.name), +v.fpts);
  const fr = new Map((R.publicForecast.rows || []).map(r => [r.nk, r])), wl = new Map((R.publicForecast.workload || []).map(w => [w.nk, w]));
  // Team projected targets = sum of workload targets over the team's players in this run.
  const teamHat = new Map(); for (const [nk, w] of wl) { const r = fr.get(nk); if (!r || !(w.targets > 0)) continue; const tm = T(r.team); teamHat.set(tm, (teamHat.get(tm) || 0) + w.targets); }
  for (const p of Object.values(R.playerResults)) { if (p.pos !== 'WR' || !(p.salary >= MIN_SAL)) continue; const nk = norm(p.name), tm = T(p.team), key = wk + '|' + tm + '|' + nk; audit.wrRows++;
    if (seen.has(key)) { audit.dupSkipped++; continue; } const a = act.get(nk); if (a == null) { audit.noActual++; continue; } seen.add(key);
    const r = fr.get(p.nk) || fr.get(nk) || {}, opp = T(r.opp), gl = GL.get(wk + '|' + [tm, opp].sort().join('-')); if (!gl) { audit.noLine++; continue; }
    const s = st.get(key), w = wl.get(p.nk) || wl.get(nk); if (!s) audit.noStats++; if (!w) audit.noWorkload++;
    const tw = teamWeek.get('2026|' + wk + '|' + tm), role = share26.get(tm + '|' + nk);
    rows.push({ wk, slate: f.includes('main') ? 'Main' : 'Afternoon', gid: gl.gid, team: tm, opp, total: gl.total, band: gl.total >= 44.5 && gl.total <= 48.5, salary: p.salary, proj: +p.proj, sd: +p.stdDev, actual: a, res: a - p.proj,
      tgtHat: w && w.targets > 0 ? w.targets : null, teamTgtHat: teamHat.get(tm) || null, tgt: s ? s.tgt : null, teamTgt: tw ? tw.tgt : null, teamAtt: tw ? tw.att : null, teamAttPrior: teamPrior(tm, wk),
      nonTd: s ? s.nonTd : null, recTd: s ? s.recTd : null, rush: s ? s.rush : null, role: role ? (role.rank <= 3 ? 'WR' + role.rank : 'WR4+') : 'no 2026 targets' }); } }
// Prior WR TD points per target (league, 2025) to split the projection's points per target into non-TD and TD parts.
const lg = [...wrHist.values()].flat().filter(r => r.y === 2025), tdPerTgt = 6 * lg.reduce((s, r) => s + r.recTd, 0) / lg.reduce((s, r) => s + r.tgt, 0);
// ---- decomposition (rows with projected and actual targets): residual = volume + share + efficiency + TD + rushing + leftover
for (const r of rows) { if (r.tgtHat == null || r.tgt == null || !(r.tgtHat >= 2) || !(r.teamTgtHat > 0) || !(r.teamTgt > 0)) continue;
  const pptHat = r.proj / r.tgtHat, tdHat = Math.min(tdPerTgt, pptHat * 0.45), nonTdHat = pptHat - tdHat, sHat = r.tgtHat / r.teamTgtHat, s = r.tgt / r.teamTgt;
  r.dVolume = pptHat * (r.teamTgt - r.teamTgtHat) * sHat; r.dShare = pptHat * r.teamTgt * (s - sHat);
  r.dEff = r.tgt > 0 ? r.tgt * (r.nonTd / r.tgt - nonTdHat) : 0; r.dTd = r.tgt > 0 ? r.tgt * (6 * r.recTd / r.tgt - tdHat) : 0; r.dRush = r.rush; r.dLeft = r.res - (r.dVolume + r.dShare + r.dEff + r.dTd + r.dRush); r.dec = true;
  r.dTargets = r.dVolume + r.dShare; r.tgtMiss = r.tgt - r.tgtHat; r.teamTgtMiss = r.teamTgt - r.teamTgtHat; r.attVsPrior = r.teamAttPrior != null ? r.teamAtt - r.teamAttPrior : null; }
// ---- bootstrap by game
const games = [...new Set(rows.map(r => r.gid))], byG = new Map(games.map(g => [g, rows.filter(r => r.gid === g)]));
const boot = (R, stat, B = 1000) => { const gs = [...new Set(R.map(r => r.gid))], by = new Map(gs.map(g => [g, R.filter(r => r.gid === g)])), o = []; for (let b = 0; b < B; b++) { const S = []; for (let i = 0; i < gs.length; i++) S.push(...by.get(gs[Math.floor(rnd() * gs.length)])); const v = stat(S); if (Number.isFinite(v)) o.push(v); } return o.length > 2 ? sdv(o) : null; };
const est = (R, f) => { const v = R.filter(r => f(r) != null && Number.isFinite(f(r))); return v.length ? { n: v.length, games: new Set(v.map(r => r.gid)).size, mean: mean(v.map(f)), se: boot(v, S => mean(S.map(f))) } : null; };
const diff = (R, f) => { const v = R.filter(r => f(r) != null && Number.isFinite(f(r))); return v.length ? { mean: mean(v.filter(r => r.band).map(f)) - mean(v.filter(r => !r.band).map(f)), se: boot(v, S => { const a = S.filter(r => r.band), b = S.filter(r => !r.band); return a.length && b.length ? mean(a.map(f)) - mean(b.map(f)) : NaN; }) } : null; };
const BAND = rows.filter(r => r.band), REST = rows.filter(r => !r.band);
const COMP = { 'residual (actual − proj)': r => r.res, 'targets: team volume': r => r.dec ? r.dVolume : null, 'targets: share': r => r.dec ? r.dShare : null, 'efficiency (non-TD points per target)': r => r.dec ? r.dEff : null, 'TDs': r => r.dec ? r.dTd : null, 'rushing': r => r.dec ? r.dRush : null, 'leftover (DK vs nflverse rounding, unmatched)': r => r.dec ? r.dLeft : null,
  'residual, decomposed rows only': r => r.dec ? r.res : null, 'targets − projected targets': r => r.dec ? r.tgtMiss : null, 'team targets − projected team targets': r => r.dec ? r.teamTgtMiss : null, 'team pass attempts − prior average': r => r.attVsPrior };
const step1 = {}; for (const [k, f] of Object.entries(COMP)) step1[k] = { band: est(BAND, f), rest: est(REST, f), bandMinusRest: diff(rows, f) };
// ---- Step 2: which WRs
const salT = s => s < 4000 ? 'under $4,000' : s < 5500 ? '$4,000–5,499' : s < 7000 ? '$5,500–6,999' : '$7,000+';
const pj = rows.map(r => r.proj).sort((a, b) => a - b), q1 = pj[Math.floor(pj.length / 3)], q2 = pj[Math.floor(2 * pj.length / 3)], prT = p => p < q1 ? 'low' : p < q2 ? 'mid' : 'high';
const step2 = { projTierCuts: [q1, q2], role: {}, salary: {}, projTier: {} };
for (const [k, sel] of [['role', r => r.role], ['salary', r => salT(r.salary)], ['projTier', r => prT(r.proj)]]) for (const g of [...new Set(rows.map(sel))].sort()) { const R = rows.filter(r => sel(r) === g); step2[k][g] = { band: est(R.filter(r => r.band), r => r.res), rest: est(R.filter(r => !r.band), r => r.res), bandShareOfExcess: null, bandRows: R.filter(r => r.band).length }; }
const bandExcessTotal = BAND.reduce((s, r) => s + r.res, 0) - BAND.length * mean(REST.map(r => r.res));
for (const k of ['role', 'salary', 'projTier']) for (const g of Object.keys(step2[k])) { const sel = k === 'role' ? r => r.role : k === 'salary' ? r => salT(r.salary) : r => prT(r.proj); const B2 = BAND.filter(r => sel(r) === g); step2[k][g].bandShareOfExcess = (B2.reduce((s, r) => s + r.res, 0) - B2.length * mean(REST.map(r => r.res))) / bandExcessTotal; }
// Leave out the top 3 WR games.
const top3 = BAND.slice().sort((a, b) => b.res - a.res).slice(0, 3); step2.bandWithoutTop3 = est(BAND.filter(r => !top3.includes(r)), r => r.res);
// ---- Step 3: which teams and games
const bandGames = [...new Set(BAND.map(r => r.gid))], teamApps = new Map(); for (const g of bandGames) for (const tm of new Set(BAND.filter(r => r.gid === g).map(r => r.team))) teamApps.set(tm, (teamApps.get(tm) || 0) + 1);
const teamRows = [...new Set(BAND.map(r => r.team))].map(tm => { const B2 = BAND.filter(r => r.team === tm), O = REST.filter(r => r.team === tm); return { team: tm, bandGames: teamApps.get(tm), bandWrRows: B2.length, bandResSum: B2.reduce((s, r) => s + r.res, 0), bandResMean: mean(B2.map(r => r.res)), restResMean: O.length ? mean(O.map(r => r.res)) : null, restWrRows: O.length,
  bandTeamTgtMiss: mean(B2.filter(r => r.dec).map(r => r.teamTgtMiss).filter(Number.isFinite)) }; }).sort((a, b) => b.bandResSum - a.bandResSum);
const totRes = BAND.reduce((s, r) => s + r.res, 0), cum = teamRows.reduce((a, t, i) => { a.push((i ? a[i - 1] : 0) + t.bandResSum); return a; }, []);
const step3 = { bandGames: bandGames.length, bandTeamGames: [...teamApps.values()].reduce((a, b) => a + b, 0), distinctTeams: teamApps.size, teamsWith3PlusBandGames: [...teamApps.values()].filter(v => v >= 3).length, appearanceCounts: Object.fromEntries([1, 2, 3, 4, 5, 6].map(k => [k, [...teamApps.values()].filter(v => v === k).length])),
  shareOfBandResidualFromTop5Teams: cum[4] / totRes, shareFromTop10: cum[9] / totRes, teams: teamRows, perGame: bandGames.map(g => { const R = BAND.filter(r => r.gid === g); return { gid: g, wk: R[0].wk, total: R[0].total, wrRows: R.length, resSum: R.reduce((s, r) => s + r.res, 0) }; }).sort((a, b) => b.resSum - a.resSum) };
// ---- Step 4: did the same teams' WRs beat projection outside the band? Team effect: band-team WRs in non-band games vs other teams.
const heavy = new Set(teamRows.filter(t => t.bandGames >= 2).map(t => t.team)), top5 = new Set(teamRows.slice(0, 5).map(t => t.team));
const step4 = { bandTeams2Plus: [...heavy], top5Teams: [...top5],
  outsideBand: { bandTeams2Plus: est(REST.filter(r => heavy.has(r.team)), r => r.res), otherTeams: est(REST.filter(r => !heavy.has(r.team)), r => r.res), top5Teams: est(REST.filter(r => top5.has(r.team)), r => r.res) },
  teamCorrelation: (() => { const ts = [...new Set(rows.map(r => r.team))].map(tm => ({ tm, b: BAND.filter(r => r.team === tm), o: REST.filter(r => r.team === tm) })).filter(x => x.b.length >= 3 && x.o.length >= 3); const a = ts.map(x => mean(x.b.map(r => r.res))), b = ts.map(x => mean(x.o.map(r => r.res))); const ma = mean(a), mb = mean(b); return { teams: ts.length, corr: a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0) * b.reduce((s, v) => s + (v - mb) ** 2, 0)) }; })(),
  teamVolumeOutsideBand: { bandTeams2Plus: est(REST.filter(r => heavy.has(r.team) && r.dec), r => r.teamTgtMiss), otherTeams: est(REST.filter(r => !heavy.has(r.team) && r.dec), r => r.teamTgtMiss) } };
// ---- Step 5: would weeks 1–6 have caught it? Band vs rest by week group, and a team-level carry-over test.
const step5 = { byWeeks: {} }; for (const [k, sel] of [['weeks 1–6', r => r.wk <= 6], ['weeks 7–8', r => r.wk >= 7]]) { const R = rows.filter(sel); step5.byWeeks[k] = { bandRows: R.filter(r => r.band).length, bandGames: new Set(R.filter(r => r.band).map(r => r.gid)).size, ...Object.fromEntries(['residual (actual − proj)', 'targets: team volume', 'targets: share', 'efficiency (non-TD points per target)', 'TDs'].map(c => [c, diff(R, COMP[c])])) }; }
{ const ts = [...new Set(rows.map(r => r.team))].map(tm => { const a = rows.filter(r => r.team === tm && r.wk <= 6), b = rows.filter(r => r.team === tm && r.wk >= 7); return { tm, x: a.length ? mean(a.map(r => r.res)) : null, xv: a.filter(r => r.dec).length ? mean(a.filter(r => r.dec).map(r => r.teamTgtMiss)) : null, y: b.length ? mean(b.map(r => r.res)) : null }; }).filter(t => t.x != null && t.y != null && t.xv != null);
  const corr = (a, b) => { const ma = mean(a), mb = mean(b); return a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0) * b.reduce((s, v) => s + (v - mb) ** 2, 0)); };
  const bootC = (fx) => { const o = []; for (let b = 0; b < 1000; b++) { const S = Array.from({ length: ts.length }, () => ts[Math.floor(rnd() * ts.length)]); o.push(corr(S.map(fx), S.map(t => t.y))); } return sdv(o.filter(Number.isFinite)); };
  step5.teamCarryOver = { teams: ts.length, wk16ResidualVsWk78Residual: { mean: corr(ts.map(t => t.x), ts.map(t => t.y)), se: bootC(t => t.x) }, wk16TeamTargetMissVsWk78Residual: { mean: corr(ts.map(t => t.xv), ts.map(t => t.y)), se: bootC(t => t.xv) } }; }
audit.statsWeeks2026 = [...new Set([...st.values()].map(r => r.wk))].sort((a, b) => a - b); audit.noStatsInSample = rows.filter(r => r.tgt == null).length; audit.noStatsMeanActual = rows.filter(r => r.tgt == null).length ? mean(rows.filter(r => r.tgt == null).map(r => r.actual)) : null;
step5.perWeek = [1, 2, 3, 4, 5, 6, 7, 8].map(w => { const R = rows.filter(r => r.wk === w); return { week: w, bandGames: new Set(R.filter(r => r.band).map(r => r.gid)).size, bandRows: R.filter(r => r.band).length, band: est(R.filter(r => r.band), r => r.res), rest: est(R.filter(r => !r.band), r => r.res), bandMinusRest: diff(R, r => r.res) }; });
step3.bandGamesPositive = step3.perGame.filter(g => g.resSum > 0).length;
const out = { builtAt: new Date().toISOString(), minSalary: MIN_SAL, audit, rows: rows.length, decomposed: rows.filter(r => r.dec).length, bandRows: BAND.length, bandGameCount: bandGames.length, games: games.length, tdPerTargetPrior: tdPerTgt, step1, step2, step3, step4, step5 };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1)); console.log(JSON.stringify({ audit, rows: rows.length, decomposed: out.decomposed, band: BAND.length, bandGames: bandGames.length, games: games.length, bandRes: step1['residual (actual − proj)'].band, restRes: step1['residual (actual − proj)'].rest }));
