// Vale unit 1103CL-N: the real shape of an RB1's bad games (nflverse 2021–25, DK scoring). Public data only; writes aggregates only.
// Usage: node unit-n.cjs <nflverse dir: stats_player_week_YYYY.csv, snap_counts_YYYY.csv, play_by_play_YYYY.csv.gz> <games.csv> <out.json> <spec.json>
const fs = require('fs'), path = require('path'), zlib = require('zlib');
const [, , ND, GAMES, OUT, SPEC] = process.argv;
const FIT = [2021, 2022, 2023, 2024], CHECK = 2025, K_SHRINK = 4, MIN_E = 5;
// ---- CSV (quote-aware)
function splitLine(l) { const o = []; let cur = '', q = false; for (let i = 0; i < l.length; i++) { const ch = l[i]; if (q) { if (ch === '"') { if (l[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += ch; } else if (ch === '"') q = true; else if (ch === ',') { o.push(cur); cur = ''; } else cur += ch; } o.push(cur); return o; }
function readCSV(text, cols) { const lines = text.split(/\r?\n/), H = splitLine(lines[0]), ix = cols.map(c => H.indexOf(c)); const out = [];
  for (let i = 1; i < lines.length; i++) { if (!lines[i]) continue; const f = splitLine(lines[i]); const r = {}; cols.forEach((c, j) => r[c] = ix[j] >= 0 ? f[ix[j]] : undefined); out.push(r); } return out; }
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const num = x => (x === '' || x == null || x === 'NA') ? 0 : +x;
const mean = a => a.reduce((s, x) => s + x, 0) / a.length, sdv = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
let seed = 1107; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const randn = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
// ---- schedules: closing lines, finals
const G = new Map(); for (const r of readCSV(fs.readFileSync(GAMES, 'utf8'), ['game_id', 'season', 'game_type', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'spread_line', 'total_line']))
  if (+r.season >= 2021 && +r.season <= 2025 && r.game_type === 'REG' && r.home_score !== 'NA' && r.home_score !== '') G.set(r.game_id, { season: +r.season, week: +r.week, home: r.home_team, away: r.away_team, spread: num(r.spread_line), total: num(r.total_line) });
// ---- halftime score from pbp (max running score in quarters 1–2)
const HALF = new Map();
for (const y of [2021, 2022, 2023, 2024, 2025]) { const t = zlib.gunzipSync(fs.readFileSync(path.join(ND, 'play_by_play_' + y + '.csv.gz'))).toString(); const lines = t.split('\n'), H = splitLine(lines[0]), gi = H.indexOf('game_id'), qi = H.indexOf('qtr'), hi = H.indexOf('total_home_score'), ai = H.indexOf('total_away_score');
  for (let i = 1; i < lines.length; i++) { if (!lines[i]) continue; const f = splitLine(lines[i]); if (!(+f[qi] <= 2)) continue; const g = f[gi], h = num(f[hi]), a = num(f[ai]); const cur = HALF.get(g) || { h: 0, a: 0 }; HALF.set(g, { h: Math.max(cur.h, h), a: Math.max(cur.a, a) }); } }
// ---- weekly stats (REG), DK points
const SC = ['player_id', 'player_display_name', 'position', 'season', 'week', 'season_type', 'game_id', 'team', 'opponent_team', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sack_fumbles_lost', 'passing_2pt_conversions', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions', 'receptions', 'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost', 'receiving_2pt_conversions', 'special_teams_tds', 'fantasy_points_ppr'];
const dk = r => { const py = num(r.passing_yards), ry = num(r.rushing_yards), cy = num(r.receiving_yards);
  return 0.04 * py + 4 * num(r.passing_tds) - num(r.passing_interceptions) + (py >= 300 ? 3 : 0) + 0.1 * ry + 6 * num(r.rushing_tds) + (ry >= 100 ? 3 : 0) + num(r.receptions) + 0.1 * cy + 6 * num(r.receiving_tds) + (cy >= 100 ? 3 : 0)
    - (num(r.rushing_fumbles_lost) + num(r.receiving_fumbles_lost) + num(r.sack_fumbles_lost)) + 2 * (num(r.passing_2pt_conversions) + num(r.rushing_2pt_conversions) + num(r.receiving_2pt_conversions)) + 6 * num(r.special_teams_tds); };
const stat = new Map(); // game|team|norm -> row
const byPlayerSeason = new Map(); // player_id|season -> [{week, dk}]
const nameToId = new Map(); // season|team|norm -> player_id
let ppCheck = { n: 0, diff: 0 };
for (const y of [2020, 2021, 2022, 2023, 2024, 2025]) for (const r of readCSV(fs.readFileSync(path.join(ND, 'stats_player_week_' + y + '.csv'), 'utf8'), SC)) {
  if (r.season_type !== 'REG') continue; const d = dk(r), wk = +r.week; if (r.position === 'RB') { ppCheck.n++; ppCheck.diff += d - num(r.fantasy_points_ppr); }
  const k = r.player_id + '|' + y; if (!byPlayerSeason.has(k)) byPlayerSeason.set(k, []); byPlayerSeason.get(k).push({ week: wk, dk: d });
  if (r.position === 'RB') { stat.set(r.game_id + '|' + r.team + '|' + norm(r.player_display_name), { id: r.player_id, dk: d, carries: num(r.carries), team: r.team }); nameToId.set(y + '|' + r.team + '|' + norm(r.player_display_name), r.player_id); } }
// ---- snap counts (RB, REG)
const snaps = new Map(); // game|team -> [{nk, pct}]
const snapHist = new Map(); // season|team|norm -> [{week, pct}]
for (const y of [2020, 2021, 2022, 2023, 2024, 2025]) for (const r of readCSV(fs.readFileSync(path.join(ND, 'snap_counts_' + y + '.csv'), 'utf8'), ['game_id', 'season', 'game_type', 'week', 'player', 'position', 'team', 'offense_snaps', 'offense_pct'])) {
  if (r.game_type !== 'REG' || r.position !== 'RB') continue; let pct = num(r.offense_pct); if (pct > 1.0001) pct /= 100; const nk = norm(r.player), k = r.game_id + '|' + r.team;
  if (!snaps.has(k)) snaps.set(k, []); snaps.get(k).push({ nk, pct, week: +r.week });
  const hk = y + '|' + r.team + '|' + nk; if (!snapHist.has(hk)) snapHist.set(hk, []); snapHist.get(hk).push({ week: +r.week, pct }); }
// Prior snap share: this season's earlier games, else last season (any team for last season is not tracked; same team only).
const priorSnap = (y, team, nk, wk) => { const a = (snapHist.get(y + '|' + team + '|' + nk) || []).filter(x => x.week < wk); if (a.length) return mean(a.map(x => x.pct)); const b = snapHist.get((y - 1) + '|' + team + '|' + nk) || []; return b.length ? mean(b.map(x => x.pct)) : null; };
// Expectation: this season's earlier games shrunk toward last season with weight K_SHRINK games.
const expectation = (id, y, wk) => { const cur = (byPlayerSeason.get(id + '|' + y) || []).filter(x => x.week < wk), last = byPlayerSeason.get(id + '|' + (y - 1)) || [];
  if (!last.length && cur.length < 2) return null; if (!last.length) return mean(cur.map(x => x.dk)); const ml = mean(last.map(x => x.dk)); return (cur.reduce((s, x) => s + x.dk, 0) + K_SHRINK * ml) / (cur.length + K_SHRINK); };
// ---- RB1 per team-game
const rows = [];
for (const [gid, g] of G) for (const [team, opp, isHome] of [[g.home, g.away, true], [g.away, g.home, false]]) {
  const sn = snaps.get(gid + '|' + team) || []; const cands = new Map();
  for (const s of sn) { const id = nameToId.get(g.season + '|' + team + '|' + s.nk) || nameToId.get((g.season - 1) + '|' + team + '|' + s.nk); if (!id) continue; cands.set(s.nk, { nk: s.nk, id, pct: s.pct }); }
  const list = [...cands.values()].map(c => ({ ...c, E: expectation(c.id, g.season, g.week) })).filter(c => c.E != null);
  if (!list.length) continue; const rb1 = list.sort((a, b) => b.E - a.E)[0]; if (rb1.E < MIN_E) continue;
  const st = stat.get(gid + '|' + team + '|' + rb1.nk), actual = st ? st.dk : 0;
  const rbCarries = sn.map(s => { const x = stat.get(gid + '|' + team + '|' + s.nk); return { nk: s.nk, c: x ? x.carries : 0 }; }), totC = rbCarries.reduce((a, x) => a + x.c, 0);
  const other40 = totC > 0 && rbCarries.some(x => x.nk !== rb1.nk && x.c / totC >= 0.4);
  const ps = priorSnap(g.season, team, rb1.nk, g.week), half = HALF.get(gid), ts = isHome ? -g.spread : g.spread;
  const trail14 = half ? ((isHome ? half.a - half.h : half.h - half.a) >= 14) : null;
  rows.push({ season: g.season, week: g.week, cl: g.season + '-' + g.week, team, E: rb1.E, actual, r: actual / rb1.E, snapPct: rb1.pct, priorSnap: ps, earlyExit: ps != null && ps >= 0.6 && rb1.pct < 0.4, trail14, committee: other40, teamSpread: ts, total: g.total, dog: ts > 0 });
}
const fit = rows.filter(r => FIT.includes(r.season)), chk = rows.filter(r => r.season === CHECK);
// Tiers: thirds of E on the fit set, applied to 2025.
const Es = fit.map(r => r.E).sort((a, b) => a - b), t1 = Es[Math.floor(Es.length / 3)], t2 = Es[Math.floor(2 * Es.length / 3)];
const tierOf = E => E < t1 ? 'low' : E < t2 ? 'mid' : 'top'; for (const r of rows) r.tier = tierOf(r.E);
const TIERS = ['low', 'mid', 'top'];
// ---- distributions
const Z = p => { // inverse normal (Acklam)
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239], b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572], c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783], d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - 0.02425) return -Z(1 - p); const q = p - 0.5, r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); };
const Phi = x => { const t = 1 / (1 + 0.2316419 * Math.abs(x)), dd = 0.3989423 * Math.exp(-x * x / 2), p = dd * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return x > 0 ? 1 - p : p; };
const qEmp = (a, p) => { const s = a.slice().sort((x, y) => x - y), h = (s.length - 1) * p, lo = Math.floor(h); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (h - lo); };
const QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], MID = []; for (let q = 0.10; q <= 0.9001; q += 0.025) MID.push(+q.toFixed(3));
const fitLN = rs => { const x = MID.map(q => Z(q)), y = MID.map(q => Math.log(Math.max(1e-6, qEmp(rs, q)))), mx = mean(x), my = mean(y); const s = x.reduce((a, v, i) => a + (v - mx) * (y[i] - my), 0) / x.reduce((a, v) => a + (v - mx) ** 2, 0); return { mu: my - s * mx, sigma: s }; };
const lnQ = (f, p) => Math.exp(f.mu + f.sigma * Z(p)), lnCdf = (f, r) => r <= 0 ? 0 : Phi((Math.log(r) - f.mu) / f.sigma);
// Gamma (shape k, scale th): regularized lower incomplete gamma by series / continued fraction.
const lgam = x => { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const v of c) s += v / ++y; return -t + Math.log(2.5066282746310005 * s / x); };
const gP = (k, x) => { if (x <= 0) return 0; if (x < k + 1) { let s = 1 / k, t = s; for (let n = 1; n < 500; n++) { t *= x / (k + n); s += t; if (t < s * 1e-12) break; } return s * Math.exp(-x + k * Math.log(x) - lgam(k)); }
  let b = x + 1 - k, c = 1e300, d = 1 / b, h = d; for (let i = 1; i < 500; i++) { const an = -i * (i - k); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-12) break; } return 1 - Math.exp(-x + k * Math.log(x) - lgam(k)) * h; };
const gQ = (k, th, p) => { let lo = 0, hi = 50 * th * Math.max(1, k); for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (gP(k, m / th) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
const fitGamma = rs => { const em = MID.map(q => qEmp(rs, q)); let best = null; for (let k = 0.5; k <= 30; k *= 1.03) { const base = MID.map(q => gQ(k, 1, q)); const th = em.reduce((a, v, i) => a + v * base[i], 0) / base.reduce((a, v) => a + v * v, 0); const err = em.reduce((a, v, i) => a + (v - th * base[i]) ** 2, 0); if (!best || err < best.err) best = { k, th, err }; } return best; };
const bootW = (R, stat, B = 400) => { const cls = [...new Set(R.map(r => r.cl))], by = new Map(cls.map(c => [c, R.filter(r => r.cl === c)])), o = []; for (let b = 0; b < B; b++) { const S = []; for (let i = 0; i < cls.length; i++) S.push(...by.get(cls[Math.floor(rnd() * cls.length)])); const v = stat(S); if (Number.isFinite(v)) o.push(v); } return o.length > 2 ? sdv(o) : null; };
const est = (R, stat, B) => ({ mean: stat(R), se: bootW(R, stat, B) });
// ---- Step 1: quantiles of actual ÷ expectation vs shapes fitted on the middle 80%
const step1 = {}; const fits = {};
for (const t of ['all', ...TIERS]) { const R = fit.filter(r => t === 'all' || r.tier === t), rs = R.map(r => r.r), L = fitLN(rs), Gm = fitGamma(rs); fits[t] = { ln: L, gamma: Gm };
  step1[t] = { n: R.length, meanRatio: est(R, S => mean(S.map(r => r.r))), sdRatio: est(R, S => sdv(S.map(r => r.r))), lognormalSd: Math.sqrt((Math.exp(L.sigma ** 2) - 1) * Math.exp(2 * L.mu + L.sigma ** 2)), meanShortfall: est(R, S => mean(S.map(r => r.actual - r.E))), lognormal: L, gamma: { shape: Gm.k, scale: Gm.th }, quantiles: {} };
  for (const q of QS) step1[t].quantiles['p' + String(Math.round(q * 100)).padStart(2, '0')] = { real: est(R, S => qEmp(S.map(r => r.r), q), 300), lognormal: lnQ(L, q), gamma: gQ(Gm.k, Gm.th, q) };
  step1[t].belowFittedP10 = est(R, S => mean(S.map(r => r.r < lnQ(L, 0.10) ? 1 : 0))); step1[t].belowFittedP05 = est(R, S => mean(S.map(r => r.r < lnQ(L, 0.05) ? 1 : 0)));
  step1[t].aboveFittedP95 = est(R, S => mean(S.map(r => r.r > lnQ(L, 0.95) ? 1 : 0))); step1[t].atOrBelowZero = est(R, S => mean(S.map(r => r.actual <= 0 ? 1 : 0)));
  step1[t].belowGammaP10 = est(R, S => mean(S.map(r => r.r < gQ(Gm.k, Gm.th, 0.10) ? 1 : 0))); }
// ---- Step 2: causes of the low tail (below the tier's fitted lognormal p10), assigned in order early exit > game script > committee > residual.
for (const r of rows) r.tail = r.r < lnQ(fits[r.tier].ln, 0.10);
const cause = r => r.earlyExit ? 'early exit' : r.trail14 ? 'game script (trailing 14+ at half)' : r.committee ? 'committee (another RB 40%+ of RB carries)' : 'residual';
const CAUSES = ['early exit', 'game script (trailing 14+ at half)', 'committee (another RB 40%+ of RB carries)', 'residual'];
const step2 = {}; for (const t of ['all', ...TIERS]) { const R = fit.filter(r => t === 'all' || r.tier === t), Tl = R.filter(r => r.tail); step2[t] = { tailGames: Tl.length, tailRate: est(R, S => mean(S.map(r => r.tail ? 1 : 0))), shares: {}, rateInTail: {}, rateOutsideTail: {} };
  for (const c of CAUSES) step2[t].shares[c] = est(Tl, S => mean(S.map(r => cause(r) === c ? 1 : 0)));
  for (const [k, f] of [['early exit', r => r.earlyExit], ['game script', r => r.trail14 === true], ['committee', r => r.committee]]) { step2[t].rateInTail[k] = est(Tl, S => mean(S.map(r => f(r) ? 1 : 0))); step2[t].rateOutsideTail[k] = est(R.filter(r => !r.tail), S => mean(S.map(r => f(r) ? 1 : 0))); } }
// ---- Step 3: do spread / total / dog status shift the tail probability? Logistic fit on 2021–24, scored on 2025.
const logit = (X, y, l2 = 1e-4) => { let b = new Array(X[0].length).fill(0); for (let it = 0; it < 50; it++) { const g = new Array(b.length).fill(0), Hm = b.map(() => new Array(b.length).fill(0));
    for (let i = 0; i < X.length; i++) { const z = X[i].reduce((s, v, j) => s + v * b[j], 0), p = 1 / (1 + Math.exp(-z)); for (let j = 0; j < b.length; j++) { g[j] += (y[i] - p) * X[i][j]; for (let k = 0; k < b.length; k++) Hm[j][k] -= p * (1 - p) * X[i][j] * X[i][k]; } }
    for (let j = 0; j < b.length; j++) { g[j] -= l2 * b[j]; Hm[j][j] -= l2; } const n = b.length, M = Hm.map((r, i) => [...r, -g[i]]); for (let i = 0; i < n; i++) { let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k; [M[i], M[p]] = [M[p], M[i]]; for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; } }
    const x = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } b = b.map((v, j) => v + x[j]); if (Math.max(...x.map(Math.abs)) < 1e-9) break; } return b; };
const tierD = r => [r.tier === 'mid' ? 1 : 0, r.tier === 'top' ? 1 : 0];
const MODELS = { 'tier only': r => [1, ...tierD(r)], 'tier + team spread': r => [1, ...tierD(r), r.teamSpread / 7], 'tier + total': r => [1, ...tierD(r), (r.total - 45) / 7], 'tier + underdog': r => [1, ...tierD(r), r.dog ? 1 : 0], 'tier + spread + total': r => [1, ...tierD(r), r.teamSpread / 7, (r.total - 45) / 7] };
const ll = (b, R, f) => mean(R.map(r => { const p = 1 / (1 + Math.exp(-f(r).reduce((s, v, j) => s + v * b[j], 0))); return -(r.tail ? Math.log(p) : Math.log(1 - p)); }));
const brier = (b, R, f) => mean(R.map(r => { const p = 1 / (1 + Math.exp(-f(r).reduce((s, v, j) => s + v * b[j], 0))); return ((r.tail ? 1 : 0) - p) ** 2; }));
const fitCls = [...new Set(fit.map(r => r.cl))], fitBy = new Map(fitCls.map(c => [c, fit.filter(r => r.cl === c)]));
const step3 = { models: {}, byGroup: {} }; const base = logit(fit.map(MODELS['tier only']), fit.map(r => r.tail ? 1 : 0));
for (const [k, f] of Object.entries(MODELS)) { const b = logit(fit.map(f), fit.map(r => r.tail ? 1 : 0)); const bs = []; for (let i = 0; i < 150; i++) { const S = []; for (let j = 0; j < fitCls.length; j++) S.push(...fitBy.get(fitCls[Math.floor(rnd() * fitCls.length)])); bs.push(logit(S.map(f), S.map(r => r.tail ? 1 : 0))); }
  step3.models[k] = { coef: b, coefSe: b.map((_, j) => sdv(bs.map(x => x[j]))), heldOutLogLoss: ll(b, chk, f), heldOutBrier: brier(b, chk, f), heldOutLogLossGainVsTierOnly: est(chk, S => ll(base, S, MODELS['tier only']) - ll(b, S, f), 300) }; }
for (const [k, sel] of [['favorite by 7+', r => r.teamSpread <= -7], ['favorite by <7', r => r.teamSpread < 0 && r.teamSpread > -7], ['underdog by <7', r => r.teamSpread > 0 && r.teamSpread < 7], ['underdog by 7+', r => r.teamSpread >= 7], ['total under 42', r => r.total < 42], ['total 42–47', r => r.total >= 42 && r.total <= 47], ['total over 47', r => r.total > 47]]) {
  const A = fit.filter(sel), Bk = chk.filter(sel); step3.byGroup[k] = { fit: { n: A.length, ...est(A, S => mean(S.map(r => r.tail ? 1 : 0))) }, check2025: { n: Bk.length, ...est(Bk, S => mean(S.map(r => r.tail ? 1 : 0))) } }; }
// ---- Step 4: mixture spec, fitted by maximum likelihood on 2021–24 with the mean held at 1 × expectation; checked on 2025.
// ratio ~ (1 − p) · LogNormal(mu_t, sigma_t) + p · Normal(m, s), with p = logistic(a_t + beta · teamSpread / 7) and mu_t set so the mixture mean is exactly 1.
const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * 2.5066282746);
const unpack = th => ({ a: { low: th[0], mid: th[1], top: th[2] }, beta: th[3], m: 1 / (1 + Math.exp(-th[4])), s: Math.exp(th[5]), sig: { low: Math.exp(th[6]), mid: Math.exp(th[7]), top: Math.exp(th[8]) }, M: th.length > 9 ? { low: Math.exp(th[9]), mid: Math.exp(th[10]), top: Math.exp(th[11]) } : { low: 1, mid: 1, top: 1 } });
const comp = (P, r) => { const p = 1 / (1 + Math.exp(-(P.a[r.tier] + P.beta * r.teamSpread / 7))), sig = P.sig[r.tier], mainMean = (P.M[r.tier] - p * P.m) / (1 - p); if (!(mainMean > 0)) return null; return { p, sig, mu: Math.log(mainMean) - sig * sig / 2, m: P.m, s: P.s }; };
const nll = th => { const P = unpack(th); let s = 0; if (th.length > 9 && Object.values(P.M).some(v => v < 0.5 || v > 2)) return 1e12; for (const r of fit) { const c = comp(P, r); if (!c) return 1e12; const x = r.r, f = (1 - c.p) * (x > 0 ? Math.exp(-0.5 * ((Math.log(x) - c.mu) / c.sig) ** 2) / (x * c.sig * 2.5066282746) : 0) + c.p * npdf(x, c.m, c.s); s -= Math.log(Math.max(f, 1e-300)); } return s; };
function nelder(f, x0, iters = 4000) { const n = x0.length; let S = [x0.slice()]; for (let i = 0; i < n; i++) { const x = x0.slice(); x[i] += 0.5; S.push(x); } let F = S.map(f);
  for (let it = 0; it < iters; it++) { const o = F.map((v, i) => i).sort((a, b) => F[a] - F[b]); S = o.map(i => S[i]); F = o.map(i => F[i]); const c = new Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += S[i][j] / n;
    const ref = c.map((v, j) => v + (v - S[n][j])), fr = f(ref); if (fr < F[0]) { const ex = c.map((v, j) => v + 2 * (v - S[n][j])), fe = f(ex); if (fe < fr) { S[n] = ex; F[n] = fe; } else { S[n] = ref; F[n] = fr; } } else if (fr < F[n - 1]) { S[n] = ref; F[n] = fr; }
    else { const con = c.map((v, j) => v + 0.5 * (S[n][j] - v)), fc = f(con); if (fc < F[n]) { S[n] = con; F[n] = fc; } else { for (let i = 1; i <= n; i++) { S[i] = S[i].map((v, j) => S[0][j] + 0.5 * (v - S[0][j])); F[i] = f(S[i]); } } }
    if (Math.abs(F[n] - F[0]) < 1e-7 * (1 + Math.abs(F[0]))) break; } return { x: S[0], f: F[0] }; }
// Fit by matching quantiles (the floor/ceiling rates are what the sim must get right): groups = tier × favorite/underdog.
const QG = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98], WQ = q => (q === 0.10 || q === 0.05 || q === 0.95) ? 3 : 1;
const groups = []; for (const t of TIERS) for (const dg of [false, true]) { const R = fit.filter(r => r.tier === t && r.dog === dg); groups.push({ R, xq: QG.map(q => qEmp(R.map(r => r.r), q)) }); }
const cdfRow = (Q, r, x) => { const c = comp(Q, r); if (!c) return NaN; return (1 - c.p) * (x > 0 ? Phi((Math.log(x) - c.mu) / c.sig) : 0) + c.p * Phi((x - c.m) / c.s); };
const qobj = th => { const Q = unpack(th); if (th.length > 9 && Object.values(Q.M).some(v => v < 0.5 || v > 2)) return 1e9; let o = 0; for (const g of groups) for (let i = 0; i < QG.length; i++) { const F = mean(g.R.map(r => cdfRow(Q, r, g.xq[i]))); if (!Number.isFinite(F)) return 1e9; o += WQ(QG[i]) * (F - QG[i]) ** 2; } return o; };
let best = null; for (const start of [[-2.2, -2.2, -2.2, 0, -1.5, -1.5, -0.8, -0.9, -1], [-1.5, -1.5, -1.5, 0.2, -2, -1.2, -0.9, -0.9, -0.9], [-2.5, -2.3, -2, 0, -1, -2, -0.7, -0.8, -1]]) { const r1 = nelder(qobj, start, 5000); const r2 = nelder(qobj, r1.x, 5000); if (!best || r2.f < best.f) best = r2; }
const P = unpack(best.x);
let bestB = null; for (const st of [[...best.x, 0, 0, 0], [...best.x, 0.2, 0, -0.08], [-1.5, -1.5, -1.5, 0, -1, -1.5, -0.8, -0.9, -1, 0.2, 0, -0.08]]) { const r1 = nelder(qobj, st, 6000); const r2 = nelder(qobj, r1.x, 6000); if (!bestB || r2.f < bestB.f) bestB = r2; }
const PB = unpack(bestB.x);
// Single lognormal with the mean held at 1 (sigma by tier fitted on the middle 80%), for comparison.
const single = t => { const s = fits[t].ln.sigma; return { mu: -s * s / 2, sigma: s }; };
const mixCdfOf = Q => (r, x) => { const c = comp(Q, r); return (1 - c.p) * (x > 0 ? Phi((Math.log(x) - c.mu) / c.sig) : 0) + c.p * Phi((x - c.m) / c.s); };
const mixCdf = (r, x) => { const c = comp(P, r); return (1 - c.p) * (x > 0 ? Phi((Math.log(x) - c.mu) / c.sig) : 0) + c.p * Phi((x - c.m) / c.s); };
const mixQ = (r, q) => { let lo = -3, hi = 20; for (let i = 0; i < 70; i++) { const m = (lo + hi) / 2; if (mixCdf(r, m) < q) lo = m; else hi = m; } return (lo + hi) / 2; };
const check = (R, cdf) => ({ floorRate: est(R, S => mean(S.map(r => cdf(r, r.r) < 0.10 ? 1 : 0))), ceilingRate: est(R, S => mean(S.map(r => cdf(r, r.r) > 0.95 ? 1 : 0))), p25Rate: est(R, S => mean(S.map(r => cdf(r, r.r) < 0.25 ? 1 : 0))), medianRate: est(R, S => mean(S.map(r => cdf(r, r.r) < 0.50 ? 1 : 0))) });
const singleCdf = (r, x) => { const L = single(r.tier); return lnCdf(L, x); };
const step4 = { fitMethod: 'quantile matching (p02–p98, weight 3 on p05/p10/p95) by tier × favorite/underdog on 2021–24', params: P, objective: best.f, paramsFreeMean: PB, objectiveFreeMean: bestB.f, fitGames: fit.length, check2025: { games: chk.length, mixture: {}, mixtureFreeMean: {}, singleLognormalMean1: {} }, fit2124: { mixture: {}, mixtureFreeMean: {}, singleLognormalMean1: {} } };
for (const t of ['all', ...TIERS]) { const C = chk.filter(r => t === 'all' || r.tier === t), F2 = fit.filter(r => t === 'all' || r.tier === t);
  step4.check2025.mixture[t] = { n: C.length, ...check(C, mixCdf) }; step4.check2025.mixtureFreeMean[t] = { n: C.length, ...check(C, mixCdfOf(PB)) }; step4.fit2124.mixtureFreeMean[t] = { n: F2.length, ...check(F2, mixCdfOf(PB)) }; step4.check2025.singleLognormalMean1[t] = { n: C.length, ...check(C, singleCdf) };
  step4.fit2124.mixture[t] = { n: F2.length, ...check(F2, mixCdf) }; step4.fit2124.singleLognormalMean1[t] = { n: F2.length, ...check(F2, singleCdf) }; }
// Shape table for a typical RB1 per tier (team spread −3 and +7): bust probability, main-part mean and quantiles vs the single lognormal.
step4.shapes = {}; for (const t of TIERS) for (const sp of [-7, -3, 3, 7]) { const r = { tier: t, teamSpread: sp }, c = comp(P, r), L = single(t);
  step4.shapes[t + ' / spread ' + (sp > 0 ? '+' : '') + sp] = { bustProb: c.p, mainMean: Math.exp(c.mu + c.sig * c.sig / 2), mainSigma: c.sig, mix: Object.fromEntries([0.05, 0.10, 0.50, 0.90, 0.95].map(q => ['p' + String(Math.round(q * 100)).padStart(2, '0'), mixQ(r, q)])), single: Object.fromEntries([0.05, 0.10, 0.50, 0.90, 0.95].map(q => ['p' + String(Math.round(q * 100)).padStart(2, '0'), lnQ(L, q)])), mixSd: Math.sqrt((1 - c.p) * (Math.exp(c.sig * c.sig) - 1) * Math.exp(2 * c.mu + c.sig * c.sig) + (1 - c.p) * (Math.exp(c.mu + c.sig * c.sig / 2) - 1) ** 2 + c.p * (c.s * c.s + (c.m - 1) ** 2)), singleSd: Math.sqrt((Math.exp(L.sigma ** 2) - 1)) }; }
const out = { builtAt: new Date().toISOString(), counts: { rb1Games: rows.length, fit: fit.length, check2025: chk.length, halftimeMissing: rows.filter(r => r.trail14 == null).length, priorSnapMissing: rows.filter(r => r.priorSnap == null).length }, ppCheck: { rbRows: ppCheck.n, meanDkMinusPpr: ppCheck.diff / ppCheck.n }, tierCuts: [t1, t2], kShrink: K_SHRINK, minE: MIN_E, step1, step2, step3, step4 };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
const spec = { unit: '1103CL-N', builtAt: out.builtAt, data: 'nflverse 2021–25 REG (fit 2021–24, check 2025), DK scoring', expectation: 'player mean this season before the game, shrunk toward last season with weight ' + K_SHRINK + ' games; RB1 = the team RB with the highest expectation; expectation >= ' + MIN_E,
  tiers: { cuts_dk_points: [t1, t2], names: ['low', 'mid', 'top'] }, model: 'ratio = actual / expectation ~ (1 - p) * LogNormal(mu_t, sigma_t) + p * Normal(m, s); p = 1 / (1 + exp(-(a_t + beta * teamSpread / 7))); teamSpread > 0 = underdog; mu_t = ln((1 - p*m)/(1 - p)) - sigma_t^2/2 so the mean is exactly 1',
  params: { a: P.a, beta: P.beta, m: P.m, s: P.s, sigma: P.sig }, meanHeldAt: 1, freeMeanVariant: { note: 'same form with the mixture mean per tier fitted instead of held at 1; shows the mean shift the data want', a: PB.a, beta: PB.beta, m: PB.m, s: PB.s, sigma: PB.sig, mean: PB.M }, check2025: { meanHeldAt1: step4.check2025.mixture, freeMean: step4.check2025.mixtureFreeMean }, typicalShapes: step4.shapes };
fs.writeFileSync(SPEC, JSON.stringify(spec, null, 1));
console.log(JSON.stringify({ counts: out.counts, ppCheck: out.ppCheck, tierCuts: out.tierCuts, params: P }));
