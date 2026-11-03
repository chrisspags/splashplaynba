// Vale unit 1103CL-D: what wins DK NFL Classic Afternoon Only and Main, and how the pre-lock sim's winner shares compare.
// Inputs (public): nfl-dk-{main,afternoon-only}-<dg>.json (results.playerResults, publicForecast, topStacks), nfl-dk-results/<date>.json.
// Usage: node unit-d.cjs <data dir> <out.json>. Aggregates only (no lineups, users or entries).
const fs = require('fs'), path = require('path');
const D = process.argv[2], OUT = process.argv[3];
const J = f => JSON.parse(fs.readFileSync(path.join(D, f), 'utf8'));
const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const TEAM = { JAC: 'JAX', LA: 'LAR', WSH: 'WAS' }, T = t => { const u = String(t || '').toUpperCase(); return TEAM[u] || u; };
const contestsByDG = {}; for (const f of fs.readdirSync(path.join(D, 'nfl-dk-results'))) for (const c of J('nfl-dk-results/' + f).contests) if (!c.isShowdown) (contestsByDG[c.draftGroupId] = contestsByDG[c.draftGroupId] || []).push(c);
const POS = ['QB', 'RB', 'WR', 'TE', 'DST'], bandOf = s => s < 4000 ? 'under $4,000' : s < 7000 ? '$4,000-$6,999' : '$7,000+', BANDS = ['under $4,000', '$4,000-$6,999', '$7,000+'];
const slates = [], players = [], stacks = [], coverage = { lineups: 0, lineupsUsed: 0, unmatched: 0, contests: 0 };
for (const f of fs.readdirSync(D).filter(f => /^nfl-dk-(main|afternoon-only)-\d+\.json$/.test(f))) {
  const w = J(f), id = w.identity, R = w.results, cs = contestsByDG[id.rawSlateId]; if (!cs) continue;
  const type = /afternoon/.test(f) ? 'Afternoon' : 'Main', week = +String(id.weekKey).replace(/.*W/, '');
  const P = new Map(); for (const p of Object.values(R.playerResults)) P.set(norm(p.name), { ...p, team: T(p.team) });
  const opp = {}, line = {}; for (const r of (R.publicForecast && R.publicForecast.rows) || []) { opp[T(r.team)] = T(r.opp); line[norm(r.name)] = r; }
  const gameOf = t => [t, opp[t]].sort().join('-');
  const sl = { dg: id.rawSlateId, type, week, date: id.date, contests: [] }, pAgg = new Map(), stAgg = new Map();
  for (const c of cs) {
    const S = c.summary || {}, cut = S.topThresholds && S.topThresholds.top1pct;
    const tops = (c.topFieldLineups || []).filter(l => l.players && l.players.length === 9 && (cut == null || l.points >= cut - 1e-6)); if (!tops.length) continue;
    coverage.contests++; const own = new Map(Object.entries(S.fieldOwnership || {}).map(([k, v]) => [norm(k), +v])), ow = nk => own.get(nk) || 0;
    const L = [];
    for (const l of tops) { coverage.lineups++; const ids = l.players.map(x => norm(x.name)); if (ids.some(nk => !P.has(nk))) { coverage.unmatched++; continue; } coverage.lineupsUsed++; L.push(ids); }
    if (!L.length) continue;
    const pc = nk => P.get(nk), share = f => L.reduce((s, x) => s + f(x), 0) / L.length;
    const feats = ids => { const qb = ids.find(nk => pc(nk).pos === 'QB'), qt = qb ? pc(qb).team : null;
      const stackN = qb ? ids.filter(nk => nk !== qb && (pc(nk).pos === 'WR' || pc(nk).pos === 'TE') && pc(nk).team === qt).length : 0;
      const bring = qb ? ids.some(nk => ['RB', 'WR', 'TE'].includes(pc(nk).pos) && pc(nk).team === opp[qt]) : false;
      const gcount = {}; for (const nk of ids) { const g = gameOf(pc(nk).team); gcount[g] = (gcount[g] || 0) + 1; }
      const dst = ids.find(nk => pc(nk).pos === 'DST'), rbDst = dst ? ids.some(nk => pc(nk).pos === 'RB' && pc(nk).team === pc(dst).team) : false;
      const sal = ids.reduce((s, nk) => s + pc(nk).salary, 0), so = ids.reduce((s, nk) => s + ow(nk), 0);
      return { qb, qt, stackN, bring, games: Object.keys(gcount).length, maxGame: Math.max(...Object.values(gcount)), rbDst, left: 50000 - sal, so,
        over30: ids.filter(nk => ow(nk) > 30).length, under5: ids.filter(nk => ow(nk) < 5).length }; };
    const F = L.map(feats), sf = g => F.reduce((s, x) => s + g(x), 0) / F.length;
    const m = { entries: S.totalEntries, topLineups: L.length, top: {
      'QB naked (0 same-team WR/TE)': sf(x => x.stackN === 0 ? 1 : 0), 'QB + 1 WR/TE': sf(x => x.stackN === 1 ? 1 : 0), 'QB + 2 WR/TE': sf(x => x.stackN === 2 ? 1 : 0), 'QB + 3+ WR/TE': sf(x => x.stackN >= 3 ? 1 : 0),
      'bring-back (opposing RB/WR/TE)': sf(x => x.bring ? 1 : 0), 'games used': sf(x => x.games), 'max players from one game': sf(x => x.maxGame), 'RB + DST same team': sf(x => x.rbDst ? 1 : 0),
      'salary left ($)': sf(x => x.left), 'salary left <= $200 (near-max)': sf(x => x.left <= 200 ? 1 : 0), 'salary left >= $1,000': sf(x => x.left >= 1000 ? 1 : 0),
      'sum of field own (%)': sf(x => x.so), 'players over 30% own (count)': sf(x => x.over30), 'players under 5% own (count)': sf(x => x.under5) }, field: {} };
    const all = [...P.keys()];
    m.field['salary left ($)'] = 50000 - all.reduce((s, nk) => s + ow(nk) / 100 * pc(nk).salary, 0);
    m.field['sum of field own (%)'] = all.reduce((s, nk) => s + ow(nk) ** 2 / 100, 0);
    m.field['players over 30% own (count)'] = all.filter(nk => ow(nk) > 30).reduce((s, nk) => s + ow(nk) / 100, 0);
    m.field['players under 5% own (count)'] = all.filter(nk => ow(nk) < 5).reduce((s, nk) => s + ow(nk) / 100, 0);
    sl.contests.push(m);
    for (const nk of all) { const a = pAgg.get(nk) || pAgg.set(nk, { real: [], own: [] }).get(nk); a.real.push(share(x => x.includes(nk) ? 1 : 0)); a.own.push(ow(nk) / 100); }
    // real share of top lineups by team-level QB stack shape (matching topStacks' team groups)
    const teams = [...new Set(all.map(nk => pc(nk).team))];
    // topStacks team groups are four exclusive shapes per team: QB+1 or QB+2+ (same-team WR/TE), with or without a bring-back.
    for (const t of teams) for (const [shape, test] of [['QB+1, no bring-back', x => x.qt === t && x.stackN === 1 && !x.bring], ['QB+1 + bring-back', x => x.qt === t && x.stackN === 1 && x.bring],
      ['QB+2+, no bring-back', x => x.qt === t && x.stackN >= 2 && !x.bring], ['QB+2+ + bring-back', x => x.qt === t && x.stackN >= 2 && x.bring]]) {
      const k = t + '|' + shape, a = stAgg.get(k) || stAgg.set(k, []).get(k); a.push(sf(x => test(x) ? 1 : 0)); }
  }
  if (!sl.contests.length) continue;
  const avg = a => a.reduce((s, x) => s + x, 0) / a.length;
  sl.top = {}; sl.field = {}; for (const k of Object.keys(sl.contests[0].top)) { sl.top[k] = avg(sl.contests.map(c => c.top[k])); const u = sl.contests.map(c => c.field[k]).filter(x => x != null); sl.field[k] = u.length ? avg(u) : null; }
  slates.push(sl);
  for (const [nk, a] of pAgg) { const p = P.get(nk), r = line[nk] || {}; players.push({ slate: sl.dg, type, week, pos: p.pos, salary: p.salary, proj: p.proj, ceiling: p.ceiling, stdDev: p.stdDev, exposure: p.exposure / 100,
    implied: r.implied != null ? +r.implied : null, total: r.total != null ? +r.total : null, spread: r.spread != null ? +r.spread : null, real: avg(a.real), field: avg(a.own) }); }
  // sim stack groups (team kind): lift = top10Rate relative to the 10/N baseline; real lift = real top share / sim field share
  const N = (R.topStacks && R.topStacks.fieldSize) || null;
  for (const g of ((R.topStacks && R.topStacks.groups) || []).filter(g => g.kind === 'team')) {
    const shape = (g.shape === 'QB+1' ? 'QB+1' : 'QB+2+') + (g.bringback ? ' + bring-back' : ', no bring-back'), k = T(g.team) + '|' + shape, real = stAgg.get(k);
    if (!real || !N) continue;
    stacks.push({ slate: sl.dg, type, week, team: T(g.team), shape, simFieldShare: g.fieldShare / 100, simCandidateShare: g.candidateShare / 100, simTop10Rate: g.top10Rate / 100, simTop10Lift: (g.top10Rate / 100) / (10 / N), simRoi: g.roi, real: avg(real) });
  }
}
// ---- slate bootstrap ----
const SL = slates.map(s => s.dg), sIx = new Map(SL.map((d, i) => [d, i]));
let seed = 1104; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const sdv = a => { const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
function boot(rows, keyOf, stat, B = 1000) { const ks = [...new Set(rows.map(keyOf))], by = new Map(ks.map(k => [k, rows.filter(r => keyOf(r) === k)]));
  const est0 = stat(rows), bs = []; for (let b = 0; b < B; b++) { const s = []; for (let i = 0; i < ks.length; i++) s.push(...by.get(ks[Math.floor(rnd() * ks.length)])); const v = stat(s); if (v != null && Number.isFinite(v)) bs.push(v); }
  return { n: rows.length, clusters: ks.length, mean: est0, se: bs.length > 1 ? sdv(bs) : null }; }
const meanOf = f => s => { const v = s.map(f).filter(x => x != null && Number.isFinite(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const anatomy = {}; for (const type of ['Afternoon', 'Main']) { const ss = slates.filter(s => s.type === type); anatomy[type] = {};
  for (const k of Object.keys(ss[0].top)) anatomy[type][k] = { top: boot(ss, s => s.dg, meanOf(s => s.top[k])), field: ss[0].field[k] != null ? boot(ss, s => s.dg, meanOf(s => s.field[k])) : null,
    topMinusField: ss[0].field[k] != null ? boot(ss, s => s.dg, meanOf(s => s.top[k] - s.field[k])) : null }; }
anatomy.afternoonMinusMain = {}; for (const k of Object.keys(slates[0].top)) { const a = slates.filter(s => s.type === 'Afternoon'), m = slates.filter(s => s.type === 'Main');
  const est = (A, M) => meanOf(s => s.top[k])(A) - meanOf(s => s.top[k])(M), bs = []; for (let b = 0; b < 1000; b++) { const A = a.map(() => a[Math.floor(rnd() * a.length)]), M = m.map(() => m[Math.floor(rnd() * m.length)]); bs.push(est(A, M)); }
  anatomy.afternoonMinusMain[k] = { mean: est(a, m), se: sdv(bs) }; }
// ---- sim vs real (players) ----
function ols(rows, yf, xf) { const fit = s => { let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0; for (const r of s) { const x = xf(r), y = yf(r); if (x == null || y == null) continue; n++; sx += x; sy += y; sxx += x * x; sxy += x * y; }
  const b = (sxy - sx * sy / n) / (sxx - sx * sx / n); return { a: sy / n - b * sx / n, b }; };
  const f0 = fit(rows), ks = [...new Set(rows.map(r => r.slate))], by = new Map(ks.map(k => [k, rows.filter(r => r.slate === k)])), bs = [];
  for (let i = 0; i < 1000; i++) { const s = []; for (let j = 0; j < ks.length; j++) s.push(...by.get(ks[Math.floor(rnd() * ks.length)])); bs.push(fit(s)); }
  return { n: rows.length, slates: ks.length, intercept: f0.a, interceptSe: sdv(bs.map(x => x.a)), slope: f0.b, slopeSe: sdv(bs.map(x => x.b)) }; }
const simReal = { calibration: {}, bias: {} };
for (const type of ['all', 'Afternoon', 'Main']) { const s = players.filter(r => type === 'all' || r.type === type);
  simReal.calibration[type] = { simExposure: ols(s, r => r.real, r => r.exposure), fieldOwn: ols(s, r => r.real, r => r.field) }; simReal.bias[type] = {};
  const groups = { all: () => true, ...Object.fromEntries(POS.map(p => ['pos ' + p, r => r.pos === p])), ...Object.fromEntries(BANDS.map(b => ['salary ' + b, r => bandOf(r.salary) === b])) };
  for (const [k, sel] of Object.entries(groups)) { const g = s.filter(sel); if (!g.length) continue;
    simReal.bias[type][k] = { real: boot(g, r => r.slate, meanOf(r => r.real)), sim: boot(g, r => r.slate, meanOf(r => r.exposure)), field: boot(g, r => r.slate, meanOf(r => r.field)),
      realMinusSim: boot(g, r => r.slate, meanOf(r => r.real - r.exposure)), realMinusField: boot(g, r => r.slate, meanOf(r => r.real - r.field)) }; } }
// ---- sim vs real (QB stacks, team groups) ----
const lg = x => Math.log(Math.max(x, 1e-4));
const spearman = (a, b) => { const rk = v => { const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]), r = new Array(v.length); let i = 0; while (i < o.length) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2; i = j + 1; } return r; };
  const x = rk(a), y = rk(b), n = a.length, mx = (n - 1) / 2; let s = 0, sx = 0, sy = 0; for (let i = 0; i < n; i++) { s += (x[i] - mx) * (y[i] - mx); sx += (x[i] - mx) ** 2; sy += (y[i] - mx) ** 2; } return sx && sy ? s / Math.sqrt(sx * sy) : null; };
const stackRes = {}; for (const type of ['all', 'Afternoon', 'Main']) { const s = stacks.filter(r => type === 'all' || r.type === type);
  const bySlate = {}; for (const r of s) (bySlate[r.slate] = bySlate[r.slate] || []).push(r);
  const rhoField = Object.values(bySlate).map(v => spearman(v.map(r => r.simFieldShare), v.map(r => r.real))).filter(x => x != null), rhoLift = Object.values(bySlate).map(v => spearman(v.map(r => r.simTop10Lift), v.map(r => r.real / Math.max(r.simFieldShare, 1e-4)))).filter(x => x != null);
  const byShape = {}; for (const sh of ['QB+1, no bring-back', 'QB+1 + bring-back', 'QB+2+, no bring-back', 'QB+2+ + bring-back']) { const g = s.filter(r => r.shape === sh);
    byShape[sh] = { groups: g.length, realTopShareSum: boot(Object.values(bySlate).map(v => ({ slate: v[0].slate, v: v.filter(r => r.shape === sh).reduce((a, r) => a + r.real, 0) })), r => r.slate, meanOf(r => r.v)),
      simFieldShareSum: boot(Object.values(bySlate).map(v => ({ slate: v[0].slate, v: v.filter(r => r.shape === sh).reduce((a, r) => a + r.simFieldShare, 0) })), r => r.slate, meanOf(r => r.v)) }; }
  stackRes[type] = { groups: s.length, slates: Object.keys(bySlate).length, withinSlateSpearman_simFieldShare_vs_realTopShare: rhoField.reduce((a, b) => a + b, 0) / rhoField.length,
    withinSlateSpearman_simTop10Lift_vs_realLift: rhoLift.reduce((a, b) => a + b, 0) / rhoLift.length, logLiftCalibration: ols(s.filter(r => r.simFieldShare > 0), r => lg(r.real / r.simFieldShare), r => lg(r.simTop10Lift)), byShape }; }
// Within-slate rank agreement of the real top-lineup share with each signal (does the sim rank players better than the field's own?).
const rankAgree = {}; for (const type of ['Afternoon', 'Main']) { const by = {}; for (const r of players.filter(r => r.type === type)) (by[r.slate] = by[r.slate] || []).push(r);
  const m = f => { const v = Object.values(by).map(s => spearman(s.map(f), s.map(r => r.real))).filter(x => x != null); return { slates: v.length, meanRho: v.reduce((a, b) => a + b, 0) / v.length, se: sdv(v) / Math.sqrt(v.length) }; };
  rankAgree[type] = { simExposure: m(r => r.exposure), fieldOwn: m(r => r.field), proj: m(r => r.proj), ceiling: m(r => r.ceiling) }; }
// Slate-level afternoon/main key features (aggregates per slate).
const perSlate = slates.map(s => ({ dg: s.dg, type: s.type, week: s.week, date: s.date, topLineups: s.contests.reduce((a, c) => a + c.topLineups, 0), salaryLeftTop: s.top['salary left ($)'], salaryLeftField: s.field['salary left ($)'], nearMaxTop: s.top['salary left <= $200 (near-max)'],
  qb2bring: s.top['bring-back (opposing RB/WR/TE)'], rbDst: s.top['RB + DST same team'], maxGame: s.top['max players from one game'] }));
// ---- held-out predictors: fit weeks 1-6, check weeks 7-9, per slate type ----
function lsq(X, y) { const k = X[0].length, A = Array.from({ length: k }, () => new Float64Array(k)), b = new Float64Array(k);
  for (let i = 0; i < X.length; i++) for (let p = 0; p < k; p++) { b[p] += X[i][p] * y[i]; for (let q = 0; q < k; q++) A[p][q] += X[i][p] * X[i][q]; }
  for (let p = 0; p < k; p++) A[p][p] += 1e-6;
  for (let c = 0; c < k; c++) { let piv = c; for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r; [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
    for (let r = 0; r < k; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let q = c; q < k; q++) A[r][q] -= f * A[c][q]; b[r] -= f * b[c]; } }
  return Array.from(b, (v, i) => v / A[i][i]); }
const posD = r => POS.map(p => r.pos === p ? 1 : 0);
const MODELS = { 'position only': r => posD(r), 'position + proj + salary': r => [...posD(r), r.proj, r.salary / 1000], 'position + sim exposure': r => [...posD(r), r.exposure],
  'position + proj + salary + sim exposure': r => [...posD(r), r.proj, r.salary / 1000, r.exposure], 'all pre-lock fields': r => [...posD(r), r.proj, r.salary / 1000, r.ceiling, r.stdDev, r.exposure, r.implied || 0, r.total || 0, r.spread || 0] };
const pred = {}, coef = {};
for (const type of ['Afternoon', 'Main', 'all']) { pred[type] = {}; const tr = players.filter(r => (type === 'all' || r.type === type) && r.week <= 6), ho = players.filter(r => (type === 'all' || r.type === type) && r.week >= 7);
  for (const [name, f] of Object.entries(MODELS)) { const beta = lsq(tr.map(r => [1, ...f(r)]), tr.map(r => r.real)), yh = r => [1, ...f(r)].reduce((s, x, i) => s + x * beta[i], 0);
    const mse = s => s.reduce((a, r) => a + (yh(r) - r.real) ** 2, 0) / s.length, vy = s => { const m = s.reduce((a, r) => a + r.real, 0) / s.length; return s.reduce((a, r) => a + (r.real - m) ** 2, 0) / s.length; };
    const bs = {}; for (const r of ho) (bs[r.slate] = bs[r.slate] || []).push(r); const rh = Object.values(bs).map(v => spearman(v.map(yh), v.map(r => r.real))).filter(x => x != null);
    pred[type][name] = { trainRows: tr.length, holdoutRows: ho.length, holdoutSlates: Object.keys(bs).length, trainR2: 1 - mse(tr) / vy(tr), holdoutR2: 1 - mse(ho) / vy(ho), holdoutWithinSlateSpearman: rh.reduce((a, b) => a + b, 0) / rh.length };
    if (name === 'all pre-lock fields') { // standardized slopes (per 1 sd of each field) on the training weeks
      const names = ['proj', 'salary', 'ceiling', 'stdDev', 'exposure', 'implied', 'total', 'spread'], sdOf = k => { const v = tr.map(r => k === 'salary' ? r.salary / 1000 : (r[k] || 0)); return sdv(v); };
      coef[type] = Object.fromEntries(names.map((k, i) => [k, +(beta[1 + POS.length + i] * sdOf(k)).toFixed(4)])); } } }
const out = { builtAt: new Date().toISOString(), slates: slates.map(s => ({ dg: s.dg, type: s.type, week: s.week, date: s.date, contests: s.contests.length, topLineups: s.contests.reduce((a, c) => a + c.topLineups, 0) })),
  coverage, playerSlates: players.length, stackGroups: stacks.length, anatomy, simReal, stacks: stackRes, rankAgreement: rankAgree, perSlate, predictors: pred, standardizedCoefficients: coef };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(JSON.stringify({ slates: slates.length, byType: slates.reduce((m, s) => (m[s.type] = (m[s.type] || 0) + 1, m), {}), coverage, playerSlates: players.length, stacks: stacks.length }));
