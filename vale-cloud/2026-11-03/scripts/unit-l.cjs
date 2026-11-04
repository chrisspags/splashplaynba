// Vale unit 1103CL-L: does the published NBA sim aim at the ownership band that wins?
// Reads origin/main directly (read-only git): dk-results/<night>.json and every committed version of nba-dk-*.json.
// A run qualifies for a results night only if its slateId is that night's contest draft group, its simUpdated is before the lock
// in its slateLabel (unit F's rule), and it carries no in-game fields (liveProj / liveFinal). Stokastic fields are never read.
// Usage: node unit-l.cjs <repo dir> <out.json> [--dry-run-postlock]   (the dry-run flag only checks mechanics; its output is not evidence)
const { execFileSync } = require('child_process'), fs = require('fs');
const REPO = process.argv[2], OUT = process.argv[3], DRY = process.argv.includes('--dry-run-postlock');
const git = (...a) => execFileSync('git', ['-C', REPO, ...a], { maxBuffer: 1 << 30 }).toString();
const show = (c, f) => { try { return JSON.parse(git('show', c + ':' + f)); } catch (e) { return null; } };
const norm = s => String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/\b(jr|sr|ii|iii|iv|v)\b\.?/g, '').replace(/[^a-z]/g, '');
const mean = a => a.reduce((x, y) => x + y, 0) / a.length, sd = a => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
let seed = 1106; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const BANDS = [['under 5%', 0, 5], ['5–15%', 5, 15], ['15–30%', 15, 30], ['30%+', 30, 1e9]], bandOf = o => BANDS.find(([, lo, hi]) => o >= lo && o < hi)[0], BN = BANDS.map(b => b[0]);
const parseET = (date, label) => { const m = /(\d+)\/(\d+) (\d+):(\d+)(AM|PM) Eastern/.exec(label || ''); if (!m) return null; const h = +m[3] % 12 + (m[5] === 'PM' ? 12 : 0); return date + 'T' + String(h).padStart(2, '0') + ':' + m[4]; };
const parseUpd = s => { const m = /(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+) (AM|PM)/.exec(s || ''); if (!m) return null; const h = +m[4] % 12 + (m[7] === 'PM' ? 12 : 0); return m[3] + '-' + m[1].padStart(2, '0') + '-' + m[2].padStart(2, '0') + 'T' + String(h).padStart(2, '0') + ':' + m[5]; };
// ---- results nights and their Classic draft groups
const nightsAll = git('ls-tree', '--name-only', 'origin/main', 'dk-results/').trim().split('\n').map(f => f.replace('dk-results/', '').replace('.json', '')).filter(d => d >= '2026-10-20').sort();
const results = Object.fromEntries(nightsAll.map(n => [n, show('origin/main', 'dk-results/' + n + '.json')]));
const dgNight = new Map(); for (const [n, R] of Object.entries(results)) for (const c of R.contests) if (!c.isShowdown) dgNight.set(String(c.draftGroupId), n);
// ---- every committed version of the NBA run files whose slateId is a results draft group
const versions = [], audit = { versionsChecked: 0, matchingDraftGroup: 0, rejectedPostLock: 0, rejectedLiveFields: 0 };
for (const line of git('log', '--format=%H', 'origin/main', '--', 'nba-dk-main.json', 'nba-dk-night.json', 'nba-dk-early.json', 'nba-dk-late-night.json', 'nba-dk-turbo.json').trim().split('\n').filter(Boolean)) {
  for (const f of ['nba-dk-main.json', 'nba-dk-night.json', 'nba-dk-early.json', 'nba-dk-late-night.json', 'nba-dk-turbo.json']) { const j = show(line, f); audit.versionsChecked++; if (!j || !dgNight.has(String(j.slateId))) continue; audit.matchingDraftGroup++;
    const lock = parseET(j.slateDate, j.slateLabel), upd = parseUpd(j.simUpdated), live = (j.playerResults || []).some(p => 'liveProj' in p || 'liveFinal' in p);
    const pre = lock && upd && upd < lock; if (!pre) audit.rejectedPostLock++; if (live) audit.rejectedLiveFields++;
    if ((pre && !live) || DRY) versions.push({ commit: line.slice(0, 10), file: f, dg: String(j.slateId), night: dgNight.get(String(j.slateId)), upd, preLock: !!pre, live, players: (j.playerResults || []).map(p => ({ nk: norm(p.name), exp: +(p.simExp != null ? p.simExp : p.winPct) || 0, proj: +p.proj || 0 })) }); } }
// Latest qualifying version per draft group (dedupe identical commits).
const pick = new Map(); for (const v of versions) { const k = v.dg, cur = pick.get(k); if (!cur || v.upd > cur.upd) pick.set(k, v); }
// ---- per contest: real top-1% slot share and points by band; sim winner share and expected points by band
const rows = [];
for (const [dg, v] of pick) { const R = results[v.night];
  for (const c of R.contests.filter(c => String(c.draftGroupId) === dg && !c.isShowdown)) { const S = c.summary, line = S.topThresholds.top1pct;
    const own = new Map(Object.entries(S.fieldOwnership || {}).map(([k, x]) => [norm(k), +x])), act = new Map(Object.values(c.playerActuals || {}).map(a => [norm(a.name), +a.fpts]));
    const tops = (c.topFieldLineups || []).filter(l => l.players && l.players.length === 8 && l.points >= line - 1e-6).map(l => l.players.map(p => norm(p.name))).filter(ids => ids.every(nk => act.has(nk)));
    if (!tops.length) continue; const sumOwn = [...own.values()].reduce((a, b) => a + b, 0);
    const r = { night: v.night, dg, contest: c.contestId, topLineups: tops.length, real: {}, sim: {} };
    for (const b of BN) { const inB = nk => bandOf(own.get(nk) || 0) === b;
      r.real[b] = { slotShare: mean(tops.map(ids => ids.filter(inB).length / 8)), fieldSlotShare: [...own].filter(([nk]) => inB(nk)).reduce((a, [, o]) => a + o, 0) / sumOwn,
        excessPoints: mean(tops.map(ids => ids.filter(inB).reduce((a, nk) => a + act.get(nk), 0))) - [...own].filter(([nk]) => inB(nk)).reduce((a, [nk, o]) => a + o / 100 * (act.get(nk) || 0), 0) * 800 / sumOwn };
      const ps = v.players.filter(p => inB(p.nk)), totExp = v.players.reduce((a, p) => a + p.exp, 0);
      r.sim[b] = { slotShare: ps.reduce((a, p) => a + p.exp, 0) / totExp, excessPoints: ps.reduce((a, p) => a + (8 * p.exp / totExp - (own.get(p.nk) || 0) / 100 * 800 / sumOwn) * p.proj, 0) }; }
    rows.push(r); } }
const nights = [...new Set(rows.map(r => r.night))].sort();
const bootN = (R, stat, B = 1000) => { const ns = [...new Set(R.map(r => r.night))]; if (ns.length < 2) return null; const o = []; for (let b = 0; b < B; b++) { const S = []; for (let i = 0; i < ns.length; i++) { const n = ns[Math.floor(rnd() * ns.length)]; for (const r of R) if (r.night === n) S.push(r); } o.push(stat(S)); } return sd(o); };
const est = (R, stat) => R.length ? { mean: stat(R), se: bootN(R, stat) } : null;
const out = { builtAt: new Date().toISOString(), dryRun: DRY, resultsNights: nightsAll, audit, qualifyingRuns: [...pick.values()].map(v => ({ night: v.night, dg: v.dg, file: v.file, commit: v.commit, simUpdated: v.upd, preLock: v.preLock, liveFields: v.live })), nights, contests: rows.length,
  step1: Object.fromEntries(BN.map(b => [b, { sim: est(rows, R => mean(R.map(r => r.sim[b].slotShare))), real: est(rows, R => mean(R.map(r => r.real[b].slotShare))), field: est(rows, R => mean(R.map(r => r.real[b].fieldSlotShare))), simMinusReal: est(rows, R => mean(R.map(r => r.sim[b].slotShare - r.real[b].slotShare))) }])),
  step2: Object.fromEntries(BN.map(b => [b, { simExpectedExcess: est(rows, R => mean(R.map(r => r.sim[b].excessPoints))), realExcess: est(rows, R => mean(R.map(r => r.real[b].excessPoints))), simMinusReal: est(rows, R => mean(R.map(r => r.sim[b].excessPoints - r.real[b].excessPoints))) }])) };
// Step 3: leave-one-night-out — real band excess (per night and band) on the sim's band tilt (sim slot share − field slot share).
const nb = []; for (const n of nights) { const R = rows.filter(r => r.night === n); for (const b of BN) nb.push({ night: n, band: b, x: mean(R.map(r => r.sim[b].slotShare - r.real[b].fieldSlotShare)), y: mean(R.map(r => r.real[b].excessPoints)) }); }
if (nights.length >= 3) { const o = []; for (const n of nights) { const tr = nb.filter(z => z.night !== n), te = nb.filter(z => z.night === n), mx = mean(tr.map(z => z.x)), my = mean(tr.map(z => z.y)); const b = tr.reduce((a, z) => a + (z.x - mx) * (z.y - my), 0) / tr.reduce((a, z) => a + (z.x - mx) ** 2, 0); for (const z of te) o.push({ night: n, y: z.y, yhat: my + b * (z.x - mx), base: my }); }
  const r2 = O => 1 - O.reduce((a, z) => a + (z.y - z.yhat) ** 2, 0) / O.reduce((a, z) => a + (z.y - z.base) ** 2, 0); out.step3 = { nightBands: nb.length, heldOutR2: r2(o), heldOutR2Se: bootN(o, r2) }; }
else out.step3 = { nightBands: nb.length, heldOutR2: null, note: 'needs 3 or more qualifying nights' };
// Step 4: the band the sim misses most (largest |sim − real| slot share, and in points).
if (rows.length) { const by = k => BN.map(b => ({ band: b, gap: out[k][b].simMinusReal.mean })).sort((a, c) => Math.abs(c.gap) - Math.abs(a.gap))[0]; out.step4 = { slotShare: by('step1'), points: by('step2') }; }
fs.writeFileSync(OUT, JSON.stringify(out, null, 1)); console.log(JSON.stringify({ dryRun: DRY, audit, qualifying: out.qualifyingRuns.length, nights, contests: rows.length }));
