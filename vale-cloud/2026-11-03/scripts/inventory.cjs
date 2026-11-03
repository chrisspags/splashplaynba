// Vale unit 1103CL, step 1: what the public data repo publishes per DK NFL run, and whether the sim's per-player draw
// inputs are among it. Usage: node inventory.cjs <dir holding the repo's nfl-dk-*.json pointer files>. Prints aggregates only.
const fs = require('fs'), path = require('path'), D = process.argv[2];
const files = fs.readdirSync(D).filter(f => /^nfl-dk-(main|early-only|afternoon-only|showdown)-\d+\.json$/.test(f));
const byWeek = {}, schemas = {}, prKeys = new Set(), rowKeys = new Set(), resKeys = new Set(), privates = new Set();
let draws = 0;
for (const f of files) {
  const w = JSON.parse(fs.readFileSync(path.join(D, f), 'utf8')), id = w.identity || {}, type = f.replace(/^nfl-dk-|-\d+\.json$/g, '');
  schemas[w.schema] = (schemas[w.schema] || 0) + 1;
  const wk = id.weekKey || '?'; byWeek[wk] = byWeek[wk] || {}; byWeek[wk][type] = (byWeek[wk][type] || 0) + 1;
  const R = w.results || {}; Object.keys(R).forEach(k => resKeys.add(k));
  for (const p of Object.values(R.playerResults || {})) Object.keys(p).forEach(k => prKeys.add(k));
  for (const r of (R.publicForecast && R.publicForecast.rows) || []) Object.keys(r).forEach(k => rowKeys.add(k));
  if (w.private) Object.keys(w.private).forEach(k => privates.add(k));
  const s = JSON.stringify(w); if (/"q21"|"reserveKernels"|"kickerKernels"|"aMarg"|"inputSnapshot"|"outcomeReplay"|"workerSource"/.test(s)) draws++;
}
console.log(JSON.stringify({ pointers: files.length, schemas, byWeek, resultsKeys: [...resKeys].sort(), playerResultsFields: [...prKeys].sort(),
  publicForecastRowFields: [...rowKeys].sort(), privateBlockFields: [...privates].sort(), pointersCarryingDrawInputs: draws }, null, 1));
