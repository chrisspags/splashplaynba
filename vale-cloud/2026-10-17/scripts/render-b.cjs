// Render the Unit B coverage table and off-par list from unit-b.json as Markdown.
const B=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
const pc=o=>o.share==null?'—':(100*o.share).toFixed(1)+'% ('+(100*o.se).toFixed(1)+')'+(o.z!=null&&Math.abs(o.z)>2?' **†**':'');
const L=['| Group | Cell | Players | Above p95 (par 5) | Above p90* (par 10) | Below p10 (par 10) | n for p10/p5 | Below p5* (par 5) |','|---|---|---:|---:|---:|---:|---:|---:|'];
const names={all:'All',salary:'Salary',position:'Position',gameTotal:'Game total',slateType:'Slate type',period:'Period'};
for(const [g,cells] of Object.entries(B.table))for(const [n,c] of Object.entries(cells))L.push('| '+names[g]+' | '+n+' | '+c.players+' | '+pc(c.above95)+' | '+pc(c.above90)+' | '+pc(c.below10)+' | '+c.below10.n+' | '+pc(c.below5)+' |');
console.log(L.join('\n'));console.log();
const M=['| Cell | Metric | Share (se) | Par | z | n / slates | Before 10/1: share (se), n | Since 10/1: share (se), n |','|---|---|---:|---:|---:|---:|---:|---:|'];
const q=o=>(100*o.share).toFixed(1)+'% ('+(100*o.se).toFixed(1)+'), '+o.n;
for(const o of B.offPar)M.push('| '+names[o.group]+': '+o.cell+' | '+o.metric+' | '+(100*o.share).toFixed(1)+'% ('+(100*o.se).toFixed(1)+') | '+(100*o.par)+'% | '+o.z.toFixed(2)+' | '+o.n+' / '+o.slates+' | '+q(o.byPeriod['before 10/1'])+' | '+q(o.byPeriod['since 10/1'])+' |');
console.log(M.join('\n'));
