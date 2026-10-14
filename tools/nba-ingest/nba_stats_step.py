#!/usr/bin/env python3
"""The stats step in the data job (Vale's nba_stats.py, 10/13, run exactly as he hands it over).

Chris, 10/13, through Monty: "make sure all the old colab datapoints that connected to on-site data displays are now linked to
our new data storage. so everything under research as well as the stats page overall".

Once a run, after the projection step, it unpacks the reviewed step from nba_projection_bundle.py and runs
    NBA_ROOT=<repo> NBA_CACHE=<scratch> python3 nba_stats.py --out <scratch file> --season <season>
then, when the answer is a whole stats file for that season, publishes it as nba-stats.json: one row a player by ESPN id, his
season numbers (this season's once he has five games in it, else last season's; the row says which), the substitution chain, and
five numbers that have no source yet, written empty. The page's Build press fills the Stats tab and the Matchups columns from it.

What it writes: nba-stats.json and its entry in the run's status file under "stats". Nothing else, and never data.json.

It is soft, like the projection step: when it cannot make the file (numpy missing, the two past seasons not in the repo, the step
failing, an answer that is not a stats file for this season) it writes no file, says why in the status file, and the run goes on.
It never writes a thinner file: when the file in the repo is for the same season and holds more players than this run would
write, the file is left as it is and the status says so.

Usage (a step of run.py):  python3 nba_stats_step.py --season 2026-27 --repo . --cache .nba-cache"""
import json,os,shutil,subprocess,sys,time,argparse
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nba_profiles as P  # noqa: E402
import nba_project_step as S  # noqa: E402

FORMAT='nba-stats-step-v1'
OUT='nba-stats.json'

def whole(doc,season):
    """None when doc is a stats file for the season; else why it is not, in plain words."""
    if not isinstance(doc,dict) or doc.get('format')!=FORMAT:return 'the step answered with something that is not a stats file'
    if doc.get('season')!=season:return 'the step answered for season %s, not %s'%(str(doc.get('season'))[:12],season)
    rows=doc.get('rows')
    if not isinstance(rows,list) or not rows:return 'the stats file has no rows'
    if any(not isinstance(r,dict) or not r.get('espnId') or not r.get('name') for r in rows):return 'a row of the stats file has no ESPN id or no name'
    if len({str(r['espnId']) for r in rows})!=len(rows):return 'the stats file lists a player twice'
    return None

def held_rows(repo,season):
    """How many players the stats file in the repo holds for this season (0 when there is none, or it is another season's)."""
    try:d=json.load(open(os.path.join(repo,OUT)))
    except Exception:return 0
    return len(d['rows']) if isinstance(d,dict) and d.get('season')==season and isinstance(d.get('rows'),list) else 0

def make(repo,season,scratch,stats_py,python=sys.executable,run=subprocess.run):
    """Runs the step. Returns the status entry; publishes nba-stats.json only for a whole answer that is not thinner."""
    t0=time.time();tmp=os.path.join(scratch,'stats-out.json');hist=os.path.join(scratch,'history')
    if os.path.exists(tmp):os.remove(tmp)
    shutil.rmtree(hist,ignore_errors=True)   # never a parsed history left by an earlier run
    try:
        r=run([python,stats_py,'--out',tmp,'--season',season],capture_output=True,text=True,timeout=900,env={**os.environ,'NBA_ROOT':os.path.abspath(repo),'NBA_CACHE':hist})
        if r.returncode:return {'wrote':False,'why':'the step exited %s: %s'%(r.returncode,((r.stderr or r.stdout or '').strip().splitlines() or ['no message'])[-1][:200])}
        try:doc=json.load(open(tmp))
        except Exception as e:return {'wrote':False,'why':'the step wrote no readable file (%s)'%type(e).__name__}
        why=whole(doc,season)
        if why:return {'wrote':False,'why':why}
        rows=doc['rows'];held=held_rows(repo,season)
        count={'rows':len(rows),'onThisSeason':sum(1 for x in rows if x.get('season')==season),'withChain':sum(1 for x in rows if x.get('substitutionChain')),
               'noSourceYet':[str(x) for x in (doc.get('noSourceYet') or [])][:12]}
        if len(rows)<held:return {'wrote':False,**count,'why':'the file in the repo holds %d players for %s and this run would write %d; left as it is'%(held,season,len(rows))}
        # published as the step wrote it, byte for byte, so the file in the repo is the file a direct run of the step gives
        shutil.copyfile(tmp,os.path.join(repo,OUT))
        return {'wrote':True,**count,'seconds':round(time.time()-t0,1)}
    finally:
        shutil.rmtree(hist,ignore_errors=True)   # the parsed history is about 120 MB: never left for the workflow's cache to carry
        if os.path.exists(tmp):os.remove(tmp)

def main(argv=None,bundle=None,numpy=S.numpy_version,run=subprocess.run):
    ap=argparse.ArgumentParser(description='Run the NBA stats step and publish nba-stats.json')
    ap.add_argument('--season',required=True);ap.add_argument('--repo',default='.');ap.add_argument('--cache',default='.nba-cache')
    a=ap.parse_args(argv)
    if bundle is None:import nba_projection_bundle as bundle
    entry={'step':bundle.BUNDLE_ID[:12],'season':a.season}
    try:
        nv=numpy();entry['numpy']=nv;miss=S.missing_history(a.repo)
        if not nv:entry.update(wrote=False,why='numpy is not installed for this run')
        elif miss:entry.update(wrote=False,why='the history the stats are made from is not in the repo: '+', '.join(miss))
        else:
            scratch=os.path.join(a.cache,'stats');code=os.path.join(scratch,'step');shutil.rmtree(code,ignore_errors=True);os.makedirs(code,exist_ok=True)
            try:stats_py=os.path.join(os.path.dirname(bundle.extract(code)),'nba_stats.py')
            except Exception as e:stats_py=None;entry.update(wrote=False,why='the step could not be unpacked: %s'%str(e)[:160])
            if stats_py and not os.path.exists(stats_py):entry.update(wrote=False,why='the step that was unpacked has no nba_stats.py')
            elif stats_py:entry.update(make(a.repo,a.season,scratch,stats_py,run=run))
    except Exception as e:entry.update(wrote=False,why='%s: %s'%(type(e).__name__,str(e)[:200]))
    print(json.dumps(entry));P.note(a.repo,entry,'stats')
    return 0

if __name__=='__main__':
    sys.exit(main())
