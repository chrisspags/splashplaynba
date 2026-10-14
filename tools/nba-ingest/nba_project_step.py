#!/usr/bin/env python3
"""The projection step in the data job (Vale's step, run exactly as he hands it over; opening-night plan, Thursday's piece).

For each date the tonight step wrote in this run that has joined DraftKings rows, it unpacks the reviewed step from
nba_projection_bundle.py and runs
    NBA_ROOT=<repo> NBA_CACHE=<scratch> python3 run_step.py --tonight nba-tonight/<date>.json --out <scratch file>
        [--injury <file>] [--lineups <file>] [--typed <file>] --team-minutes 241
then, when the answer is a whole projection file for that date, publishes it as nba-proj/<date>.json.

What the step is handed, each press (Vale 10/13): the league's newest injury report, read by the step's own nba_injury.py fetch
(at most eight header requests and two files, under its own name); the NBA's lineups file for the date, one plain read; and the
minutes Chris typed for that slate date, from minutes-data.json. Each is optional: when the league does not answer, or a file is
not there, the step runs without it and the status file says which and why. Nothing is disguised and nothing is retried.

What it writes: nba-proj/<date>.json, its entry in the run's status file under "projection", and the night's record of the news it
was handed (Vale's DAILY-RECORD.md, item 3): nba-record/<date>/injury-<UTC time>.json, the league's report exactly as the reader
answered it (or its "unavailable" answer with the reason), and lineups-<UTC time>.json, the lineups file as read (or that it was
not read, and why). A new file each press, never one replaced. Nothing else. It never writes
data.json: the page builds that from this file when an admin presses Build (Monty 10/12: nothing writes data.json except a build
Chris pressed). The step itself refuses that name too.

It is soft, like the tonight step in the job: a date that cannot be projected (numpy missing, the two past seasons not in the
repo, the step failing, an answer that is not a projection file for that date) writes no file for that date, says why in the
status file, and the run's history and tonight files are still committed.

The step keeps a parsed copy of the history in its scratch folder and reuses it when it is there, so the scratch folder is
emptied before every run: a run always reads the history that is in the repo now.

Usage (a step of run.py):  python3 nba_project_step.py --repo . --cache .nba-cache [--date 2026-10-20]"""
import json,os,shutil,subprocess,sys,time,argparse
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nba_profiles as P  # noqa: E402

LINEUPS='https://stats.nba.com/js/data/leaders/00_daily_lineups_{}.json'   # the NBA's own daily lineups file, one per US Eastern date
AGENT='splashplay-nba-data-job (python urllib)'
TEAM_MINUTES='241'
PAST=('2024-25','2025-26')   # the seasons the model looks back across; both must be in the repo as the job writes them
FORMAT_KEYS=('date','games','rows')

def numpy_version(python=sys.executable):
    """numpy's version in the interpreter that will run the step, or None when it is not installed there."""
    try:r=subprocess.run([python,'-c','import numpy;print(numpy.__version__)'],capture_output=True,text=True,timeout=60)
    except Exception:return None
    return r.stdout.strip() if r.returncode==0 and r.stdout.strip() else None

def missing_history(repo):
    """What the model needs from the repo and is not there, as plain words; [] when it is all there."""
    out=[]
    for s in PAST:
        if not os.path.exists(os.path.join(repo,'nba-events',s,'index.json')):out.append('nba-events/%s'%s)
        if not os.path.exists(os.path.join(repo,'nba-stints',s+'.json.gz')):out.append('nba-stints/%s.json.gz'%s)
    return out

def wanted_dates(repo,date=None):
    """(dates to project, dates passed over with the reason). With no --date: the dates the tonight step wrote in this run."""
    if date:return [date],[]
    try:td=(json.load(open(os.path.join(repo,'nba-ingest-status.json'))).get('tonight') or {}).get('dates') or []
    except Exception:td=[]
    go=[];skip=[]
    for x in td:
        if not isinstance(x,dict) or not x.get('date'):continue
        if not x.get('wrote'):skip.append({'date':x['date'],'wrote':False,'why':'the tonight file for this date was not written in this run'})
        elif not x.get('slateRowsJoined'):skip.append({'date':x['date'],'wrote':False,'why':'no DraftKings slate rows are joined for this date'})
        else:go.append(x['date'])
    return go,skip

def plain_get(url,timeout=8):
    """One plain request under the job's own name; no retry. Returns the parsed JSON or raises."""
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':AGENT,'Accept':'application/json'}),timeout=timeout) as r:return json.loads(r.read())

def injury_file(scratch,run_step,python=sys.executable,run=subprocess.run):
    """(path or None, what to say). The step's own reader fetches the league's report; "unavailable" is an answer and is handed on."""
    out=os.path.join(scratch,'injury.json')
    if os.path.exists(out):os.remove(out)
    try:run([python,os.path.join(os.path.dirname(run_step),'nba_injury.py'),'fetch','--out',out],capture_output=True,text=True,timeout=120)
    except Exception as e:return None,'not read (%s)'%type(e).__name__
    try:d=json.load(open(out))
    except Exception:return None,'not read (the reader wrote no file)'
    return out,('read' if d.get('ok') else str(d.get('error') or 'unavailable')[:160])

def lineups_file(date,scratch,get=plain_get):
    """(path or None, what to say). One plain read of the NBA's lineups file for the date."""
    out=os.path.join(scratch,'lineups-'+date+'.json')
    if os.path.exists(out):os.remove(out)
    try:d=get(LINEUPS.format(date.replace('-','')))
    except Exception as e:return None,'lineups file not read (%s)'%(str(e)[:80] or type(e).__name__)
    if not isinstance(d,dict) or not isinstance(d.get('games'),list):return None,'lineups file not read (not the file we know)'
    json.dump(d,open(out,'w'));return out,'read, %d game(s)'%len(d['games'])

# ESPN's DraftKings player props for one game (Vale's order, props item G; README of the props step 10/14): every item of every page
PROPS_URL='https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e}/competitions/{e}/odds/100/propBets?limit=1000&page={p}'
PROPS_MAX_PAGES=10

def props_file(date,repo,scratch,get=plain_get):
    """(path or None, what it said). ESPN's DraftKings player props for the date's NBA games, read on the run his update starts (never
    a clock), every page of each game, kept whole as {event id: {count, pageCount, items}} for the step's --props (nba_props.from_feed
    reads the items, in the order the feed listed them: the step's tripwire judges that order). A game whose read fails or that has no
    props is named; with no props at all the step runs without them, as before."""
    out=os.path.join(scratch,'props-'+date+'.json')
    if os.path.exists(out):os.remove(out)
    try:games=[g for g in json.load(open(os.path.join(repo,'nba-tonight',date+'.json'))).get('games') or [] if isinstance(g,dict) and g.get('nbaGame') and g.get('eventId')]
    except Exception as e:return None,{'used':False,'why':'the tonight file could not be read (%s)'%type(e).__name__}
    have={};none=[];failed=[]
    for g in games:
        e=str(g['eventId']);items=[];page=1;pages=1
        try:
            while page<=pages and page<=PROPS_MAX_PAGES:
                j=get(PROPS_URL.format(e=e,p=page))
                if not isinstance(j,dict) or not isinstance(j.get('items'),list):raise ValueError('not the feed we know')
                items+=j['items'];pages=int(j.get('pageCount') or 1);page+=1
        except Exception as x:failed.append({'eventId':e,'why':(str(x)[:80] or type(x).__name__)});continue
        if items:have[e]={'count':len(items),'pageCount':pages,'items':items}
        else:none.append(e)
    say={'games':len(games),'withProps':len(have),'items':sum(v['count'] for v in have.values()),'noProps':none,'failed':failed}
    if not have:say.update(used=False,why='no game had props' if games else 'no NBA game in the tonight file');return None,say
    json.dump(have,open(out,'w'));say['used']=True;return out,say

def typed_file(date,repo,scratch):
    """(path or None, how many). The minutes he typed for THIS slate date, by ESPN id, from the minutes file in the repo."""
    out=os.path.join(scratch,'typed-'+date+'.json')
    if os.path.exists(out):os.remove(out)
    try:d=json.load(open(os.path.join(repo,'minutes-data.json')))
    except Exception:return None,0
    if not isinstance(d,dict) or d.get('slateDate')!=date:return None,0
    t={}
    for p in d.get('players') or []:
        if not isinstance(p,dict):continue   # an entry that is not a player's row is passed over (Monty 10/13: it used to stop the step)
        v=p.get('typedMinutes')
        if p.get('espnId') and isinstance(v,(int,float)) and not isinstance(v,bool) and 0<=v<=48:t[str(p['espnId'])]=v
    if not t:return None,0
    json.dump(t,open(out,'w'));return out,len(t)

def record_news(repo,date,stamp,inj_path,lu_path,lu_say,props_path=None,props_say=None):
    """The news as it reached us, for the night's record: one new file each, per date, never replaced. Returns the names written.
    The props pull is kept whole beside them (props-<UTC time>.json), or its "not used" answer when there was none."""
    d=os.path.join(repo,'nba-record',date);os.makedirs(d,exist_ok=True);wrote=[]
    for name,src,alt in (('injury',inj_path,{'ok':False,'error':'the injury reader wrote no answer'}),('lineups',lu_path,{'read':False,'why':lu_say}),('props',props_path,dict(props_say or {'used':False,'why':'not read'}))):
        fn='%s-%s.json'%(name,stamp);dst=os.path.join(d,fn)
        if os.path.exists(dst):continue
        if src and os.path.exists(src):shutil.copyfile(src,dst)
        else:
            with open(dst,'w') as f:json.dump(alt,f)
        wrote.append(fn)
    return wrote

def project(date,repo,scratch,run_step,python=sys.executable,run=subprocess.run,handed=()):
    """Runs the step for one date. Returns the status entry; publishes nba-proj/<date>.json only for a whole answer."""
    t0=time.time();tonight=os.path.join(repo,'nba-tonight',date+'.json');tmp=os.path.join(scratch,'out-'+date+'.json')
    if not os.path.exists(tonight):return {'date':date,'wrote':False,'why':'nba-tonight/%s.json is not in the repo'%date}
    if os.path.exists(tmp):os.remove(tmp)
    hist=os.path.join(scratch,'history');shutil.rmtree(hist,ignore_errors=True)   # never a parsed history left by an earlier run
    try:
        r=run([python,run_step,'--tonight',tonight,'--out',tmp,*handed,'--team-minutes',TEAM_MINUTES],capture_output=True,text=True,timeout=900,env={**os.environ,'NBA_ROOT':os.path.abspath(repo),'NBA_CACHE':hist})
        if r.returncode:return {'date':date,'wrote':False,'why':'the step exited %s: %s'%(r.returncode,((r.stderr or r.stdout or '').strip().splitlines() or ['no message'])[-1][:200])}
        try:doc=json.load(open(tmp))
        except Exception as e:return {'date':date,'wrote':False,'why':'the step wrote no readable file (%s)'%type(e).__name__}
        if not isinstance(doc,dict) or any(k not in doc for k in FORMAT_KEYS) or doc.get('date')!=date or not isinstance(doc.get('rows'),list) or not isinstance(doc.get('games'),list):
            return {'date':date,'wrote':False,'why':'the step answered with something that is not a projection file for %s'%date}
        rows=[x for x in doc['rows'] if isinstance(x,dict)]
        # published as the step wrote it, byte for byte, so the file in the repo is the file a direct run of the step gives
        dst=os.path.join(repo,'nba-proj',date+'.json');os.makedirs(os.path.dirname(dst),exist_ok=True);shutil.copyfile(tmp,dst)
        return {'date':date,'wrote':True,'games':len(doc['games']),'rows':len(rows),'pool15':sum(1 for x in rows if (x.get('proj') or 0)>=15),
                'notes':len(doc.get('notes') or []),'seconds':round(time.time()-t0,1)}
    finally:
        # the parsed history is about 120 MB: never left behind for the workflow's cache to carry from run to run
        shutil.rmtree(hist,ignore_errors=True)
        if os.path.exists(tmp):os.remove(tmp)

def main(argv=None,bundle=None,numpy=numpy_version,run=subprocess.run,get=plain_get,now=None):
    ap=argparse.ArgumentParser(description='Run the NBA projection step for the dates the tonight step wrote')
    ap.add_argument('--repo',default='.');ap.add_argument('--cache',default='.nba-cache');ap.add_argument('--date',help='one US Eastern date, YYYY-MM-DD (default: the dates the tonight step wrote in this run)')
    a=ap.parse_args(argv)
    if bundle is None:import nba_projection_bundle as bundle
    dates,done=wanted_dates(a.repo,a.date);entry={'step':bundle.BUNDLE_ID[:12]}
    why=None
    if dates:
        nv=numpy();entry['numpy']=nv;miss=missing_history(a.repo)
        if not nv:why='numpy is not installed for this run'
        elif miss:why='the history the model reads is not in the repo: '+', '.join(miss)
    if dates and why:done+=[{'date':d,'wrote':False,'why':why} for d in dates]
    elif dates:
        scratch=os.path.join(a.cache,'projection');code=os.path.join(scratch,'step');shutil.rmtree(code,ignore_errors=True);os.makedirs(code,exist_ok=True)
        try:run_step=bundle.extract(code)
        except Exception as e:run_step=None;done+=[{'date':d,'wrote':False,'why':'the step could not be unpacked: %s'%str(e)[:160]} for d in dates]
        inj,inj_say=injury_file(scratch,run_step,run=run) if run_step else (None,None)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now) if now is not None else time.gmtime())
        if run_step:entry['injuryReport']=inj_say
        for d in dates if run_step else []:
            try:
                lu,lu_say=lineups_file(d,scratch,get);ty,ty_n=typed_file(d,a.repo,scratch);pr,pr_say=props_file(d,a.repo,scratch,get)
                handed=[x for pair in (('--injury',inj),('--lineups',lu),('--typed',ty),('--props',pr)) if pair[1] for x in pair]
                try:rec=record_news(a.repo,d,stamp,inj,lu,lu_say,pr,pr_say)       # before the step runs: the news is kept even when the step fails
                except Exception as x:rec=None;rec_why='%s: %s'%(type(x).__name__,str(x)[:120])
                e=project(d,a.repo,scratch,run_step,run=run,handed=handed);e['lineups']=lu_say;e['typed']=ty_n;e['props']=pr_say
                if rec is None:e['recordNotSaved']=rec_why
                else:e['recorded']=rec
                done.append(e)
            except Exception as e:done.append({'date':d,'wrote':False,'why':'%s: %s'%(type(e).__name__,str(e)[:200])})
            finally:
                for f in ('lineups-'+d+'.json','typed-'+d+'.json','props-'+d+'.json'):
                    if os.path.exists(os.path.join(scratch,f)):os.remove(os.path.join(scratch,f))
        if os.path.exists(os.path.join(scratch,'injury.json')):os.remove(os.path.join(scratch,'injury.json'))
    entry['dates']=sorted(done,key=lambda x:x['date'])
    for x in entry['dates']:print(json.dumps(x))
    P.note(a.repo,entry,'projection')
    return 0

if __name__=='__main__':
    sys.exit(main())
