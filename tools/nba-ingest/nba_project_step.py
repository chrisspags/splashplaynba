#!/usr/bin/env python3
"""The projection step in the data job (Vale's step, run exactly as he hands it over; opening-night plan, Thursday's piece).

For each date the tonight step wrote in this run that has joined DraftKings rows, it unpacks the reviewed step from
nba_projection_bundle.py and runs
    NBA_ROOT=<repo> NBA_CACHE=<scratch> python3 run_step.py --tonight nba-tonight/<date>.json --out <scratch file>
        [--injury <file>] [--lineups <file>] [--typed <file>] --team-minutes 241
then, when the answer is a whole projection file for that date, publishes it as nba-proj/<date>.json.

What the step is handed, each press (Vale 10/13): the league's newest injury report, read by the step's own nba_injury.py fetch
(at most eight header requests and two files, under its own name); the NBA's lineups file for the date, one plain read; and the
minutes Chris typed for that slate date. Each is optional: when the league does not answer, or a file is not there, the step runs
without it and the status file says which and why. Nothing is disguised and nothing is retried. A file that is not there yet says
so (order 086, Monty's A4): "injury report not posted yet (...)" when all eight stamps answer 403 and the fixed old report answers
200 (a ninth header request), and "lineups file not posted yet for <date> (...)" on a 404 or a redirect that ends in no JSON.

HIS TYPED MINUTES ARE PRIVATE (order 072; Chris 10/14: his edits are private, free numbers stay public). They live in the private
repo (nba-typed-minutes/<date>.json), not in minutes-data.json any more. This job asks the site for them (nba-typed-for-job.js, at
NBA_TYPED_URL) with the identity token GitHub signs for this run, holds the answer in one temporary file OUTSIDE the job's cache
folder (the workflow caches .nba-cache), hands it to the step and removes it. The status file says only whether they were read, never
which or how many. What is published is the step's file less his typed fields and the notes that speak of them (public_copy): each
row's minutes and points are the fit around what he typed, which is the product, and they stay.

What it writes: nba-proj/<date>.json, nba-model-inputs.json (the saved stats the page's projection step reads, order 084), its
entry in the run's status file under "projection", and the night's record of the news it
was handed (Vale's DAILY-RECORD.md, item 3): nba-public-record/<date>/injury-<UTC time>.json, the league's report exactly as the reader
answered it (or its "unavailable" answer with the reason), and lineups-<UTC time>.json, the lineups file as read (or that it was
not read, and why). A new file each press, never one replaced. That folder is the PUBLIC repo's: nba-record/ is the private
repo's (it carries Stokastic's numbers; Monty's HOLD of 8caf95e, 10/14), and this job never writes there. Nothing else. It never writes
data.json: the page builds that from this file when an admin presses Build (Monty 10/12: nothing writes data.json except a build
Chris pressed). The step itself refuses that name too.

THE CHAIN CHECK (Vale's orders 078 and 081): before a date's file is published, node runs the step's nba_chain_check.js on the
PUBLIC copy, the exact bytes that are then published, with the page's own chain module (assets/nba-chain.js, carried in the bundle)
and his typed minutes handed in as the page's minutes edits (they live in his private store, and the published file carries nothing
of them). Every row must carry its components; nba-chain.js must give every row's possessions, stat line, DraftKings and FanDuel
points to 1e-6 of the step's and its published minutes exactly; and every hover step explain() prints must add up from its own
numbers, in a chain, to the cell. Otherwise the file is not published and the status says which row and which stat (or which field
is missing), naming no row and no number when his typed minutes were in the check. No node, no check, no file.

THE MODEL CHECK (Vale's orders 080 and 084; SPEC-v3 section 3, "one copy of the math, enforced"): after the step answers and before
the public copy is made, the step's nba_model_inputs.py writes the saved stats for that date (nba-model-inputs.json) from the step's
own reads (the same repo as NBA_ROOT, the history the step has just parsed as NBA_CACHE) and is handed nothing else: no tonight
file, no news, no props, no typed minutes. Then node runs the step's nba_model_check.js on the step's own file: the page's projection
step (assets/nba-model.js, carried in the bundle beside nba-chain.js) on those saved stats and the files the step was handed must give
it, every row and field, numbers to 1e-6, from the same night of stats. When it does not, or it cannot say, NOTHING is published (no
nba-proj file for any date, no saved stats): the projections STOP. Order 085 (Monty's review of 084): only the projections stop. Each
date gets a `stop` entry in the status file (whether the check failed or could not be checked, which check, which field; the row only
on a night his minutes were not read; whether an earlier run's nba-proj/<date>.json is still there), a date held back by another
date's stop says so, this step exits 0, and run.py goes on (stats, minutes): the run commits the history, tonight's slate and the
status as on any night, and the last good nba-proj/<date>.json stays as it was. The Data tab and Build show the stop as one red line.
"Could not be checked" (the check is not there, did not finish, gave no answer, exited 2, the saved stats could not be written, no node)
is never a pass. The log says which check, which row and which field; on a night his minutes were read, which check and which field
only (the log is public). When every date agrees, the dates' files are published and so is the saved stats file of the first of them,
after a last guard: nothing in it outside its format, no key or string that speaks of typed minutes, no key that speaks of props or
Stokastic.

It is soft otherwise, like the tonight step in the job: a date that cannot be projected (numpy missing, the two past seasons not in
the repo, the step failing, an answer that is not a projection file for that date, the chain check failing) writes no file for that
date, says why in the status file, and the run's history and tonight files are still committed.

The step keeps a parsed copy of the history in its scratch folder and reuses it when it is there, so the scratch folder is
emptied before every run: a run always reads the history that is in the repo now.

Usage (a step of run.py):  python3 nba_project_step.py --repo . --cache .nba-cache [--date 2026-10-20]"""
import json,os,re,shutil,subprocess,sys,tempfile,time,argparse
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

def node_version(which=shutil.which):
    """(node's path, its version) for the chain check, or None when node is not installed."""
    p=which('node')
    if not p:return None
    try:r=subprocess.run([p,'--version'],capture_output=True,text=True,timeout=60)
    except Exception:return None
    return (p,r.stdout.strip()) if r.returncode==0 and r.stdout.strip() else None

def chain_check(out,stepdir,node,run=subprocess.run,edits=None):
    """(ok, what it said): nba_chain_check.js from the unpacked step, with the bundle's nba-chain.js, on the file to be published;
    edits: the typed-minutes file the step was handed, given to the check as the page's minutes edits."""
    chk=os.path.join(stepdir,'nba_chain_check.js')
    if not os.path.exists(chk):return False,{'why':'the chain check is not in the step'}
    try:r=run([node,chk,out,os.path.join(stepdir,'nba-chain.js')]+(['--edits',edits] if edits else []),capture_output=True,text=True,timeout=300)
    except Exception as e:return False,{'why':'the chain check did not finish (%s)'%type(e).__name__}
    try:d=json.loads((r.stdout or '').strip().splitlines()[-1])
    except Exception:return False,{'why':'the chain check gave no answer (exit %s): %s'%(r.returncode,((r.stderr or '').strip().splitlines() or ['no message'])[-1][:200])}
    if not isinstance(d,dict):return False,{'why':'the chain check gave no answer'}
    return (r.returncode==0 and d.get('ok') is True),d

INPUTS_FILE='nba-model-inputs.json'      # the saved stats (orders 080, 084), at the top of the repo, beside data.json's other inputs
INPUTS_FORMAT='nba-model-inputs-v1'
INPUTS_KEYS=('format','stats','season','seasons','players','teams','acc','plk','tmk','lgk','onoff','stints','benchPerMinute','benchLinePerMinute','ownFit')
NOT_PUBLIC_KEY=re.compile(r'typed|prop|stok',re.I)     # what the saved stats never carry: his minutes, the props, Stokastic's numbers
WRITER_NOT_GIVEN=('NBA_TYPED_URL','ACTIONS_ID_TOKEN_REQUEST_URL','ACTIONS_ID_TOKEN_REQUEST_TOKEN')   # it cannot ask the site for his minutes

def saved_stats(date,repo,scratch,stepdir,hist,python=sys.executable,run=subprocess.run):
    """(inputs file, stats file, None) or (None, None, why): the step's nba_model_inputs.py writes the saved stats for `date` and the
    night of stats they are ({through, lastGame, games}) from the step's own reads: the repo as NBA_ROOT, the history the step has
    just parsed as NBA_CACHE. It is handed nothing else (no tonight file, no news, no props, no typed minutes)."""
    inp=os.path.join(scratch,'inputs-'+date+'.json');sta=os.path.join(scratch,'stats-'+date+'.json')
    for f in (inp,sta):
        if os.path.exists(f):os.remove(f)      # never a file an earlier run left
    w=os.path.join(stepdir,'nba_model_inputs.py')
    if not os.path.exists(w):return None,None,'the saved-stats writer is not in the step'
    env={k:v for k,v in os.environ.items() if k not in WRITER_NOT_GIVEN};env.update(NBA_ROOT=os.path.abspath(repo),NBA_CACHE=hist)
    try:r=run([python,w,'--out',inp,'--date',date,'--stats-out',sta],capture_output=True,text=True,timeout=900,env=env);why=None
    except Exception as e:r=None;why='the saved-stats writer did not finish (%s)'%type(e).__name__
    both=os.path.exists(inp) and os.path.exists(sta)
    if r is not None and (r.returncode or not both):why='the saved-stats writer exited %s and wrote %s'%(r.returncode,'them' if both else 'not both files')
    if why:
        for f in (inp,sta):
            if os.path.exists(f):os.remove(f)      # nothing half-written is left in the scratch folder
        return None,None,why
    return inp,sta,None

def model_check(out,inputs,stats,tonight,handed,stepdir,node,run=subprocess.run):
    """(ok, what it said): nba_model_check.js from the unpacked step (nba-model.js and nba-chain.js beside it, as the bundle carries
    them) on the step's own file: the page's projection step on the saved stats and the files the step was handed, with the same flags
    and team total, must give it, and from the same night of stats. Exit 0 with ok is a pass and exit 1 a disagreement, as the check
    promises; anything else (not in the step, did not finish, no answer, exit 2 or any other code) could not be checked (`cannot`), never
    a pass (order 085)."""
    chk=os.path.join(stepdir,'nba_model_check.js')
    if not os.path.exists(chk):return False,{'check':'load','why':'the model check is not in the step','cannot':True}
    try:r=run([node,chk,'--step',out,'--inputs',inputs,'--stats',stats,'--tonight',tonight,*handed,'--team-minutes',TEAM_MINUTES],capture_output=True,text=True,timeout=300)
    except Exception as e:return False,{'check':'run','why':'the model check did not finish (%s)'%type(e).__name__,'cannot':True}
    try:d=json.loads((r.stdout or '').strip().splitlines()[-1])
    except Exception:return False,{'check':'run','why':'the model check gave no answer (exit %s)'%r.returncode,'cannot':True}
    if not isinstance(d,dict):return False,{'check':'run','why':'the model check gave no answer','cannot':True}
    if r.returncode==0 and d.get('ok') is True:return True,d
    if r.returncode==1 and d.get('ok') is False:return False,d
    return False,dict(d,check=d.get('check') or 'run',cannot=True,exit=r.returncode,why=d.get('why') or None)

def last_good(repo,date):
    """Whether an earlier run's nba-proj/<date>.json is in the repo: the projections a stop leaves up (order 085)."""
    return os.path.exists(os.path.join(repo,'nba-proj',date+'.json'))

def split_at(at):
    """(row or team, field) of a path the model check names: rows[9: A Name (BOS, ESPN 1)].chain.final.dk -> ('A Name (BOS, ESPN 1)',
    'chain.final.dk'); stats.games -> (None, 'stats.games'). Any index left in the field is []."""
    at=str(at or '');m=re.match(r'^(?:rows|teams)\[(?:\d+: )?([^\]]*)\]\.?',at)
    rest=at[m.end():] if m else re.sub(r'^file\.','',at)
    return (m.group(1) or None) if m else None,re.sub(r'\[[^\]]*\]','[]',rest) or None

CANNOT_WORDS={'load':'it could not read its files or load the model','run':'it did not run to the end'}

def stop_of(said,ty,repo,date):
    """The date's stop entry for the status file (order 085), which the Data tab and Build turn into one red line: result "failed" (the
    check ran and the page's model does not give the job's file) or "not checked" (it could not say, never a pass); the check; the field
    (rows[].x as x); the row or team only on a night his minutes were NOT read (ty: his typed file), as the log names it; and lastGood:
    an earlier run's nba-proj/<date>.json is still there. On a night his minutes were read a reason the check itself wrote is not
    carried (it could name a file of his): the check's own words for what could not run stand in."""
    s={'result':'not checked' if said.get('cannot') else 'failed','check':str(said.get('check') or 'run')}
    bad=[b for b in (said.get('bad') or []) if isinstance(b,dict)]
    if s['result']=='failed' and bad:
        if ty:f=field_only(bad[0].get('at'));s['field']=f if f.startswith('a field') else split_at(f)[1]
        else:
            row,s['field']=split_at(bad[0].get('at'))
            if row:s['row']=row
    elif s['result']=='not checked':
        w=said.get('why')
        if 'exit' in said and not w:w='its answer and its exit code (%s) do not agree'%said['exit']
        elif ty and 'exit' in said:w=CANNOT_WORDS.get(s['check'],'it could not run')
        s['why']=str(w or 'no reason given')[:200]
    elif not ty and said.get('why'):s['why']=str(said['why'])[:200]
    s['lastGood']=last_good(repo,date)
    return {k:v for k,v in s.items() if v is not None}

def field_only(at):
    """A path the model check names, without the row or team it is on (rows[9: A Name (BOS, ...)].min -> rows[].min): what a public
    log may say on a night his minutes were read. A path that speaks of typed minutes, or a field the public copy does not carry
    (CHAIN_MINUTES_PRIVATE, chain.final.min), is not named at all."""
    p=re.sub(r'\[[^\]]*\]','[]',str(at or ''))
    if TYPED_WORD.search(p) or re.search(r'\.chain\.minutes\.(%s)\b|\.chain\.final\.min\b'%'|'.join(CHAIN_MINUTES_PRIVATE),p):return 'a field the public file does not carry'
    return p

INPUTS_SHAPE={'stats':('through','lastGame','games','bySeason','onoffBefore','stintsSince','rule','signature'),'acc':('PL','TM','OP','LG'),
              'onoff':('regs','players'),'stints':('ids','days','day','sec','on'),
              'benchLinePerMinute':('PTS','TPM','TRB','AST','STL','BLK','TOV')}      # the members of stats, of each season's or team's entry, and the bench line (order 086)

def inputs_trace(doc,own=None):
    """Where the saved stats hold anything but the job's reads of the public history (order 084), by path: a member outside the format
    (at the top, in stats, in each season's totals or team's on/off and stints, in the bench stand-in's line, or ownFit other than the
    step's own_fit.json `own`), and at any depth a key or string that speaks of typed minutes or a key that speaks of props or
    Stokastic. [] when there is none. Its numbers are the history's own and are not judged here: a typed number can equal a box
    score's minutes by chance."""
    if not isinstance(doc,dict):return ['the file is not an object']
    hits=['.'+str(k) for k in doc if k not in INPUTS_KEYS]+['.%s (missing)'%k for k in INPUTS_KEYS if k not in doc]
    if doc.get('format')!=INPUTS_FORMAT:hits.append('.format')
    shape=lambda o,keys,path:hits.extend([path] if not isinstance(o,dict) else [path+'.'+str(k) for k in o if k not in keys]+[path+'.%s (missing)'%k for k in keys if k not in o])
    shape(doc.get('stats'),INPUTS_SHAPE['stats'],'.stats'); shape(doc.get('benchLinePerMinute'),INPUTS_SHAPE['benchLinePerMinute'],'.benchLinePerMinute')
    for m in ('acc','onoff','stints'):
        for k,v in (doc.get(m) or {}).items() if isinstance(doc.get(m),dict) else [(None,None)]:shape(v,INPUTS_SHAPE[m],'.%s.%s'%(m,k))
    if own is not None and doc.get('ownFit')!=own:hits.append('.ownFit (not the step\'s own_fit.json)')
    def walk(o,path):
        if isinstance(o,dict):
            for k,v in o.items():
                if NOT_PUBLIC_KEY.search(str(k)):hits.append(path+'.'+str(k))
                walk(v,path+'.'+str(k))
        elif isinstance(o,list):
            for i,v in enumerate(o):walk(v,'%s[%d]'%(path,i))
        elif isinstance(o,str) and TYPED_WORD.search(o):hits.append(path)
    walk(doc,'');return hits

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
    """One plain request under the job's own name; no retry. Returns the parsed JSON or raises. An answer that is not JSON raises as
    it always did, marked .moved when it came from another address than the one asked (a redirect; order 086, the lineups file)."""
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':AGENT,'Accept':'application/json'}),timeout=timeout) as r:
        b=r.read()
        try:return json.loads(b)
        except ValueError as e:e.moved=r.geturl()!=url;raise

def not_posted(e,url):
    """Order 086 (Monty's A4 of 10/15): the read failed because the file is not there yet, not because anything blocked it: HTTP 404,
    or a redirect that ends in no JSON (the NBA sends a date with no lineups file on to its error page: measured 10/15, 2026-10-20
    answered 302 to /error/, which ends in a 404 page; 2026-10-14, a game day, answered 200)."""
    a=vars(e)       # instance attributes only: on Python 3.9 getattr on an HTTPError made without a body raises KeyError, not the default
    code=a.get('code')
    if code==404 or a.get('moved'):return True        # a 404, or an answer from another address that is not JSON
    return code is not None and (a.get('url') or a.get('filename') or url)!=url      # an HTTP error at the end of a redirect: no JSON either

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
    url=LINEUPS.format(date.replace('-',''))
    try:d=get(url)
    except Exception as e:return None,('lineups file not posted yet for %s (the NBA posts it on game day)'%date if not_posted(e,url) else 'lineups file not read (%s)'%(str(e)[:80] or type(e).__name__))
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

TYPED_AUDIENCE='splashplay-nba-typed-minutes'     # the audience nba-typed-for-job.js accepts, and nothing else

def ask_json(url,headers,body=None,timeout=15):
    """One request, no retry: GET, or POST of a JSON body. The parsed answer, or raises."""
    import urllib.request
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(url,data=data,method='GET' if body is None else 'POST',headers={**headers,'User-Agent':AGENT,'Accept':'application/json',**({'Content-Type':'application/json'} if body is not None else {})})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read())

def typed_from_site(date,env=None,ask=ask_json):
    """({ESPN id: minutes} or None, what to say). The minutes he typed for THIS slate date, asked of the site with the run's GitHub
    identity token. None (and why) when this run cannot ask or the site does not answer; the step then runs without them."""
    env=os.environ if env is None else env
    url=env.get('NBA_TYPED_URL');req_url=env.get('ACTIONS_ID_TOKEN_REQUEST_URL');req_tok=env.get('ACTIONS_ID_TOKEN_REQUEST_TOKEN')
    if not url:return None,'not asked: this run was not given the site\'s address'
    if not url.startswith('https://'):return None,'not asked: the site\'s address is not https'
    if not (req_url and req_tok):return None,'not read: this run has no GitHub identity token'
    try:jwt=(ask(req_url+('&' if '?' in req_url else '?')+'audience='+TYPED_AUDIENCE,{'Authorization':'bearer '+req_tok}) or {}).get('value')
    except Exception as e:return None,'not read: GitHub gave no identity token (%s)'%(type(e).__name__)
    if not isinstance(jwt,str) or not jwt:return None,'not read: GitHub gave no identity token'
    try:ans=ask(url,{'Authorization':'Bearer '+jwt},{'date':date})
    except Exception as e:return None,'not read: the site did not answer (%s)'%(str(getattr(e,'code','') or type(e).__name__))
    if not isinstance(ans,dict) or ans.get('ok') is not True or ans.get('date')!=date or not isinstance(ans.get('typed'),dict):return None,'not read: the site\'s answer is not the one we know'
    t={}
    for k,v in ans['typed'].items():      # the step's own rule again: minutes from 0 to 48 by ESPN id, nothing else
        if str(k) and isinstance(v,(int,float)) and not isinstance(v,bool) and 0<=v<=48:t[str(k)]=v
    return t,('read' if t else 'none for this date')

def typed_file(date,folder,env=None,ask=ask_json):
    """(path or None, what to say). His typed minutes for the date, in a file in `folder` (a temporary folder outside the cache) for
    the step's --typed; no file when there are none or they could not be read."""
    out=os.path.join(folder,'typed-'+date+'.json')
    if os.path.exists(out):os.remove(out)
    t,say=typed_from_site(date,env,ask)
    if not t:return None,say
    with open(out,'w') as f:json.dump(t,f)
    return out,say

PRIVATE_ROW_KEYS=('typed','typedMin')
TYPED_WORD=re.compile(r'typed',re.I)
# order 081: what the chain components (078) carry of his typed minutes. chain.minutes.typed is his number; .refit is the refit around
# it; .used and chain.final.min are each row's used minutes to the last digit, which for a row he typed IS his number. They go from
# every row alike (a field gone from his rows only would name them); the page has his minutes from his private store, and the chain
# check hands them in, so nothing needs them. teamChain.minutes.typed (the refit's log) goes too.
CHAIN_MINUTES_PRIVATE=('typed','refit','used')

def _scrub(o):
    """o without any key that speaks of typed minutes, and with any such string made null (order 081: chain and teamChain)"""
    if isinstance(o,dict):return {k:_scrub(v) for k,v in o.items() if not TYPED_WORD.search(str(k))}
    if isinstance(o,list):return [_scrub(v) for v in o]
    if isinstance(o,str) and TYPED_WORD.search(o):return None
    return o

def public_copy(doc):
    """The step's file as it is published: every row without his typed fields (typed, typedMin), and the notes without the ones that
    speak of a minute he typed (they name his numbers and his players). A row he typed to 0 also has the step's reason for it taken off
    (whyOut "typed 0 minutes"; found by running the real step, proof/ab-fit.py): he stays out at 0, with no reason given, as a row the
    job has no reason for. Each row's minutes and points stay: they are the fit around what he typed. modelMin, the model's own number
    before that fit, stays too (a free number, which the page shows as the model's). The chain components (order 078) go without what
    they carry of his minutes (CHAIN_MINUTES_PRIVATE, chain.final.min, teamChain.minutes.typed) and without any key or string in chain
    or teamChain that speaks of typed minutes (the market note "his minutes were typed" is made null)."""
    out=dict(doc)
    def row(r):
        if not isinstance(r,dict):return r
        r={k:v for k,v in r.items() if k not in PRIVATE_ROW_KEYS}
        if isinstance(r.get('whyOut'),str) and TYPED_WORD.search(r['whyOut']):r['whyOut']=None
        if isinstance(r.get('chain'),dict):
            ch=dict(r['chain'])
            if isinstance(ch.get('minutes'),dict):ch['minutes']={k:v for k,v in ch['minutes'].items() if k not in CHAIN_MINUTES_PRIVATE}
            if isinstance(ch.get('final'),dict):ch['final']={k:v for k,v in ch['final'].items() if k!='min'}
            r['chain']=_scrub(ch)
        return r
    def team(t):
        return dict(t,teamChain=_scrub(t['teamChain'])) if isinstance(t,dict) and isinstance(t.get('teamChain'),dict) else t
    out['rows']=[row(r) for r in doc['rows']]
    if isinstance(doc.get('notes'),list):out['notes']=[n for n in doc['notes'] if not TYPED_WORD.search(str(n))]
    if isinstance(doc.get('teams'),list):out['teams']=[team(t) for t in doc['teams']]
    return out

def typed_trace(doc,values=()):
    """Where a file speaks of typed minutes (order 081), by path: every key or string anywhere in it that matches TYPED_WORD, and every
    number in the chain components (rows[].chain, teams[]) equal to one of `values` (his typed minutes). The rows' own fields are v1.2's
    published product, rounded: a row he typed publishes his minutes as its min, and an unrelated rounded field can equal his number
    by chance (on 10/20 Stephon Castle's AST% is 33.5), so their numbers are not counted. The job refuses to publish on words; numbers
    are for the tests, since a typed number can also equal one of the chain's own (a 25.5 points line, the 40-minute ceiling)."""
    hits=[];vals={float(v) for v in values}
    def walk(o,path,comp):
        if isinstance(o,dict):
            for k,v in o.items():
                if TYPED_WORD.search(str(k)):hits.append(path+'.'+str(k))
                walk(v,path+'.'+str(k),comp or k=='chain' or path=='' and k=='teams')
        elif isinstance(o,list):
            for i,v in enumerate(o):walk(v,'%s[%d]'%(path,i),comp)
        elif isinstance(o,str):
            if TYPED_WORD.search(o):hits.append(path)
        elif comp and isinstance(o,(int,float)) and not isinstance(o,bool) and float(o) in vals:hits.append(path)
    walk(doc,'',False);return hits

def record_news(repo,date,stamp,inj_path,lu_path,lu_say,props_path=None,props_say=None):
    """The news as it reached us, for the night's record: one new file each, per date, never replaced. Returns the names written.
    The props pull is kept whole beside them (props-<UTC time>.json), or its "not used" answer when there was none."""
    d=os.path.join(repo,'nba-public-record',date);os.makedirs(d,exist_ok=True);wrote=[]
    for name,src,alt in (('injury',inj_path,{'ok':False,'error':'the injury reader wrote no answer'}),('lineups',lu_path,{'read':False,'why':lu_say}),('props',props_path,dict(props_say or {'used':False,'why':'not read'}))):
        fn='%s-%s.json'%(name,stamp);dst=os.path.join(d,fn)
        if os.path.exists(dst):continue
        if src and os.path.exists(src):shutil.copyfile(src,dst)
        else:
            with open(dst,'w') as f:json.dump(alt,f)
        wrote.append(fn)
    return wrote

def project(date,repo,scratch,run_step,python=sys.executable,run=subprocess.run,handed=(),node=None,ready=None):
    """Runs the step for one date and returns the status entry. A whole answer that both checks agree with is left ready to publish
    (ready[date] = (its public copy, the saved stats it was checked on), in scratch/ready); main() publishes them only when no date
    stopped the run. An answer the model check does not agree with returns "stop"."""
    t0=time.time();tonight=os.path.join(repo,'nba-tonight',date+'.json');tmp=os.path.join(scratch,'out-'+date+'.json');ptmp=os.path.join(scratch,'pub-'+date+'.json')
    if ready is None:ready={}
    if not os.path.exists(tonight):return {'date':date,'wrote':False,'why':'nba-tonight/%s.json is not in the repo'%date}
    for f in (tmp,ptmp):
        if os.path.exists(f):os.remove(f)
    hist=os.path.join(scratch,'history');shutil.rmtree(hist,ignore_errors=True)   # never a parsed history left by an earlier run
    inp=sta=None
    try:
        r=run([python,run_step,'--tonight',tonight,'--out',tmp,*handed,'--team-minutes',TEAM_MINUTES],capture_output=True,text=True,timeout=900,env={**os.environ,'NBA_ROOT':os.path.abspath(repo),'NBA_CACHE':hist})
        if r.returncode:
            last=((r.stderr or r.stdout or '').strip().splitlines() or ['no message'])[-1][:200]
            if TYPED_WORD.search(last):last='its last line speaks of typed minutes, so it is not shown here'      # the status file is public
            return {'date':date,'wrote':False,'why':'the step exited %s: %s'%(r.returncode,last)}
        try:doc=json.load(open(tmp))
        except Exception as e:return {'date':date,'wrote':False,'why':'the step wrote no readable file (%s)'%type(e).__name__}
        if not isinstance(doc,dict) or any(k not in doc for k in FORMAT_KEYS) or doc.get('date')!=date or not isinstance(doc.get('rows'),list) or not isinstance(doc.get('games'),list):
            return {'date':date,'wrote':False,'why':'the step answered with something that is not a projection file for %s'%date}
        ty=handed[list(handed).index('--typed')+1] if '--typed' in handed else None
        # THE MODEL CHECK (orders 080, 084): the page's projection step on the saved stats must give the step's own file, before any
        # of it is made public. A date it does not agree with, or cannot say about, stops the projections (order 085: only them).
        t1=time.time();stepdir=os.path.dirname(run_step);inp,sta,why=saved_stats(date,repo,scratch,stepdir,hist,python,run)
        if why:return {'date':date,'wrote':False,'stop':dict(result='not checked',check='writer',why=why,lastGood=last_good(repo,date)),'why':'model check: the saved stats could not be written: '+why}
        ok,said=model_check(tmp,inp,sta,tonight,handed,stepdir,node,run)
        mc={'seconds':round(time.time()-t1,1),**{k:said.get(k) for k in ('rows','teams','fields','ms') if said.get(k) is not None}}
        if not ok:
            bad=[b for b in (said.get('bad') or []) if isinstance(b,dict)];stop=stop_of(said,ty,repo,date)
            if stop['result']=='not checked':
                return {'date':date,'wrote':False,'stop':stop,'why':'model check: %s: could not be checked: %s'%(stop['check'],stop['why']),'modelCheck':mc}
            if ty:return {'date':date,'wrote':False,'stop':stop,'why':'model check: %s: a difference in %s (it ran with his typed minutes, so no row and no number is named here)'%(said.get('check') or 'failed',field_only(bad[0].get('at')) if bad else 'the file'),'modelCheck':mc}
            return {'date':date,'wrote':False,'stop':stop,'why':'model check: %s: %s'%(said.get('check') or 'failed',str(said.get('why') or 'it did not agree')[:400]),'modelCheck':dict(mc,bad=[{k:b.get(k) for k in ('at','step','model')} for b in bad[:3]])}
        mc['worst']=(said.get('worst') or {}).get('diff')
        pub=public_copy(doc);rows=[x for x in pub['rows'] if isinstance(x,dict)]
        tr=typed_trace(pub) if ty else []      # a last guard on words, on a night his minutes were read (order 081): nothing published speaks of them
        if tr:return {'date':date,'wrote':False,'why':'the public copy still speaks of typed minutes in %d place(s), so it is not published'%len(tr)}
        # published as the step writes its file (json, indent 0), less his typed fields: the file a direct run of the step gives, without them.
        # The chain check reads exactly these bytes, with his typed minutes as the page's edits, before they are published (order 081).
        with open(ptmp,'w') as f:json.dump(pub,f,indent=0)
        ok,said=chain_check(ptmp,os.path.dirname(run_step),node,run,edits=ty) if node else (False,{'why':'node is not installed for this run'})
        if not ok:
            if ty:return {'date':date,'wrote':False,'why':'chain check: %s (it ran with his typed minutes, so no row or number is named here)'%str(said.get('kind') or 'it did not agree'),
                          'chainCheck':{k:said.get(k) for k in ('rows','checked') if said.get(k) is not None}}
            return {'date':date,'wrote':False,'why':'chain check: '+str(said.get('why') or 'failed')[:300],'chainCheck':{k:said.get(k) for k in ('rows','checked','bad','missing') if said.get(k) is not None}}
        rd=os.path.join(scratch,'ready');os.makedirs(rd,exist_ok=True);rp=os.path.join(rd,'proj-'+date+'.json');ri=os.path.join(rd,'inputs-'+date+'.json')
        shutil.move(ptmp,rp);shutil.move(inp,ri);ready[date]=(rp,ri)      # published by main() when no date stops the run
        return {'date':date,'wrote':True,'games':len(pub['games']),'rows':len(rows),'pool15':sum(1 for x in rows if (x.get('proj') or 0)>=15),
                'notes':len(pub.get('notes') or []),'seconds':round(time.time()-t0,1),'modelCheck':mc,
                'chainCheck':{'rows':said.get('rows'),'checked':said.get('checked'),'steps':said.get('steps'),'worst':(said.get('worst') or {}).get('diff')}}
    finally:
        # the parsed history is about 120 MB: never left behind for the workflow's cache to carry from run to run
        shutil.rmtree(hist,ignore_errors=True)
        for f in (tmp,ptmp,inp,sta):
            if f and os.path.exists(f):os.remove(f)

def publish(repo,ready,entry,done,own=None):
    """Publishes what project() left ready, when no date stopped the run: each date's public copy as nba-proj/<date>.json, and the saved
    stats of the first of them as nba-model-inputs.json (after its last guard, inputs_trace; own: the step's own_fit.json). Returns why
    the projections stopped, or None. A date that agreed but is held back by another date's stop gets a stop entry that says so (order
    085: result "held", by which dates), so the Data tab and Build treat it as the slate's date too and never open another."""
    stops=[e for e in done if e.get('stop')]
    if not stops and ready:
        first=min(ready);raw=open(ready[first][1],'rb').read()
        try:tr=inputs_trace(json.loads(raw),own)
        except Exception:tr=['the file is not JSON']
        if tr:
            entry['modelInputs']={'published':False,'why':'the saved stats hold something outside the public history in %d place(s), so nothing is published'%len(tr)}
            e=[x for x in done if x.get('date')==first and x.get('wrote')][0]
            e.update(wrote=False,stop=dict(result='failed',check='guard',why='the saved stats hold something outside the public history',lastGood=last_good(repo,first)),why='the saved stats\' last guard: '+entry['modelInputs']['why'])
            stops=[e]
    if stops:
        by=[x['date'] for x in stops]
        for e in done:
            if e.get('wrote'):e.update(wrote=False,stop=dict(result='held',by=by,lastGood=last_good(repo,e['date'])),why='not published: the model check stopped the run (%s)'%', '.join(by))
        return '; '.join('%s: %s'%(x['date'],x['why']) for x in stops)
    for d,(rp,ri) in sorted(ready.items()):
        dst=os.path.join(repo,'nba-proj',d+'.json');os.makedirs(os.path.dirname(dst),exist_ok=True);shutil.copyfile(rp,dst)
    if ready:
        st=json.loads(raw).get('stats') or {}
        with open(os.path.join(repo,INPUTS_FILE),'wb') as f:f.write(raw)
        entry['modelInputs']={'published':True,'date':first,'through':st.get('through'),'lastGame':st.get('lastGame'),'games':st.get('games'),'bytes':len(raw)}
    return None

def main(argv=None,bundle=None,numpy=numpy_version,run=subprocess.run,get=plain_get,now=None,env=None,ask=ask_json,node=node_version):
    ap=argparse.ArgumentParser(description='Run the NBA projection step for the dates the tonight step wrote')
    ap.add_argument('--repo',default='.');ap.add_argument('--cache',default='.nba-cache');ap.add_argument('--date',help='one US Eastern date, YYYY-MM-DD (default: the dates the tonight step wrote in this run)')
    a=ap.parse_args(argv)
    if bundle is None:import nba_projection_bundle as bundle
    dates,done=wanted_dates(a.repo,a.date);entry={'step':bundle.BUNDLE_ID[:12]}
    why=None;stopped=None
    if dates:
        nv=numpy();entry['numpy']=nv;nd=node();entry['node']=nd[1] if nd else None;miss=missing_history(a.repo)
        if not nv:why='numpy is not installed for this run'
        elif not nd:why='node is not installed for this run, so the chain check and the model check cannot run and no file is published'
        elif miss:why='the history the model reads is not in the repo: '+', '.join(miss)
    if dates and why:done+=[{'date':d,'wrote':False,'why':why,**({'stop':dict(result='not checked',check='node',why='node is not installed for this run',lastGood=last_good(a.repo,d))} if nv and not nd else {})} for d in dates]
    elif dates:
        scratch=os.path.join(a.cache,'projection');code=os.path.join(scratch,'step');shutil.rmtree(code,ignore_errors=True);os.makedirs(code,exist_ok=True)
        ready={};rdir=os.path.join(scratch,'ready');shutil.rmtree(rdir,ignore_errors=True)     # what both checks agreed with, published below
        try:run_step=bundle.extract(code)
        except Exception as e:run_step=None;done+=[{'date':d,'wrote':False,'why':'the step could not be unpacked: %s'%str(e)[:160]} for d in dates]
        inj,inj_say=injury_file(scratch,run_step,run=run) if run_step else (None,None)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now) if now is not None else time.gmtime())
        if run_step:entry['injuryReport']=inj_say
        tdir=tempfile.mkdtemp(prefix='nba-typed-')       # his typed minutes: outside the cache folder, removed below
        for d in dates if run_step else []:
            try:
                lu,lu_say=lineups_file(d,scratch,get);ty,ty_say=typed_file(d,tdir,env,ask);pr,pr_say=props_file(d,a.repo,scratch,get)
                handed=[x for pair in (('--injury',inj),('--lineups',lu),('--typed',ty),('--props',pr)) if pair[1] for x in pair]
                try:rec=record_news(a.repo,d,stamp,inj,lu,lu_say,pr,pr_say)       # before the step runs: the news is kept even when the step fails
                except Exception as x:rec=None;rec_why='%s: %s'%(type(x).__name__,str(x)[:120])
                e=project(d,a.repo,scratch,run_step,run=run,handed=handed,node=nd[0],ready=ready);e['lineups']=lu_say;e['typed']=ty_say;e['props']=pr_say
                if rec is None:e['recordNotSaved']=rec_why
                else:e['recorded']=rec
                done.append(e)
            except Exception as e:done.append({'date':d,'wrote':False,'why':'%s: %s'%(type(e).__name__,str(e)[:200])})
            finally:
                for f in ('lineups-'+d+'.json','props-'+d+'.json'):
                    if os.path.exists(os.path.join(scratch,f)):os.remove(os.path.join(scratch,f))
        shutil.rmtree(tdir,ignore_errors=True)
        if os.path.exists(os.path.join(scratch,'injury.json')):os.remove(os.path.join(scratch,'injury.json'))
        try:own=json.load(open(os.path.join(code,'own_fit.json')))
        except Exception:own=None
        stopped=publish(a.repo,ready,entry,done,own);shutil.rmtree(rdir,ignore_errors=True)
    entry['dates']=sorted(done,key=lambda x:x['date'])
    for x in entry['dates']:print(json.dumps(x))
    P.note(a.repo,entry,'projection')
    if stopped:      # order 085: only the projections stop; run.py goes on, and the run commits the history, the slate and this status
        print('NBA projection STOPPED, nothing published: '+stopped,flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
