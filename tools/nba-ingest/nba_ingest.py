#!/usr/bin/env python3
"""NBA daily ingest (Wes 9/29, for Vale 17d0cd01 / Keel 82197): ESPN public API -> the season tables the projection engine
(assets/nba-proj-engine.js) reads, and the play-by-play stints the WOWY (on/off) rates come from. Stdlib only.

The math is Vale's reference code, unchanged; only paths became arguments:
  fetch()         <- espn_fetch.py   scoreboard per date -> summary per FINAL event, gzipped, ~1 request/s, resumable
  build_tables()  <- build_tables.py Basketball-Reference formulas: per-100, USG%, TRB%, AST%, team/opp per-100, league
  build_stints()  <- wowy.py build   on-court fives and stints with every event attributed
Changes for a daily run: a date is re-read from the scoreboard while any of its games is not final, or is in the last
two days (Vale's fetcher read each date once, so a game indexed before it finished was never fetched); and the All-Star
games and the NBA Cup Championship, which ESPN files as regular season, are left out of the tables and stints; and the
stints count what the box counts (a missed end-of-period heave is not an attempt, 'Traveling' is a turnover, and a zero-length
stint between two same-second subs keeps its events).

Usage (from the data repo root, in the GitHub Actions workflow):
  python3 nba_ingest.py --season 2026-27 --cache .nba-cache --out .
Writes nba-tables/<season>-<type>.json (2 = regular season, 1 = preseason), nba-stints/<season>.json.gz, nba-events/<season>/
(one file per final event: header, boxscore, pickcenter; added to, never rewritten) and nba-ingest-status.json (what this run
fetched and built, the requests it sent to ESPN and the seconds it took). The table and stint bytes depend only on the games fetched,
so a day without new games leaves them unchanged.
Exit 1 on any failed request after retries: the run fails loudly and the next run resumes where it stopped."""
import json,gzip,os,re,sys,time,glob,io,argparse,datetime,collections,urllib.request

SEASON_WINDOWS={'2024-25':('2024-10-01','2025-06-25'),'2025-26':('2025-10-01','2026-06-25'),'2026-27':('2026-09-25','2027-06-30')}
UA={'Accept':'application/json'}
SCOREBOARD='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}'
SUMMARY='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}'

REQUESTS=[0]   # every request this run sent to ESPN, retries included (the status file reports it)
def http_get(url,tries=3,sleep=time.sleep):
    for i in range(tries):
        REQUESTS[0]+=1
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=30) as r:return json.loads(r.read())
        except Exception:
            if i==tries-1:raise
            sleep(3*(i+1))

# ---------------------------------------------------------------- fetch (espn_fetch.py)
def fetch(season,cache,start=None,end=None,today=None,get=http_get,sleep=time.sleep,log=print):
    """Incremental: returns (new summaries saved, dates read). Raises on a request that fails after retries."""
    s0,e0=SEASON_WINDOWS.get(season,(None,None))
    start=datetime.date.fromisoformat(start or s0);today=today or datetime.date.today()
    end=min(datetime.date.fromisoformat(end or e0),today)
    D=os.path.join(cache,season);os.makedirs(D,exist_ok=True)
    idxf=os.path.join(D,'index.json');idx=json.load(open(idxf)) if os.path.exists(idxf) else {}
    d=start;n_new=0;n_dates=0;recent=today-datetime.timedelta(days=2)
    while d<=end:
        ds=d.strftime('%Y%m%d')
        stale=ds not in idx or d>=recent or any(e['status']!='STATUS_FINAL' and not e['status'].startswith(('STATUS_POSTPONED','STATUS_CANCELED')) for e in idx[ds])
        if stale:
            sb=get(SCOREBOARD.format(ds));sleep(1);n_dates+=1
            idx[ds]=[{'id':e['id'],'name':e.get('shortName'),'status':e['status']['type']['name'],'season':e.get('season',{}),'date':e.get('date')} for e in sb.get('events',[])]
            with open(idxf,'w') as f:json.dump(idx,f)
        for e in idx[ds]:
            fp=os.path.join(D,f"{e['id']}.json.gz")
            if os.path.exists(fp) or e['status']!='STATUS_FINAL':continue
            s=get(SUMMARY.format(e['id']));sleep(1)
            keep={k:s.get(k) for k in ('header','boxscore','plays','gameInfo','pickcenter','injuries')}
            with gzip.open(fp,'wt') as f:json.dump(keep,f)
            n_new+=1
            if n_new%50==0:log(ds,'fetched',n_new)
        d+=datetime.timedelta(days=1)
    return n_new,n_dates

# ESPN files the All-Star games (STARS / STRIPES / WORLD) and the NBA Cup Championship as REGULAR season (and preseason
# exhibitions can face non-NBA clubs). None counts in NBA stats, so neither the tables nor the stints use them
# (Wes 82672, Monty 6ff62c80). Everything else is Vale's reference unchanged.
NBA30=frozenset('ATL BKN BOS CHA CHI CLE DAL DEN DET GS HOU IND LAC LAL MEM MIA MIL MIN NO NY OKC ORL PHI PHX POR SA SAC TOR UTAH WSH'.split())
def nba_game(s):
    h=s.get('header') or {}
    if re.search(r'cup championship',str(h.get('gameNote') or ''),re.I):return False
    bx=(s.get('boxscore') or {}).get('players') or []
    return all((side.get('team') or {}).get('abbreviation') in NBA30 for side in bx)
def summaries(cache,season):
    for fp in sorted(glob.glob(os.path.join(cache,season,'*.json.gz'))):
        yield os.path.basename(fp).split('.')[0],json.load(gzip.open(fp,'rt'))

# ---------------------------------------------------------------- per-event history (Vale 10/12, opening-night plan piece 2)
# One small file per fetched FINAL event, kept in the data repo so a model can replay a season from files alone: the event's header,
# boxscore and pickcenter exactly as ESPN gave them (ESPN athlete ids untouched; no play-by-play). A file that exists is never
# written again, so the history is only added to. index.json lists every event file: id, tip time, season type, the two teams, and
# whether it counts as an NBA game by the same rule the tables use (an exhibition against a non-NBA club, an All-Star game and
# the Cup Championship do not).
EVENT_KEEP=('header','boxscore','pickcenter')
def write_events(cache,season,out):
    """Returns (files written this run, files there in all)."""
    O=os.path.join(out,'nba-events',season);n=0
    for fp in sorted(glob.glob(os.path.join(cache,season,'*.json.gz'))):
        dst=os.path.join(O,os.path.basename(fp))
        if os.path.exists(dst):continue
        s=json.load(gzip.open(fp,'rt'));write_gz(dst,{k:s.get(k) for k in EVENT_KEEP});n+=1
    idx=[]
    for fp in sorted(glob.glob(os.path.join(O,'*.json.gz'))):
        s=json.load(gzip.open(fp,'rt'));h=s.get('header') or {};c=(h.get('competitions') or [{}])[0] or {}
        side={x.get('homeAway'):(x.get('team') or {}).get('abbreviation') for x in c.get('competitors') or []}
        idx.append({'id':os.path.basename(fp)[:-8],'date':c.get('date'),'seasonType':str((h.get('season') or {}).get('type') or ''),'away':side.get('away'),'home':side.get('home'),'nbaGame':nba_game(s)})
    if idx:write_json(os.path.join(O,'index.json'),sorted(idx,key=lambda e:(e['date'] or '',e['id'])))
    return n,len(idx)

# ---------------------------------------------------------------- tables (build_tables.py)
def _mk(s):
    try:a,b=s.split('-');return float(a),float(b)
    except Exception:return 0.0,0.0
def _f(x):
    try:return float(x)
    except Exception:return 0.0
def _mins(x):
    if x in (None,'','--'):return 0.0
    if ':' in str(x):m,s=str(x).split(':');return float(m)+float(s)/60
    return _f(x)
def build_tables(cache,season,stype='2'):
    games=[];skipped=collections.Counter()
    for gid,s in summaries(cache,season):
        h=s.get('header') or {}
        st=str(((h.get('season') or {}).get('type')))
        if stype!='all' and st!=stype:skipped['type '+st]+=1;continue
        if not nba_game(s):skipped['not an NBA game']+=1;continue
        bx=(s.get('boxscore') or {}).get('players') or []
        if len(bx)!=2:skipped['no box']+=1;continue
        comp=(h.get('competitions') or [{}])[0];date=comp.get('date','')[:10]
        teams=[]
        for side in bx:
            abbr=side['team']['abbreviation'];stats=side['statistics'][0];lab=stats.get('labels') or stats.get('names')
            rows=[]
            for a in stats['athletes']:
                if a.get('didNotPlay') or not a.get('stats') or not (a.get('athlete') or {}).get('id'):continue
                v=dict(zip(lab,a['stats']));fgm,fga=_mk(v.get('FG'));tpm,tpa=_mk(v.get('3PT'));ftm,fta=_mk(v.get('FT'))
                rows.append(dict(id=a['athlete']['id'],name=a['athlete']['displayName'],pos=(a['athlete'].get('position') or {}).get('abbreviation'),starter=bool(a.get('starter')),
                    MP=_mins(v.get('MIN')),PTS=_f(v.get('PTS')),FGM=fgm,FGA=fga,TPM=tpm,TPA=tpa,FTM=ftm,FTA=fta,ORB=_f(v.get('OREB')),DRB=_f(v.get('DREB')),TRB=_f(v.get('REB')),
                    AST=_f(v.get('AST')),STL=_f(v.get('STL')),BLK=_f(v.get('BLK')),TOV=_f(v.get('TO')),PF=_f(v.get('PF'))))
            T={k:sum(r[k] for r in rows) for k in ('MP','PTS','FGM','FGA','TPM','TPA','FTM','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV')}
            teams.append(dict(abbr=abbr,players=rows,T=T))
        a,b=teams
        def poss(t,o):
            orbp=t['ORB']/(t['ORB']+o['DRB']) if (t['ORB']+o['DRB']) else 0
            return t['FGA']+0.4*t['FTA']-1.07*orbp*(t['FGA']-t['FGM'])+t['TOV']
        P=0.5*(poss(a['T'],b['T'])+poss(b['T'],a['T']))
        games.append(dict(id=gid,date=date,type=st,poss=P,teams=teams))
    pl=collections.defaultdict(lambda:collections.Counter());pinfo={}
    tm=collections.defaultdict(lambda:collections.Counter());op=collections.defaultdict(lambda:collections.Counter())
    for g in games:
        for i,t in enumerate(g['teams']):
            o=g['teams'][1-i]['T'];T=t['T']
            tm[t['abbr']].update({**{k:T[k] for k in T},'G':1,'POSS':g['poss']})
            op[t['abbr']].update({**{k:o[k] for k in o},'G':1,'POSS':g['poss']})
            for r in t['players']:
                if r['MP']<=0:continue
                k=r['id'];pinfo[k]=dict(name=r['name'],pos=r['pos'],team=t['abbr'])
                c=pl[k];c['G']+=1;c['GS']+=r['starter']
                for s in ('MP','PTS','FGM','FGA','TPM','TPA','FTM','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF'):c[s]+=r[s]
                c['tMP']+=T['MP'];c['tFGA']+=T['FGA'];c['tFTA']+=T['FTA'];c['tTOV']+=T['TOV'];c['tFGM']+=T['FGM'];c['tTRB']+=T['TRB'];c['oTRB']+=o['TRB']
                c['possOn']+=g['poss']*r['MP']/(T['MP']/5) if T['MP'] else 0
                c['uNum']+=(r['FGA']+0.44*r['FTA']+r['TOV'])*(T['MP']/5);c['uDen']+=r['MP']*(T['FGA']+0.44*T['FTA']+T['TOV'])
                c['rNum']+=r['TRB']*(T['MP']/5);c['rDen']+=r['MP']*(T['TRB']+o['TRB'])
                c['aDen']+=(r['MP']/(T['MP']/5))*T['FGM']-r['FGM'] if T['MP'] else 0
    players={}
    for k,c in pl.items():
        p=c['possOn']
        players[k]=dict(**pinfo[k],G=c['G'],GS=c['GS'],MP=round(c['MP'],1),MPG=round(c['MP']/c['G'],2),POSS=round(p,1),
            per100={s:round(100*c[s]/p,3) if p else None for s in ('PTS','TPM','TRB','AST','STL','BLK','TOV','FGA','FTA','ORB','DRB')},
            USG=round(100*c['uNum']/c['uDen'],2) if c['uDen'] else None,TRBp=round(100*c['rNum']/c['rDen'],2) if c['rDen'] else None,
            ASTp=round(100*c['AST']/c['aDen'],2) if c['aDen']>0 else None,TS=round(c['PTS']/(2*(c['FGA']+0.44*c['FTA'])),4) if (c['FGA']+c['FTA']) else None)
    teams={t:dict(G=c['G'],POSS=round(c['POSS'],1),pace=round(c['POSS']/c['G'],2),per100={s:round(100*c[s]/c['POSS'],3) for s in ('PTS','TPM','TRB','AST','STL','BLK','TOV','FGA','FGM','FTA','ORB','DRB')}) for t,c in tm.items()}
    opps={t:dict(G=c['G'],per100={s:round(100*c[s]/c['POSS'],3) for s in ('PTS','TPM','TRB','AST','STL','BLK','TOV','FGA','FGM','FTA','ORB','DRB')}) for t,c in op.items()}
    L=collections.Counter();LP=0
    for c in tm.values():
        LP+=c['POSS']
        for s in ('PTS','TPM','TRB','AST','STL','BLK','TOV','FGA','FGM','FTA','ORB','DRB'):L[s]+=c[s]
    league={s:round(100*L[s]/LP,3) for s in L}
    return dict(season=season,type=stype,games=len(games),league=league,teams=teams,opps=opps,players=players),dict(skipped)

# ---------------------------------------------------------------- stints (wowy.py build)
def _secs(period,clock):
    c=str((clock or {}).get('displayValue','0'))
    if ':' in c:m,s=c.split(':');rem=int(m)*60+float(s)
    else:rem=float(c or 0)
    plen=720 if period<=4 else 300
    start=(period-1)*720 if period<=4 else 2880+(period-5)*300
    return start+(plen-rem)
def parse_game(s):
    bx=s['boxscore']['players'];tid={};names={};roster={}
    for side in bx:
        t=side['team'];tid[str(t['id'])]=t['abbreviation'];roster[t['abbreviation']]=set()
        for a in side['statistics'][0]['athletes']:
            if not (a.get('athlete') or {}).get('id'):continue
            i=str(a['athlete']['id']);names[i]=a['athlete']['displayName'];roster[t['abbreviation']].add(i)
    starters={t:[str(a['athlete']['id']) for side in bx if side['team']['abbreviation']==t for a in side['statistics'][0]['athletes'] if a.get('starter') and (a.get('athlete') or {}).get('id')] for t in roster}
    # sorted: a set's order follows the process hash seed, so unsorted iteration made the output bytes vary run to run
    team_of={i:t for t,ids in roster.items() for i in sorted(ids)}
    plays=s.get('plays') or []
    periods=sorted({p['period']['number'] for p in plays})
    teams=list(roster)
    start5={}
    prev_end={t:set(starters[t]) for t in teams}
    for per in periods:
        pp=[p for p in plays if p['period']['number']==per]
        for t in teams:
            known=[];subbed_in=set()
            for p in pp:
                parts=[str(x['athlete']['id']) for x in (p.get('participants') or []) if x.get('athlete')]
                if p.get('type',{}).get('text')=='Substitution':
                    if len(parts)>=2 and team_of.get(parts[0])==t:
                        inn,out=parts[0],parts[1]
                        if out not in subbed_in and out not in known:known.append(out)
                        subbed_in.add(inn)
                    continue
                for x in parts:
                    if team_of.get(x)==t and x not in subbed_in and x not in known:known.append(x)
            five=known[:5] if per>1 else list(starters[t])[:5]
            if per>1 and len(five)<5:
                for x in sorted(prev_end[t]):
                    if len(five)>=5:break
                    if x not in five and x not in subbed_in:five.append(x)
            start5[(per,t)]=set(five[:5])
        for t in teams:
            on=set(start5[(per,t)])
            for p in pp:
                if p.get('type',{}).get('text')!='Substitution':continue
                parts=[str(x['athlete']['id']) for x in (p.get('participants') or []) if x.get('athlete')]
                if len(parts)>=2 and team_of.get(parts[0])==t:on.discard(parts[1]);on.add(parts[0])
            prev_end[t]=on
    stints=[];cur=None
    # a stint of zero length still keeps its events (free throws often fall between two subs at the same clock time)
    keep=lambda st:st['t1']>st['t0'] or any(st['team'][t] for t in teams)
    def new_stint(on,t0):
        return {'on':{t:set(on[t]) for t in teams},'t0':t0,'t1':t0,'team':{t:collections.Counter() for t in teams},'pl':collections.defaultdict(collections.Counter)}
    on={}
    for per in periods:
        pp=[p for p in plays if p['period']['number']==per]
        on={t:set(start5[(per,t)]) for t in teams}
        cur=new_stint(on,_secs(per,{'displayValue':'12:00' if per<=4 else '5:00'}))
        for p in pp:
            tt=p.get('type',{}).get('text') or '';ts=_secs(per,p.get('clock'))
            parts=[str(x['athlete']['id']) for x in (p.get('participants') or []) if x.get('athlete')]
            team=tid.get(str((p.get('team') or {}).get('id')))
            if tt=='Substitution':
                if len(parts)>=2 and parts[0] in team_of:
                    t=team_of[parts[0]]
                    if parts[1] in on[t] or parts[0] not in on[t]:
                        cur['t1']=ts
                        if keep(cur):stints.append(cur)
                        on[t].discard(parts[1]);on[t].add(parts[0])
                        cur=new_stint(on,ts)
                continue
            cur['t1']=ts
            if not team:continue
            ev=collections.Counter();low=(p.get('text') or '').lower()
            if tt.startswith('Free Throw'):
                ev['FTA']=1
                if p.get('scoringPlay'):ev['FTM']=1;ev['PTS']=1
                shooter=parts[0] if parts else None
            elif p.get('shootingPlay') and 'Heave' in tt and not p.get('scoringPlay'):
                shooter=None   # a missed end-of-period heave is not a field-goal attempt in the box; a block on it still counts
                if len(parts)>1 and 'block' in low:
                    b=parts[1];bt=team_of.get(b)
                    if bt:cur['pl'][b]['BLK']+=1;cur['team'][bt]['BLK']+=1
            elif p.get('shootingPlay'):
                three=(p.get('pointsAttempted')==3) or ('three point' in low) or (p.get('scoreValue')==3)
                ev['FGA']=1;ev['TPA']=1 if three else 0
                if p.get('scoringPlay'):ev['FGM']=1;ev['TPM']=1 if three else 0;ev['PTS']=p.get('scoreValue') or (3 if three else 2)
                shooter=parts[0] if parts else None
            else:shooter=None
            if shooter:
                cur['team'][team].update(ev);cur['pl'][shooter].update(ev)
                if ev.get('FGM') and len(parts)>1 and 'assist' in low:cur['team'][team]['AST']+=1;cur['pl'][parts[1]]['AST']+=1
                if ev.get('FGA') and not ev.get('FGM') and len(parts)>1 and 'block' in low:
                    b=parts[1];bt=team_of.get(b)
                    if bt:cur['pl'][b]['BLK']+=1;cur['team'][bt]['BLK']+=1
            elif 'Offensive Rebound' in tt:
                cur['team'][team]['ORB']+=1
                if parts and team_of.get(parts[0])==team:cur['pl'][parts[0]]['ORB']+=1
            elif 'Defensive Rebound' in tt:
                cur['team'][team]['DRB']+=1
                if parts and team_of.get(parts[0])==team:cur['pl'][parts[0]]['DRB']+=1
            elif 'urnover' in tt or 'turnover' in low or tt=='Traveling':   # ESPN's 'Traveling' says neither
                cur['team'][team]['TOV']+=1
                if parts and team_of.get(parts[0])==team:cur['pl'][parts[0]]['TOV']+=1
                if len(parts)>1 and 'steal' in low:
                    st=parts[1];stt=team_of.get(st)
                    if stt:cur['pl'][st]['STL']+=1;cur['team'][stt]['STL']+=1
        cur['t1']=_secs(per,{'displayValue':'0:00'})
        if keep(cur):stints.append(cur)
    out=[]
    for st_ in stints:
        out.append({'s':round(st_['t1']-st_['t0'],1),'on':{t:sorted(v) for t,v in st_['on'].items()},'team':{t:dict(c) for t,c in st_['team'].items()},
                    'pl':{k:dict(v) for k,v in st_['pl'].items() if v}})
    return {'teams':teams,'names':names,'team_of':team_of,'stints':out}
def build_stints(cache,season,types=('2','3'),log=print,excluded=None):
    games=[];skipped=[];excluded=excluded if excluded is not None else []
    for gid,s in summaries(cache,season):
        h=s.get('header') or {}
        st=str(((h.get('season') or {}).get('type')))
        if st not in types:continue
        if not nba_game(s):excluded.append(gid);continue
        try:g=parse_game(s)
        except Exception as e:skipped.append([gid,str(e)[:120]]);log('skip',gid,e);continue
        g['id']=gid;g['type']=st;g['date']=((h.get('competitions') or [{}])[0].get('date') or '')[:10]
        games.append(g)
    return games,skipped

# ---------------------------------------------------------------- outputs
def write_json(path,obj):
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    data=json.dumps(obj).encode()
    if os.path.exists(path) and open(path,'rb').read()==data:return False
    with open(path,'wb') as f:f.write(data)
    return True
def write_gz(path,obj):
    """Deterministic gzip (mtime 0, no file name), so unchanged content is unchanged bytes."""
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    buf=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',fileobj=buf,mtime=0) as g:g.write(json.dumps(obj).encode())
    data=buf.getvalue()
    if os.path.exists(path) and open(path,'rb').read()==data:return False
    with open(path,'wb') as f:f.write(data)
    return True

def main(argv=None):
    ap=argparse.ArgumentParser(description='NBA daily ingest: ESPN -> season tables + stints')
    ap.add_argument('--season',required=True);ap.add_argument('--cache',default='.nba-cache');ap.add_argument('--out',default='.')
    ap.add_argument('--from',dest='start');ap.add_argument('--to',dest='end');ap.add_argument('--no-fetch',action='store_true')
    ap.add_argument('--types',default='2,1',help='season types to tabulate: 2 regular, 1 preseason, 3 playoffs')
    a=ap.parse_args(argv)
    t0=time.time();status={'season':a.season,'startedAt':datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    if not a.no_fetch:
        n_new,n_dates=fetch(a.season,a.cache,a.start,a.end)
        status.update(newGames=n_new,datesRead=n_dates)
    status['tables']={}
    for st in [x.strip() for x in a.types.split(',') if x.strip()]:
        tab,skipped=build_tables(a.cache,a.season,st)
        if not tab['games']:continue
        changed=write_json(os.path.join(a.out,'nba-tables',f'{a.season}-{st}.json'),tab)
        status['tables'][st]={'games':tab['games'],'players':len(tab['players']),'teams':len(tab['teams']),'changed':changed,'skipped':skipped}
    excluded=[];games,skipped=build_stints(a.cache,a.season,excluded=excluded)
    status['stints']={'games':len(games),'stints':sum(len(g['stints']) for g in games),'skipped':skipped,'notNbaGames':excluded,
        'changed':write_gz(os.path.join(a.out,'nba-stints',f'{a.season}.json.gz'),games) if games else False}
    ev_new,ev_all=write_events(a.cache,a.season,a.out);status['events']={'written':ev_new,'total':ev_all}
    status['requests']=REQUESTS[0]
    status['seconds']=round(time.time()-t0,1)
    print(json.dumps(status))
    write_json(os.path.join(a.out,'nba-ingest-status.json'),status)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print('NBA ingest FAILED:',type(e).__name__,str(e)[:300],file=sys.stderr);sys.exit(1)
