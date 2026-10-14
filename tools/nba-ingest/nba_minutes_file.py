#!/usr/bin/env python3
"""minutes-data.json for the slate the projection step just projected (Vale 10/12, 11:28 PM). Stdlib only.

Why: the NBA page's sim takes each player's game log from minutes-data.json (seasonGames when it has five rows or more, else
last10Games: his own variance, and the kernel's points per minute). The published file is June's two Finals teams, so on opening
night a sim would run with no game log for nearly every player unless this file is written.

For every slate player the projection file has joined to an ESPN athlete id, it writes, BY THAT ID, from the event files in the
repo (nba-events/<season>/, regular season and playoffs, NBA games only):
  seasonGames   never fewer than 25 games played while he has them (Vale 10/13: five games give a spread that is off by about a
                third either way, 25 bring that to a seventh): this season's rows first, then his most recent of last season
                until 25 played are in it, whatever team he played for. Newest first. A game his team's box score lists him in
                and he did not play keeps its row with the zero keys.
  last10Games   his last ten games played, across the two seasons, newest first. l5Avg, l10Avg: his minutes over the last five
                and the last ten of those. Minutes are whole, as ESPN's box score gives them.
  projectedMinutes  the projection step's minutes. confidence and reasoning: from the step's row (same team, new to team, or no
                NBA game; the model or the bench-minute stand-in).
  typedMinutes  a minute Chris typed FOR THIS SLATE DATE (Vale's ruling A). It lives in this file, on his row, with typedAt; the
                job hands it to the step, which holds it and fits his team around it; here the row keeps his number as
                projectedMinutes with the model's beside it (modelMinutes) and says "Typed 30; the model has 15.4." When the
                league's own Out or Inactive beats it the row is Out, keeps what he typed, and says so (typedBeatenBy). A typed
                minute of another slate date is not carried.
  injuryStatus  "Out" for a row the step has as unavailable, else "Available".
A row that did not join is not written (it has no ESPN id and no minutes).

A row of this file NEVER carries substitutionChain (Vale's ruling, 10/13 evening). The page's lineup builder reads that key off
this file for a handcuff bonus; the Colab never wrote it here, so that bonus has never run in a build, and a chain says who comes
on when a player goes off, which is as often the star as his backup. A lineup rule is not switched on by a data file: a test checks
that no row has the key.

player-profiles.json (Vale 10/13, the held profiles step un-held this way): the Minutes tab's fallback file is written here, at
every press, from the SAME rows as the minutes file, in the Colab's shape (name, team, pos, price, currentProjMin, l5Avg, l10Avg,
last10Games, seasonGames, substitutionChain, opponent), so the two can never disagree. The chain lives HERE and in nba-stats.json
only: the stats step's chain for the player, an entry kept only when both players are on the same team in tonight's file
(nba-tonight/<date>.json), a replacement on the slate named as the slate names him. The page's fallback from this file does not
pass the chain on.

It never writes a thinner file: if either file in the repo holds more of this slate's players than would be written, it writes
neither and says so. The two are written under temporary names and then put in place together (write_pair): if the second cannot
be put in place, the first is put back as it was. It writes minutes-data.json, player-profiles.json and its entry in the run's status file, under "minutes";
nothing else. It is soft: when it cannot write, the reason is in the status file and the run goes on.

Usage (a step of run.py):  python3 nba_minutes_file.py --season 2026-27 --repo . [--date 2026-10-20]"""
import json,os,sys,argparse,datetime
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nba_ingest as I  # noqa: E402
import nba_profiles as P  # noqa: E402

LOG_GAMES=25
TYPES=('2','3')    # regular season and playoffs; the preseason is not a game log (Vale tested it and left it out)
ZERO=dict(pts=0,reb=0,ast=0,dkfp=0.0)

def previous(season):
    y=int(season[:4])-1;return '%d-%02d'%(y,(y+1)%100)

def logs(repo,season):
    """ESPN athlete id -> his rows of one season, newest first: every final NBA game whose box score lists him."""
    out={}
    for gid,s in I.summaries(os.path.join(repo,'nba-events'),season):
        h=s.get('header') or {}
        if str((h.get('season') or {}).get('type')) not in TYPES or not I.nba_game(s):continue
        bx=(s.get('boxscore') or {}).get('players') or []
        if len(bx)!=2:continue
        try:day=datetime.datetime.fromisoformat((((h.get('competitions') or [{}])[0]).get('date') or '').replace('Z','+00:00')).astimezone(P.ET).date().isoformat()
        except Exception:continue
        for side in bx:
            st=((side.get('statistics') or [{}])[0]) or {};lab=st.get('labels') or st.get('names') or []
            for a in st.get('athletes') or []:
                i=str((a.get('athlete') or {}).get('id') or '')
                if not i:continue
                v=dict(zip(lab,a.get('stats') or []));m=round(P._mins(v.get('MIN')),1) if not a.get('didNotPlay') else 0
                if m>0:
                    pts,reb,ast=int(P._num(v.get('PTS'))),int(P._num(v.get('REB'))),int(P._num(v.get('AST')))
                    row=dict(date=day,minutes=m,started=bool(a.get('starter')),pts=pts,reb=reb,ast=ast,
                             dkfp=P.dkfp(pts,reb,ast,int(P._num(v.get('STL'))),int(P._num(v.get('BLK'))),int(P._num(v.get('TO'))),P._made(v.get('3PT'))))
                else:row=dict(date=day,minutes=0,started=False,**ZERO)   # listed and did not play: the zero keys stay (Vale's point 4)
                out.setdefault(i,[]).append((day,gid,row))
    return {i:[r for _,_,r in sorted(v,key=lambda x:(x[0],x[1]),reverse=True)] for i,v in out.items()}

def confidence(row):
    """high: the model, on the team he last played for. medium: the model, new to this team. low: no model number for him."""
    if row.get('basis')!='model':return 'low'
    return 'high' if row.get('kind')=='same team' else 'medium'

def reasoning(row):
    kind={'same team':'Same team as his last game','new to team':'New to this team','no NBA game':'No NBA game yet'}.get(row.get('kind'),str(row.get('kind') or ''))
    basis={'model':'minutes from the model','bench-minute stand-in':'a bench-minute stand-in, not the model','not playing':'not playing'}.get(row.get('basis'),str(row.get('basis') or ''))
    return (kind+'; '+basis+'.') if kind or basis else None

def _g(x):
    return ('%.1f'%x).rstrip('0').rstrip('.') if isinstance(x,(int,float)) else str(x)

def chains(repo,date,slate_name):
    """(ESPN id -> his chain for tonight, why there is none). The stats row's chain; an entry is kept only when the player and his
    replacement are on the same team in tonight's file. slate_name: ESPN id -> the slate's own name for a player on it."""
    try:st=json.load(open(os.path.join(repo,'nba-stats.json')))
    except Exception:return {},'nba-stats.json is not in the repo'
    try:tn=json.load(open(os.path.join(repo,'nba-tonight',date+'.json')))
    except Exception:return {},'nba-tonight/%s.json is not in the repo'%date
    team={}
    for t,L in ((tn.get('rosters') if isinstance(tn,dict) else None) or {}).items():
        for p in L if isinstance(L,list) else []:
            if isinstance(p,dict) and p.get('espnId'):team[str(p['espnId'])]=t
    if not team:return {},'the tonight file has no rosters'
    out={}
    for r in (st.get('rows') if isinstance(st,dict) else None) or []:
        if not isinstance(r,dict) or not r.get('espnId'):continue
        me=team.get(str(r['espnId']))
        if not me:continue
        ch=[]
        for c in r.get('substitutionChain') or []:
            if not isinstance(c,dict) or not c.get('replacement') or not c.get('espnId'):continue
            pct=c.get('pct')
            if isinstance(pct,bool) or not isinstance(pct,(int,float)) or not pct>0 or team.get(str(c['espnId']))!=me:continue
            ch.append({'replacement':slate_name.get(str(c['espnId'])) or str(c['replacement']),'espnId':str(c['espnId']),'pct':pct})
        if ch:out[str(r['espnId'])]=ch
    return out,None

def player(row,cur,prev,old=None):
    played=lambda g:[x for x in g if x['minutes']>0]
    season=list(cur);n=len(played(cur))
    for g in prev:                      # his most recent of last season, until 25 played are in the log
        if n>=LOG_GAMES:break
        season.append(g);n+=1 if g['minutes']>0 else 0
    last=(played(cur)+played(prev))[:10];m5=[x['minutes'] for x in last[:5]];m10=[x['minutes'] for x in last]
    l10=list(last)
    while len(l10)<10:l10.append({'date':'','minutes':0,'started':False})   # the page's table has ten columns
    mins=row.get('min');model=row.get('modelMin') if row.get('modelMin') is not None else mins
    was=old.get('typedMinutes') if isinstance(old,dict) and isinstance(old.get('typedMinutes'),(int,float)) and not isinstance(old.get('typedMinutes'),bool) else None
    # The step held it (the row is in at his number), or something beat it and the row is out at 0. In the second case the number he
    # typed is the one already in the file, NOT the row's 0 (Monty 10/13: writing that 0 would hand the step a typed 0 on the next
    # press and keep the player out after the league cleared him: a zero nobody typed).
    held=row.get('typed') and row.get('available')
    typed=mins if held else (was if was is not None else (mins if row.get('typed') else None))
    why=reasoning(row);extra={}
    if typed is not None:
        beaten=not row.get('available') and typed>0
        extra=dict(typedMinutes=typed,typedAt=(old or {}).get('typedAt') if was==typed else None,modelMinutes=model)
        if beaten:extra['typedBeatenBy']=str(row.get('whyOut') or 'out');why='Typed %s; he is out (%s).'%(_g(typed),extra['typedBeatenBy'])
        else:why='Typed %s; the model has %s.'%(_g(typed),_g(model))
    return dict(name=row['name'],espnId=str(row['espnId']),team=row.get('team') or '',pos=row.get('pos') or '',price=int(P._num(row.get('salary'))),
        projectedMinutes=mins,confidence=confidence(row),reasoning=why,injuryStatus='Available' if row.get('available') else 'Out',**extra,
        l5Avg=round(sum(m5)/len(m5),1) if m5 else 0.0,gp5=len(m5),l10Avg=round(sum(m10)/len(m10),1) if m10 else 0.0,currentProjMin=mins,
        last10Games=l10,seasonGames=season)

PROFILE_KEYS=('name','team','pos','price','currentProjMin','l5Avg','l10Avg','last10Games','seasonGames')      # a profile's own copy of the minutes row

def profiles_of(doc,opp,chain):
    """player-profiles.json in the Colab's shape, from the minutes file's own rows. opp: ESPN id -> tonight's opponent; chain:
    ESPN id -> his substitution chain for tonight, which goes on the profile and NOT on the minutes row."""
    return {'generatedAt':doc['lastUpdated'],'season':doc['season'],'slateDate':doc['slateDate'],'source':'nba_minutes_file.py, the same rows as minutes-data.json',
            'profiles':[{**{k:p[k] for k in PROFILE_KEYS},'substitutionChain':list(chain.get(p['espnId']) or []),'opponent':opp.get(p['espnId'],'')} for p in doc['players']]}

def pick_date(repo,date=None,today=None):
    """The projected date this file is for: --date, else the nearest projected date from today on, else the latest one."""
    if date:return date
    try:pd=(json.load(open(os.path.join(repo,'nba-ingest-status.json'))).get('projection') or {}).get('dates') or []
    except Exception:pd=[]
    ok=sorted(x['date'] for x in pd if isinstance(x,dict) and x.get('wrote') and x.get('date'))
    if not ok:return None
    today=today or datetime.datetime.now(datetime.timezone.utc).astimezone(P.ET).strftime('%Y-%m-%d')
    ahead=[d for d in ok if d>=today]
    return ahead[0] if ahead else ok[-1]

def build(repo,season,date,now=None):
    """Returns (minutes document or None, profiles document or None, status entry). Both documents or neither."""
    try:proj=json.load(open(os.path.join(repo,'nba-proj',date+'.json')))
    except Exception:return None,None,{'wrote':False,'date':date,'why':'nba-proj/%s.json is not in the repo'%date}
    rows=[r for r in proj.get('rows') or [] if isinstance(r,dict) and r.get('name') and r.get('espnId')]
    if not rows:return None,None,{'wrote':False,'date':date,'why':'the projection file has no joined slate player'}
    cur=logs(repo,season);prev=logs(repo,previous(season))
    try:olddoc=json.load(open(os.path.join(repo,'minutes-data.json')))
    except Exception:olddoc={}
    # a typed minute is carried for ITS slate date only: the old file's rows count when that file was written for this date
    same=isinstance(olddoc,dict) and olddoc.get('slateDate')==date
    oldby={str(p.get('espnId')):p for p in (olddoc.get('players') or []) if isinstance(p,dict) and p.get('espnId')} if same else {}
    chain,no_chain=chains(repo,date,{str(r['espnId']):r['name'] for r in rows})
    players=[player(r,cur.get(str(r['espnId']),[]),prev.get(str(r['espnId']),[]),oldby.get(str(r['espnId']))) for r in rows]
    players.sort(key=lambda x:(x['team'],-(x['projectedMinutes'] or 0),x['name']))
    five=lambda p:sum(1 for g in p['seasonGames'] if g['minutes']>0 and g.get('dkfp') is not None)>=5
    pool=[r['name'] for r in rows if (r.get('proj') or 0)>=15];by={p['name']:p for p in players}
    entry={'date':date,'players':len(players),'withFiveGames':sum(1 for p in players if five(p)),'out':sum(1 for p in players if p['injuryStatus']=='Out'),
           'pool15':len(pool),'pool15Short':sorted(n for n in pool if not five(by[n])),
           'typed':sum(1 for p in players if p.get('typedMinutes') is not None),'typedBeaten':sorted(p['name'] for p in players if p.get('typedBeatenBy')),
           'profilesWithChain':sum(1 for p in players if chain.get(p['espnId'])),**({'noChain':no_chain} if no_chain else {})}
    # never a thinner file: the players of THIS slate the file in the repo already holds must all still be there
    old=(olddoc.get('players') or []) if isinstance(olddoc,dict) else []
    slate={P.canon_name(r['name']) for r in proj.get('rows') or [] if isinstance(r,dict) and r.get('name')}
    held=sum(1 for p in old if isinstance(p,dict) and P.canon_name(p.get('name') or '') in slate)
    if len(players)<held:return None,None,{'wrote':False,**entry,'why':'the file in the repo holds %d players of this slate and this run would write %d; left as it is'%(held,len(players))}
    # the same rule for the profiles file, and both or neither: the two files are written from the same rows and must never disagree
    try:oldprof=json.load(open(os.path.join(repo,'player-profiles.json')))
    except Exception:oldprof={}
    heldp=sum(1 for p in ((oldprof.get('profiles') if isinstance(oldprof,dict) else None) or []) if isinstance(p,dict) and P.canon_name(p.get('name') or '') in slate)
    if len(players)<heldp:return None,None,{'wrote':False,**entry,'why':'player-profiles.json in the repo holds %d players of this slate and this run would write %d; both files left as they are'%(heldp,len(players))}
    stamp=(now or datetime.datetime.now(datetime.timezone.utc)).strftime('%Y-%m-%dT%H:%M:%SZ')
    doc={'lastUpdated':stamp,'season':season,'slateDate':date,'draftGroups':sorted({str(r['draftGroup']) for r in rows if r.get('draftGroup')}),
         'source':'nba_minutes_file.py from nba-proj/%s.json'%date,'players':players}
    return doc,profiles_of(doc,{str(r['espnId']):str(r.get('opp') or '') for r in rows},chain),{'wrote':True,'profiles':True,**entry}

def write_pair(repo,doc,prof,replace=os.replace):
    """Both files or neither (Monty 10/13: two writes one after the other left the first on disk when the second failed, with a
    status that said nothing was written). Each is written whole under a temporary name beside it; only then are the two put in
    place. If the second cannot be put in place the first is put back as it was (removed if it was not there) and the error goes
    up, so the status says nothing was written and that is true. The bytes are the ones write_json writes."""
    paths=[os.path.join(repo,'minutes-data.json'),os.path.join(repo,'player-profiles.json')];tmps=[x+'.writing' for x in paths]
    try:
        for t,obj in zip(tmps,(doc,prof)):
            with open(t,'wb') as f:f.write(json.dumps(obj).encode())
        was=open(paths[0],'rb').read() if os.path.exists(paths[0]) else None
        replace(tmps[0],paths[0])
        try:replace(tmps[1],paths[1])
        except Exception:
            if was is None:os.remove(paths[0])
            else:
                with open(paths[0],'wb') as f:f.write(was)
            raise
    finally:
        for t in tmps:
            if os.path.exists(t):os.remove(t)

def main(argv=None,now=None,replace=os.replace):
    ap=argparse.ArgumentParser(description='Write minutes-data.json and player-profiles.json for the projected slate: game logs by ESPN id, the step\'s minutes and status; the substitution chain on the profiles only')
    ap.add_argument('--season',required=True);ap.add_argument('--repo',default='.');ap.add_argument('--date',help='the projected date (default: the nearest one the projection step wrote in this run)')
    a=ap.parse_args(argv)
    date=pick_date(a.repo,a.date,now.astimezone(P.ET).strftime('%Y-%m-%d') if now else None)
    if not date:
        entry={'wrote':False,'why':'no projection file was written in this run'}
    else:
        try:
            doc,prof,entry=build(a.repo,a.season,date,now)
            if doc is not None and prof is not None:write_pair(a.repo,doc,prof,replace)
        except Exception as e:entry={'wrote':False,'date':date,'why':'%s: %s'%(type(e).__name__,str(e)[:200])}
    print(json.dumps(entry));P.note(a.repo,entry,'minutes')
    return 0

if __name__=='__main__':
    sys.exit(main())
