#!/usr/bin/env python3
"""Tonight's NBA slate from ESPN, for the model step (Vale 10/12, opening-night plan piece 2). Stdlib only.

For one US Eastern date (today unless --date is given) it writes nba-tonight/<date>.json:
  games    each scheduled or played NBA event of that date: ESPN event id, tip time (UTC, as ESPN gives it), status, season type,
           the two teams (ESPN abbreviation and team id), and the line ESPN shows (provider, its wording, spread, total), read from
           the scoreboard, or from the event's own page when the scoreboard carries none. No line = "line": null, never a guess.
  rosters  each of those teams' CURRENT roster: ESPN athlete id, name, position, ESPN's status word. So a rookie or a new signing
           who has no game yet exists for the model.
  slate    every row of the NBA DraftKings slates in dk-slates.json, joined to an ESPN athlete id BY NAME among the rosters of the
           row's own team. A row joins only when exactly one roster player of that team has its name; every row that did not join
           is listed under "unjoined" with the reason. None is guessed. Each row keeps DraftKings' own team code ("team", NYK) and
           status word ("dkStatus", exactly as the slate file has it) and names ESPN's code for the same team ("espnTeam", NY), which
           is the code the games and rosters above use (Vale 10/12).
Only the slates that start on the file's date are read (a slate's date is its first tip in US Eastern time); the others are listed
with read false. A player on two slates of the date is one row, from the classic slate when there is one; "draftGroup" says which.
With no --date it writes a file for today and for each date an NBA slate starts on, from yesterday through fourteen days ahead.
It reads nothing but ESPN's public scoreboard, event and roster pages (about one request a second) and dk-slates.json. The
injury report and the lineups file are NOT read here: the model step reads those itself.

Usage (from the data repo root, as a step of run.py):  python3 nba_tonight.py --repo . --out . [--date 2026-10-13]"""
import json,os,re,sys,time,argparse,datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nba_ingest as I  # noqa: E402
import nba_profiles as P  # noqa: E402

ROSTER='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{}/roster'
ET=ZoneInfo('America/New_York')

def eastern_date(now=None):
    return (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(ET).strftime('%Y-%m-%d')

def _line(odds,source):
    for o in odds or []:
        if not isinstance(o,dict):continue
        if o.get('details') is None and o.get('overUnder') is None and o.get('spread') is None:continue
        return {'provider':(o.get('provider') or {}).get('name'),'details':o.get('details'),'spread':o.get('spread'),'total':o.get('overUnder'),'from':source}
    return None

def slate_date(sl):
    """The US Eastern date of a DraftKings slate's first tip (minStartTime), or None when the slate does not say."""
    try:return datetime.datetime.fromisoformat(re.sub(r'\.\d+','',str(sl.get('minStartTime') or '')).replace('Z','+00:00')).astimezone(ET).strftime('%Y-%m-%d')
    except Exception:return None

def nba_slates(repo):
    """(key, slate) for every NBA slate in dk-slates.json, classic slates first; None when the file cannot be read."""
    try:k=json.load(open(os.path.join(repo,'dk-slates.json')))
    except Exception:return None
    sl=[(str(key),s) for key,s in ((k.get('slates') if isinstance(k,dict) else None) or {}).items() if isinstance(s,dict) and str(s.get('sportId'))=='4' and isinstance(s.get('players'),list)]
    return sorted(sl,key=lambda x:(bool(x[1].get('showdown')),x[0]))

def slate_dates(repo,today):
    """The dates NBA slates start on, from yesterday (a slate still being played after midnight) through fourteen days ahead
    (DraftKings posted the 10/20 opening night on 10/12)."""
    t=datetime.date.fromisoformat(today);lo=(t-datetime.timedelta(days=1)).isoformat();hi=(t+datetime.timedelta(days=14)).isoformat()
    return sorted({d for d in (slate_date(s) for _,s in nba_slates(repo) or []) if d and lo<=d<=hi})

def build(date,repo,get=I.http_get,sleep=time.sleep):
    """Returns (document, requests sent). Raises on a request that fails after retries, so a partial file is never written."""
    n=[0]
    def ask(url):
        n[0]+=1;r=get(url);sleep(1);return r
    sb=ask(I.SCOREBOARD.format(date.replace('-','')))
    games=[];teams={}
    for e in sb.get('events') or []:
        c=(e.get('competitions') or [{}])[0] or {};side={}
        for x in c.get('competitors') or []:
            t=x.get('team') or {};side[x.get('homeAway')]={'abbr':t.get('abbreviation'),'espnTeamId':str(t.get('id') or ''),'name':t.get('displayName')}
        if not side.get('home') or not side.get('away'):continue
        nba=side['home']['abbr'] in I.NBA30 and side['away']['abbr'] in I.NBA30
        line=_line(c.get('odds'),'scoreboard')
        if line is None and nba:line=_line(ask(I.SUMMARY.format(e['id'])).get('pickcenter'),'event')
        games.append({'eventId':str(e['id']),'tip':e.get('date'),'status':((e.get('status') or {}).get('type') or {}).get('name'),'seasonType':str((e.get('season') or {}).get('type') or ''),
                      'away':side['away'],'home':side['home'],'nbaGame':nba,'line':line})
        if nba:
            for s in (side['away'],side['home']):teams[s['abbr']]=s['espnTeamId']
    rosters={}
    for abbr in sorted(teams):
        r=ask(ROSTER.format(teams[abbr]));rosters[abbr]=[]
        for a in r.get('athletes') or []:
            if not isinstance(a,dict) or not a.get('id'):continue
            rosters[abbr].append({'espnId':str(a['id']),'name':a.get('fullName') or a.get('displayName'),'pos':(a.get('position') or {}).get('abbreviation'),'status':(a.get('status') or {}).get('name')})
        rosters[abbr].sort(key=lambda a:(a['name'] or '',a['espnId']))
    slate=join_slate(repo,rosters,date)
    doc={'format':'nba-tonight-v1','date':date,'readAt':datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),'requests':n[0],
         'games':sorted(games,key=lambda g:(g['tip'] or '',g['eventId'])),'rosters':rosters,'slate':slate}
    return doc,n[0]

def join_slate(repo,rosters,date=None):
    """Every NBA DraftKings slate row, joined by name within its own team's roster. Exactly one match or it is listed unjoined.
    A row with no player of its name on the team gets one more look, for a shortened first name (see the second pass)."""
    found=nba_slates(repo)
    if found is None:return {'source':'dk-slates.json','found':False,'slates':[],'rows':[],'unjoined':[]}
    by={}   # (team, canonical name) -> roster players
    for abbr,ps in rosters.items():
        for a in ps:by.setdefault((abbr,P.canon_name(a['name'])),[]).append(a)
    anyteam={}
    for (abbr,c),ps in by.items():anyteam.setdefault(c,[]).extend((abbr,a) for a in ps)
    rows=[];unjoined=[];slates=[];seen=set();taken=set();later=[]
    for key,sl in found:
        # a slate of another date is listed and not read: its players would otherwise be joined to, or refused by, this date's rosters
        sd=slate_date(sl);read=not (date and sd and sd!=date)
        slates.append({'key':key,'label':sl.get('label'),'players':len(sl['players']),'date':sd,'read':read})
        if not read:continue
        for p in sl['players']:
            if not isinstance(p,dict) or not p.get('name'):continue
            team=P.espn_team(p.get('team'));c=P.canon_name(p['name']);ident=(team,c)
            if ident in seen:continue
            seen.add(ident)
            row={'name':p['name'],'team':str(p.get('team') or '').upper().strip(),'espnTeam':team,'pos':p.get('pos') or '','salary':p.get('salary'),'dkStatus':p.get('status'),'draftGroup':key}
            hit=by.get(ident,[])
            if len(hit)==1:taken.add(hit[0]['espnId']);rows.append({**row,'espnId':hit[0]['espnId'],'espnName':hit[0]['name']});continue
            if team not in rosters:why='his team does not play on this date'
            elif len(hit)>1:why='%d players on his team have this name'%len(hit)
            else:later.append((row,c));continue
            unjoined.append({**row,'why':why})
    # Second pass, after every exact join: a shortened first name (10/20 slate: DraftKings "Christopher Cenac Jr.", ESPN "Chris Cenac Jr.").
    # Same team, the same last name, one first name the beginning of the other (two letters or more), exactly one such roster player,
    # and he is not the exact match of another row. The row says how it joined. Anything else stays unjoined and is never guessed.
    for row,c in later:
        f,rest=c.split()[0],c.split()[1:];cand=[]
        for a in rosters[row['espnTeam']]:
            x=P.canon_name(a['name']).split()
            if len(x)>1 and x[1:]==rest and x[0]!=f and min(len(x[0]),len(f))>=2 and (x[0].startswith(f) or f.startswith(x[0])) and a['espnId'] not in taken:cand.append(a)
        if len(cand)==1:taken.add(cand[0]['espnId']);rows.append({**row,'espnId':cand[0]['espnId'],'espnName':cand[0]['name'],'how':'short first name'});continue
        other=sorted({a for a,_ in anyteam.get(c,[])})
        unjoined.append({**row,'why':'no player on his team has this name'+(' (a player with it is on %s)'%', '.join(other) if other else '')})
    return {'source':'dk-slates.json','found':True,'slates':slates,'rows':sorted(rows,key=lambda r:(r['team'],r['name'])),'unjoined':sorted(unjoined,key=lambda r:(r['team'],r['name']))}

def main(argv=None,get=I.http_get,sleep=time.sleep,now=None):
    ap=argparse.ArgumentParser(description="Tonight's NBA games, lines, rosters and the DraftKings name join, from ESPN")
    ap.add_argument('--repo',default='.');ap.add_argument('--out',default='.')
    ap.add_argument('--date',help='one US Eastern date, YYYY-MM-DD (default: today and each date an NBA slate starts on)')
    ap.add_argument('--soft',action='store_true',help='a date that fails is reported in the status file and the exit is 0 (the job passes this)')
    a=ap.parse_args(argv)
    if a.date:datetime.date.fromisoformat(a.date);dates=[a.date]
    else:today=eastern_date(now);dates=sorted({today,*slate_dates(a.repo,today)})
    done=[]
    for date in dates:
        t0=time.time()
        try:doc,n=build(date,a.repo,get=get,sleep=sleep)
        except Exception as e:
            # nothing is written for this date: never a partial file. In the job (--soft) the other dates and the run's history still go.
            if not a.soft:raise
            done.append({'date':date,'wrote':False,'why':'%s: %s'%(type(e).__name__,str(e)[:200])});print(json.dumps(done[-1]));continue
        I.write_json(os.path.join(a.out,'nba-tonight',date+'.json'),doc)
        nba=[g for g in doc['games'] if g['nbaGame']]
        done.append({'date':date,'wrote':True,'games':len(nba),'withLine':sum(1 for g in nba if g['line']),'teams':len(doc['rosters']),'rosterPlayers':sum(len(v) for v in doc['rosters'].values()),
                     'slateRowsJoined':len(doc['slate']['rows']),'joinedByShortFirstName':sum(1 for r in doc['slate']['rows'] if r.get('how')),'slateRowsUnjoined':len(doc['slate']['unjoined']),'requests':n,'seconds':round(time.time()-t0,1)})
        print(json.dumps(done[-1]))
    P.note(a.out,{'dates':done},'tonight')
    return 0

if __name__=='__main__':
    sys.exit(main())
