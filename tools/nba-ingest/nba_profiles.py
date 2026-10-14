#!/usr/bin/env python3
"""NBA player profiles from ESPN (Wes 9/29; part 2 of Vale 17d0cd01). Replaces the Colab's player-profiles.json and the
history half of minutes-data.json, in the shapes nba/index.html reads:
  - the sim's per-player sigma comes from minutes-data players[].seasonGames[].dkfp (games with minutes > 0), keyed by
    the slate's EXACT names;
  - the Minutes tab reads l5Avg / l10Avg / gp5 / last10Games / currentProjMin.
Rules are the Colab's (cells 3 and 5):
  - a player's games are his TEAM's final games, newest first, 0 minutes when he did not play (no stats on those rows);
    the NBA Cup Championship and the All-Star games (ESPN files both as regular season) are left out, as in the NBA's own
    logs the Colab read;
  - L5 / L10 average the non-zero minutes of the last 5 / 10 team games, zeroed when he played <= 2 of 5 / <= 4 of 10;
    with under 3 such games, his own last 10 regular-season appearances are used instead when they have more;
  - dkfp = PTS + 1.25 REB + 1.5 AST + 2 STL + 2 BLK - 0.5 TOV + 0.5 3PM; +1.5 when two of PTS/REB/AST/STL/BLK reach 10,
    +3 more when three do;
  - substitutionChain = up to 4 players who came in when he went out (this season's substitutions), share >= 5%;
  - currentProjMin = L5, else the slate's minutes (data.json myProj "min"). The Colab used DK's FPPG there for players
    known only from dk-slates.json; that is points, not minutes, so those players get 0 instead;
  - minutes-data players carry no substitutionChain (the Colab never wrote it there, so the sim's handcuff bonus has
    never fired; adding it would change the sim, which is a decision, not an ingest detail).
The slate is data.json myProj (the names the sim uses) plus NBA DraftKings slates in dk-slates.json (sportId 4). The
minutes agent's fields (projectedMinutes, confidence, reasoning, injuryStatus) in the existing minutes-data.json are kept
by name, as the Colab did.
Usage (data repo root): python3 nba_profiles.py --season 2026-27 --cache .nba-cache --repo . [--types 2,3]"""
import os,re,sys,json,argparse,datetime,unicodedata,collections
from zoneinfo import ZoneInfo
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from nba_ingest import summaries,write_json,nba_game,NBA30

ET=ZoneInfo('America/New_York')
# DraftKings / NBA / Basketball-Reference team codes -> ESPN's
TEAM={'GSW':'GS','NOP':'NO','NOR':'NO','NYK':'NY','SAS':'SA','UTA':'UTAH','WAS':'WSH','PHO':'PHX','BRK':'BKN','CHO':'CHA','CHH':'CHA'}
def espn_team(t):t=(t or '').upper().strip();return TEAM.get(t,t)
# the Colab's canon_name, aliases included
_ALIASES={'alex sarr':'alexandre sarr','nic claxton':'nicolas claxton','cam thomas':'cameron thomas','kj martin':'kenyon martin',
 'kj martin jr':'kenyon martin','gg jackson':'gregory jackson','gg jackson ii':'gregory jackson','ej harkless':'elijah harkless',
 'ron holland':'ronald holland','bub carrington':'carlton carrington','hansen yang':'yang hansen','lj cryer':'lj cryer','jimmy butler iii':'jimmy butler'}
def normalize_name(name):
    nfkd=unicodedata.normalize('NFKD',name or '');s=''.join(c for c in nfkd if not unicodedata.combining(c))
    s=s.replace('.','').replace("'",'').replace('-',' ')
    s=re.sub(r'\b(jr|sr|iii|iv|ii|v)\s*$','',s,flags=re.IGNORECASE)
    return ' '.join(s.split()).strip()
def canon_name(name):n=normalize_name(name).lower();return _ALIASES.get(n,n)

def _num(x):
    try:return float(x)
    except Exception:return 0.0
def _made(x):
    try:return int(str(x).split('-')[0])
    except Exception:return 0
def _mins(x):
    if x in (None,'','--'):return 0.0
    if ':' in str(x):m,s=str(x).split(':');return float(m)+float(s)/60
    return _num(x)
def dkfp(pts,reb,ast,stl,blk,tov,fg3m):
    v=pts+1.25*reb+1.5*ast+2*stl+2*blk-0.5*tov+0.5*fg3m
    cats=sum(1 for s in (pts,reb,ast,stl,blk) if s>=10)
    if cats>=2:v+=1.5
    if cats>=3:v+=3
    return round(v,1)

def collect(cache,season,types=('2','3')):
    """Every final game of the season types: team timelines, player lines, substitution counts, latest team per player."""
    team_games=collections.defaultdict(list);lines=collections.defaultdict(dict);names={};latest={};dates={};subs=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for gid,s in summaries(cache,season):
        h=s.get('header') or {}
        if str(((h.get('season') or {}).get('type'))) not in types:continue
        if not nba_game(s):continue   # All-Star games, the Cup Championship, exhibitions (nba_ingest.nba_game)
        bx=(s.get('boxscore') or {}).get('players') or []
        if len(bx)!=2:continue
        comp=(h.get('competitions') or [{}])[0]
        try:day=datetime.datetime.fromisoformat((comp.get('date') or '').replace('Z','+00:00')).astimezone(ET).date().isoformat()
        except Exception:continue
        team_of={};dates[gid]=(day,str(((h.get('season') or {}).get('type'))))
        for side in bx:
            abbr=side['team']['abbreviation'];team_games[abbr].append((day,gid))
            st=side['statistics'][0];lab=st.get('labels') or st.get('names') or []
            for a in st.get('athletes') or []:
                i=str((a.get('athlete') or {}).get('id') or '')
                if not i:continue
                names[i]=a['athlete'].get('displayName') or names.get(i,'');team_of[i]=abbr
                if (day,gid)>=latest.get(i,('',''))[:2]:latest[i]=(day,gid,abbr)
                if a.get('didNotPlay') or not a.get('stats'):continue
                v=dict(zip(lab,a['stats']))
                lines[i][gid]=dict(minutes=round(_mins(v.get('MIN')),1),started=bool(a.get('starter')),pts=int(_num(v.get('PTS'))),reb=int(_num(v.get('REB'))),
                    ast=int(_num(v.get('AST'))),stl=int(_num(v.get('STL'))),blk=int(_num(v.get('BLK'))),tov=int(_num(v.get('TO'))),fg3m=_made(v.get('3PT')))
        for p in s.get('plays') or []:
            if (p.get('type') or {}).get('text')!='Substitution':continue
            parts=[str(x['athlete']['id']) for x in (p.get('participants') or []) if x.get('athlete')]
            if len(parts)>=2 and parts[0] in team_of and team_of.get(parts[1])==team_of[parts[0]]:
                subs[team_of[parts[0]]][parts[1]][parts[0]]+=1        # participants: [player in, player out]
    for t in team_games:team_games[t]=sorted(set(team_games[t]),reverse=True)
    return dict(team_games=dict(team_games),lines=dict(lines),names=names,team={i:v[2] for i,v in latest.items()},subs=subs,dates=dates)

def load_slate(repo):
    """data.json myProj first (the names the sim uses), then NBA DraftKings slates; one entry per canonical name."""
    out=[];seen=set()
    def add(name,team,pos,price,minutes,opp,source):
        c=canon_name(name)
        if not name or c in seen:return
        seen.add(c);out.append(dict(name=name,team=team or '',pos=pos or '',price=int(_num(price)),min=_num(minutes),opp=opp or '',source=source))
    try:d=json.load(open(os.path.join(repo,'data.json')))
    except Exception:d={}
    for p in d.get('myProj') or []:
        if isinstance(p,dict) and p.get('name') and p.get('name')!='0':add(p['name'],p.get('team'),p.get('pos'),p.get('price'),p.get('min'),p.get('opp'),'data.json')
    try:k=json.load(open(os.path.join(repo,'dk-slates.json')))
    except Exception:k={}
    for key,sl in (k.get('slates') or {}).items():
        if str(sl.get('sportId'))!='4' or not isinstance(sl.get('players'),list):continue
        for p in sl['players']:add(p.get('name'),p.get('team'),p.get('pos'),p.get('salary'),0,'','dk-slates.json')
    return out

def match(slate,col):
    """slate name -> ESPN id: same canonical name on his team; else the only such name anywhere (a trade); else the only
    player on his team with that last name and first initial. Unmatched players keep an empty history."""
    by_canon=collections.defaultdict(list)
    for i,n in col['names'].items():by_canon[canon_name(n)].append(i)
    out={}
    for p in slate:
        c=canon_name(p['name']);t=espn_team(p['team']);ids=by_canon.get(c,[])
        hit=[i for i in ids if col['team'].get(i)==t]
        if len(hit)!=1 and len(ids)==1:hit=ids
        if len(hit)!=1 and c:
            parts=c.split();hit=[i for i,n in col['names'].items() if col['team'].get(i)==t and canon_name(n).split()[-1:]==parts[-1:] and canon_name(n)[:1]==c[:1]]
        if len(hit)==1:out[p['name']]=hit[0]
    return out

def profile(p,pid,col):
    t=espn_team(p['team']);games=col['team_games'].get(t,[])   # his slate team's games, as the Colab did (DK knows his current team)
    L=col['lines'].get(pid,{}) if pid else {}
    season=[]
    for day,gid in games:
        g=L.get(gid);e={'date':day,'minutes':g['minutes'] if g else 0,'started':bool(g and g['started'])}
        if g:e.update(pts=g['pts'],reb=g['reb'],ast=g['ast'],dkfp=dkfp(g['pts'],g['reb'],g['ast'],g['stl'],g['blk'],g['tov'],g['fg3m']))
        season.append(e)
    last10=[e['minutes'] for e in season[:10]]+[0.0]*max(0,10-len(season))
    # Colab cell 3: under 3 games with minutes in his team's last 10, L5/L10 come from his own last 10 regular-season appearances
    # (trades, injuries, call-ups) when those have more; last10Games still shows the team's games.
    nz=sum(1 for m in last10 if m>0)
    if nz<3 and L:
        reg=[g for g in L if col['dates'].get(g,('',''))[1]=='2']   # his regular-season game log, as the Colab read
        own=[L[g]['minutes'] for g in sorted(reg,key=lambda g:(col['dates'][g][0],g),reverse=True)[:10]]
        if sum(1 for m in own if m>0)>nz:last10=own+[0.0]*(10-len(own))
    l5g=[m for m in last10[:5] if m>0];l10g=[m for m in last10 if m>0]
    l5=round(sum(l5g)/len(l5g),1) if l5g else 0.0;l10=round(sum(l10g)/len(l10g),1) if l10g else 0.0
    if len(l5g)<=2:l5=0.0
    if len(l10g)<=4:l10=0.0
    chain=[]
    if pid:
        cnt=col['subs'].get(t,{}).get(pid,collections.Counter());tot=sum(cnt.values())
        for inn,n in sorted(cnt.items(),key=lambda x:(-x[1],x[0]))[:4]:
            if tot and n/tot>=0.05:chain.append({'replacement':col['names'].get(inn,inn),'pct':round(n/tot,3)})
    l10games=season[:10]
    while len(l10games)<10:l10games.append({'date':'','minutes':0,'started':False})
    return dict(name=p['name'],team=p['team'],pos=p['pos'],price=p['price'],currentProjMin=l5 or p['min'] or 0,l5Avg=l5,l10Avg=l10,gp5=len(l5g),
        last10Games=l10games,seasonGames=season or l10games,substitutionChain=chain,opponent=p['opp'])

def build(cache,season,repo,types=('2','3'),now=None):
    now=now or datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y-%m-%dT%H:%M:%SZ')
    col=collect(cache,season,types);slate=load_slate(repo);ids=match(slate,col)
    # chain names: the slate's own name for a matched replacement, so the sim's name lookup finds him
    back={v:k for k,v in ids.items()}
    profs=[]
    for p in slate:
        pr=profile(p,ids.get(p['name']),col)
        for c in pr['substitutionChain']:
            rid=next((i for i,n in col['names'].items() if n==c['replacement']),None)
            if rid in back:c['replacement']=back[rid]
        profs.append(pr)
    try:prev={p['name']:p for p in json.load(open(os.path.join(repo,'minutes-data.json'))).get('players',[])}
    except Exception:prev={}
    players=[]
    for pr in profs:
        o=prev.get(pr['name'],{})
        players.append(dict(name=pr['name'],team=pr['team'],pos=pr['pos'],price=pr['price'],projectedMinutes=o.get('projectedMinutes'),confidence=o.get('confidence'),
            reasoning=o.get('reasoning'),injuryStatus=o.get('injuryStatus') or 'Available',l5Avg=pr['l5Avg'],gp5=pr['gp5'],l10Avg=pr['l10Avg'],
            currentProjMin=pr['currentProjMin'],last10Games=pr['last10Games'],seasonGames=pr['seasonGames']))
    players.sort(key=lambda x:(x['team'],-(x['projectedMinutes'] or x['l5Avg'] or 0)))
    profiles={'generatedAt':stamp,'season':season,'profiles':[{k:v for k,v in p.items() if k!='gp5'} for p in profs]}
    minutes={'lastUpdated':stamp,'season':season,'players':players}
    # withGames: slate players matched to a player with a box-score line in the games read. None means this run has no history for
    # anyone on the slate (before the season's first final game, or a slate of another season), and main() then writes nothing.
    report={'slate':len(slate),'matched':len(ids),'withGames':sum(1 for p in slate if col['lines'].get(ids.get(p['name']))),
        'unmatched':sorted(p['name'] for p in slate if p['name'] not in ids),'games':sum(len(v) for v in col['team_games'].values())//2}
    return profiles,minutes,report

def note(repo,entry,key='profiles'):
    """The run's status file (nba_ingest.py wrote it one step earlier) says what this step did. Run alone, with no status file, this makes none."""
    fp=os.path.join(repo,'nba-ingest-status.json')
    try:st=json.load(open(fp))
    except Exception:return
    if isinstance(st,dict):st[key]=entry;write_json(fp,st)

# Vale 10/12, after this step was compared with the published June 14 files (not a reproduction): it writes nothing, regular season
# included, until he rules on what the page takes from these two files. The job runs it without --write, so it only reports.
HELD='held: this step writes neither file until what the page takes from them is settled (Vale 10/12)'

def main(argv=None):
    ap=argparse.ArgumentParser(description='NBA player profiles + minutes history from ESPN')
    ap.add_argument('--season',required=True);ap.add_argument('--cache',default='.nba-cache');ap.add_argument('--repo',default='.')
    ap.add_argument('--types',default='2,3')
    ap.add_argument('--write',action='store_true',help='write the two files (the job does not pass this: see HELD)')
    a=ap.parse_args(argv)
    profiles,minutes,report=build(a.cache,a.season,a.repo,tuple(x.strip() for x in a.types.split(',') if x.strip()))
    counts={k:report[k] for k in ('slate','matched','withGames','games')}
    # Vale 10/12: a file the page reads is never replaced by a thinner one. With no slate player matched to a game (a press before
    # the regular season's first final game), both files would lose every history they hold, so they are left exactly as they are.
    why=None
    if not report['slate']:why='no NBA slate in data.json or dk-slates.json'
    elif not report['withGames']:why='no slate player matched a game of %s (season types %s): %d on the slate, %d games read'%(a.season,a.types,report['slate'],report['games'])
    held=not why and not a.write
    if held:why=HELD
    if why:
        print('NBA profiles: '+why+'; nothing written');note(a.repo,{'wrote':False,'why':why,**({'held':True} if held else {}),**counts});print(json.dumps(report));return 0
    write_json(os.path.join(a.repo,'player-profiles.json'),profiles);write_json(os.path.join(a.repo,'minutes-data.json'),minutes)
    note(a.repo,{'wrote':True,**counts,'unmatched':len(report['unmatched'])})
    print(json.dumps(report));return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print('NBA profiles FAILED:',type(e).__name__,str(e)[:300],file=sys.stderr);sys.exit(1)
