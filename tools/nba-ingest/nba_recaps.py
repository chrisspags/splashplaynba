#!/usr/bin/env python3
"""NBA game recaps from ESPN (Wes 9/29; part 2b of Vale 17d0cd01). Replaces the Colab's cell 6 (recaps.json for the Game
Recaps tab and the sim's stint windows), in the shape nba/index.html reads:
  {version: 1, updated, windowDays: 60, games: [{gameId, date, away, home, periods, players, subs, events}]}
  - away / home: {tricode (NBA style: NYK, SAS, GSW...), name, teamId, score, stats: official team totals}
  - players: every athlete in the box (a DNP with zeros), 23 fields: name, playerId, team, teamId, position, started,
    MIN, PTS, FGM, FGA, FG3M, FG3A, FTM, FTA, REB, OREB, DREB, AST, STL, BLK, TOV, PF, PLUS_MINUS
  - subs: {period, secInPeriod (seconds ELAPSED in the period), out, outId, in, inId, teamId}; ESPN lists [in, out]
  - events: [period, secInPeriod, code, name, teamId, playerId], codes as the Colab: M2/X2/M3/X3 field goals, MF/XF free
    throws, OR/DR rebounds (credited to a player), AS assist (on the made shot), TO turnover, ST steal, BL block,
    PF personal foul (technical fouls excluded)
Ids are ESPN's; the page only matches ids within one game. Games are the final regular-season and playoff NBA games
(nba_ingest.nba_game: no All-Star, no Cup Championship; no play-in, as the Colab) of the last 60 days, newest first; a game
with under 10 player-minutes is left out.
Usage (data repo root): python3 nba_recaps.py --season 2026-27 --cache .nba-cache --repo . [--today YYYY-MM-DD]"""
import os,re,sys,json,argparse,datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from nba_ingest import summaries,write_json,nba_game

ET=ZoneInfo('America/New_York');WINDOW_DAYS=60
TRI={'NY':'NYK','SA':'SAS','GS':'GSW','NO':'NOP','UTAH':'UTA','WSH':'WAS'}
def tricode(a):return TRI.get(a,a)
def _i(x):
    try:return int(float(x))
    except Exception:return 0
def _pair(x):
    try:a,b=str(x).split('-');return int(a),int(b)
    except Exception:return 0,0
def _min(x):
    if x in (None,'','--'):return 0.0
    if ':' in str(x):m,s=str(x).split(':');return round(float(m)+float(s)/60,2)
    try:return round(float(x),2)
    except Exception:return 0.0
def _elapsed(period,clock):
    try:
        rem=0.0
        for x in str((clock or {}).get('displayValue','0') or '0').split(':'):rem=rem*60+float(x)   # 'M:SS', 'SS.s'
    except Exception:rem=0.0
    return max(0,int((720 if period<=4 else 300)-rem))

def recap(gid,s):
    h=s.get('header') or {};comp=(h.get('competitions') or [{}])[0]
    try:day=datetime.datetime.fromisoformat((comp.get('date') or '').replace('Z','+00:00')).astimezone(ET).date().isoformat()
    except Exception:return None
    bx=(s.get('boxscore') or {}).get('players') or []
    if len(bx)!=2:return None
    side_of={str(c['team']['id']):c for c in comp.get('competitors') or []}
    players=[];team_of={};name_of={}
    for side in bx:
        t=side['team'];tid=_i(t['id']);tri=tricode(t.get('abbreviation',''))
        st=side['statistics'][0];lab=st.get('labels') or []
        for a in st.get('athletes') or []:
            ath=a.get('athlete') or {};pid=str(ath.get('id') or '')
            if not pid:continue
            v=dict(zip(lab,a.get('stats') or []));fgm,fga=_pair(v.get('FG'));tpm,tpa=_pair(v.get('3PT'));ftm,fta=_pair(v.get('FT'))
            team_of[pid]=tid;name_of[pid]=ath.get('displayName') or ''
            players.append({'name':name_of[pid],'playerId':_i(pid),'team':tri,'teamId':tid,'position':(ath.get('position') or {}).get('abbreviation') or '',
                'started':bool(a.get('starter')),'MIN':_min(v.get('MIN')),'PTS':_i(v.get('PTS')),'FGM':fgm,'FGA':fga,'FG3M':tpm,'FG3A':tpa,'FTM':ftm,'FTA':fta,
                'REB':_i(v.get('REB')),'OREB':_i(v.get('OREB')),'DREB':_i(v.get('DREB')),'AST':_i(v.get('AST')),'STL':_i(v.get('STL')),'BLK':_i(v.get('BLK')),
                'TOV':_i(v.get('TO')),'PF':_i(v.get('PF')),'PLUS_MINUS':_i(v.get('+/-'))})
    if sum(p['MIN'] for p in players)<10.0:return None   # pre-tipoff placeholder, as the Colab dropped
    teams={}
    for t in (s.get('boxscore') or {}).get('teams') or []:
        tm=t.get('team') or {};d={x.get('name'):x.get('displayValue') for x in t.get('statistics') or []}
        fgm,fga=_pair(d.get('fieldGoalsMade-fieldGoalsAttempted'));tpm,tpa=_pair(d.get('threePointFieldGoalsMade-threePointFieldGoalsAttempted'));ftm,fta=_pair(d.get('freeThrowsMade-freeThrowsAttempted'))
        c=side_of.get(str(tm.get('id')),{});tid=_i(tm.get('id'))
        teams[t.get('homeAway') or c.get('homeAway')]={'tricode':tricode(tm.get('abbreviation','')),'name':tm.get('displayName') or '','teamId':tid,'score':_i(c.get('score')),
            'stats':{'FGM':fgm,'FGA':fga,'FG3M':tpm,'FG3A':tpa,'FTM':ftm,'FTA':fta,'REB':_i(d.get('totalRebounds')),'OREB':_i(d.get('offensiveRebounds')),'DREB':_i(d.get('defensiveRebounds')),
                'AST':_i(d.get('assists')),'STL':_i(d.get('steals')),'BLK':_i(d.get('blocks')),'TOV':_i(d.get('totalTurnovers') or d.get('turnovers')),'PF':_i(d.get('fouls')),
                'PTS':_i(c.get('score')),'MIN':round(sum(p['MIN'] for p in players if p['teamId']==tid),2)}}
    if set(teams)!={'home','away'}:return None
    subs=[];events=[];periods=4
    def ev(per,sec,code,pid):
        if pid and pid in team_of:events.append([per,sec,code,name_of[pid],team_of[pid],_i(pid)])
    for p in s.get('plays') or []:
        per=int((p.get('period') or {}).get('number') or 0)
        if not per:continue
        periods=max(periods,per);sec=_elapsed(per,p.get('clock'))
        tt=(p.get('type') or {}).get('text') or '';low=(p.get('text') or '').lower()
        parts=[str(x['athlete']['id']) for x in (p.get('participants') or []) if x.get('athlete')]
        if tt=='Substitution':
            if len(parts)>=2 and parts[0] in team_of:
                subs.append({'period':per,'secInPeriod':sec,'out':name_of.get(parts[1],''),'outId':_i(parts[1]),'in':name_of[parts[0]],'inId':_i(parts[0]),'teamId':team_of[parts[0]]})
            continue
        if not parts:continue
        if tt.startswith('Free Throw'):ev(per,sec,'MF' if p.get('scoringPlay') else 'XF',parts[0])
        elif p.get('shootingPlay') and 'Heave' in tt and not p.get('scoringPlay'):   # a missed end-of-period heave is not an attempt in the box; a block on it still counts
            if len(parts)>1 and 'block' in low:ev(per,sec,'BL',parts[1])
        elif p.get('shootingPlay'):
            three=(p.get('pointsAttempted')==3) or (p.get('scoreValue')==3) or ('three point' in low)
            if p.get('scoringPlay'):
                ev(per,sec,'M3' if three else 'M2',parts[0])
                if len(parts)>1 and 'assist' in low:ev(per,sec,'AS',parts[1])
            else:
                ev(per,sec,'X3' if three else 'X2',parts[0])
                if len(parts)>1 and 'block' in low:ev(per,sec,'BL',parts[1])
        elif tt=='Offensive Rebound':ev(per,sec,'OR',parts[0])
        elif tt=='Defensive Rebound':ev(per,sec,'DR',parts[0])
        elif 'urnover' in tt or tt=='Traveling':   # ESPN's 'Traveling' plays say neither 'turnover' nor carry a separate turnover play
            ev(per,sec,'TO',parts[0])
            if len(parts)>1 and 'steal' in low:ev(per,sec,'ST',parts[1])
        elif ('Foul' in tt or tt=='Offensive Charge') and 'Technical' not in tt:ev(per,sec,'PF',parts[0])
    return {'gameId':str(gid),'date':day,'away':teams['away'],'home':teams['home'],'periods':periods,'players':players,'subs':subs,'events':events}

def build(cache,season,today=None,now=None,types=('2','3')):
    today=today or datetime.datetime.now(ET).date();now=now or datetime.datetime.now(datetime.timezone.utc)
    cutoff=(today-datetime.timedelta(days=WINDOW_DAYS)).isoformat();games=[];left_out=[]
    for gid,s in summaries(cache,season):
        if str((((s.get('header') or {}).get('season') or {}).get('type'))) not in types or not nba_game(s):continue   # regular season + playoffs, as the Colab
        comp=((s.get('header') or {}).get('competitions') or [{}])[0]
        if (comp.get('date') or '')[:10]<cutoff:continue   # cheap pre-filter on the UTC date; exact cutoff below
        g=recap(gid,s)
        if g is None:left_out.append(gid);continue
        if cutoff<=g['date']<=today.isoformat():games.append(g)
    games.sort(key=lambda g:(g['date'],g['gameId']),reverse=True)
    return {'version':1,'updated':now.isoformat(),'windowDays':WINDOW_DAYS,'games':games},left_out

def main(argv=None):
    ap=argparse.ArgumentParser(description='NBA game recaps (last 60 days) from ESPN')
    ap.add_argument('--season',required=True);ap.add_argument('--cache',default='.nba-cache');ap.add_argument('--repo',default='.')
    ap.add_argument('--today',help='YYYY-MM-DD (ET); default today')
    ap.add_argument('--types',default='2,3',help='ESPN season types: 2 regular, 3 playoffs (the Colab); 1 preseason, 5 play-in')
    a=ap.parse_args(argv)
    out,left=build(a.cache,a.season,datetime.date.fromisoformat(a.today) if a.today else None,types=tuple(x.strip() for x in a.types.split(',') if x.strip()))
    if not out['games']:print(json.dumps({'games':0,'note':'no final NBA game in the last %d days; recaps.json left as is'%WINDOW_DAYS}));return 0
    write_json(os.path.join(a.repo,'recaps.json'),out)
    print(json.dumps({'games':len(out['games']),'leftOut':left}));return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print('NBA recaps FAILED:',type(e).__name__,str(e)[:300],file=sys.stderr);sys.exit(1)
