# Vale, Wed 10/14/2026. THE PRIVATE NEXT-MORNING CARD for one NBA slate date: DAILY-RECORD.md items 9 to 13 and the validator, read only
# from what the night saved and what happened. It runs in the PRIVATE workflow (Monty's review of 8caf95e) and is written under the private
# repo's nba-record/<date>/, so Chris never presses anything for it.
#   NBA_ROOT=<job files> python3 nba_card.py <date> <data repo dir> [--out <dir>] [--step <folder holding nbalib.py>] [--private] [--entries <file>]
# Reads: nba-record/<date>/<draft group>/build-*.json and sim-*.json; nba-record/<date>/injury-*.json, lineups-*.json, news-*.json;
# nba-proj/<date>.json (the step's file); dk-results/<date>.json (contest results, v2 or v3); the job's game files through nbalib (box scores).
# --private adds his money for every contest he entered: from the --entries file (the private repo's nba-my-lineups/results/<date>.json,
# {contests:[...]}) first, else the results' myLineups (v3), else the v2 leaderboard's user entries. Without --private no entry is read at all.
# Writes: <out>/card-<date>.json and prints the card as text, never into the data repo. Rules: a missing input is a validator line, never a crash;
# a slate player whose team played and who has no box line played 0 and scored 0 (DraftKings); "last build before lock" = the latest build before the first tip.
import sys,os,json,glob,re,math,argparse,datetime,unicodedata,collections
HERE=os.path.dirname(os.path.abspath(__file__))
PRIVATE=False   # --private: add his money (the private workflow passes it)
ENTRIES=None    # --entries: his private entries file {contests:[...]}; read only with --private
nk=lambda s:re.sub(r"[^a-z]","",unicodedata.normalize("NFD",s or "").lower().replace(" jr.","").replace(" jr","").replace(" iii","").replace(" ii",""))
E2D={"NY":"NYK","SA":"SAS","GS":"GSW","NO":"NOP","UTAH":"UTA","WSH":"WAS","PHO":"PHX","BKN":"BKN"}
STAMP=re.compile(r"-(\d{8}T\d{6}Z)\.json$")
def stamp_of(p):
    m=STAMP.search(p); return datetime.datetime.strptime(m.group(1),"%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc) if m else None
def iso(s):
    if not s: return None
    s=str(s).replace("Z","+00:00")
    try: d=datetime.datetime.fromisoformat(s)
    except ValueError: return None
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)
def season_of(date): y,m=int(date[:4]),int(date[5:7]); a=y if m>=8 else y-1; return "%d-%02d"%(a,(a+1)%100)
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def stats(errs):
    e=[x for x in errs if x is not None and math.isfinite(x)]
    if not e: return dict(n=0)
    n=len(e); m=sum(e)/n; return dict(n=n,mean=round(m,2),mae=round(sum(abs(x) for x in e)/n,2),rmse=round(math.sqrt(sum(x*x for x in e)/n),2))
def mins_to(a,b): return None if (a is None or b is None) else round((b-a).total_seconds()/60.0,1)

def box_for(date,step_dir):
    """{espnId: dict(min, dk, starter, name, team, out)} and the set of teams that played, from the job's game files (nbalib)."""
    if step_dir and step_dir not in sys.path: sys.path.insert(0,step_dir)
    try: import nbalib
    except Exception as e: return None,set(),"box scores: nbalib not importable ("+str(e)+")"
    s=season_of(date)
    if s not in nbalib.SEASONS: nbalib.SEASONS=tuple(nbalib.SEASONS)+(s,)        # as the projection step does: this season's game files too
    try: D=nbalib.load()
    except Exception as e: return None,set(),"box scores: the game files did not load ("+str(e)+")"
    B={}; teams=set(); n=0
    for g in D["games"]:
        if g["date"]!=date: continue
        n+=1
        for t in g["teams"]:
            ab=E2D.get(t["abbr"],t["abbr"]); teams.add(ab)
            for r in t["rows"]:
                B[str(r["id"])]=dict(min=float(r["MP"] or 0),dk=float(nbalib.dkpts(r)) if r["MP"]>0 else 0.0,starter=bool(r["starter"]),name=r.get("name"),team=ab,out=bool(r.get("out")))
    return B,teams,("box scores: %d games, %d player lines"%(n,len(B)) if n else "box scores: no game of this date in the game files yet")

def walk(o):
    if isinstance(o,dict):
        yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o,list):
        for v in o: yield from walk(v)
def first_seen(files,name,statuses=("Out","Doubtful")):
    """earliest file (by its stamp) in which a record names this player with one of these statuses; for news files, any mention"""
    key=nk(name)
    for p in sorted(files,key=lambda p:stamp_of(p) or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)):
        d=load(p)
        if d is None: continue
        if "/news-" in p:
            if key and key in nk(json.dumps(d)): return stamp_of(p)
            continue
        for x in walk(d):
            nm=x.get("name") or x.get("listed") or x.get("player")
            if isinstance(nm,str) and nk(nm)==key and str(x.get("status","")) in statuses: return stamp_of(p)
    return None

def dig(o,*ks):
    for k in ks: o=o.get(k) if isinstance(o,dict) else None
    return o
def num(x):
    try: v=float(x)
    except (TypeError,ValueError): return None
    return v if math.isfinite(v) else None
def is_sd(c): return bool(dig(c,"isShowdown") or dig(c,"summary","isShowdown"))     # v3 sets it at the contest's top level, v2 in summary
def listed(x): return list(x.values()) if isinstance(x,dict) else list(x) if isinstance(x,list) else []
def won_of(e):
    """one entry's winnings: DK's v2 rows carry [{value}], the v3 ingest (extractMyEntries) writes a number"""
    w=dig(e,"winnings")
    if isinstance(w,dict): w=[w]
    if isinstance(w,list): return sum(num(dig(x,"value") if isinstance(x,dict) else x) or 0.0 for x in w)
    return num(w) or 0.0
def money(date,res,entries):
    """his money for every contest of the date he entered, Classic and Showdown: (money or {}, validator text). Only with --private."""
    ef=load(entries) if entries else None; looked=[]
    if not entries: looked.append("no --entries file given")
    elif not isinstance(dig(ef,"contests"),list): looked.append("--entries %s (missing or not {contests:[...]})"%entries); ef=None
    elif dig(ef,"date") not in (None,date): looked.append("--entries %s (its date is %s: not used)"%(entries,ef["date"])); ef=None
    else: looked.append("--entries "+entries)
    ec=[x for x in dig(ef,"contests") or [] if isinstance(x,dict)]; used=set()
    def match(c):                                   # by contestId, else by draft group + contest name; each file contest used once
        for same in (lambda x:x.get("contestId") is not None and str(x.get("contestId"))==str(c.get("contestId")),
                     lambda x:str(x.get("draftGroupId"))==str(c.get("draftGroupId")) and x.get("contestName")==c.get("contestName")):
            for i,x in enumerate(ec):
                if i not in used and same(x): used.add(i); return x
        return None
    rc=[c for c in listed(dig(res,"contests")) if isinstance(c,dict)]
    looked+=["dk-results/%s.json myLineups"%date,"dk-results/%s.json v2 leaderboard"%date] if res else ["dk-results/%s.json (missing)"%date]
    got=[]
    for c in rc:
        x=match(c)
        for src,es in (("entries file",listed(dig(x,"myLineups"))),("results myLineups",listed(c.get("myLineups"))),
                       ("results v2 leaderboard",listed(dig(c,"leaderboard","leaderBoardUserEntries","entryByEntryKey")))):
            if es: got.append((c,x,src,es)); break
    got+=[(x,x,"entries file",listed(x.get("myLineups"))) for i,x in enumerate(ec) if i not in used and listed(x.get("myLineups"))]
    M=[]; nofee=0
    for c,x,src,es in got:
        fee=next((f for f in (num(dig(c,"contestDetail","entryFee")),num(c.get("entryFee")),num(dig(x,"contestDetail","entryFee")),num(dig(x,"entryFee"))) if f is not None),None)
        if fee is None: nofee+=1; fee=0.0
        n=len(es); won=sum(won_of(e) for e in es)
        M.append(dict(contest=c.get("contestName"),contestId=c.get("contestId"),draftGroupId=c.get("draftGroupId"),showdown=is_sd(c),
                      entries=n,fees=round(fee*n,2),won=round(won,2),net=round(won-fee*n,2),source=src))
    if not M: return {},"no entries found (looked in: %s)"%", ".join(looked)
    T=dict(entries=sum(m["entries"] for m in M),**{k:round(sum(m[k] for m in M),2) for k in ("fees","won","net")})
    return dict(contests=M,total=T),"%d entries in %d contests (%s)"%(T["entries"],len(M),", ".join("%s %d"%kv for kv in collections.Counter(m["source"] for m in M).items()))+(
        "; NO entry fee found for %d contest(s), counted as $0"%nofee if nofee else "")

def card(date,repo,step_dir):
    rec=os.path.join(repo,"nba-record",date); C=dict(date=date,format="nba-card-v1",made=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        validator=[],groups={},minutes={},points={},sat=[],surprise=[],news=[],ownership={},money={},rules={})
    V=C["validator"]
    dgs=sorted(d for d in os.listdir(rec) if os.path.isdir(os.path.join(rec,d))) if os.path.isdir(rec) else []
    injury=glob.glob(os.path.join(rec,"injury-*.json")); lineups=glob.glob(os.path.join(rec,"lineups-*.json")); news=glob.glob(os.path.join(rec,"news-*.json"))
    step=load(os.path.join(repo,"nba-proj",date+".json")); res=load(os.path.join(repo,"dk-results",date+".json"))
    B,played,boxnote=box_for(date,step_dir)
    if not dgs: V.append("1 build record: MISSING (no nba-record/%s/<draft group>/ folder)"%date)
    primary=None
    for dg in dgs:
        builds=sorted(glob.glob(os.path.join(rec,dg,"build-*.json")),key=stamp_of); sims=sorted(glob.glob(os.path.join(rec,dg,"sim-*.json")),key=stamp_of)
        if not builds: V.append("1 build record, draft group %s: MISSING"%dg); continue
        last_any=load(builds[-1]) or {}
        tips=[iso(g.get("tip")) for g in last_any.get("games",[]) if iso(g.get("tip"))]; tip=min(tips) if tips else None
        pre=[p for p in builds if tip is None or (stamp_of(p) and stamp_of(p)<tip)]; lock=pre[-1] if pre else None
        simpre=[p for p in sims if tip is None or (stamp_of(p) and stamp_of(p)<tip)]
        G=dict(builds=len(builds),buildsAfterFirstTip=len(builds)-len(pre),firstTip=tip.strftime("%Y-%m-%dT%H:%MZ") if tip else None,
               lastBuildBeforeLock=os.path.basename(lock) if lock else None,minutesBeforeTip=mins_to(stamp_of(lock),tip) if lock else None,
               sims=len(sims),lastSimBeforeLock=os.path.basename(simpre[-1]) if simpre else None,games=len(last_any.get("games",[])))
        C["groups"][dg]=G
        V.append("1 build record, draft group %s: %s"%(dg,("ok, "+G["lastBuildBeforeLock"]+", %s min before the first tip"%G["minutesBeforeTip"]) if lock else "MISSING before the first tip (%d after it)"%len(builds)))
        V.append("5 sim record, draft group %s: %s"%(dg,("ok, "+G["lastSimBeforeLock"]) if simpre else "MISSING before the first tip"))
        if lock and (primary is None or G["games"]>C["groups"][primary[0]]["games"]): primary=(dg,lock,simpre[-1] if simpre else None,tip)
    V.append("2 typed edits: "+("in the build record" if primary and "typed" in (load(primary[1]) or {}) else "not seen"))
    V.append("3 news: injury reads %d, lineups reads %d, page news checks %d"%(len(injury),len(lineups),len(news))+("" if injury else " (NO injury read saved)"))
    V.append("7 "+boxnote)
    V.append("8 results: "+("dk-results/%s.json, %d contests"%(date,len(res.get("contests",[]))) if res else "MISSING (no dk-results/%s.json)"%date))
    if PRIVATE: C["money"],book=money(date,res,ENTRIES)
    else: book="private (nba-my-lineups/); this card does not read it"+(" (--entries ignored without --private)" if ENTRIES else "")
    V.append("6 book: "+book)
    if not primary: return C
    dg,lockp,simp,tip=primary; b=load(lockp); s=load(simp) if simp else None; lock_at=stamp_of(lockp)
    C["primary"]=dict(draftGroup=dg,build=os.path.basename(lockp),sim=os.path.basename(simp) if simp else None)
    stok=sum(1 for r in b["rows"] if r.get("stokMin") is not None)
    V.append("4 Stokastic at the last build: %d of %d players with Stokastic minutes"%(stok,len(b["rows"])))
    if B is None or not played: return C
    rows=[r for r in b["rows"] if r.get("team") in played]
    E=collections.defaultdict(list); P=collections.defaultdict(list); join=0
    for r in rows:
        a=B.get(str(r.get("espnId")))
        am=a["min"] if a else 0.0; ad=a["dk"] if a else 0.0; join+=a is not None
        pm=float(r.get("min") or 0); proj=float(r.get("proj") or 0)
        if pm<=0 and am<=0: continue
        st="starters" if (a and a["starter"]) else "bench"
        E["all"].append(pm-am); E[st].append(pm-am)
        if r.get("modelMin") is not None: E["model (before typed and Stokastic)"].append(float(r["modelMin"])-am)
        if r.get("stokMin") is not None: E["Stokastic"].append(float(r["stokMin"])-am); E["ours where Stokastic exists"].append(pm-am)
        if r.get("typed") and r.get("typedMin") is not None: E["typed by Chris"].append(float(r["typedMin"])-am)
        sal=r.get("salary") or 0; tier="$8,000+" if sal>=8000 else "$6,000-7,900" if sal>=6000 else "$4,000-5,900" if sal>=4000 else "under $4,000"
        P["all"].append(proj-ad); P[tier].append(proj-ad)
        if pm>=10 and am==0:
            fs=first_seen(injury,r["name"]); fn=first_seen(news,r["name"])
            C["sat"].append(dict(name=r["name"],team=r["team"],projMin=pm,proj=proj,buildKnew=bool(r.get("leagueStatus") or r.get("whyOut")),
                firstInjuryReport=fs.strftime("%H:%MZ") if fs else None,reportBeforeBuild=bool(fs and lock_at and fs<lock_at),minutesReportBeforeTip=mins_to(fs,tip),
                firstNewsCheck=fn.strftime("%H:%MZ") if fn else None))
        if pm<10 and am>=20: C["surprise"].append(dict(name=r["name"],team=r["team"],projMin=pm,min=am,dk=round(ad,2)))
    C["minutes"]={k:stats(v) for k,v in E.items()}; C["points"]={k:stats(v) for k,v in P.items()}
    C["joined"]=dict(buildRowsOnTeamsThatPlayed=len(rows),withBoxLine=join)
    miss=sorted(((float(r.get("proj") or 0)-(B.get(str(r.get("espnId")),{}).get("dk",0.0)),r["name"]) for r in rows if float(r.get("proj") or 0)>0),key=lambda x:x[0])
    C["biggestUnder"]=[dict(name=n,miss=round(m,2)) for m,n in miss if m<0][:5]; C["biggestOver"]=[dict(name=n,miss=round(m,2)) for m,n in miss[::-1] if m>0][:5]
    # ownership: our forecast (sim record) against the field in the largest Classic contest of this draft group
    if res and s:
        cl=[c for c in res.get("contests",[]) if str(c.get("draftGroupId"))==str(dg) and not is_sd(c)]
        if cl:
            c=max(cl,key=lambda c:c["summary"].get("totalEntries",0)); fo={nk(k):v for k,v in (c["summary"].get("fieldOwnership") or {}).items()}
            pairs=[(float(p["own"]),float(fo.get(nk(p["name"]),0.0)),p["name"],p.get("ownSource","")) for p in s.get("players",[]) if p.get("own") is not None]
            C["ownership"]=dict(contest=c.get("contestName"),entries=c["summary"].get("totalEntries"),**stats([a-b for a,b,_,_ in pairs]),
                sources=dict(collections.Counter(src.split(" (")[0] for *_,src in pairs)),
                over=[dict(name=n,ours=a,field=b) for a,b,n,_ in sorted(pairs,key=lambda x:x[1]-x[0]) if a>b][:5],
                under=[dict(name=n,ours=a,field=b) for a,b,n,_ in sorted(pairs,key=lambda x:x[0]-x[1]) if a<b][:5])
    # what each live NBA rule did tonight (RULES-AND-WAY-OUTS.md, N1 to N6)
    R=C["rules"]
    if step:
        pr=[r for r in step.get("rows",[]) if r.get("props")]
        R["N1 props"]=dict(players=len(pr),meanMove=round(sum(float(r.get("proj",0))-float(r.get("projRaw",r.get("proj",0))) for r in pr)/len(pr),2) if pr else 0.0)
        R["N2 tripwire"]=[n for n in step.get("notes",[]) if "price order" in str(n).lower() or "tripwire" in str(n).lower()][:2]
    R["N4 Stokastic blend"]=dict(blended=sum(1 for r in b["rows"] if r.get("blended")),typed=sum(1 for r in b["rows"] if r.get("typed")))
    if s:
        k=((s.get("run") or {}).get("kernel") or {}); R["N3 minutes seam"]=k.get("stepMinutes"); R["N6 late swap"]=(s.get("run") or {}).get("lateSwap")
        R["sim"]=dict(worlds=(s.get("run") or {}).get("worlds"),settlementOk=((s.get("run") or {}).get("settlementCheck") or {}).get("ok"))
    return C

def text(C):
    L=["NBA card %s (made %s)"%(C["date"],C["made"])]
    for v in C["validator"]: L.append("  "+v)
    if C.get("primary"): L.append("Read from draft group %s, %s"%(C["primary"]["draftGroup"],C["primary"]["build"]))
    def row(title,d):
        for k,v in d.items():
            if v.get("n"): L.append("  %-38s n %3d  mean %+6.2f  MAE %5.2f  RMSE %5.2f"%(k,v["n"],v["mean"],v["mae"],v["rmse"]))
    if C["minutes"]: L.append("MINUTES (projected minus played; + = we had him too high)"); row("m",C["minutes"])
    if C["points"]: L.append("DK POINTS (projected minus scored)"); row("p",C["points"])
    if C["sat"]:
        L.append("SAT (we had 10+ minutes, he played 0):")
        for x in C["sat"]:
            rep=("report first %s, %s min before tip, %s the last build"%(x["firstInjuryReport"],x["minutesReportBeforeTip"],"before" if x["reportBeforeBuild"] else "after")) if x["firstInjuryReport"] else "in no saved report"
            L.append("  %s %s %.1f min: build knew %s; %s; page news %s"%(x["name"],x["team"],x["projMin"],"yes" if x["buildKnew"] else "NO",rep,x["firstNewsCheck"] or "none"))
    if C["surprise"]: L.append("PLAYED 20+ WHERE WE HAD UNDER 10: "+", ".join("%s %.0f->%.0f"%(x["name"],x["projMin"],x["min"]) for x in C["surprise"]))
    if C.get("biggestUnder"): L.append("UNDER (scored most above us): "+", ".join("%s %+.1f"%(x["name"],x["miss"]) for x in C["biggestUnder"]))
    if C.get("biggestOver"): L.append("OVER (scored most below us): "+", ".join("%s %+.1f"%(x["name"],x["miss"]) for x in C["biggestOver"]))
    o=C["ownership"]
    if o.get("n"): L.append("OWNERSHIP vs %s (%s entries): ours minus field, mean %+.1f, MAE %.1f; sources %s"%(o["contest"],o["entries"],o["mean"],o["mae"],o["sources"]))
    if C["money"]:
        t=C["money"]["total"]; L.append("MONEY: %d entries, fees $%.2f, won $%.2f, net $%+.2f"%(t["entries"],t["fees"],t["won"],t["net"]))
        for m in C["money"]["contests"]: L.append("  %s%s: %d entries, fees $%.2f, won $%.2f, net $%+.2f (%s)"%(m["contest"]," [Showdown]" if m["showdown"] else "",m["entries"],m["fees"],m["won"],m["net"],m["source"]))
    if C["rules"]: L.append("RULES TONIGHT: "+json.dumps(C["rules"]))
    return "\n".join(L)

if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("date"); a.add_argument("repo"); a.add_argument("--out",default=None)
    a.add_argument("--step",default=os.environ.get("NBA_STEP") or os.path.expanduser("~/Documents/Mara/nba-model-vale-2026-10-11/projection-step/step"))
    a.add_argument("--private",action="store_true",help="include his money; ONLY for a copy written to the private repo")
    a.add_argument("--entries",default=None,help="his private entries file {contests:[...]} (nba-my-lineups/results/<date>.json); read only with --private")
    x=a.parse_args(); PRIVATE=x.private; ENTRIES=x.entries; C=card(x.date,x.repo,x.step)
    if x.out: os.makedirs(x.out,exist_ok=True); json.dump(C,open(os.path.join(x.out,"card-%s.json"%x.date),"w"),indent=1)
    print(text(C))
