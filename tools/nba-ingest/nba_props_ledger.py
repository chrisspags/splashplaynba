# Vale, Wed 10/14/2026. THE PRIVATE PROPS LEDGER (Chris 10/14: "do not show season long results. save them in the back end for me to
# review but do not publish to users unless we know they are good. this remains a SALES business").
# For one date: the rows the members' props page showed (rebuilt exactly as the page builds them: main lines = exactly two items of the
# same type and line for a player and stat, first listed = over; milestones = whole-number thresholds listed once), our number = the
# step's lineRaw (never the props-informed line), the page's four numbers from page_math, and what happened (the job's box scores).
#   NBA_ROOT=<job files> python3 nba_props_ledger.py <date> <props pull: the latest nba-record/<date>/props-<UTC>.json before the first tip>
#        <step file: nba-proj/<date>.json> [--out <dir>] [--actuals <json: {espnId: {MP, PTS, TRB, AST, TPM, STL, BLK}}> (tests only)]
# Writes <out>/props-ledger-<date>.json (PRIVATE: Chris only, never members) and prints the night's summary.
# Grading: over if the stat is above the line, under if below, push if equal (stake back); a player who did not play = void (DraftKings
# voids the bet). A side is "taken" when its EV (or AdjEV) is above 0; it returns the price's payout on a win, -1 on a loss, 0 on push/void.
import sys,os,json,math,argparse,collections
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE)); import page_math as PM
TOTALS={"Total Points":("points",("PTS",)),"Total Rebounds":("rebounds",("TRB",)),"Total Assists":("assists",("AST",)),
        "Total 3-Point Field Goals":("threes",("TPM",)),"Total Points, Rebounds, and Assists":("pts+reb+ast",("PTS","TRB","AST")),
        "Total Points and Assists":("pts+ast",("PTS","AST")),"Total Points and Rebounds":("pts+reb",("PTS","TRB")),
        "Total Assists and Rebounds":("ast+reb",("AST","TRB")),"Total Steals":("steals",("STL",)),"Total Blocks":("blocks",("BLK",)),
        "Total Steals and Blocks":("stl+blk",("STL","BLK"))}
MILES={"Points Milestones":"points","Rebounds Milestones":"rebounds","Assists Milestones":"assists","3-Point Field Goals Milestones":"threes",
       "Points + Assists + Rebounds Milestones":"pts+reb+ast","Points + Assists Milestones":"pts+ast","Points + Rebounds Milestones":"pts+reb"}
KEYS={v[0]:v[1] for v in TOTALS.values()}
def num(x):
    try: v=float(str(x).strip()); return v if math.isfinite(v) else None
    except (TypeError,ValueError): return None
def athlete(it):
    try: return it["athlete"]["$ref"].split("/athletes/")[1].split("?")[0].split("/")[0]
    except (KeyError,TypeError,IndexError,AttributeError): return None
def line_of(it):
    a=num(((it.get("current") or {}).get("target") or {}).get("value")); b=num(((it.get("odds") or {}).get("total") or {}).get("value"))
    return None if (a is not None and b is not None and a!=b) else (a if a is not None else b)
def page_rows(pull):
    """main lines and milestones exactly as the members' page builds them (Mara's loader 89e51217, equal on the real 10/20 pull)"""
    G=collections.defaultdict(list)
    for ev,resp in (pull or {}).items():
        for it in (resp or {}).get("items") or []:
            a=athlete(it); lab=(it.get("type") or {}).get("name"); tid=(it.get("type") or {}).get("id")
            if not a or not isinstance(lab,str): continue
            stat=TOTALS[lab][0] if lab in TOTALS else MILES.get(lab)
            if not stat: continue
            G[(a,stat)].append(dict(label=lab,tid=str(tid),line=line_of(it),price=num(((it.get("odds") or {}).get("american") or {}).get("value")),event=ev))
    mains,miles=[],[]
    for (a,stat),items in G.items():
        L=[x for x in items if x["label"] in TOTALS]
        if len(L)==2 and L[0]["tid"]==L[1]["tid"] and L[0]["line"] is not None and L[0]["line"]>=0 and L[0]["line"]==L[1]["line"]:
            mains.append(dict(athlete=a,stat=stat,line=L[0]["line"],over=L[0]["price"],under=L[1]["price"],event=L[0]["event"]))
        cnt=collections.Counter((x["label"],x["line"]) for x in items if x["label"] in MILES)
        for x in items:
            if x["label"] in MILES and x["line"] is not None and float(x["line"]).is_integer() and x["line"]>=1 and cnt[(x["label"],x["line"])]==1:
                miles.append(dict(athlete=a,stat=stat,threshold=x["line"],price=x["price"],event=x["event"]))
    return mains,miles
def actuals_from_box(date):
    import nbalib
    y=int(date[:4]); s=("%d-%02d"%(y,(y+1)%100)) if int(date[5:7])>=8 else ("%d-%02d"%(y-1,y%100))
    if s not in nbalib.SEASONS: nbalib.SEASONS=tuple(nbalib.SEASONS)+(s,)
    A={}
    for g in nbalib.load()["games"]:
        if g["date"]!=date: continue
        for t in g["teams"]:
            for r in t["rows"]: A[str(r["id"])]={k:float(r.get(k) or 0) for k in ("MP","PTS","TRB","AST","TPM","STL","BLK")}
    return A
def settle(side,price,result):
    if side is None: return None
    if result in ("push","void"): return 0.0
    return PM.pays(price) if result==side else -1.0
def ledger(date,pull,step,A):
    ours={}
    for r in step.get("rows",[]):
        lr=r.get("lineRaw")
        if r.get("espnId") and isinstance(lr,dict): ours[str(r["espnId"])]=(r.get("name"),lr)
    mains,miles=page_rows(pull); out=[]; mk_num={}
    for m in mains:
        nm,lr=ours.get(m["athlete"],(None,None))
        if lr is None or m["over"] is None or m["under"] is None or any(k not in lr for k in KEYS[m["stat"]]): continue
        o=sum(lr[k] for k in KEYS[m["stat"]]); R=PM.row(m["stat"],m["line"],m["over"],m["under"],o); mk_num[(m["athlete"],m["stat"])]=R["marketsNumber"]
        a=A.get(m["athlete"]); played=bool(a and a["MP"]>0); v=sum(a[k] for k in KEYS[m["stat"]]) if played else None
        res="void" if not played else ("over" if v>m["line"] else "under" if v<m["line"] else "push")
        ev=max((("over",R["evOver"]),("under",R["evUnder"])),key=lambda x:x[1]); adj=max((("over",R["adjEvOver"]),("under",R["adjEvUnder"])),key=lambda x:x[1])
        evs=ev[0] if ev[1]>0 else None; adjs=adj[0] if adj[1]>0 else None
        out.append(dict(kind="line",athlete=m["athlete"],name=nm,stat=m["stat"],line=m["line"],over=m["over"],under=m["under"],ours=round(o,2),
            withProps=R["withProps"],evOver=R["evOver"],evUnder=R["evUnder"],adjEvOver=R["adjEvOver"],adjEvUnder=R["adjEvUnder"],actual=v,result=res,
            evSide=evs,evUnits=settle(evs,m["over"] if evs=="over" else m["under"],res),adjSide=adjs,adjUnits=settle(adjs,m["over"] if adjs=="over" else m["under"],res),
            blindOverUnits=settle("over",m["over"],res),blindUnderUnits=settle("under",m["under"],res)))
    for m in miles:
        nm,lr=ours.get(m["athlete"],(None,None))
        if lr is None or m["price"] is None or any(k not in lr for k in KEYS[m["stat"]]): continue
        o=sum(lr[k] for k in KEYS[m["stat"]]); R=PM.milestone_row(m["stat"],m["threshold"],m["price"],o,mk_num.get((m["athlete"],m["stat"])))
        a=A.get(m["athlete"]); played=bool(a and a["MP"]>0); v=sum(a[k] for k in KEYS[m["stat"]]) if played else None
        res="void" if not played else ("yes" if v>=m["threshold"] else "no")
        evs="yes" if R["ev"]>0 else None; adjs="yes" if (R["adjEv"] is not None and R["adjEv"]>0) else None
        out.append(dict(kind="milestone",athlete=m["athlete"],name=nm,stat=m["stat"],threshold=m["threshold"],price=m["price"],ours=round(o,2),
            ev=R["ev"],adjEv=R["adjEv"],actual=v,result=res,evSide=evs,evUnits=settle(evs,m["price"],res),adjSide=adjs,adjUnits=settle(adjs,m["price"],res),
            blindUnits=settle("yes",m["price"],res)))
    return out
def summary(rows):
    def s(key,kind=None):
        u=[r[key] for r in rows if r.get(key) is not None and (kind is None or r["kind"]==kind) and r["result"]!="void"]
        if not u: return dict(n=0)
        m=sum(u)/len(u); se=math.sqrt(sum((x-m)**2 for x in u)/(len(u)-1))/math.sqrt(len(u)) if len(u)>1 else float("nan")
        return dict(n=len(u),units=round(sum(u),2),roi=round(100*m,1),se=round(100*se,1))
    return {"lines":dict(shown=sum(1 for r in rows if r["kind"]=="line"),graded=sum(1 for r in rows if r["kind"]=="line" and r["result"]!="void"),
                         evSide=s("evUnits","line"),adjEvSide=s("adjUnits","line"),blindOver=s("blindOverUnits"),blindUnder=s("blindUnderUnits")),
            "milestones":dict(shown=sum(1 for r in rows if r["kind"]=="milestone"),evSide=s("evUnits","milestone"),adjEvSide=s("adjUnits","milestone"),blindYes=s("blindUnits"))}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("date"); ap.add_argument("pull"); ap.add_argument("step"); ap.add_argument("--out"); ap.add_argument("--actuals")
    x=ap.parse_args(); pull=json.load(open(x.pull)); step=json.load(open(x.step))
    A=json.load(open(x.actuals)) if x.actuals else actuals_from_box(x.date)
    rows=ledger(x.date,pull,step,A); S=summary(rows)
    if x.out: os.makedirs(x.out,exist_ok=True); json.dump(dict(format="nba-props-ledger-v1",private=True,date=x.date,pull=os.path.basename(x.pull),summary=S,rows=rows),open(os.path.join(x.out,"props-ledger-%s.json"%x.date),"w"),indent=0)
    print(json.dumps(S,indent=1))
