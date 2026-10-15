#!/usr/bin/env python3
"""THE PRIVATE PROPS LEDGER AND THE MORNING CARD, run by the PRIVATE data repo's own workflow (workflow/nba-private-record.yml), never by
the public job. The card (nba_card_step.py, Vale's nba_card.py) runs after the ledger, each soft on its own; see that file.
Chris 10/14: "do not show season long results. save them in the back end for me to review but do not publish to users unless we know
they are good. this remains a SALES business". Vale's nba_props_ledger.py runs unchanged (one copy of the math), graded on the box
scores the public job read. The public job cannot write it: its only token writes the public repo (Shep, Vale, Monty 10/14).

For yesterday (US Eastern, from the run's own time), from the public data repo checked out read-only after this press's ingest run:
  the props pull: the latest nba-public-record/<date>/props-<UTC>.json stamped before the slate's first tip (the earliest tip in
    nba-tonight/<date>.json) that is a pull ({event id: {count, pageCount, items}}), not the job's "not used" answer;
  our numbers: nba-proj/<date>.json, the step's file.
It unpacks nbalib from the reviewed bundle (the ledger reads the box scores through it) and runs
    NBA_ROOT=<public> NBA_CACHE=<scratch> PYTHONPATH=<unpacked step> python3 nba_props_ledger.py <date> <pull> <step file> --out <scratch>
then keeps props-ledger-<date>.json as <private>/nba-my-lineups/<date>/props-ledger.json, beside his exported lineups (an admin-only
path). A later press replaces it: the same lines, graded on fuller box scores. A ledger with no graded line (no box score of that date
in the job's files yet, so every line would read "void") is not written.

It prints one JSON line saying what each did ({"ledger": ..., "card": ...}) and lists what it wrote in the outputs file the workflow
commits from (nothing when nothing was written). It exits 0 when there is nothing to grade: a quiet day is not a failure.

Usage: python3 nba_private_record.py --public <public checkout> --private <private checkout> --cache <scratch> --outputs-file <file>
                                     [--date YYYY-MM-DD]"""
import json,os,re,shutil,subprocess,sys,time,argparse,datetime
from zoneinfo import ZoneInfo
HERE=os.path.dirname(os.path.abspath(__file__))

ET=ZoneInfo('America/New_York')
TIMEOUT=900
STAMP=re.compile(r'^props-(\d{8}T\d{6}Z)\.json$')
OUT='nba-my-lineups/{}/props-ledger.json'

def yesterday(now=None):
    d=(now or datetime.datetime.now(datetime.timezone.utc)).astimezone(ET).date()
    return (d-datetime.timedelta(days=1)).isoformat()

def utc(s):
    try:d=datetime.datetime.fromisoformat(str(s).replace('Z','+00:00'))
    except (TypeError,ValueError):return None
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)

def first_tip(public,date):
    """The slate's first tip from the tonight file the public job wrote for the date, or None."""
    try:games=json.load(open(os.path.join(public,'nba-tonight',date+'.json'))).get('games') or []
    except Exception:return None
    tips=[utc(g.get('tip')) for g in games if isinstance(g,dict) and g.get('nbaGame') and g.get('tip')]
    tips=[x for x in tips if x]
    return min(tips) if tips else None

def is_pull(d):
    return isinstance(d,dict) and d.get('used') is not False and any(isinstance(v,dict) and isinstance(v.get('items'),list) for v in d.values())

def pick_pull(public,date,tip):
    """(path, its stamp) of the latest pull stamped before the first tip, or (None, why)."""
    folder=os.path.join(public,'nba-public-record',date)
    try:names=sorted(os.listdir(folder))
    except Exception:return None,'no nba-public-record/%s folder'%date
    pre=[]
    for n in names:
        m=STAMP.match(n)
        if not m:continue
        at=datetime.datetime.strptime(m.group(1),'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc)
        if at<tip:pre.append((at,n))
    for at,n in sorted(pre,reverse=True):
        try:d=json.load(open(os.path.join(folder,n)))
        except Exception:continue
        if is_pull(d):return os.path.join(folder,n),n
    return None,'no props pull saved before the first tip (%s)'%tip.strftime('%Y-%m-%dT%H:%MZ')

def ledger(date,public,private,cache,bundle,python=sys.executable,run=subprocess.run):
    """Grades one date. Returns (what it did, the private path it wrote or None)."""
    t0=time.time();e={'date':date,'wrote':False}
    step=os.path.join(public,'nba-proj',date+'.json')
    if not os.path.exists(step):e['why']='no nba-proj/%s.json: no slate was projected that day'%date;return e,None
    tip=first_tip(public,date)
    if tip is None:e['why']='no first tip for %s in nba-tonight/%s.json'%(date,date);return e,None
    pull,said=pick_pull(public,date,tip)
    if pull is None:e['why']=said;return e,None
    e['pull']=said
    scratch=os.path.join(cache,'ledger');code=os.path.join(scratch,'step');hist=os.path.join(scratch,'history');out=os.path.join(scratch,'out')
    for d in (code,hist,out):shutil.rmtree(d,ignore_errors=True)
    os.makedirs(code,exist_ok=True);os.makedirs(out,exist_ok=True)
    try:
        try:bundle.extract(code)
        except Exception as x:e['why']='the step could not be unpacked: %s'%str(x)[:160];return e,None
        try:r=run([python,os.path.join(HERE,'nba_props_ledger.py'),date,pull,step,'--out',out],capture_output=True,text=True,timeout=TIMEOUT,
                  env={**os.environ,'NBA_ROOT':os.path.abspath(public),'NBA_CACHE':hist,'PYTHONPATH':code})
        except subprocess.TimeoutExpired:e['why']='the ledger ran past %d seconds'%TIMEOUT;return e,None
        if r.returncode:e['why']='the ledger exited %s: %s'%(r.returncode,((r.stderr or r.stdout or '').strip().splitlines() or ['no message'])[-1][:200]);return e,None
        src=os.path.join(out,'props-ledger-%s.json'%date)
        try:doc=json.load(open(src))
        except Exception as x:e['why']='the ledger wrote no readable file (%s)'%type(x).__name__;return e,None
        if not isinstance(doc,dict) or doc.get('format')!='nba-props-ledger-v1' or doc.get('date')!=date or doc.get('private') is not True or not isinstance(doc.get('rows'),list):
            e['why']='the ledger answered with something that is not the private ledger for %s'%date;return e,None
        rows=[x for x in doc['rows'] if isinstance(x,dict)];graded=sum(1 for x in rows if x.get('result')!='void')
        e.update(lines=sum(1 for x in rows if x.get('kind')=='line'),milestones=sum(1 for x in rows if x.get('kind')=='milestone'),graded=graded)
        if not graded:e['why']='no line could be graded: no box score of %s in the job\'s files yet'%date;return e,None
        rel=OUT.format(date);dst=os.path.join(private,rel);os.makedirs(os.path.dirname(dst),exist_ok=True)
        shutil.copyfile(src,dst)   # as the ledger wrote it, byte for byte
        e.update(wrote=True,file=rel,seconds=round(time.time()-t0,1))
        return e,rel
    finally:
        for d in (code,hist,out):shutil.rmtree(d,ignore_errors=True)

def main(argv=None,bundle=None,run=subprocess.run,now=None):
    ap=argparse.ArgumentParser(description="Grade yesterday's props into the private ledger")
    ap.add_argument('--public',required=True);ap.add_argument('--private',required=True);ap.add_argument('--cache',required=True)
    ap.add_argument('--outputs-file',required=True);ap.add_argument('--date',help='one US Eastern date, YYYY-MM-DD (default: yesterday)')
    a=ap.parse_args(argv)
    if a.date and not re.match(r'^\d{4}-\d{2}-\d{2}$',a.date):ap.error('--date must look like 2026-10-20')
    date=a.date or yesterday(now);wrote=[];said={}
    soft=lambda x:{'date':date,'wrote':False,'why':'%s: %s'%(type(x).__name__,str(x)[:200])}
    try:
        sys.path.insert(0,HERE)
        if bundle is None:import nba_projection_bundle as bundle
        import nba_card_step as card_step
    except Exception as x:bundle=card_step=None;said['ledger']=said['card']=soft(x)
    if bundle is not None:
        try:e,rel=ledger(date,a.public,a.private,a.cache,bundle,run=run)
        except Exception as x:e,rel=soft(x),None
        said['ledger']=e;wrote+=[rel] if rel else []
        try:c,rel=card_step.card(date,a.public,a.private,a.cache,bundle,run=run,stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now.timestamp()) if now is not None else time.gmtime()))
        except Exception as x:c,rel=soft(x),None
        said['card']=c;wrote+=[rel] if rel else []
    with open(a.outputs_file,'w') as f:f.write(''.join(p+'\n' for p in wrote))
    print(json.dumps(said))
    return 0

if __name__=='__main__':
    sys.exit(main())
