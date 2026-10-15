#!/usr/bin/env python3
"""The morning card, in the PRIVATE data repo's own workflow (Vale 10/14: DAILY-RECORD.md items 9 to 13, his card/README.md; Monty's
HOLD of 8caf95e). Chris presses nothing for it. The card is graded from the night's record, which carries Stokastic's minutes and
ownership, so it never runs in the public job: nba_private_record.py runs it beside the props ledger, and it is kept in the private repo.

For yesterday (US Eastern), when the night left a record (the public repo's nba-proj/<date>.json or the private repo's nba-record/<date>/),
it puts together, in its scratch folder, the one folder Vale's card reads, from both checkouts:
    nba-record/<date>/     the private repo's (each draft group's build, sim and stok files, the admin's news checks), and the league's
                           injury report and the lineups file the public job kept in the public repo's nba-public-record/<date>/;
    nba-proj/<date>.json, dk-results/<date>.json   from the public repo;
unpacks the reviewed step from nba_projection_bundle.py (for nbalib, which reads the box scores in the public checkout) and runs Vale's
card unchanged (68d176f73104, his order 071), with his money:
    NBA_ROOT=<public> NBA_CACHE=<scratch> python3 nba_card.py <date> <that folder> --out <scratch> --step <unpacked step>
        --private --entries <private>/nba-my-lineups/results/<date>.json
then keeps its card-<date>.json as nba-record/<date>/card-<UTC time>.json in the PRIVATE checkout: a new file each press, never one
replaced. --private is passed only here, where the file can only go to the private repo.

Soft: a day with no record, the step not unpacking, the card failing or answering with something that is not the card for the date
writes no file and says why in what it returns (nba_private_record.py prints it in the run's log). Nothing is retried. The parsed
history the card's nbalib makes is about 120 MB, so its scratch folder is emptied before and after. nbalib imports numpy: the private
workflow installs the public job's pinned numpy before this runs."""
import json,os,re,shutil,subprocess,sys,time,datetime
from zoneinfo import ZoneInfo
HERE=os.path.dirname(os.path.abspath(__file__))

ET=ZoneInfo('America/New_York')
TIMEOUT=900
NEWS=re.compile(r'^(?:injury|lineups)-\d{8}T\d{6}Z\.json$')    # what the card takes from the public job's record of the night
ENTRIES='nba-my-lineups/results/{}.json'                         # his entries file, in the private checkout (read only with --private)
OUT='nba-record/{}/card-{}.json'

def yesterday(now=None):
    """The US Eastern date before the run's own date."""
    d=(now or datetime.datetime.now(datetime.timezone.utc)).astimezone(ET).date()
    return (d-datetime.timedelta(days=1)).isoformat()

def had_slate(public,private,date):
    """True when the night left a record to grade: the step's file, or the night's record in the private repo."""
    return os.path.exists(os.path.join(public,'nba-proj',date+'.json')) or os.path.isdir(os.path.join(private,'nba-record',date))

def stage(date,public,private,root):
    """The folder the card reads as its data repo, put together from both checkouts; nothing in either checkout is touched."""
    rec=os.path.join(root,'nba-record',date);src=os.path.join(private,'nba-record',date)
    if os.path.isdir(src):shutil.copytree(src,rec)
    else:os.makedirs(rec)
    pub=os.path.join(public,'nba-public-record',date)
    for n in sorted(os.listdir(pub)) if os.path.isdir(pub) else []:
        if NEWS.match(n):shutil.copyfile(os.path.join(pub,n),os.path.join(rec,n))
    for rel in ('nba-proj/%s.json'%date,'dk-results/%s.json'%date):
        s=os.path.join(public,rel)
        if os.path.isfile(s):os.makedirs(os.path.dirname(os.path.join(root,rel)),exist_ok=True);shutil.copyfile(s,os.path.join(root,rel))

def card(date,public,private,cache,bundle,python=sys.executable,run=subprocess.run,stamp=None):
    """Runs the card for one date. Returns (the status entry, the private path it wrote or None)."""
    t0=time.time();e={'date':date,'wrote':False}
    if not had_slate(public,private,date):e['why']='no record of a slate on %s (no nba-proj file, no nba-record folder)'%date;return e,None
    scratch=os.path.join(cache,'card');code=os.path.join(scratch,'step');hist=os.path.join(scratch,'history');out=os.path.join(scratch,'out');root=os.path.join(scratch,'root')
    for d in (code,hist,out,root):shutil.rmtree(d,ignore_errors=True)
    os.makedirs(code,exist_ok=True);os.makedirs(out,exist_ok=True)
    try:
        try:bundle.extract(code)
        except Exception as x:e['why']='the step could not be unpacked: %s'%str(x)[:160];return e,None
        stage(date,public,private,root)
        try:r=run([python,os.path.join(HERE,'nba_card.py'),date,os.path.abspath(root),'--out',out,'--step',code,
                   '--private','--entries',os.path.join(os.path.abspath(private),ENTRIES.format(date))],capture_output=True,text=True,timeout=TIMEOUT,
                  env={**os.environ,'NBA_ROOT':os.path.abspath(public),'NBA_CACHE':hist})
        except subprocess.TimeoutExpired:e['why']='the card ran past %d seconds'%TIMEOUT;return e,None
        if r.returncode:e['why']='the card exited %s: %s'%(r.returncode,((r.stderr or r.stdout or '').strip().splitlines() or ['no message'])[-1][:200]);return e,None
        src=os.path.join(out,'card-%s.json'%date)
        try:doc=json.load(open(src))
        except Exception as x:e['why']='the card wrote no readable file (%s)'%type(x).__name__;return e,None
        if not isinstance(doc,dict) or doc.get('date')!=date or not isinstance(doc.get('validator'),list):
            e['why']='the card answered with something that is not the card for %s'%date;return e,None
        rel=OUT.format(date,stamp or time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));dst=os.path.join(private,rel)
        if os.path.exists(dst):e['why']='%s is already in the record (never replaced)'%rel;return e,None
        os.makedirs(os.path.dirname(dst),exist_ok=True)
        shutil.copyfile(src,dst)   # as the card wrote it, byte for byte
        e.update(wrote=True,file=rel,validator=len(doc['validator']),missing=sum(1 for v in doc['validator'] if 'MISSING' in str(v)),seconds=round(time.time()-t0,1))
        return e,rel
    finally:
        for d in (code,hist,out,root):shutil.rmtree(d,ignore_errors=True)
