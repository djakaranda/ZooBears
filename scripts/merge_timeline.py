import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.ledger import TimelineLedger
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--pred',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();calls=read_jsonl(a.input);pred={c['call_id']:c for c in read_jsonl(a.pred)};ledger=TimelineLedger()
    for call in sorted(calls,key=lambda c:c['started_at']):ledger.ingest({k:v for k,v in call.items() if k!='facts'},pred.get(call['call_id'],{'facts':[]}))
    write_json(a.output,ledger.snapshot())
