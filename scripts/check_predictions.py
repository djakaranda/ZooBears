"""Validate all prediction records against the minimal case contract."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.io import read_jsonl,write_json


def check(gold,pred):
    inputs={c['call_id']:c for c in gold}
    errors=[]; seen=set(); n=0
    for call in pred:
        cid=call.get('call_id')
        if cid not in inputs or cid in seen:
            errors.append({'call_id':cid,'error':'Unknown or duplicate ID'}); continue
        seen.add(cid)
        turns={t['idx']:t for t in inputs[cid]['turns']}
        for i,f in enumerate(call.get('facts',[])):
            n+=1
            try:
                assert f['attribute'] in {'service_state','tariff','ticket_status','action','agreement'}
                assert f['polarity'] in [-1,1]
                assert isinstance(f['value'],str) and f['value']
                assert isinstance(f['object']['name'],str) and f['object']['name']
                assert f['object']['type'] in {'service','client','device','ticket'}
                datetime.fromisoformat(f['valid_from'])
                if f['valid_to'] is not None:
                    datetime.fromisoformat(f['valid_to'])
                    assert f['valid_to']>=f['valid_from']
                source=f['source']; turn=turns[source['turn_idx']]
                assert turn['speaker']==f['speaker']
                lo,hi=source['span']
                assert 0<=lo<=hi<=len(turn['text'])
                assert turn['text'][lo:hi]==source['quote']
            except (KeyError,TypeError,ValueError,AssertionError) as exc:
                errors.append({'call_id':cid,'fact_index':i,'error':type(exc).__name__})
    missing=set(inputs)-seen
    result={'calls':len(pred),'facts':n,'missing_calls':len(missing),'errors':errors,'valid':not errors and not missing}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('input'); parser.add_argument('pred')
    parser.add_argument('--output',default='reports/schema_check.json'); args=parser.parse_args()
    result=check(read_jsonl(args.input),read_jsonl(args.pred)); write_json(args.output,result)
    print(result); raise SystemExit(0 if result['valid'] else 1)
