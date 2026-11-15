"""Choose the neural date-reference policy on the internal dev only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from compactfacts.io import read_jsonl,write_json,write_jsonl
from compactfacts.pipeline import assemble_facts
from compactfacts.date_ranker import apply_date_model,preserve_visit_window
from compactfacts.date_pointers import apply_date_pointers
from evaluate import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',required=True)
    parser.add_argument('--raw-predictions',required=True)
    parser.add_argument('--prefix',required=True)
    args=parser.parse_args()
    dev=read_jsonl('tmp/splits/dev.jsonl')
    cache={c['call_id']:c['candidates'] for c in read_jsonl(args.candidates)}
    raw={c['call_id']:c for c in read_jsonl(args.raw_predictions)}
    assert set(cache)==set(raw)=={c['call_id'] for c in dev}
    ranker=joblib.load('artifacts/date_ranker_v21.joblib')
    results={}
    for mode in ['off','starts','conservative','full']:
        predictions=[]
        for gold in dev:
            call={k:v for k,v in gold.items() if k!='facts'}
            candidates=copy.deepcopy(cache[call['call_id']])
            p=assemble_facts(call,candidates,None,agreement_evidence=False)
            apply_date_model(call,p,ranker,{'action','agreement','tariff','service_state','ticket_status'})
            preserve_visit_window(call,p)
            apply_date_pointers(call,p,candidates,mode)
            if mode=='conservative':
                assert p['facts']==raw[call['call_id']]['facts'],f"Cache/inference parity failed: {call['call_id']}"
            predictions.append(p)
        score=evaluate(dev,predictions,'tmp/splits/dev.jsonl',True)
        write_json(f'{args.prefix}_{mode}.json',score)
        write_jsonl(f'{args.prefix}_{mode}_pred.jsonl',predictions)
        results[mode]=score['official']
        print(mode,results[mode],flush=True)
    write_json(args.prefix+'.json',{'metrics':results,'selected_on':'internal account-disjoint dev',
        'conservative_cache_parity':True,'candidate_source':args.candidates})


if __name__=='__main__':main()
