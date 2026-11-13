"""Secondary metrics: full object names and object-specific current states.

Official scores remain in evaluate.py. This diagnostic deliberately does not
claim equivalence to the jury scorer.
"""
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'data/raw')]
import case_metrics as official
from compactfacts.io import read_jsonl,write_json


def name_similarity(a,b):
    return SequenceMatcher(None,' '.join(official._tokens(a)),' '.join(official._tokens(b))).ratio()


def pairs(gold,pred):
    options=[]
    for gi,g in enumerate(gold):
        for pi,p in enumerate(pred):
            if g['attribute']!=p['attribute'] or (g.get('polarity',1)>0)!=(p.get('polarity',1)>0):continue
            n=name_similarity(g['object']['name'],p['object']['name']);v=official._value_sim(g['value'],p['value'])
            if n>=.8 and v>=.85:options.append((n*v,gi,pi))
    options.sort(reverse=True);used_g=set();used_p=set();out=[]
    for score,gi,pi in options:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi);used_p.add(pi);out.append((gold[gi],pred[pi]))
    return out


def evaluate(gold,pred):
    predicted={c['call_id']:c for c in pred};n_gold=n_pred=matched=dated=grounded=state_total=state_ok=0
    by_attr=defaultdict(lambda:defaultdict(int));state_types=defaultdict(lambda:defaultdict(int))
    for call in gold:
        gf=call['facts'];pf=predicted.get(call['call_id'],{}).get('facts',[])
        n_gold+=len(gf);n_pred+=len(pf);matches=pairs(gf,pf);matched+=len(matches)
        for f in gf:by_attr[f['attribute']]['gold']+=1
        for f in pf:by_attr[f['attribute']]['pred']+=1
        for g,p in matches:
            by_attr[g['attribute']]['matched']+=1
            correct=official._day(g['valid_from'])==official._day(p['valid_from']) and official._day(g['valid_to'])==official._day(p['valid_to'])
            dated+=correct;by_attr[g['attribute']]['correct_dates']+=correct
            grounded+=g['source']['turn_idx']==p['source']['turn_idx']
        active={}
        for f in gf:
            if f['attribute'] in official.SINGLE_VALUED and f['valid_to'] is None:
                k=(f['object']['id'],f['attribute'])
                if k not in active or (f.get('valid_from') or '')>(active[k].get('valid_from') or ''):active[k]=f
        for g in active.values():
            state_total+=1
            category='changed' if any(f['object']['id']==g['object']['id'] and f['attribute']==g['attribute'] and f['valid_to'] is not None for f in gf) else 'static'
            state_types[category]['total']+=1
            candidates=[p for p in pf if p['attribute']==g['attribute'] and p['valid_to'] is None and name_similarity(g['object']['name'],p['object']['name'])>=.8]
            if candidates:
                latest=max(candidates,key=lambda p:p.get('valid_from') or '')
                correct=official._value_sim(g['value'],latest['value'])>=.85
                state_ok+=correct;state_types[category]['correct']+=correct
    return {'definition':'Full-name similarity >= .8, value >= .85; per-object state; not the official scorer',
        'calls':len(gold),'gold_facts':n_gold,'pred_facts':n_pred,'matched':matched,
        'fact_precision':matched/n_pred if n_pred else 0,'fact_recall':matched/n_gold if n_gold else 0,
        'temporal_recall_over_ALL_gold':dated/n_gold if n_gold else 0,
        'temporal_accuracy_among_matches':dated/matched if matched else None,
        'grounding_turn_among_matches':grounded/matched if matched else None,
        'state_accuracy_per_object':state_ok/state_total if state_total else None,
        'state_static_vs_changed':{k:dict(v,accuracy=v.get('correct',0)/v['total']) for k,v in state_types.items()},
        'per_attribute':{k:dict(v,precision=v.get('matched',0)/max(v['pred'],1),recall=v.get('matched',0)/max(v['gold'],1)) for k,v in by_attr.items()}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('gold');parser.add_argument('pred');parser.add_argument('--output',required=True)
    args=parser.parse_args();metrics=evaluate(read_jsonl(args.gold),read_jsonl(args.pred));write_json(args.output,metrics);print(metrics)
