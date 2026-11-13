"""Official scores unchanged, plus diagnostics and a stricter object check."""
import argparse
from collections import Counter
import copy
from difflib import SequenceMatcher
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'data/raw')]
import case_metrics as official
from compactfacts.io import read_jsonl, write_json


def evaluate(gold, pred):
    gd = {c['call_id']:c for c in gold}; pd={c['call_id']:c for c in pred}
    result={'official':official.evaluate(gd,pd),'per_attribute':{},'errors':[]}
    matched_total = strict_total = temporal_correct = 0
    for attr in ['service_state','tariff','ticket_status','action','agreement']:
        ga = {k:{**v,'facts':[f for f in v['facts'] if f['attribute']==attr]} for k,v in gd.items()}
        pa = {k:{**v,'facts':[f for f in v['facts'] if f['attribute']==attr]} for k,v in pd.items()}
        result['per_attribute'][attr]=official.evaluate(ga,pa)
    for cid,g in gd.items():
        pf = pd.get(cid,{}).get('facts',[])
        pairs=official.match_facts(g['facts'],pf)
        matched_total += len(pairs)
        for gf,p,_ in pairs:
            strict_total += SequenceMatcher(None,gf['object']['name'].lower(),p['object']['name'].lower()).ratio() >= .8
            temporal_correct += official._day(gf.get('valid_from'))==official._day(p.get('valid_from')) and official._day(gf.get('valid_to'))==official._day(p.get('valid_to'))
        matched_ids={id(gf) for gf,_,_ in pairs}
        if len(result['errors'])<30:
            for f in g['facts']:
                if id(f) not in matched_ids:
                    source=next((t for t in g['turns'] if t['idx']==f['source']['turn_idx']),None)
                    result['errors'].append({'call_id':cid,'missed_gold':f,'source_turn':source,'predictions':pf})
                    break
    result['diagnostics']={'matched_pairs_passing_full_object_name_similarity':strict_total,
        'official_matched_pairs':matched_total,
        'full_name_agreement_among_official_matches':strict_total/matched_total if matched_total else 0,
        'correct_temporal_matches_divided_by_ALL_gold_facts':temporal_correct/sum(len(c['facts']) for c in gold),
        'note':'Full-name agreement filters official matches; it is diagnostic, not an independent optimal rematching metric. Official temporal_recall is conditional on matched facts.'}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('gold'); parser.add_argument('pred'); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=evaluate(read_jsonl(args.gold),read_jsonl(args.pred))
    write_json(args.output,result)
    print(result['official'])
    print(result['diagnostics'])
