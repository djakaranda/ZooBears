"""Freeze a group-held-out development split and diagnose dates using train only."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random
import sys

from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl
from compactfacts.temporal import normalize_date


if __name__=='__main__':
    calls=read_jsonl(ROOT/'data/raw/train.jsonl')
    groups=[(c.get('series') or {}).get('account_id') or c['call_id'].split(':')[0] for c in calls]
    fit_idx,dev_idx=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=42).split(calls,groups=groups))
    fit=[calls[i] for i in fit_idx];dev=[calls[i] for i in dev_idx]
    write_jsonl(ROOT/'tmp/splits/fit.jsonl',fit);write_jsonl(ROOT/'tmp/splits/dev.jsonl',dev)
    manifest={'seed':42,'source_sha256':hashlib.sha256((ROOT/'data/raw/train.jsonl').read_bytes()).hexdigest(),
              'fit_call_ids':[c['call_id'] for c in fit],'dev_call_ids':[c['call_id'] for c in dev],
              'fit_calls':len(fit),'dev_calls':len(dev),'account_overlap':len({groups[i] for i in fit_idx}&{groups[i] for i in dev_idx})}
    write_json(ROOT/'reports/dev_split.json',manifest)
    counters=defaultdict(Counter);examples=defaultdict(list)
    for c in dev:
        for f in c['facts']:
            a=f['attribute'];s=f['source']['turn_idx']
            i=next(i for i,t in enumerate(c['turns']) if t['idx']==s)
            text=c['turns'][i]['text']
            parsed,explicit=normalize_date(text,c['started_at'],future=a=='agreement')
            if not explicit and a in {'tariff','agreement'}:parsed=c['started_at']
            counters[a]['gold_facts']+=1
            counters[a]['date_expression_found_in_source']+=explicit
            counters[a]['source_parser_correct_start_day']+=parsed[:10]==f['valid_from'][:10]
            any_matching=False
            for t in c['turns']:
                date,e=normalize_date(t['text'],c['started_at'],future=a=='agreement')
                any_matching |= e and date[:10]==f['valid_from'][:10]
            counters[a]['some_call_turn_parses_to_gold_start']+=any_matching
            counters[a]['start_equals_call_day']+=f['valid_from'][:10]==c['started_at'][:10]
            counters[a]['closed_intervals']+=f['valid_to'] is not None
            if parsed[:10]!=f['valid_from'][:10] and len(examples[a])<12:
                examples[a].append({'call_id':c['call_id'],'started_at':c['started_at'],'value':f['value'],
                  'gold_from':f['valid_from'],'gold_to':f['valid_to'],'parsed':parsed,'explicit':explicit,
                  'source':text,'context':[t for t in c['turns'][max(0,i-1):i+2]]})
    write_json(ROOT/'reports/temporal_diagnosis.json',{'split':'held-out train dev; reference sources used for diagnosis only',
        'note':'No matching parse is evidence about this parser, not proof that the date is unobservable.',
        'attributes':{k:dict(v) for k,v in counters.items()},'examples':dict(examples)})
    print({k:v for k,v in manifest.items() if not k.endswith('_ids')})
    print({k:dict(v) for k,v in counters.items()})
