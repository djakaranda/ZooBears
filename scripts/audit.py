"""Read-only dataset audit; the reconstructed vocabulary uses train ONLY."""
from collections import Counter
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from compactfacts.io import read_jsonl, write_json


def account(call):
    return (call.get("series") or {}).get("account_id") or call["call_id"].split(":")[0]


def audit(calls):
    problems = Counter()
    examples = []
    for call in calls:
        turns = {t['idx']:t for t in call['turns']}
        for fact in call['facts']:
            source = fact.get('source') or {}
            turn = turns.get(source.get('turn_idx'))
            if turn is None:
                problems['missing_source_turn'] += 1
                continue
            text = turn['text']
            quote = source.get('quote')
            span = source.get('span')
            issues = []
            if quote and quote not in text:
                problems['quote_not_exact_substring'] += 1; issues.append('quote_not_exact_substring')
            if span:
                problems['facts_with_span'] += 1
                if not 0 <= span[0] < span[1] <= len(text):
                    problems['span_out_of_bounds'] += 1; issues.append('span_out_of_bounds')
                elif quote and text[span[0]:span[1]] != quote:
                    problems['span_quote_disagreement'] += 1
            if fact.get('speaker') != turn['speaker']:
                problems['speaker_source_disagreement'] += 1
            if fact['object'].get('type') not in {'client','service','device','ticket'}:
                problems['object_type_outside_schema'] += 1
            if issues and len(examples)<12:
                examples.append({'call_id':call['call_id'],'issues':issues,'fact':fact,'actual_source':text})
    return {'calls':len(calls),'turns':sum(len(c['turns']) for c in calls),
            'facts':sum(len(c['facts']) for c in calls),'attributes':dict(Counter(f['attribute'] for c in calls for f in c['facts'])),
            'unique_values':len({(f['attribute'],f['value']) for c in calls for f in c['facts']}),
            'observations':dict(problems),'examples':examples}


if __name__ == '__main__':
    train, val = (read_jsonl('data/raw/'+s+'.jsonl') for s in ['train','val'])
    train_values = {(f['attribute'],f['value']) for c in train for f in c['facts']}
    report = {'train':audit(train),'val':audit(val),
        'overlapping_call_ids':len({c['call_id'] for c in train}&{c['call_id'] for c in val}),
        'overlapping_accounts':len({account(c) for c in train}&{account(c) for c in val}),
        'val_facts_value_not_seen_verbatim_in_train':sum((f['attribute'],f['value']) not in train_values for c in val for f in c['facts']),
        'catalog_provided':Path('data/raw/catalog.json').exists(),
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data/raw').iterdir() if p.is_file()}}
    write_json('reports/data_audit.json',report)
    objects={}; values={}
    for c in train:
        for f in c['facts']:
            objects.setdefault(f['object']['type'],set()).add(f['object']['name'])
            values.setdefault(f['attribute'],set()).add(f['value'])
    write_json('artifacts/train_vocabulary.json',{'source':'train.jsonl only; not an official catalog',
        'objects':{k:sorted(v) for k,v in objects.items()},'values':{k:sorted(v) for k,v in values.items()}})
    print({k: v for k,v in report.items() if k not in ['train','val','sha256']})
    for split in ['train','val']: print(split, {k:v for k,v in report[split].items() if k!='examples'})
