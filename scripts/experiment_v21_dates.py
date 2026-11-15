"""Dev-only ablation of date ranking, coherent timelines and object coreference."""
import argparse
import copy
import json
from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from compactfacts.io import read_jsonl, write_json, write_jsonl
from compactfacts.date_ranker import DateModel, apply_date_model, preserve_visit_window
from compactfacts.event_dates import EventDateModel
from compactfacts.entities import canonicalize_objects
from compactfacts.pipeline import assemble_facts
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='tmp/qwen_dev_candidates.jsonl')
    parser.add_argument('--output-model', default='artifacts/qwen_v21')
    parser.add_argument('--prefix', default='reports/v21_date_ablation')
    args = parser.parse_args()
    fit = read_jsonl('tmp/splits/fit.jsonl'); dev = read_jsonl('tmp/splits/dev.jsonl')
    cached = {c['call_id']: c['candidates'] for c in read_jsonl(args.candidates)}
    assert set(cached) == {c['call_id'] for c in dev}
    event_path = Path('artifacts/event_dates_v21.joblib')
    if event_path.exists():
        event = joblib.load(event_path)
    else:
        print('Fitting event/date rankers on fit only', flush=True)
        event = EventDateModel().fit(fit)
        joblib.dump(event, event_path, compress=3)
        write_json(event_path.with_suffix('.json'), {'fit_calls': len(fit), 'dev_calls': len(dev), **event.audit})
    legacy_path = Path('artifacts/date_ranker_v21.joblib')
    if legacy_path.exists():
        legacy = joblib.load(legacy_path)
    else:
        legacy = DateModel().fit(fit); joblib.dump(legacy, legacy_path, compress=3)
    old = joblib.load('artifacts/date_ranker.joblib')
    from compactfacts.wide_dates import WideDateModel, apply_wide_dates
    from compactfacts.fit_examples import targeted_examples
    supplements=targeted_examples(json.loads(Path('data/raw/catalog.json').read_text(encoding='utf8')),42)
    wide_path=Path('artifacts/wide_dates_v21.joblib')
    import hashlib
    wide_audit=wide_path.with_suffix('.json')
    wide_fingerprint={'code_sha256':hashlib.sha256((ROOT/'src/compactfacts/wide_dates.py').read_bytes()).hexdigest(),
                      'fit_sha256':hashlib.sha256(Path('tmp/splits/fit.jsonl').read_bytes()).hexdigest(),
                      'supplement_sha256':hashlib.sha256(json.dumps(supplements,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                      'fit_calls':len(fit),'supplement_calls':len(supplements),'seed':42}
    if wide_path.exists() and wide_audit.exists() and json.loads(wide_audit.read_text(encoding='utf8'))==wide_fingerprint:
        wide=joblib.load(wide_path)
    else:
        wide=WideDateModel().fit(fit+supplements)
        joblib.dump(wide,wide_path,compress=3)
        write_json(wide_audit,wide_fingerprint)
    variants = {
        'old_ranker': (old, None, False, True),
        'rules': (None, None, False, True),
        'retrained_ranker': (legacy, None, False, True),
        'retrained_ranker_entities': (legacy, None, True, True),
        'event_dates': (None, event, False, False),
        'event_dates_timeline': (None, event, False, True),
        'event_dates_timeline_entities': (None, event, True, True),
        'retrained_event_strong': (legacy, event, False, False),
        'retrained_event_very_strong': (legacy, event, False, False),
        'wide_ranker': (wide, None, False, False),
        'wide_ranker_chains': (wide, None, False, True),
        'wide_ranker_chains_entities': (wide, None, True, True),
        'retrained_end': (legacy, event, False, False),
        'retrained_end_strong': (legacy, event, False, False),
        'retrained_end_very_strong': (legacy, event, False, False),
        'wide_ranker_end': (wide, event, False, True),
        'wide_ranker_end_strong': (wide, event, False, True),
        'wide_ranker_end_very_strong': (wide, event, False, True),
    }
    metrics = {}; predictions = {}
    for name, (ranker, event_model, entities, timeline) in variants.items():
        pred = []
        for raw in dev:
            call = {k:v for k,v in raw.items() if k != 'facts'}
            candidates = copy.deepcopy(cached[call['call_id']])
            if entities:
                account = (call.get('series') or {}).get('account_id') or call['call_id'].split(':')[0]
                candidates = canonicalize_objects(candidates, account)
            p = assemble_facts(call, candidates, None, agreement_evidence=False)
            if name.startswith('wide_ranker'):
                apply_wide_dates(call,p,ranker,chains=timeline)
            elif ranker:
                apply_date_model(call, p, ranker, {'action','agreement','tariff','service_state','ticket_status'})
                preserve_visit_window(call, p)
            if event_model:
                opts = {'start_only':True,'margin':2. if name=='retrained_event_strong' else 4.} if name.startswith('retrained_event_') else {}
                end_only=name.startswith(('retrained_end','wide_ranker_end'))
                if end_only:
                    opts={'end_only':True,'margin':4. if name.endswith('very_strong') else 2. if name.endswith('strong') else 0.}
                event_model.apply(call, p, timeline=False if end_only else timeline, **opts)
            pred.append(p)
        detail = evaluate(dev, pred, 'tmp/splits/dev.jsonl', True)
        metrics[name] = detail['official']; predictions[name] = pred
        write_json(f'{args.prefix}_{name}.json', detail)
        print(name, metrics[name], flush=True)
    def f1(m):
        p,r=m['fact_precision'],m['fact_recall'];return 2*p*r/(p+r) if p+r else 0
    base=metrics['old_ranker']
    eligible=[name for name,m in metrics.items() if m['fact_recall']>=base['fact_recall']*.98 and f1(m)>=f1(base)-.01]
    def objective(name):
        m=metrics[name];t=m['temporal_recall'] or 0;s=m['state_accuracy'] or 0
        return min(t/.4506,s/.5145), t+s, f1(m)
    selected=max(eligible,key=objective)
    ranker,event_model,entities,timeline=variants[selected]
    ranker_path = 'artifacts/date_ranker.joblib' if selected == 'old_ranker' else str(legacy_path).replace('\\','/') if ranker else None
    if selected.startswith('wide_ranker'):
        ranker_path=None
    config={'selected_on':'internal account-disjoint dev v2.1', 'variant':selected,
            'date_ranker':ranker_path,'agreement_evidence':False,
            'canonical_objects':entities,'event_date_model':str(event_path).replace('\\','/') if event_model else None}
    if selected.startswith('wide_ranker'):
        config.update(wide_date_model=str(wide_path).replace('\\','/'),wide_date_chains=timeline)
    if selected.startswith(('retrained_end','wide_ranker_end')):
        config['event_date_options']={'timeline':False,'end_only':True,
            'margin':4. if selected.endswith('very_strong') else 2. if selected.endswith('strong') else 0.}
    elif selected.startswith('retrained_event_'):
        config['event_date_options']={'timeline':False,'start_only':True,'margin':2. if selected=='retrained_event_strong' else 4.}
    elif event_model:
        config['event_date_options']={'timeline':timeline}
    write_json(Path(args.output_model)/'inference_config.json',config)
    write_json(f'{args.prefix}.json',{'metrics_version':'2.1','metrics':metrics,'selected':selected,
               'objective':'maximize the smaller normalized temporal/state score; retain fact recall and F1',
               'inference_config':config,'candidate_source':args.candidates,
               'note':'First cached ablation is exploratory; final selection must use fresh dev generation.'})
    write_jsonl(f'{args.prefix}_selected_pred.jsonl', predictions[selected])
    print('SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
