"""Select date postprocessing on internal dev only, without regenerating facts."""
import argparse
import copy
from pathlib import Path
import sys
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from compactfacts.io import read_jsonl, write_json, write_jsonl
from compactfacts.pipeline import assemble_facts
from compactfacts.date_ranker import apply_date_model, preserve_visit_window
from evaluate import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', default='tmp/splits/dev.jsonl')
    parser.add_argument('--candidates', default='tmp/qwen_dev_candidates.jsonl')
    parser.add_argument('--output', default='reports/qwen_dev_selection.json')
    args = parser.parse_args()
    gold = read_jsonl(args.dev)
    calls = {c['call_id']: {k: v for k, v in c.items() if k != 'facts'} for c in gold}
    cached = read_jsonl(args.candidates)
    assert {r['call_id'] for r in cached} == set(calls)
    ranker = load('artifacts/date_ranker.joblib')
    variants = {}; metrics = {}
    for name in ['rules', 'date_ranker', 'ranker_with_agreement_evidence']:
        predictions = []
        for row in cached:
            call = calls[row['call_id']]
            result = assemble_facts(call, copy.deepcopy(row['candidates']), None, agreement_evidence=name != 'date_ranker')
            if name != 'rules':
                apply_date_model(call, result, ranker, {'action', 'agreement', 'tariff', 'service_state', 'ticket_status'})
                preserve_visit_window(call, result)
            predictions.append(result)
        variants[name] = predictions
        detail = evaluate(gold, predictions, args.dev, True)
        metrics[name] = detail['official']
        write_json(f'reports/qwen_dev_{name}_metrics.json', detail)
        write_jsonl(f'tmp/qwen_dev_{name}_pred.jsonl', predictions)
    def score(name):
        m = metrics[name]
        p, r = m['fact_precision'] or 0, m['fact_recall'] or 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        return ((m['temporal_recall'] or 0) + (m['state_accuracy'] or 0), f1)
    selected = max(metrics, key=score)
    write_json(args.output, {'split': 'internal account-disjoint dev', 'calls': len(gold),
                             'objective': 'temporal_recall + state_accuracy; fact F1 breaks ties',
                             'metrics': metrics, 'selected': selected,
                             'date_ranker': 'artifacts/date_ranker.joblib' if selected != 'rules' else None,
                             'agreement_evidence': selected != 'date_ranker'})
    write_json('artifacts/qwen/inference_config.json', {
        'selected_on': 'internal account-disjoint dev', 'variant': selected,
        'date_ranker': 'artifacts/date_ranker.joblib' if selected != 'rules' else None,
        'agreement_evidence': selected != 'date_ranker',
    })
    write_jsonl('reports/qwen_dev_pred.jsonl', variants[selected])
    print({'selected': selected, 'metrics': metrics}, flush=True)
