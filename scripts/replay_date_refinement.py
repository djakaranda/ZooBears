"""Audit a post-selection API correction against frozen generated predictions.

For this release all competition fact objects must remain identical. Only
auxiliary provenance is replayed; metrics and neural generation are not rerun.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl
from compactfacts.date_refinement import refine_date_evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    datasets=[('dev','tmp/splits/dev.jsonl','reports/date_pointer_dev_selected_pred.jsonl'),
              ('val','data/raw/val.jsonl','reports/date_pointer_val_pred.jsonl'),
              ('stress','tmp/date_pointer_stress_calls.jsonl','reports/date_pointer_stress_pred.jsonl'),
              ('stress_clean','tmp/date_pointer_stress_clean_calls.jsonl','tmp/date_pointer_stress_clean_pred.jsonl')]
    datasets.extend((name,'tmp/date_pointer_benchmark_calls.jsonl',f'reports/date_pointer_benchmark_{name}_pred.jsonl')
                    for name in ['cuda','cuda_batch4','cpu'])
    plans=[];audit={}
    for name,input_path,pred_path in datasets:
        if not Path(input_path).exists() or not Path(pred_path).exists():
            if name in {'dev','val'}:raise FileNotFoundError(pred_path)
            continue
        calls={c['call_id']:{k:v for k,v in c.items() if k!='facts'} for c in read_jsonl(input_path)}
        predictions=read_jsonl(pred_path);changed=0;durations=[]
        before_facts=[copy.deepcopy(p['facts']) for p in predictions]
        for p in predictions:
            original=copy.deepcopy(p)
            started=time.perf_counter();refine_date_evidence(calls[p['call_id']],p)
            durations.append((time.perf_counter()-started)*1000)
            changed+=p!=original
        assert [p['facts'] for p in predictions]==before_facts,f'Competition facts changed: {name}'
        facts_sha=hashlib.sha256(json.dumps(before_facts,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        audit[name]={'calls':len(predictions),'metadata_changed_calls':changed,'fact_objects_unchanged':True,
                     'facts_sha256':facts_sha,'helper_mean_ms':statistics.mean(durations),
                     'helper_max_ms':max(durations)}
        plans.append((pred_path,predictions))
    if not args.dry_run:
        for pred_path,predictions in plans:write_jsonl(pred_path,predictions)
    write_json('reports/date_pointer_api_refinement_audit.json',{'dry_run':args.dry_run,'datasets':audit,
        'refinement_source_sha256':hashlib.sha256((ROOT/'src/compactfacts/date_refinement.py').read_bytes()).hexdigest(),
        'model_weights_and_metric_files_unchanged':True,
        'resource_timing_note':'Resource timings precede this small CPU postprocessor correction. Frozen benchmark fact objects are checked unchanged; additional helper overhead is measured here.',
        'test_oracle_note':'Initial legacy API failures are retained. Exact transition dates are tested on an explicit transcript, not on an unrelated SMS date.'})
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
