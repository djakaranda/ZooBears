"""Matched clean/noisy dev control for the frozen final model; no tuning."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.asr import augment_calls
from compactfacts.io import read_jsonl,write_json,write_jsonl


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');args=parser.parse_args()
    selection=json.loads(Path('reports/date_pointer_selection.json').read_text(encoding='utf8'))
    model=args.model or selection['deployed_model']
    if selection['selected']=='baseline' and args.model is None:
        for kind in ['stress','stress_clean']:
            shutil.copyfile(f'reports/v21_{kind}_metrics.json',f'reports/date_pointer_{kind}_metrics.json')
        write_json('reports/date_pointer_stress_status.json',{'reused_identical_fallback_measurements':True,'model':model})
        print('Reused stress measurements for the unchanged fallback')
        return
    dev=read_jsonl('tmp/splits/dev.jsonl');noisy=augment_calls(dev[:64],1.,.2,43)
    ids={c['call_id'] for c in noisy}
    write_jsonl('tmp/date_pointer_stress_calls.jsonl',noisy)
    write_jsonl('tmp/date_pointer_stress_clean_calls.jsonl',[c for c in dev if c['call_id'] in ids])
    predictions=read_jsonl('reports/date_pointer_dev_selected_pred.jsonl')
    write_jsonl('tmp/date_pointer_stress_clean_pred.jsonl',[c for c in predictions if c['call_id'] in ids])
    commands=[['scripts/predict_experiment.py','--engine','qwen','--model',model,'--base-model','artifacts/qwen_base',
        '--date-ranker','artifacts/date_ranker_v21.joblib','--input','tmp/date_pointer_stress_calls.jsonl',
        '--output','reports/date_pointer_stress_pred.jsonl','--device','cuda','--batch-size','4'],
        ['scripts/evaluate.py','tmp/date_pointer_stress_calls.jsonl','reports/date_pointer_stress_pred.jsonl',
         '--output','reports/date_pointer_stress_metrics.json','--repair-known-scorer-errors'],
        ['scripts/evaluate.py','tmp/date_pointer_stress_clean_calls.jsonl','tmp/date_pointer_stress_clean_pred.jsonl',
         '--output','reports/date_pointer_stress_clean_metrics.json','--repair-known-scorer-errors']]
    records=[];logs=Path('tmp/date_pointer_stress_logs');logs.mkdir(parents=True,exist_ok=True)
    for index,command in enumerate(commands):
        started=time.time()
        with (logs/f'{index}.log').open('w',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        records.append({'command':command,'exit_code':result.returncode,'seconds':time.time()-started})
        write_json('reports/date_pointer_stress_status.json',{'model':model,'calls':len(noisy),'seed':43,'rate':.2,'stages':records})
        if result.returncode:raise RuntimeError(f'Stress stage {index} failed; see {logs}')
    print('Stress and matched clean control complete',flush=True)


if __name__=='__main__':main()
