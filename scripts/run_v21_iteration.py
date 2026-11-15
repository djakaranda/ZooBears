"""Resumable v2.1 refinement; all choices on dev, external val after freezing."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import atexit
import importlib.metadata
import platform
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl
from compactfacts.asr import augment_calls


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true',help='Skip stages already completed successfully')
    parser.add_argument('--allow-slow-gpu',action='store_true',help='Explicitly allow starting training under a reported power cap <40 W')
    args=parser.parse_args()
    if sys.platform=='win32':
        import ctypes
        # Keep the system awake only while this process lives; leave the display free to sleep.
        # Display sleep can cap the GPU on this laptop; release on exit.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        atexit.register(ctypes.windll.kernel32.SetThreadExecutionState,0x80000000)
    environment={'python':sys.version,'platform':platform.platform(),'packages':{},'started_at':time.time()}
    for package in ['torch','transformers','peft','accelerate','numpy','scikit-learn','joblib']:
        environment['packages'][package]=importlib.metadata.version(package)
    write_json('reports/v21_environment.json',environment)
    state_path=Path('reports/v21_iteration_status.json')
    records=json.loads(state_path.read_text(encoding='utf8')) if args.resume and state_path.exists() else []
    backup_record=Path('reports/v21_original_adapter_backup.json')
    if not records or not backup_record.exists():
        source=Path('artifacts/qwen')
        backup=Path('tmp/backups')/f'qwen_before_v21_{time.time_ns()}'
        shutil.copytree(source,backup)
        write_json(backup_record,{'backup':str(backup),
                   'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}})
    else:
        original=json.loads(backup_record.read_text(encoding='utf8'))
        assert all(hashlib.sha256((Path('artifacts/qwen')/name).read_bytes()).hexdigest()==digest
                   for name,digest in original['sha256'].items()), 'Original adapter changed since pipeline start'
    logdir=Path('tmp/v21_logs');logdir.mkdir(parents=True,exist_ok=True)
    def run(name,command):
        prior=next((r for r in records if r['stage']==name and r['status']=='complete'),None)
        if prior:
            print('SKIP',name,flush=True);return
        record={'stage':name,'status':'running','command':command,'started_at':time.time()}
        records.append(record);write_json(state_path,records);print('START',name,flush=True)
        with (logdir/(name+'.log')).open('w',encoding='utf8') as f:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        record.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,elapsed_seconds=time.time()-record['started_at'])
        write_json(state_path,records)
        if result.returncode:
            print((logdir/(name+'.log')).read_text(encoding='utf8')[-5000:],flush=True)
            raise RuntimeError(f'{name} failed')
        print('DONE',name,flush=True)
    if not any(r['stage']=='train' and r['status']=='complete' for r in records):
        power=subprocess.run(['nvidia-smi','--query-gpu=enforced.power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True)
        try:cap=float(power.stdout.strip().splitlines()[0])
        except (ValueError,IndexError):cap=None
        if cap is not None and cap<40 and not args.allow_slow_gpu:
            write_json('reports/v21_hardware_gate.json',{'power_cap_watts':cap,'training_started':False,
                       'reason':'Four-call training smoke measured 49 tokens/sec; full epoch would take roughly 30 hours. Restore ordinary GPU power before running.'})
            raise SystemExit('GPU power cap below 40 W. Restore notebook power profile, then rerun --resume; --allow-slow-gpu explicitly bypasses this runtime guard.')
    # Evaluate fresh dev predictions before changing any adapter weights.
    run('dev_generate_baseline',['scripts/predict_experiment.py','--engine','qwen','--model','artifacts/qwen',
        '--base-model','artifacts/qwen_base','--date-ranker','artifacts/date_ranker_v21.joblib',
        '--input','tmp/splits/dev.jsonl','--output','reports/v21_dev_baseline_raw.jsonl',
        '--candidate-output','tmp/v21_baseline_candidates.jsonl','--device','cuda','--batch-size','4'])
    run('dev_select_baseline',['scripts/experiment_v21_dates.py','--candidates','tmp/v21_baseline_candidates.jsonl',
        '--output-model','tmp/v21_configs/baseline','--prefix','reports/v21_dev_baseline'])
    run('train',['scripts/train_qwen.py','--output','artifacts/qwen_v21','--resume-adapter','artifacts/qwen',
          '--epochs','1','--lr','5e-5','--augmentation-fraction','.08','--targeted-examples','--checkpoint-steps','300'])
    candidates=[('baseline',Path('artifacts/qwen'))]
    candidates += [(p.name,p) for p in sorted(Path('artifacts/qwen_v21/checkpoints').iterdir()) if (p/'adapter_model.safetensors').exists()]
    results={}
    for label,model in candidates:
        prefix=f'reports/v21_dev_{label}'
        raw=f'tmp/v21_{label}_dev_raw.jsonl';cache=f'tmp/v21_{label}_candidates.jsonl'
        run(f'dev_generate_{label}',['scripts/predict_experiment.py','--engine','qwen','--model',str(model),
            '--base-model','artifacts/qwen_base','--date-ranker','artifacts/date_ranker_v21.joblib',
            '--input','tmp/splits/dev.jsonl','--output',raw,'--candidate-output',cache,'--device','cuda','--batch-size','4'])
        run(f'dev_select_{label}',['scripts/experiment_v21_dates.py','--candidates',cache,
            '--output-model',f'tmp/v21_configs/{label}','--prefix',prefix])
        detail=json.loads(Path(prefix+'.json').read_text(encoding='utf8'))
        results[label]={'model':str(model),'metrics':detail['metrics'][detail['selected']],
                        'variant':detail['selected'],'config':detail['inference_config'],'prefix':prefix}
    def f1(m):
        p,r=m['fact_precision'],m['fact_recall'];return 2*p*r/(p+r) if p+r else 0
    base=results['baseline']['metrics']
    eligible=[name for name,r in results.items() if r['metrics']['fact_recall']>=base['fact_recall']*.98 and f1(r['metrics'])>=f1(base)-.01]
    def key(name):
        m=results[name]['metrics'];t=m['temporal_recall'] or 0;s=m['state_accuracy'] or 0
        return min(t/.4506,s/.5145),t+s,f1(m)
    best=max(eligible,key=key);selected=results[best]
    write_json('reports/v21_checkpoint_selection.json',{'selected':best,'selected_on':'full internal dev, metrics v2.1',
             'objective':'maximize smaller normalized temporal/state score with recall/F1 safeguards',
             'candidates':results,'external_val_used_for_selection':False})
    destination=Path('artifacts/qwen_v21')
    if (destination/'data_audit.json').exists():
        shutil.copyfile(destination/'data_audit.json',destination/'retraining_data_audit.json')
    for path in Path(selected['model']).iterdir():
        if path.is_file() and path.name not in ['inference_config.json','history.json','progress.json']:
            shutil.copyfile(path,destination/path.name)
    write_json(destination/'inference_config.json',selected['config'])
    metadata=json.loads((destination/'experiment.json').read_text(encoding='utf8'))
    metadata.update(selection='full internal dev generated temporal/state metrics, recall and F1 safeguards',
                    selected_checkpoint=best,metrics_version='2.1',
                    selected_training_data_version='v1' if best=='baseline' else 'v2.1')
    write_json(destination/'experiment.json',metadata)
    shutil.copyfile(selected['prefix']+'_selected_pred.jsonl','reports/v21_dev_selected_pred.jsonl')
    ranker=['--date-ranker',selected['config']['date_ranker']] if selected['config']['date_ranker'] else []
    common=['--engine','qwen','--model','artifacts/qwen_v21','--base-model','artifacts/qwen_base',*ranker]
    run('val_generate',['scripts/predict_experiment.py',*common,'--input','data/raw/val.jsonl',
                        '--output','reports/v21_val_pred.jsonl','--device','cuda','--batch-size','4'])
    run('val_score',['scripts/evaluate.py','data/raw/val.jsonl','reports/v21_val_pred.jsonl','--output','reports/v21_val_metrics.json'])
    run('val_schema',['scripts/check_predictions.py','data/raw/val.jsonl','reports/v21_val_pred.jsonl','--output','reports/v21_val_schema.json'])
    dev=read_jsonl('tmp/splits/dev.jsonl');write_jsonl('tmp/v21_benchmark_calls.jsonl',dev[:24])
    for name,device,extra in [('cuda','cuda',[]),('cuda_batch4','cuda',['--batch-size','4']),('cpu','cpu',[])]:
        path=f'reports/v21_benchmark_{name}_pred.jsonl'
        run('benchmark_'+name,['scripts/predict_experiment.py',*common,'--input','tmp/v21_benchmark_calls.jsonl','--output',path,'--device',device,*extra])
        run('benchmark_'+name+'_score',['scripts/evaluate.py','tmp/v21_benchmark_calls.jsonl',path,
             '--output',f'reports/v21_benchmark_{name}_metrics.json','--repair-known-scorer-errors'])
    noisy=augment_calls(dev[:64],1.,.2,43)
    ids={c['call_id'] for c in noisy}
    write_jsonl('tmp/v21_stress_calls.jsonl',noisy)
    write_jsonl('tmp/v21_stress_clean_calls.jsonl',[c for c in dev if c['call_id'] in ids])
    write_jsonl('tmp/v21_stress_clean_pred.jsonl',[c for c in read_jsonl('reports/v21_dev_selected_pred.jsonl') if c['call_id'] in ids])
    run('stress_generate',['scripts/predict_experiment.py',*common,'--input','tmp/v21_stress_calls.jsonl',
        '--output','reports/v21_stress_pred.jsonl','--device','cuda','--batch-size','4'])
    run('stress_score',['scripts/evaluate.py','tmp/v21_stress_calls.jsonl','reports/v21_stress_pred.jsonl',
        '--output','reports/v21_stress_metrics.json','--repair-known-scorer-errors'])
    run('stress_clean_score',['scripts/evaluate.py','tmp/v21_stress_clean_calls.jsonl','tmp/v21_stress_clean_pred.jsonl',
        '--output','reports/v21_stress_clean_metrics.json','--repair-known-scorer-errors'])
    run('report',['scripts/write_v21_report.py'])
    print('V2.1 iteration complete',flush=True)


if __name__=='__main__':main()
