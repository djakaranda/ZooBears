"""Train grounded date selection, freeze on dev, then evaluate the selected model."""
import argparse
import atexit
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import importlib.metadata
import platform

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if sys.platform=='win32':
        import ctypes
        # On this MSI laptop display sleep also caps the GPU at 30 W.
        # The request lasts only for this process; no power plan is changed.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        atexit.register(ctypes.windll.kernel32.SetThreadExecutionState,0x80000000)
    assert Path('reports/V21_ITERATION_REPORT.md').exists(),'Finish the fallback iteration before starting this experiment'
    write_json('reports/date_pointer_environment.json',{'python':sys.version,'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','numpy','scikit-learn','joblib']},
        'trainer_sha256':hashlib.sha256(Path('scripts/train_qwen.py').read_bytes()).hexdigest(),
        'format_sha256':hashlib.sha256(Path('src/compactfacts/qwen_date_format.py').read_bytes()).hexdigest()})
    state_path=Path('reports/date_pointer_iteration_status.json')
    records=read(state_path) if args.resume and state_path.exists() else []
    logs=Path('tmp/date_pointer_logs');logs.mkdir(parents=True,exist_ok=True)
    baseline=Path('artifacts/qwen_v21');output=Path('artifacts/qwen_dates')
    backup_path=Path('reports/date_pointer_baseline_hashes.json')
    if not records:
        write_json(backup_path,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.iterdir() if p.is_file()})
    def check_baseline():
        assert all(hashlib.sha256((baseline/name).read_bytes()).hexdigest()==digest for name,digest in read(backup_path).items()),'Fallback model changed'
    check_baseline()
    def run(name,command):
        if any(r['stage']==name and r['status']=='complete' for r in records):
            print('SKIP',name,flush=True);return
        record={'stage':name,'command':command,'status':'running','started_at':time.time()}
        records.append(record);write_json(state_path,records);print('START',name,flush=True)
        with (logs/(name+'.log')).open('w',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.time()-record['started_at'])
        write_json(state_path,records)
        if result.returncode:
            print((logs/(name+'.log')).read_text(encoding='utf8')[-6000:],flush=True)
            raise RuntimeError(name+' failed')
        print('DONE',name,flush=True)
    if not any(r['stage']=='train' and r['status']=='complete' for r in records):
        gpu=subprocess.run(['nvidia-smi','--query-gpu=enforced.power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if float(gpu.stdout.strip())<40:raise RuntimeError('GPU power cap too low for the planned training')
        write_json(output/'inference_config.json',{'date_ranker':'artifacts/date_ranker_v21.joblib',
            'agreement_evidence':False,'canonical_objects':False,'event_date_model':None,
            'date_pointer_mode':'conservative','selected_on':'candidate, not yet selected'})
    run('preflight',['scripts/preflight_date_pointers.py'])
    run('train',['scripts/train_qwen.py','--output',str(output),'--resume-adapter',str(baseline),
        '--epochs','1','--lr','5e-5','--augmentation-fraction','.08','--targeted-examples',
        '--checkpoint-steps','300','--date-pointers','--max-prompt-tokens','3072','--date-token-weight','3'])
    previous=read('reports/v21_checkpoint_selection.json')
    base_metrics=previous['candidates'][previous['selected']]['metrics']
    candidates={'baseline':{'model':baseline.as_posix(),'mode':'off','metrics':base_metrics,
        'predictions':'reports/v21_dev_selected_pred.jsonl'}}
    for model in sorted((output/'checkpoints').iterdir()):
        if not (model/'adapter_model.safetensors').exists():continue
        label=model.name;raw=f'tmp/date_pointer_{label}_dev_raw.jsonl';cache=f'tmp/date_pointer_{label}_candidates.jsonl'
        run('dev_generate_'+label,['scripts/predict_experiment.py','--engine','qwen','--model',str(model),
            '--base-model','artifacts/qwen_base','--date-ranker','artifacts/date_ranker_v21.joblib',
            '--input','tmp/splits/dev.jsonl','--output',raw,'--candidate-output',cache,'--device','cuda','--batch-size','4'])
        prefix=f'reports/date_pointer_dev_{label}'
        run('dev_select_'+label,['scripts/select_date_pointers.py','--candidates',cache,'--raw-predictions',raw,'--prefix',prefix])
        for mode,metrics in read(prefix+'.json')['metrics'].items():
            candidates[label+'_'+mode]={'model':model.as_posix(),'mode':mode,'metrics':metrics,'predictions':prefix+'_'+mode+'_pred.jsonl'}
    def f1(m):
        p,r=m['fact_precision'],m['fact_recall'];return 2*p*r/(p+r) if p+r else 0.
    eligible=[]
    for name,item in candidates.items():
        m=item['metrics']
        if (m['fact_recall']>=base_metrics['fact_recall']*.98 and f1(m)>=f1(base_metrics)-.01
            and (m['state_accuracy'] or 0)>=min(base_metrics['state_accuracy']-.005,.5145)
            and (m['temporal_recall'] or 0)>=min(base_metrics['temporal_recall']-.005,.4506)):
            eligible.append(name)
    def key(name):
        m=candidates[name]['metrics'];t=m['temporal_recall'] or 0;s=m['state_accuracy'] or 0
        return min(t/.4506,s/.5145),t+s,f1(m)
    best=max(eligible,key=key);chosen=candidates[best]
    if best=='baseline':
        deployed=baseline
    else:
        for path in Path(chosen['model']).iterdir():
            if path.is_file():shutil.copyfile(path,output/path.name)
        config=read(output/'inference_config.json');config.update(date_pointer_mode=chosen['mode'],selected_on='full internal dev')
        write_json(output/'inference_config.json',config)
        metadata=read(output/'experiment.json')
        metadata.update(selection='generated full internal dev temporal/state metrics with fact and state safeguards',
                        selected_checkpoint=best,date_pointer_mode=chosen['mode'],external_val_used_for_selection=False)
        write_json(output/'experiment.json',metadata)
        for name in ['LICENSE','UPSTREAM_SOURCE.json']:
            shutil.copyfile(Path('artifacts/qwen_base')/name,output/name)
        shutil.copyfile('data/raw/catalog.json',output/'catalog.json')
        deployed=output
    selection={'selected':best,'deployed_model':deployed.as_posix(),'eligible':eligible,'candidates':candidates,
        'selected_on':'full account-disjoint internal dev','external_val_used_for_selection':False,
        'objective':'normalized temporal/state, recall/F1 guards and retention of previously achieved state threshold'}
    write_json('reports/date_pointer_selection.json',selection)
    shutil.copyfile(chosen['predictions'],'reports/date_pointer_dev_selected_pred.jsonl')
    common=['--engine','qwen','--model',str(deployed),'--base-model','artifacts/qwen_base',
            '--date-ranker','artifacts/date_ranker_v21.joblib']
    if best=='baseline':
        shutil.copyfile('reports/v21_val_pred.jsonl','reports/date_pointer_val_pred.jsonl')
        shutil.copyfile('reports/v21_val_metrics.json','reports/date_pointer_val_metrics.json')
        shutil.copyfile('reports/v21_val_schema.json','reports/date_pointer_val_schema.json')
    else:
        run('val_generate',['scripts/predict_experiment.py',*common,'--input','data/raw/val.jsonl',
            '--output','reports/date_pointer_val_pred.jsonl','--device','cuda','--batch-size','4'])
        run('val_score',['scripts/evaluate.py','data/raw/val.jsonl','reports/date_pointer_val_pred.jsonl','--output','reports/date_pointer_val_metrics.json'])
        run('val_schema',['scripts/check_predictions.py','data/raw/val.jsonl','reports/date_pointer_val_pred.jsonl','--output','reports/date_pointer_val_schema.json'])
    dev=read_jsonl('tmp/splits/dev.jsonl');write_jsonl('tmp/date_pointer_benchmark_calls.jsonl',dev[:24])
    for name,device,extra in [('cuda','cuda',[]),('cuda_batch4','cuda',['--batch-size','4']),('cpu','cpu',[])]:
        if best=='baseline':
            for suffix in ['pred.resources.json','metrics.json']:
                shutil.copyfile(f'reports/v21_benchmark_{name}_{suffix}',f'reports/date_pointer_benchmark_{name}_{suffix}')
            continue
        pred=f'reports/date_pointer_benchmark_{name}_pred.jsonl'
        run('benchmark_'+name,['scripts/predict_experiment.py',*common,'--input','tmp/date_pointer_benchmark_calls.jsonl',
            '--output',pred,'--device',device,*extra])
        run('benchmark_'+name+'_score',['scripts/evaluate.py','tmp/date_pointer_benchmark_calls.jsonl',pred,
            '--output',f'reports/date_pointer_benchmark_{name}_metrics.json','--repair-known-scorer-errors'])
    check_baseline()
    run('report',['scripts/write_date_pointer_report.py'])
    print('Date-pointer experiment complete:',best,flush=True)


if __name__=='__main__':main()
