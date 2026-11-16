"""Isolated bounded continuation and full dev selection; no automatic deployment."""
import argparse,atexit,hashlib,json,subprocess,sys,time,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        atexit.register(ctypes.windll.kernel32.SetThreadExecutionState,0x80000000)
    baseline=Path('artifacts/qwen_dates');output=Path('artifacts/qwen_sprint')
    def intact():
        assert all(hashlib.sha256((baseline/n).read_bytes()).hexdigest()==h
                   for n,h in read('reports/sprint_baseline_hashes.json').items()),'Rollback baseline changed'
    intact()
    logs=Path('tmp/sprint_logs');logs.mkdir(parents=True,exist_ok=True)
    state=Path('reports/sprint_status.json');records=read(state) if args.resume and state.exists() else []
    def run(name,command):
        if any(r['stage']==name and r['status']=='complete' for r in records):
            print('SKIP',name,flush=True);return
        item={'stage':name,'command':command,'started_at':time.time(),'status':'running'}
        records.append(item);write_json(state,records);print('START',name,flush=True)
        with (logs/(name+'.log')).open('w',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        item.update(exit_code=result.returncode,status='complete' if result.returncode==0 else 'failed',seconds=time.time()-item['started_at'])
        write_json(state,records)
        if result.returncode:
            print((logs/(name+'.log')).read_text(encoding='utf8')[-5000:],flush=True)
            raise RuntimeError(name)
        print('DONE',name,flush=True)
    run('train',['scripts/train_qwen.py','--train','tmp/sprint_fit.jsonl','--dev','tmp/splits/dev.jsonl',
        '--output',str(output),'--resume-adapter',str(baseline),'--epochs','1','--lr','2e-5',
        '--seed','73','--max-prompt-tokens','3072','--max-target-tokens','512','--dev-loss-limit','64',
        '--date-pointers','--date-token-weight','3','--checkpoint-steps','100','--augmentation-fraction','.03'])
    for filename in ['data_audit.json','history.json']:
        target=Path('reports/sprint_training_'+filename)
        if not target.exists():shutil.copyfile(output/filename,target)
    run('baseline_dates',['scripts/select_sprint_dates.py','--predictions','reports/date_pointer_dev_selected_pred.jsonl',
                         '--prefix','reports/sprint_dates_baseline'])
    candidates={('baseline/'+name):{'model':baseline.as_posix(),**item}
                for name,item in read('reports/sprint_dates_baseline.json')['candidates'].items()}
    for model in sorted((output/'checkpoints').iterdir()):
        if not (model/'adapter_model.safetensors').exists():continue
        label=model.name;pred=f'reports/sprint_dev_{label}_pred.jsonl';metrics=f'reports/sprint_dev_{label}_metrics.json'
        run('dev_generate_'+label,['scripts/predict_experiment.py','--engine','qwen','--model',str(model),
            '--base-model','artifacts/qwen_base','--input','tmp/splits/dev.jsonl','--output',pred,
            '--candidate-output',f'tmp/sprint_{label}_candidates.jsonl','--device','cuda','--batch-size','4'])
        run('dev_score_'+label,['scripts/evaluate.py','tmp/splits/dev.jsonl',pred,'--output',metrics,'--repair-known-scorer-errors'])
        prefix='reports/sprint_dates_'+label
        run('dates_'+label,['scripts/select_sprint_dates.py','--predictions',pred,'--prefix',prefix])
        for name,item in read(prefix+'.json')['candidates'].items():
            candidates[label+'/'+name]={'model':model.as_posix(),**item}
    # Reload the inventory after all stages, including any completed policy
    # expansion performed before external validation.
    candidates={('baseline/'+name):{'model':baseline.as_posix(),**item}
                for name,item in read('reports/sprint_dates_baseline.json')['candidates'].items()}
    for model in sorted((output/'checkpoints').iterdir()):
        path=Path('reports/sprint_dates_'+model.name+'.json')
        if path.exists():
            candidates.update({model.name+'/'+name:{'model':model.as_posix(),**item}
                for name,item in read(path)['candidates'].items()})
    base=candidates['baseline/raw']['metrics']
    def f1(m):return 2*m['fact_precision']*m['fact_recall']/(m['fact_precision']+m['fact_recall'])
    eligible=[n for n,c in candidates.items() if c['metrics']['fact_recall']>=base['fact_recall']*.98
        and f1(c['metrics'])>=f1(base)-.01 and c['metrics']['temporal_recall']>=base['temporal_recall']
        and c['metrics']['state_accuracy']>=.54]
    def key(n):
        m=candidates[n]['metrics'];return min(m['temporal_recall']/.4506,m['state_accuracy']/.5145),f1(m),bool(candidates[n].get('policy',{}).get('constraints'))
    selected=max(eligible,key=key)
    selection={'selected':selected,'candidates':candidates,'eligible':eligible,'checkpoint_selection':'full internal dev only',
        'deployment_changed':False,'baseline_preserved':True,'external_val_used_for_checkpoint_selection':False}
    write_json('reports/sprint_selection.json',selection)
    print('DEV SELECTED',selected,flush=True)
    if selected!='baseline/raw':
        chosen=candidates[selected]
        for p in Path(chosen['model']).iterdir():
            if p.is_file():shutil.copyfile(p,output/p.name)
        config=read(output/'inference_config.json');config.update(selected_on='full sprint internal dev',sprint_dates=chosen['policy'])
        write_json(output/'inference_config.json',config)
        metadata=read(output/'experiment.json');metadata.update(selected_checkpoint=selected,selection='Full internal dev temporal/state with recall/F1 guards',external_val_used_for_checkpoint_selection=False)
        write_json(output/'experiment.json',metadata)
        for n in ['LICENSE','UPSTREAM_SOURCE.json','catalog.json']:shutil.copyfile(baseline/n,output/n)
        identity={'selected':selected,'weights_sha256':hashlib.sha256((output/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'inference_config':config,'sources':{n:hashlib.sha256((ROOT/'src/compactfacts'/n).read_bytes()).hexdigest()
                for n in ['qwen.py','date_pointers.py','date_refinement.py','scoped_dates.py','boundary_ranker.py','range_estimator.py','calendar_constraints.py']},
            'auxiliary_models':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [chosen['policy'].get('model'),chosen['policy'].get('ranges')] if p}}
        identity_path=Path('reports/sprint_val_identity.json')
        old_identity=read(identity_path) if identity_path.exists() else None
        if old_identity!=identity and any(r['stage']=='val_generate' and r['status']=='complete' for r in records):
            stamp=str(int(time.time()))
            for filename in ['sprint_val_pred.jsonl','sprint_val_metrics.json','sprint_val_schema.json']:
                path=Path('reports')/filename
                if path.exists():shutil.copyfile(path,Path('reports')/('previous_'+stamp+'_'+filename))
            for r in records:
                if r['stage'] in {'val_generate','val_score','val_schema'}:r['status']='superseded_after_dev_policy_refresh'
            write_json(state,records)
        write_json(identity_path,identity)
        run('val_generate',['scripts/predict_experiment.py','--engine','qwen','--model',str(output),
            '--base-model','artifacts/qwen_base','--input','data/raw/val.jsonl','--output','reports/sprint_val_pred.jsonl',
            '--device','cuda','--batch-size','4'])
        run('val_score',['scripts/evaluate.py','data/raw/val.jsonl','reports/sprint_val_pred.jsonl','--output','reports/sprint_val_metrics.json'])
        run('val_schema',['scripts/check_predictions.py','data/raw/val.jsonl','reports/sprint_val_pred.jsonl','--output','reports/sprint_val_schema.json'])
    intact();print('Sprint evaluation complete; review results before deployment.',flush=True)


if __name__=='__main__':main()
