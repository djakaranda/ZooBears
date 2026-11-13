"""Sequential fixed-input runs; invoke only after all training has completed."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw/val.jsonl');p.add_argument('--generator',default='artifacts/generator')
    p.add_argument('--limit',type=int);args=p.parse_args()
    configurations=[('encoder_cpu',['--engine','hybrid','--device','cpu']),
                    ('encoder_int8',['--engine','hybrid','--device','cpu','--quantized']),
                    ('encoder_gpu',['--engine','hybrid','--device','cuda']),
                    ('quality_gpu',['--engine','hybrid','--device','cuda','--generator',args.generator])]
    results={}
    for name,extra in configurations:
        output=f'reports/val_{name}.jsonl'
        command=[sys.executable,'scripts/predict_experiment.py','--input',args.input,'--output',output,
                 '--model','artifacts/neural','--normalizer','artifacts/cpu_baseline.joblib']+extra
        if args.limit:command+=['--limit',str(args.limit)]
        subprocess.run(command,cwd=ROOT,check=True)
        if not args.limit:
            subprocess.run([sys.executable,'scripts/evaluate.py',args.input,output,'--output',f'reports/val_{name}_metrics.json'],cwd=ROOT,check=True)
            subprocess.run([sys.executable,'scripts/evaluate_strict.py',args.input,output,'--output',f'reports/val_{name}_strict.json'],cwd=ROOT,check=True)
            subprocess.run([sys.executable,'scripts/check_predictions.py',args.input,output,'--output',f'reports/val_{name}_schema.json'],cwd=ROOT,check=True)
        results[name]=json.loads((ROOT/Path(output).with_suffix('.resources.json')).read_text(encoding='utf-8'))
    (ROOT/'reports/resource_curve.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
