"""Run the separately supplied regression suite against the real HTTP API."""
import argparse,json,subprocess,sys,time,zipfile
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import write_json,write_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--prefix',default='reports/sprint_new_suite_gpu')
    args=parser.parse_args()
    work=ROOT/'tmp/sprint_api_suite';work.mkdir(parents=True,exist_ok=True)
    calls=[];gold=[]
    with zipfile.ZipFile(ROOT/'data/regression/ZooBears_tests_v2_1.zip') as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('.input.jsonl'):calls.extend(json.loads(s) for s in archive.read(name).decode('utf8').splitlines() if s.strip())
            elif name.endswith('.gold.jsonl'):gold.extend(json.loads(s) for s in archive.read(name).decode('utf8').splitlines() if s.strip())
            elif name.endswith('/check_predictions.py') or name=='check_predictions.py':
                (work/'check_predictions.py').write_bytes(archive.read(name))
    assert len(calls)==len(gold)==43
    predictions=[];times=[]
    for c in calls:
        clean={k:v for k,v in c.items() if k!='facts'}
        request=Request(args.url+'/extract',data=json.dumps(clean,ensure_ascii=False).encode('utf8'),headers={'Content-Type':'application/json'})
        start=time.perf_counter()
        with urlopen(request,timeout=180) as response:predictions.append(json.load(response))
        times.append(time.perf_counter()-start)
    pred=args.prefix+'_pred.jsonl';report=args.prefix+'_report.json'
    write_jsonl(pred,predictions);write_jsonl(work/'gold.jsonl',gold)
    command=[sys.executable,str(work/'check_predictions.py'),str(work/'gold.jsonl'),pred,
        '--scorer','data/raw/case_metrics.py','--catalog','data/raw/catalog.json','--out',report]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf8')
    assert result.returncode in [0,1],result.stderr
    checked=json.loads(Path(report).read_text(encoding='utf8'))
    assert checked['strict']['total']==43
    write_json(args.prefix+'_http.json',{'calls':43,'url':args.url,'latency_mean_ms':float(np.mean(times)*1000),
        'latency_p95_ms':float(np.percentile(times,95)*1000),'unit':'HTTP request wall time, batch1, model already loaded',
        'checker_exit_code':result.returncode,'strict_all_passed':result.returncode==0,
        'generation_invalid_json_windows':sum(p.get('generation_diagnostics',{}).get('invalid_json_windows',0) for p in predictions)})
    print(json.dumps({'strict':{k:checked['strict'][k] for k in ['passed','total']},'official':checked['official'],'official_excluded_calls':checked['official_excluded_calls']},ensure_ascii=False))


if __name__=='__main__':main()
