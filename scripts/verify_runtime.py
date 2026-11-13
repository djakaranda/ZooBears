"""Integration verification against a real HTTP process and saved weights."""
import copy
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.model import Extractor


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--engine',choices=['baseline','hybrid'],default='baseline')
    parser.add_argument('--model',default='artifacts/cpu_baseline.joblib');parser.add_argument('--normalizer',default='artifacts/cpu_baseline.joblib')
    parser.add_argument('--generator');parser.add_argument('--output',default='reports/runtime_check.json');args=parser.parse_args()
    if args.engine=='hybrid':
        from compactfacts.hybrid import HybridExtractor
        model=HybridExtractor(args.model,args.normalizer,'cpu',args.generator)
    else:model=Extractor.load(ROOT/args.model)
    original=read_jsonl(ROOT/'data/raw/val.jsonl')[0]
    clean={k:v for k,v in original.items() if k!='facts'}
    poisoned=copy.deepcopy(clean)
    poisoned['facts']=[{'attribute':'agreement','value':'DO NOT COPY VALIDATION LABELS'}]
    expected=model.predict(clean)
    assert model.predict(poisoned)==expected
    assert model.predict({'call_id':'empty','started_at':'2026-11-14T12:00:00','turns':[]})=={'call_id':'empty','facts':[]}
    remapped=copy.deepcopy(clean)
    mapping={turn['idx']:turn['idx']*3+7 for turn in remapped['turns']}
    for turn in remapped['turns']: turn['idx']=mapping[turn['idx']]
    remapped_pred=model.predict(remapped)
    assert [f['source']['turn_idx'] for f in remapped_pred['facts']]==[mapping[f['source']['turn_idx']] for f in expected['facts']]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONIOENCODING':'utf-8'}
    command=[sys.executable,'-m','compactfacts.api','--port',str(port),'--engine',args.engine,'--model',args.model,'--normalizer',args.normalizer]
    if args.generator:command+=['--generator',args.generator]
    process=subprocess.Popen(command,cwd=ROOT,env=env,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    base=f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            if process.poll() is not None: raise RuntimeError('HTTP process exited')
            try:
                with urllib.request.urlopen(base+'/health',timeout=.5) as response:
                    assert json.load(response)['status']=='ok'
                break
            except urllib.error.URLError: time.sleep(.1)
        else: raise RuntimeError('HTTP server did not start')
        request=urllib.request.Request(base+'/extract',data=json.dumps(clean,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response: assert json.load(response)==expected
        with urllib.request.urlopen(base+'/',timeout=5) as response:
            assert response.headers['Content-Type'].startswith('text/html')
            assert b'ZooBears' in response.read()
        for payload in [b'{}',b'{broken']:
            request=urllib.request.Request(base+'/extract',data=payload,headers={'Content-Type':'application/json'})
            try:
                urllib.request.urlopen(request,timeout=5)
                raise AssertionError('Invalid input accepted')
            except urllib.error.HTTPError as error: assert error.code==400
    finally:
        process.terminate(); process.wait(timeout=10)
    write_json(ROOT/'reports/demo_input.json',clean)
    write_json(ROOT/'reports/demo_output.json',expected)
    result={'passed':['saved_model_load','gold_label_independence','empty_transcript','nonconsecutive_turn_ids',
                      'http_health','http_extraction_matches_cli','http_invalid_payloads_rejected','http_demo_page'],
            'http_process_stopped':process.poll() is not None}
    result['engine']=args.engine;result['model']=args.model
    write_json(ROOT/args.output,result)
    print(result)
