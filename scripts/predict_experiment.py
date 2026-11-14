import argparse
from pathlib import Path
import sys,time
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--engine',choices=['temporal','neural','hybrid'],default='neural')
    parser.add_argument('--model',default='tmp/neural_dev');parser.add_argument('--normalizer',default='tmp/dev_baseline.joblib')
    parser.add_argument('--device',default='cpu');parser.add_argument('--quantized',action='store_true')
    parser.add_argument('--max-length',type=int);parser.add_argument('--mode',default='balanced')
    parser.add_argument('--generator',help='Optional generator directory for hybrid engine')
    parser.add_argument('--date-ranker', help='Optional fit-trained date ranker for hybrid engine')
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.engine=='hybrid':
        from compactfacts.hybrid import HybridExtractor
        model=HybridExtractor(args.model,args.normalizer,args.device,args.generator,quantized=args.quantized,
                              date_ranker=args.date_ranker)
    elif args.engine=='neural':
        from compactfacts.neural import NeuralExtractor
        model=NeuralExtractor(args.model,args.normalizer,args.device,args.max_length,args.quantized)
    else:
        from compactfacts.pipeline import TemporalExtractor
        model=TemporalExtractor(args.normalizer)
    calls=read_jsonl(args.input)
    if args.limit:calls=calls[:args.limit]
    times=[];memory=[];predictions=[]
    for i,c in enumerate(calls):
        clean={k:v for k,v in c.items() if k!='facts'}
        if args.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize()
        start=time.perf_counter();predictions.append(model.predict(clean,args.mode))
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        times.append(time.perf_counter()-start);memory.append(psutil.Process().memory_info().rss)
        if (i+1)%50==0:print(f'{i+1}/{len(calls)}',flush=True)
    write_jsonl(args.output,predictions)
    write_json(Path(args.output).with_suffix('.resources.json'),{'engine':args.engine,'device':args.device,
        'quantized':args.quantized,'max_length':args.max_length,'calls':len(calls),'mode':args.mode,
        'latency_mean_ms':float(np.mean(times)*1000),'latency_p50_ms':float(np.percentile(times,50)*1000),
        'latency_p95_ms':float(np.percentile(times,95)*1000),'max_sampled_rss_mb':max(memory)/1024**2,
        'gpu_peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if args.device.startswith('cuda') else None,
        'note':'Sequential batch=one whole call; within-call turns batched for neural. First call included; load and IO excluded; CUDA synchronized.'})
