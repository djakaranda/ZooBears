import argparse
from pathlib import Path
import sys,time
import numpy as np
import psutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl


def process_peak_rss_mb():
    info=psutil.Process().memory_info()
    if hasattr(info,'peak_wset'):
        return info.peak_wset/1024**2
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak/(1024**2 if sys.platform=='darwin' else 1024)
    except ImportError:
        return None


def hardware_snapshot(device):
    snapshot={}
    try:
        battery=psutil.sensors_battery()
        snapshot['power_plugged']=battery.power_plugged if battery else None
    except (AttributeError,NotImplementedError):
        snapshot['power_plugged']=None
    if device.startswith('cuda'):
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu',
                                   '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
            snapshot['nvidia_smi']=result.stdout.strip()
        except (OSError,subprocess.TimeoutExpired):
            snapshot['nvidia_smi']=None
    return snapshot

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--engine',choices=['temporal','neural','hybrid','qwen'],default='neural')
    parser.add_argument('--model',default='tmp/neural_dev');parser.add_argument('--normalizer',default='tmp/dev_baseline.joblib')
    parser.add_argument('--device',default='cpu');parser.add_argument('--quantized',action='store_true')
    parser.add_argument('--max-length',type=int);parser.add_argument('--mode',default='balanced')
    parser.add_argument('--generator',help='Optional generator directory for hybrid engine')
    parser.add_argument('--date-ranker', help='Optional fit-trained date ranker for hybrid engine')
    parser.add_argument('--date-evidence', action='store_true', help='Include uncertain-start metadata; keeps competition facts unchanged')
    parser.add_argument('--base-model',default='artifacts/qwen_base')
    parser.add_argument('--candidate-output',help='Optional Qwen candidate cache for dev-only postprocessing selection')
    parser.add_argument('--batch-size',type=int,default=1,help='Qwen bulk call batch; latency reported amortized when >1')
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.engine=='qwen':
        from compactfacts.qwen import QwenExtractor
        model=QwenExtractor(args.model,args.base_model,args.device,args.date_ranker,quantized=args.quantized)
    elif args.engine=='hybrid':
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
    hardware_start=hardware_snapshot(args.device)
    times=[];memory=[];predictions=[];candidate_cache=[];generation_errors=[]
    if args.batch_size<1 or (args.batch_size!=1 and args.engine!='qwen'):
        raise ValueError('Bulk batching requires Qwen and positive batch size')
    next_progress=50
    progress={'processed_calls':0,'total_calls':len(calls),'complete':False}
    for begin in range(0,len(calls),args.batch_size):
        batch_calls=calls[begin:begin+args.batch_size]
        clean=[{k:v for k,v in c.items() if k!='facts'} for c in batch_calls]
        if args.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize()
        start=time.perf_counter()
        batch_predictions=model.predict_batch(clean,args.mode,batch_size=args.batch_size) if args.engine=='qwen' else [model.predict(clean[0],args.mode)]
        for c,clean_call,prediction in zip(batch_calls,clean,batch_predictions):
            if args.date_evidence:
                from compactfacts.date_evidence import with_date_evidence
                prediction=with_date_evidence(clean_call,prediction)
            predictions.append(prediction)
            if args.engine=='qwen':
                generation_errors.extend({'call_id':c['call_id'],**error} for error in model.last_generation_errors_by_call[c['call_id']])
            if args.candidate_output:
                if args.engine!='qwen':raise ValueError('--candidate-output requires --engine qwen')
                candidate_cache.append({'call_id':c['call_id'],'candidates':model.last_candidates_by_call[c['call_id']]})
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        times.extend([elapsed/len(batch_calls)]*len(batch_calls));memory.append(psutil.Process().memory_info().rss)
        progress={'processed_calls':len(predictions),'total_calls':len(calls),
                  'facts':sum(len(p['facts']) for p in predictions),
                  'invalid_json_windows':sum(p.get('generation_diagnostics',{}).get('invalid_json_windows',0) for p in predictions),
                  'mean_amortized_ms':float(np.mean(times)*1000),'complete':False}
        write_json(Path(args.output).with_suffix('.progress.json'),progress)
        if len(predictions)>=next_progress:
            next_progress+=50
            if args.engine=='qwen':
                bad=sum(p['generation_diagnostics']['invalid_json_windows'] for p in predictions)
                invalid_rows=sum(p['generation_diagnostics']['invalid_rows'] for p in predictions)
                print(f'{len(predictions)}/{len(calls)} facts={sum(len(p["facts"]) for p in predictions)} invalid_json_windows={bad} invalid_rows={invalid_rows} mean_ms={np.mean(times)*1000:.1f}',flush=True)
            else:print(f'{len(predictions)}/{len(calls)}',flush=True)
    write_jsonl(args.output,predictions)
    if args.candidate_output:write_jsonl(args.candidate_output,candidate_cache)
    if args.engine=='qwen':write_jsonl(Path(args.output).with_suffix('.generation_errors.jsonl'),generation_errors)
    write_json(Path(args.output).with_suffix('.progress.json'),{**progress,'complete':True})
    write_json(Path(args.output).with_suffix('.resources.json'),{'engine':args.engine,'device':args.device,
        'quantized':args.quantized,'max_length':args.max_length,'calls':len(calls),'mode':args.mode,
        'quantization_scheme':'dynamic_int8_per_channel' if args.engine=='qwen' and args.quantized else None,
        'base_parameters':getattr(model,'base_parameters',None),
        'batch_size':args.batch_size,
        'latency_unit':'request wall time' if args.batch_size==1 else 'amortized batch wall time per call; not online request latency',
        'hardware_start':hardware_start,'hardware_end':hardware_snapshot(args.device),
        'latency_mean_ms':float(np.mean(times)*1000),'latency_p50_ms':float(np.percentile(times,50)*1000),
        'calls_per_hour':3600/float(np.mean(times)),
        'wall_seconds_per_1000_calls':float(np.mean(times))*1000,
        'gpu_seconds_per_1000_calls_proxy':float(np.mean(times))*1000 if args.device.startswith('cuda') else None,
        'latency_p95_ms':float(np.percentile(times,95)*1000),'max_sampled_rss_mb':max(memory)/1024**2,
        'process_peak_rss_mb':process_peak_rss_mb(),
        'gpu_peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if args.device.startswith('cuda') else None,
        'note':f'Call batch={args.batch_size}; Qwen batches independent windows with left padding. For batch>1 reported time/call is amortized wall time, not online latency. First call included; latency excludes load and IO; CUDA synchronized. Process peak RSS includes model loading; sampled RSS measured after calls. GPU seconds proxy is synchronized wall time on a dedicated GPU, not a kernel profiler measurement.'})
