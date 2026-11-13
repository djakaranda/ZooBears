"""Ablate normalization on FIXED extracted facts without using gold labels."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_jsonl,write_json
from compactfacts.generator import ValueGenerator,prompt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--pred',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='tmp/generator_dev');p.add_argument('--attributes',default='action,agreement,tariff,service_state');p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();model=ValueGenerator(a.model,a.device);attrs=set(a.attributes.split(','))
    pred={c['call_id']:c for c in read_jsonl(a.pred)};output=[];start=time.perf_counter()
    prompts=[];facts=[]
    for c in read_jsonl(a.input):
        row=pred[c['call_id']];indices={t['idx']:i for i,t in enumerate(c['turns'])}
        for f in row['facts']:
            if f['attribute'] in attrs:
                prompts.append(prompt(c,indices[f['source']['turn_idx']],f['attribute']));facts.append(f)
        output.append(row)
    for start_idx in range(0,len(prompts),a.batch_size):
        values=model.generate_prompts(prompts[start_idx:start_idx+a.batch_size])
        for f,value in zip(facts[start_idx:start_idx+a.batch_size],values):
            if value:f['value']=value
        if start_idx%320==0:print(f'{start_idx}/{len(prompts)} values',flush=True)
    write_jsonl(a.output,output);write_json(Path(a.output).with_suffix('.generation.json'),{'calls':len(output),'seconds':time.perf_counter()-start,'attributes':sorted(attrs),'device':a.device})
