"""Screen a base-model date-only verifier on a fixed dev subset, never val."""
import argparse,copy,json,random,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl
from compactfacts.qwen_date_format import references
from compactfacts.temporal_verifier import prompt_batches,apply_reply
from compactfacts.metrics import official,scorer_context


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--limit',type=int,default=64)
    parser.add_argument('--predictions',default='reports/sprint_dev_epoch-1_pred.jsonl');args=parser.parse_args()
    calls=read_jsonl('tmp/splits/dev.jsonl');pred={c['call_id']:c for c in read_jsonl(args.predictions)}
    pool=[c for c in calls if len(references(c))>=2 and sum(f['attribute']!='action' for f in pred[c['call_id']]['facts'])>=2]
    random.Random(73).shuffle(pool);gold=pool[:args.limit]
    tokenizer=AutoTokenizer.from_pretrained('artifacts/qwen_base',local_files_only=True)
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained('artifacts/qwen_base',dtype=torch.bfloat16 if args.device=='cuda' else torch.float32,
        attn_implementation='sdpa',local_files_only=True).to(args.device).eval()
    outputs={mode:[] for mode in ['starts','conservative','open_ends']};records=[];started=time.perf_counter()
    with torch.inference_mode():
        for c in gold:
            call={k:v for k,v in c.items() if k!='facts'};base=pred[c['call_id']];results={m:copy.deepcopy(base) for m in outputs}
            for selected,ids in prompt_batches(tokenizer,call,base):
                tensor=torch.tensor([ids],device=args.device)
                output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=False,
                    max_new_tokens=100,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
                text=tokenizer.decode(output[0,len(ids):],skip_special_tokens=True)
                record={'call_id':c['call_id'],'text':text}
                for mode in outputs:
                    try:results[mode],n=apply_reply(call,results[mode],text,selected,mode=mode);record[mode+'_rows']=n
                    except (ValueError,TypeError,KeyError) as e:record[mode+'_error']=str(e)
                records.append(record)
            for mode in outputs:outputs[mode].append(results[mode])
            write_json('reports/sprint_verifier_probe.progress.json',{'calls':len(outputs['starts']),'total':len(gold),'seconds':time.perf_counter()-started})
    metrics={}
    with scorer_context('tmp/splits/dev.jsonl',True):
        metrics['before']=official.evaluate({c['call_id']:c for c in gold},{c['call_id']:pred[c['call_id']] for c in gold})
        for mode in outputs:metrics[mode]=official.evaluate({c['call_id']:c for c in gold},{p['call_id']:p for p in outputs[mode]})
    for mode,p in outputs.items():write_jsonl('tmp/verifier_probe_'+mode+'.jsonl',p)
    write_json('reports/sprint_verifier_probe.json',{'subset':'64 input-selected dev cases; exploratory, not final quality','metrics':metrics,'generation_records':records,'seconds':time.perf_counter()-started})
    print(metrics,flush=True)


if __name__=='__main__':main()
