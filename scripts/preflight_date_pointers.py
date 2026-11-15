"""Check every date-reference training target before allocating GPU training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from transformers import AutoTokenizer
import train_qwen as trainer
from compactfacts import qwen_date_format as fmt
from compactfacts.io import read_jsonl,write_json
from compactfacts.fit_examples import targeted_examples
from compactfacts.asr import augment_calls


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-prompt-tokens',type=int,default=3072)
    args=parser.parse_args()
    trainer.target_rows,trainer.windows=fmt.target_rows,fmt.windows
    fit=read_jsonl('tmp/splits/fit.jsonl')
    extra=augment_calls(fit,.08,.12,42)+targeted_examples(json.loads(Path('data/raw/catalog.json').read_text(encoding='utf8')),42)
    dev=read_jsonl('tmp/splits/dev.jsonl')
    tokenizer=AutoTokenizer.from_pretrained('artifacts/qwen_base',local_files_only=True)
    fit_data=trainer.dataset(fit+extra,tokenizer,args.max_prompt_tokens,512,3.)
    dev_data=trainer.dataset(dev,tokenizer,args.max_prompt_tokens,512,3.)
    rows=[row for c in fit+extra for row in fmt.target_rows(c,c['turns'])]
    audit={'fit_calls':len(fit),'extra_calls':len(extra),'fit_windows':len(fit_data),'dev_windows':len(dev_data),
        'fit_tokens':sum(len(s['input']) for s in fit_data),'max_sequence_tokens':max(len(s['input']) for s in fit_data+dev_data),
        'max_target_tokens':max(len(s['input'])-s['prefix'] for s in fit_data+dev_data),
        'max_date_references':max(len(fmt.references(c)) for c in fit+extra+dev),
        'positive_start_references':sum(r[6]>0 for r in rows),'positive_end_references':sum(r[7]>0 for r in rows),
        'closed_unknown_ends':sum(r[7]==-1 for r in rows),'truncated_examples':0,'external_val_read':False}
    write_json('reports/date_pointer_preflight.json',audit)
    print(audit,flush=True)


if __name__=='__main__':main()
