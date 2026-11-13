"""Temporal ablation on fixed predictions: no extraction is repeated or changed."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_jsonl
from compactfacts.pipeline import TemporalExtractor

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--pred',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();calls=read_jsonl(args.input);pred={c['call_id']:c for c in read_jsonl(args.pred)}
    write_jsonl(args.output,[TemporalExtractor.transform({k:v for k,v in c.items() if k!='facts'},pred[c['call_id']]) for c in calls])
