import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl
from compactfacts.neural import fit_neural

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',default='tmp/splits/fit.jsonl')
    parser.add_argument('--dev',default='tmp/splits/dev.jsonl')
    parser.add_argument('--output',default='tmp/neural_dev')
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-length',type=int,default=256);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--calibration',help='Frozen internal-dev experiment.json for full-train refit')
    args=parser.parse_args()
    fit_neural(read_jsonl(args.train),read_jsonl(args.dev) if args.dev!='none' else [],args.output,
               args.epochs,args.batch_size,args.max_length,args.seed,
               calibration=json.loads(Path(args.calibration).read_text(encoding='utf-8')) if args.calibration else None)
