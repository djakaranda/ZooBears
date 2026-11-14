import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl
from compactfacts.neural import BACKBONE, fit_neural

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',default='tmp/splits/fit.jsonl')
    parser.add_argument('--dev',default='tmp/splits/dev.jsonl')
    parser.add_argument('--output',default='tmp/neural_dev')
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-length',type=int,default=256);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--calibration',help='Frozen internal-dev experiment.json for full-train refit')
    parser.add_argument('--backbone', default=BACKBONE, help='Upstream model or local upstream snapshot')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--asr-fraction', type=float, default=0., help='Fraction of fit calls copied with added noise')
    parser.add_argument('--asr-rate', type=float, default=.15, help='Substitution probability per eligible letter')
    parser.add_argument('--asr-clean-control', action='store_true', help='Repeat the same selected calls with their original text')
    args=parser.parse_args()
    fit_neural(read_jsonl(args.train),read_jsonl(args.dev) if args.dev!='none' else [],args.output,
               args.epochs,args.batch_size,args.max_length,args.seed,
               backbone=args.backbone,
               calibration=json.loads(Path(args.calibration).read_text(encoding='utf-8')) if args.calibration else None,
               device_name=args.device, asr_fraction=args.asr_fraction, asr_rate=args.asr_rate,
               asr_clean_control=args.asr_clean_control)
