import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl
from compactfacts.generator import GENERATOR, train_generator
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',default='tmp/splits/fit.jsonl');p.add_argument('--dev',default='tmp/splits/dev.jsonl')
    p.add_argument('--output',default='tmp/generator_dev');p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device', choices=['cpu','cuda','mps'])
    p.add_argument('--backbone', default=GENERATOR)
    p.add_argument('--attribute', choices=['tariff'])
    a=p.parse_args();train_generator(read_jsonl(a.train),read_jsonl(a.dev) if a.dev!='none' else [],a.output,a.epochs,a.batch_size,
                                    device_name=a.device, backbone=a.backbone, attribute=a.attribute)
