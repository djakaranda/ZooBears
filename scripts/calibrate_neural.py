"""Expand calibration to cover the learned logit range on the frozen train dev."""
import argparse,json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.neural import NeuralExtractor,TurnDataset,collate,validate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='tmp/neural_dev');p.add_argument('--dev',default='tmp/splits/dev.jsonl')
    args=p.parse_args();m=NeuralExtractor(args.model,'tmp/dev_baseline.joblib','cuda')
    dataset=TurnDataset(read_jsonl(args.dev),m.tokenizer,m.vocabulary,m.max_length)
    loader=DataLoader(dataset,batch_size=64,collate_fn=collate,num_workers=0)
    thresholds,metrics=validate(m.model,loader,m.device)
    info=m.metadata;info['thresholds_before_recalibration']=info['thresholds'];info['thresholds']=thresholds
    info['recalibration']={'split':'frozen internal train dev','logit_grid':[-4,6,.2],'metrics':metrics}
    write_json(Path(args.model)/'experiment.json',info);write_json('reports/neural_calibration.json',info['recalibration'])
    print(thresholds);print(metrics)
