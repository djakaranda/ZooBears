"""Reproducible additional ASR-like corruption; not the jury adversarial split."""
import argparse
import copy
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_jsonl,write_json

CONFUSIONS={'о':'а','а':'о','е':'и','и':'е','т':'д','д':'т','п':'б','б':'п','с':'з','з':'с','в':'ф','ф':'в'}


def corrupt(calls,rate=.08,seed=43):
    rng=random.Random(seed);output=copy.deepcopy(calls)
    for call in output:
        for turn in call['turns']:
            text=[]
            for char in turn['text']:
                if char.lower() in CONFUSIONS and rng.random()<rate:
                    if rng.random()<.2:continue
                    replacement=CONFUSIONS[char.lower()]
                    text.append(replacement.upper() if char.isupper() else replacement)
                else:text.append(char)
            turn['text']=''.join(text)
        for fact in call.get('facts',[]):
            # Character boundaries no longer represent the mutated input. Preserve
            # turn grounding and semantic/date labels, disable only span scoring.
            fact['source']['span']=None;fact['source']['quote']=None
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--rate',type=float,default=.08);parser.add_argument('--seed',type=int,default=43)
    args=parser.parse_args();calls=read_jsonl(args.input);write_jsonl(args.output,corrupt(calls,args.rate,args.seed))
    write_json(Path(args.output).with_suffix('.noise.json'),{'seed':args.seed,'additional_character_error_rate':args.rate,'calls':len(calls),
        'note':'Local synthetic stress; labels unchanged except source span/quote cleared; no comparison to jury hidden ASR split.'})
