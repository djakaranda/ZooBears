"""Checks specific to saved neural artifacts, independent of validation labels."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.hybrid import HybridExtractor

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='artifacts/neural');p.add_argument('--generator')
    a=p.parse_args();model=HybridExtractor(a.model,generator=a.generator)
    call=read_jsonl('data/raw/val.jsonl')[0];clean={k:v for k,v in call.items() if k!='facts'}
    expected=model.predict(clean);injected=copy.deepcopy(clean);injected['facts']=[{'value':'UNSEEN_SECRET_LABEL'}]
    assert model.predict(injected)==expected
    before=copy.deepcopy(clean);model.predict(clean);assert clean==before
    assert model.predict({'call_id':'empty','started_at':'2026-11-14T12:00:00','turns':[]})['facts']==[]
    remapped=copy.deepcopy(clean);mapping={t['idx']:7+t['idx']*3 for t in remapped['turns']}
    for t in remapped['turns']:t['idx']=mapping[t['idx']]
    actual=model.predict(remapped)
    assert [f['source']['turn_idx'] for f in actual['facts']]==[mapping[f['source']['turn_idx']] for f in expected['facts']]
    for f in expected['facts']:
        t=next(t for t in clean['turns'] if t['idx']==f['source']['turn_idx']);lo,hi=f['source']['span']
        assert f['source']['quote']==t['text'][lo:hi]
    result={'passed':['labels_ignored','input_immutable','empty_call','nonconsecutive_turn_indices','literal_grounding'],
            'model':a.model,'generator':a.generator}
    write_json('reports/neural_contract_check.json',result);print(result)
