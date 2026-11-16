"""Reproduce the fit-only CPU boundary models and range statistics."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.boundary_ranker import BoundaryRanker,NonlinearBoundaryRanker
from compactfacts.range_estimator import RangeEstimator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['linear','trees','ranges','all'],default='all');args=parser.parse_args()
    fit=read_jsonl('tmp/splits/fit.jsonl')
    synthetic=[c for c in read_jsonl('tmp/sprint_fit.jsonl') if c['call_id'].startswith('sprint_synthetic_')]
    for kind,cls,path in [('linear',BoundaryRanker,'sprint_boundary'),('trees',NonlinearBoundaryRanker,'sprint_boundary_trees')]:
        if args.kind not in [kind,'all']:continue
        started=time.perf_counter();model=cls().fit(fit+synthetic)
        for estimator in model.models.values():
            if hasattr(estimator,'n_jobs'):estimator.n_jobs=1
        joblib.dump(model,'artifacts/'+path+'.joblib')
        write_json('reports/'+path+'_fit.json',{'source':'original fit plus new synthetic examples; no dev/val labels',
            'calls':len(fit+synthetic),'seconds':time.perf_counter()-started,'audit':model.audit,
            'source_sha256':hashlib.sha256((ROOT/'src/compactfacts/boundary_ranker.py').read_bytes()).hexdigest(),'inference_threads':1})
    if args.kind in ['ranges','all']:
        model=RangeEstimator().fit(fit);joblib.dump(model,'artifacts/sprint_ranges.joblib')
        write_json('reports/sprint_ranges_fit.json',model.stats)


if __name__=='__main__':main()
