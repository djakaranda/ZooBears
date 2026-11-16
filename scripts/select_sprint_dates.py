"""Compare a fixed small policy set on cached internal dev predictions."""
import argparse,copy,json,sys
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json,write_jsonl
from compactfacts.metrics import scorer_context,official
from compactfacts.scoped_dates import apply_sprint_policy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',required=True);parser.add_argument('--prefix',required=True)
    parser.add_argument('--only',nargs='+',help='Append named policies to an existing dev inventory')
    args=parser.parse_args();gold=read_jsonl('tmp/splits/dev.jsonl')
    calls={c['call_id']:{k:v for k,v in c.items() if k!='facts'} for c in gold}
    original=read_jsonl(args.predictions);model=joblib.load('artifacts/sprint_boundary.joblib')
    policies={'raw':{},'scope_tickets':{'scope':True}}
    for mode,margin in [('start',1.),('end',1.),('both',0.),('both',1.),('both',2.)]:
        policies[f'boundary_{mode}_{margin:g}']={'model':'artifacts/sprint_boundary.joblib','mode':mode,'margin':margin}
    policies['boundary_both_1_scope']={**policies['boundary_both_1'],'scope':True}
    ranges=joblib.load('artifacts/sprint_ranges.joblib')
    policies['range_estimate']={'ranges':'artifacts/sprint_ranges.joblib'}
    policies['range_boundary_2']={**policies['boundary_both_2'],'ranges':'artifacts/sprint_ranges.joblib'}
    policies['range_scope']={'scope':True,'ranges':'artifacts/sprint_ranges.joblib'}
    trees=joblib.load('artifacts/sprint_boundary_trees.joblib')
    for margin in [1.,2.]:
        policies[f'trees_both_{margin:g}']={'model':'artifacts/sprint_boundary_trees.joblib','mode':'both','margin':margin}
    policies['range_trees_2']={**policies['trees_both_2'],'ranges':'artifacts/sprint_ranges.joblib'}
    policies['calendar_exact']={'constraints':True}
    policies['calendar_trees_2']={**policies['trees_both_2'],'constraints':True}
    policies['calendar_range_trees_2']={**policies['range_trees_2'],'constraints':True}
    policies['calendar_interval_trees_2']={**policies['calendar_range_trees_2'],'range_method':'interval_offset'}
    policies['calendar_interval_linear_2']={**policies['range_boundary_2'],'constraints':True,'range_method':'interval_offset'}
    results={}
    if args.only:
        path=Path(args.prefix+'.json')
        if path.exists():results=json.loads(path.read_text(encoding='utf8'))['candidates']
        policies={name:policies[name] for name in args.only}
    for name,policy in policies.items():
        pred=copy.deepcopy(original)
        if policy:
            selected_model=trees if policy.get('model','').endswith('_trees.joblib') else model
            for p in pred:apply_sprint_policy(calls[p['call_id']],p,policy,selected_model,ranges)
        with scorer_context('tmp/splits/dev.jsonl',True):
            metrics=official.evaluate({c['call_id']:c for c in gold},{p['call_id']:p for p in pred})
        path=args.prefix+'_'+name+'_pred.jsonl';write_jsonl(path,pred)
        results[name]={'metrics':metrics,'predictions':path,'policy':policy}
        print(name,metrics,flush=True)
    write_json(args.prefix+'.json',{'selected_on':'internal dev only','candidates':results})


if __name__=='__main__':main()
