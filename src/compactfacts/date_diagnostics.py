"""Separate start/end accuracy using exactly the official temporal eligibility."""
from collections import Counter
from datetime import datetime

from .metrics import official, scorer_context


def boundary_metrics(gold, predictions, gold_path=None, repair_known_errors=False):
    predictions={c['call_id']:c for c in predictions}
    counts={field:Counter() for field in ('valid_from','valid_to')}
    eligible=both_correct=0
    with scorer_context(gold_path,repair_known_errors) as repairs:
        for call in gold:
            pairs=official.match_facts(call['facts'],predictions.get(call['call_id'],{}).get('facts',[]))
            if not any(g['attribute'] in official.TEMPORAL_ATTRS for g,_,_ in pairs):
                continue
            day=datetime.fromisoformat(call['started_at']).date()
            recovered=set()
            for turn in call['turns']:
                recovered.update(d.isoformat() for d in official.recover_dates(turn['text'],day))
            for g,p,_ in pairs:
                if g['attribute'] not in official.TEMPORAL_ATTRS:
                    continue
                gd={f:official._day(g.get(f)) for f in counts}
                pd={f:official._day(p.get(f)) for f in counts}
                if not (set(gd.values())-{None}) & recovered:
                    continue
                eligible+=1
                both_correct+=int(gd==pd)
                for field,counter in counts.items():
                    counter['eligible_facts']+=1
                    counter['correct_on_eligible']+=int(gd[field]==pd[field])
                    if gd[field] is not None and gd[field] in recovered:
                        counter['recoverable_nonnull_boundaries']+=1
                        counter['correct_recoverable_nonnull']+=int(gd[field]==pd[field])
        repair_audit=dict(repairs)
    fields={}
    for field,counter in counts.items():
        fields[field]={key:counter[key] for key in ('eligible_facts','correct_on_eligible',
                      'recoverable_nonnull_boundaries','correct_recoverable_nonnull')}
        fields[field]['accuracy_on_eligible']=round(counter['correct_on_eligible']/eligible,4) if eligible else None
        n=counter['recoverable_nonnull_boundaries']
        fields[field]['accuracy_on_recoverable_nonnull']=round(counter['correct_recoverable_nonnull']/n,4) if n else None
    return {'eligible_matched_facts':eligible,'both_correct':both_correct,
            'temporal_recall_recomputed':round(both_correct/eligible,4) if eligible else None,
            'fields':fields,'local_repairs':repair_audit,
            'comparison_unit':'calendar day, as in official v2.1',
            'note':'Eligible accuracy includes null ends. Recoverable-nonnull accuracy excludes null and dates absent from the official date bank. These diagnostics do not replace official scores.'}
