from copy import deepcopy
from datetime import date
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.entities import canonicalize_objects
from compactfacts.metrics import official, scorer_context


class V21Tests(unittest.TestCase):
    def rows(self, names):
        return [{'object':{'id':str(i),'type':'device','name':name},'index':i} for i,name in enumerate(names)]

    def test_unique_aliases_join_without_mutating_input(self):
        rows=self.rows(['мой роутер','маршрутизатор','роутер TP Link']);before=deepcopy(rows)
        result=canonicalize_objects(rows,'account')
        self.assertEqual(len({r['object']['id'] for r in result}),1)
        self.assertEqual(rows,before)

    def test_generic_does_not_bridge_two_numbered_routers(self):
        result=canonicalize_objects(self.rows(['роутер','роутер 1','роутер 2']),'a')
        self.assertEqual(len({r['object']['id'] for r in result}),3)

    def test_official_crash_repair_is_opt_in_and_restored(self):
        if getattr(official,'VERSION',None) != '2.1':self.skipTest('Requires supplied v2.1')
        original=official.recover_dates
        with scorer_context(repair_known_errors=True) as audit:
            dates=official.recover_dates('через две недели',date(2026,11,15))
            self.assertIn(date(2026,11,29),dates)
            self.assertEqual(audit['recover_dates_type_errors'],1)
        self.assertIs(official.recover_dates,original)

    def test_supplement_sources_are_real_and_training_only(self):
        import json
        from compactfacts.fit_examples import targeted_examples
        from compactfacts.qwen_format import target_rows
        catalog=json.loads((Path(__file__).resolve().parents[1]/'data/raw/catalog.json').read_text(encoding='utf8'))
        calls=targeted_examples(catalog)
        self.assertEqual(len(calls),112)
        for call in calls:
            self.assertTrue(call['call_id'].startswith('fit_supplement:'))
            turns={t['idx']:t for t in call['turns']}
            self.assertEqual(len(target_rows(call,call['turns'])),len(call['facts']))
            for fact in call['facts']:
                source=fact['source'];lo,hi=source['span']
                self.assertEqual(turns[source['turn_idx']]['text'][lo:hi],source['quote'])
                if fact['valid_to']:
                    self.assertLessEqual(fact['valid_from'],fact['valid_to'])

    def test_scorer_date_cache_does_not_share_mutable_results(self):
        original=official.recover_dates
        with scorer_context():
            result=official.recover_dates('вчера',date(2026,11,15))
            expected=set(result)
            result.clear()
            self.assertEqual(official.recover_dates('вчера',date(2026,11,15)),expected)
        self.assertIs(official.recover_dates,original)

    def test_boundary_diagnostics_separate_wrong_start_from_correct_open_end(self):
        from compactfacts.date_diagnostics import boundary_metrics
        fact={'object':{'id':'x','type':'service','name':'интернет'},'attribute':'service_state',
              'value':'работает','polarity':1,'speaker':'client',
              'valid_from':'2026-11-14T00:00:00','valid_to':None}
        call={'call_id':'test','started_at':'2026-11-15T12:00:00',
              'turns':[{'idx':0,'speaker':'client','text':'Интернет работает со вчера.'}], 'facts':[fact]}
        prediction={'call_id':'test','facts':[{**deepcopy(fact),'valid_from':'2026-11-15T12:00:00'}]}
        result=boundary_metrics([call],[prediction])
        self.assertEqual(result['eligible_matched_facts'],1)
        self.assertEqual(result['temporal_recall_recomputed'],0.)
        self.assertEqual(result['fields']['valid_from']['accuracy_on_eligible'],0.)
        self.assertEqual(result['fields']['valid_to']['accuracy_on_eligible'],1.)
        self.assertEqual(result['fields']['valid_to']['recoverable_nonnull_boundaries'],0)

    def test_boundary_diagnostics_skip_unmatched_calls_like_official_scorer(self):
        from compactfacts.date_diagnostics import boundary_metrics
        call={'call_id':'empty','started_at':'2026-11-15T12:00:00','facts':[],
              'turns':[{'idx':0,'speaker':'client','text':'Через две недели.'}]}
        result=boundary_metrics([call],[{'call_id':'empty','facts':[]}])
        self.assertEqual(result['eligible_matched_facts'],0)
        self.assertIsNone(result['temporal_recall_recomputed'])

    def test_end_only_keeps_start_and_does_not_reopen_history(self):
        from compactfacts.event_dates import EventDateModel
        class Vectorizer:
            def transform(self,features):return features
        class Scores:
            def __init__(self,values):self.values=values
            def decision_function(self,features):return self.values
        call={'started_at':'2026-11-15T12:00:00',
              'turns':[{'idx':0,'speaker':'client','text':'Интернет не работал до вчера.'}]}
        fact={'attribute':'service_state','value':'не работал','polarity':1,
              'object':{'id':'x','name':'интернет','type':'service'},
              'valid_from':'2026-11-12T00:00:00','valid_to':None,
              'source':{'turn_idx':0,'span':[0,19]}}
        model=EventDateModel();model.vectorizers={'end':Vectorizer()};model.models={'end':Scores([0.,5.])}
        prediction={'facts':[deepcopy(fact)]}
        model.apply(call,prediction,end_only=True,timeline=False)
        self.assertEqual(prediction['facts'][0]['valid_from'],fact['valid_from'])
        self.assertEqual(prediction['facts'][0]['valid_to'],'2026-11-14T00:00:00')
        model.models['end']=Scores([5.,0.])
        model.apply(call,prediction,end_only=True,timeline=False)
        self.assertEqual(prediction['facts'][0]['valid_to'],'2026-11-14T00:00:00')
        prediction['facts'][0].update(valid_from=call['started_at'],valid_to=None)
        model.models['end']=Scores([0.,5.])
        model.apply(call,prediction,end_only=True,timeline=False)
        self.assertIsNone(prediction['facts'][0]['valid_to'])

    def test_wide_dates_do_not_borrow_another_ticket_or_invent_interval_day(self):
        from compactfacts.wide_dates import WideDateModel
        call={'started_at':'2026-11-15T12:00:00','turns':[
            {'idx':0,'speaker':'operator','text':'Заявка 1234 в работе.'},
            *[{'idx':i,'speaker':'client','text':'Подожду.'} for i in range(1,6)],
            {'idx':6,'speaker':'operator','text':'Заявка 9999 в работе со вчера.'},
            {'idx':7,'speaker':'operator','text':'Заявка 1234 в работе с прошлого месяца.'}]}
        fact={'attribute':'ticket_status','value':'в работе','object':{'name':'заявка 1234'},
              'source':{'span':[12,20]}}
        model=WideDateModel()
        opts=model.choices(call,0,fact)
        self.assertEqual(len(opts),1)  # only observation anchor, no invented date
        call['turns'][7]['text']='Заявка 1234 в работе со вчера.'
        opts=model.choices(call,0,fact)
        self.assertEqual(len(opts),2)
        self.assertEqual(opts[-1].date,'2026-11-14T00:00:00')


if __name__=='__main__':unittest.main()
