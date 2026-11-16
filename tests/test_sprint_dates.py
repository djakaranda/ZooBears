from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.boundary_ranker import BoundaryRanker
from compactfacts.scoped_dates import apply_scoped_dates
from compactfacts.range_estimator import RangeEstimator
from compactfacts.date_evidence import with_date_evidence


def example():
    call={'call_id':'test:ranker','started_at':'2026-11-14T15:00:00','turns':[
        {'idx':2,'speaker':'operator','text':'Заявка 7001 принята пять дней назад.'},
        {'idx':8,'speaker':'operator','text':'Вчера заявка 7001 в работе.'}],'facts':[]}
    for idx,value,start,end in [(2,'принята','2026-11-09T00:00:00','2026-11-13T00:00:00'),(8,'в работе','2026-11-13T00:00:00',None)]:
        call['facts'].append({'object':{'id':'t','type':'ticket','name':'заявка 7001'},'attribute':'ticket_status',
            'value':value,'polarity':1,'valid_from':start,'valid_to':end,'source':{'turn_idx':idx}})
    return call


class SprintDateTests(unittest.TestCase):
    def test_unobservable_closed_end_is_skipped_not_relabelled_open(self):
        call=example();other=deepcopy(call);other['facts'][0]['valid_to']='2026-11-11T00:00:00'
        model=BoundaryRanker().fit([call,other])
        self.assertEqual(model.audit['end_unobservable_skipped'],1)
        self.assertEqual(model.audit['end_facts'],3)

    def test_reference_bank_ignores_gold_dates(self):
        call=example();model=BoundaryRanker();before=model.bank(call)
        call['facts'][0]['valid_from']='1990-01-01T00:00:00'
        self.assertEqual(model.bank(call),before)

    def test_year_is_not_mistaken_for_another_ticket_number(self):
        call=example();call['turns'].append({'idx':14,'speaker':'operator','text':'Дата обращения 12 апреля 2026 года.'})
        model=BoundaryRanker();model._prepare(call,call['facts'])
        options=model.options(call,call['facts'][0],'start',model.bank(call))
        self.assertTrue(any(o.date and o.date.startswith('2026-04-12') for o in options))
        call['turns'][-1]['text']='Заявка 8999 принята 12 апреля 2026 года.'
        self.assertFalse(any(o.index==2 for o in model.options(call,call['facts'][0],'start',model.bank(call))))

    def test_clause_dates_close_only_the_same_object_chain(self):
        call=example();text='Заявка 7001 принята пять дней назад, а вчера в работе.'
        call['turns']=[{'idx':2,'speaker':'operator','text':text}]
        facts=deepcopy(call['facts'])
        for fact in facts:
            at=text.index(fact['value']);fact['source']={'turn_idx':2,'span':[at,at+len(fact['value'])]}
            fact.update(valid_from=call['started_at'],valid_to=None)
        prediction={'facts':facts};apply_scoped_dates(call,prediction,chains=True)
        self.assertEqual(facts[0]['valid_from'],'2026-11-09T00:00:00')
        self.assertEqual(facts[0]['valid_to'],'2026-11-13T00:00:00')
        self.assertEqual(facts[1]['valid_from'],'2026-11-13T00:00:00')

    def range_example(self):
        call=example();call['started_at']='2026-11-18T15:00:00'
        call['turns']=[{'idx':2,'speaker':'operator','text':'Заявка 7001 принята на прошлой неделе.'}]
        fact=deepcopy(call['facts'][0]);fact.update(valid_from=call['started_at'],valid_to=None)
        pointer={'object_id':'t','attribute':'ticket_status','turn_idx':2,'value':'принята','polarity':1,
                 'start':None,'end':None,'start_unknown':False,'closed_unknown':False}
        model=RangeEstimator();model.stats={'week/previous/ticket_status':{'total':20,'inside':20,'delta_mode':-7,'offset_mode':2}}
        return call,{'facts':[fact],'date_pointer_evidence':[pointer]},model

    def test_range_point_estimate_remains_explicitly_uncertain_in_api(self):
        call,prediction,model=self.range_example();model.apply(call,prediction)
        self.assertEqual(prediction['facts'][0]['valid_from'],'2026-11-11T00:00:00')
        evidence=with_date_evidence(call,prediction)['temporal_evidence'][0]['start']
        self.assertEqual(evidence['kind'],'range')
        self.assertEqual(evidence['estimate_method'],'fit_modal_offset')
        self.assertLessEqual(evidence['earliest'],evidence['point_estimate'])
        self.assertLess(evidence['point_estimate'],evidence['latest_exclusive'])

    def test_range_policy_preserves_existing_in_range_date_and_requires_support(self):
        call,prediction,model=self.range_example();prediction['facts'][0]['valid_from']='2026-11-10T00:00:00'
        model.apply(call,prediction)
        self.assertEqual(prediction['facts'][0]['valid_from'],'2026-11-10T00:00:00')
        prediction['facts'][0]['valid_from']=call['started_at'];model.stats['week/previous/ticket_status']['total']=5
        model.apply(call,prediction)
        self.assertEqual(prediction['facts'][0]['valid_from'],call['started_at'])


if __name__=='__main__':unittest.main()
