from copy import deepcopy
import unittest

from compactfacts.date_refinement import refine_date_evidence
from compactfacts.date_evidence import with_date_evidence
from compactfacts.qwen_date_format import references


class DateRefinementTests(unittest.TestCase):
    def visit(self):
        call={'started_at':'2026-11-14T15:00:00','turns':[
            {'idx':2,'speaker':'operator','text':'Давайте оформим выезд специалиста в четверг после двух до 18:00.'},
            {'idx':8,'speaker':'client','text':'Да, выезд специалиста в четверг подходит.'},
            {'idx':14,'speaker':'operator','text':'Отлично, фиксирую выезд мастера в четверг после двух.'}]}
        ref=next(r for r in references(call) if r['value']=='2026-11-19T14:00:00')
        fact={'object':{'id':'c','type':'client','name':'клиент'},'attribute':'agreement',
              'value':'выезд специалиста','polarity':1,'speaker':'operator',
              'valid_from':ref['value'],'valid_to':None,'source':{'turn_idx':14}}
        pointer={'object_id':'c','attribute':'agreement','value':fact['value'],'polarity':1,
                 'turn_idx':14,'start':ref,'end':None,'closed_unknown':False}
        return call,{'facts':[fact],'date_pointer_evidence':[pointer]}

    def test_operator_reconfirmation_preserves_the_accepted_offer_window(self):
        call,prediction=self.visit()
        refine_date_evidence(call,prediction)
        self.assertEqual(prediction['facts'][0]['valid_to'],'2026-11-19T18:00:00')
        end=prediction['date_pointer_evidence'][0]['end']
        self.assertEqual(end['turn_idx'],2)
        lo,hi=end['span'];self.assertEqual(call['turns'][0]['text'][lo:hi],end['quote'])
        before=deepcopy(prediction)
        refine_date_evidence(call,prediction)
        self.assertEqual(prediction,before)

    def test_rejected_offer_does_not_supply_an_end(self):
        call,prediction=self.visit()
        call['turns'][1]['text']='Нет, выезд не подходит.'
        refine_date_evidence(call,prediction)
        self.assertIsNone(prediction['facts'][0]['valid_to'])

    def test_a_new_visit_window_prevents_inheriting_the_old_offer(self):
        call,prediction=self.visit()
        call['turns'].insert(2,{'idx':10,'speaker':'operator',
            'text':'Переносим выезд на четверг после 15:00 до 19:00.'})
        refine_date_evidence(call,prediction)
        self.assertIsNone(prediction['facts'][0]['valid_to'])

    def test_a_date_of_one_state_does_not_prove_the_start_of_another(self):
        text='Заявка принята вчера, потом в работе.'
        call={'started_at':'2026-11-14T15:00:00','turns':[{'idx':8,'speaker':'operator','text':text}]}
        ref=references(call)[0]
        facts=[];pointers=[]
        for value,start_ref in [('принята',ref),('в работе',None)]:
            at=text.index(value)
            facts.append({'object':{'id':'t','type':'ticket','name':'заявка'},'attribute':'ticket_status',
                          'value':value,'polarity':1,'speaker':'operator','valid_from':ref['value'],
                          'valid_to':None,'source':{'turn_idx':8,'span':[at,at+len(value)]}})
            pointers.append({'object_id':'t','attribute':'ticket_status','value':value,'polarity':1,
                             'turn_idx':8,'start':start_ref,'end':None,'closed_unknown':False})
        prediction={'facts':facts,'date_pointer_evidence':pointers};before=deepcopy(facts)
        refine_date_evidence(call,prediction)
        self.assertEqual(prediction['facts'],before)
        evidence=with_date_evidence(call,prediction)['temporal_evidence']
        self.assertEqual(evidence[0]['start']['kind'],'exact')
        self.assertEqual(evidence[1]['start']['kind'],'unknown')


if __name__=='__main__':unittest.main()
