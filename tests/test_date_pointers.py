from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.qwen_date_format import references,messages,target_rows,parse_rows,weighted_target
from compactfacts.date_pointers import apply_date_pointers
from compactfacts.pipeline import assemble_facts


class DatePointerTests(unittest.TestCase):
    def call(self):
        call={'call_id':'test:dates','started_at':'2026-11-14T15:00:00','turns':[
            {'idx':2,'speaker':'client','text':'Со вчера интернет не работает.'},
            {'idx':8,'speaker':'operator','text':'Сегодня интернет работает нормально.'}]}
        call['facts']=[]
        for idx,value,start,end in [(2,'не работает','2026-11-13T00:00:00','2026-11-14T00:00:00'),
                                    (8,'работает нормально','2026-11-14T00:00:00',None)]:
            call['facts'].append({'object':{'id':'test:svc','type':'service','name':'интернет'},
                'attribute':'service_state','value':value,'polarity':1,'speaker':'client' if idx==2 else 'operator',
                'valid_from':start,'valid_to':end,'source':{'turn_idx':idx}})
        return call

    def test_prompt_never_reads_gold_dates_or_labels(self):
        call=self.call();before=messages(call,call['turns'])
        call['facts']=[{'secret':'DO NOT LEAK','valid_from':'1900-01-01'}]
        self.assertEqual(before,messages(call,call['turns']))
        self.assertNotIn('DO NOT LEAK',str(before))

    def test_targets_link_start_and_end_to_real_utterances(self):
        call=self.call();refs={r['ref']:r for r in references(call)}
        rows=target_rows(call,call['turns'])
        self.assertEqual(refs[rows[0][6]]['turn_idx'],2)
        self.assertEqual(refs[rows[0][7]]['turn_idx'],8)
        self.assertEqual(rows[1][7],0)
        for reference in refs.values():
            turn=next(t for t in call['turns'] if t['idx']==reference['turn_idx'])
            lo,hi=reference['span'];self.assertEqual(turn['text'][lo:hi],reference['quote'])

    def test_observation_anchor_stays_unknown_and_reference_cache_is_not_mutable(self):
        call=self.call();call['facts'][0]['valid_from']=call['started_at']
        self.assertEqual(target_rows(call,call['turns'])[0][6],0)
        refs=references(call);expected=deepcopy(refs)
        refs[0]['value']='1900-01-01T00:00:00'
        self.assertEqual(references(call),expected)

    def test_repeated_dates_do_not_expand_the_prompt_table(self):
        call=self.call()
        call['turns'] += [{'idx':20+i,'speaker':'operator','text':'Со вчера интернет не работает.'} for i in range(100)]
        self.assertEqual(len(references(call)),2)

    def test_date_weights_do_not_change_numbers_inside_fact_values(self):
        class CharacterTokenizer:
            eos_token='<eos>'
            def __call__(self,text,**kwargs):
                return {'input_ids':[ord(c) for c in text], 'offset_mapping':[(i,i+1) for i in range(len(text))]}
        row=[2,'service_state','строка ,7,8] с запятой','service','интернет',1,3,-1]
        ids,weights=weighted_target(CharacterTokenizer(),[row],3.)
        text=''.join(chr(i) for i in ids)
        self.assertEqual(weights[text.index(',7,8]')+1],1.)
        at=text.rfind(',3,-1]')
        self.assertEqual(weights[at+1],3.)
        self.assertEqual(weights[at+3],2.)
        self.assertTrue(all(w==1 for w in weights[-len('<eos>'):]))

    def test_unknown_period_does_not_fabricate_a_day(self):
        call=self.call();call['turns']=[{'idx':2,'speaker':'client','text':'В прошлом месяце интернет не работал.'}]
        call['facts']=call['facts'][:1]
        call['facts'][0].update(valid_from='2026-10-12T00:00:00',valid_to='2026-10-20T00:00:00')
        self.assertEqual(references(call),[])
        self.assertEqual(target_rows(call,call['turns'])[0][6:],[0,-1])

    def test_invalid_reference_keeps_grounded_fact_with_legacy_fallback(self):
        call=self.call();row=target_rows(call,call['turns'])[0];row[6]=999999
        candidates,invalid=parse_rows(json.dumps([row],ensure_ascii=False),call,call['turns'])
        self.assertEqual(invalid,1);self.assertEqual(len(candidates),1)
        self.assertNotIn('date_from_ref',candidates[0])

    def test_decoded_dates_correct_both_boundaries_and_keep_provenance(self):
        call=self.call();rows=target_rows(call,call['turns'])
        candidates,invalid=parse_rows(json.dumps(rows,ensure_ascii=False),call,call['turns'])
        self.assertEqual(invalid,0)
        prediction=assemble_facts(call,candidates,None,agreement_evidence=False)
        prediction['facts'][0].update(valid_from=call['started_at'],valid_to=call['started_at'])
        original=deepcopy(candidates)
        apply_date_pointers(call,prediction,candidates)
        self.assertEqual(prediction['facts'][0]['valid_from'],'2026-11-13T00:00:00')
        self.assertEqual(prediction['facts'][0]['valid_to'],'2026-11-14T00:00:00')
        self.assertEqual(prediction['date_pointer_evidence'][0]['end']['turn_idx'],8)
        self.assertEqual(candidates,original)

    def test_api_explains_linked_date_from_another_turn_without_changing_facts(self):
        from compactfacts.date_evidence import with_date_evidence
        call={'started_at':'2026-11-14T15:00:00','turns':[
            {'idx':2,'speaker':'operator','text':'Работы завершены вчера.'},
            {'idx':8,'speaker':'client','text':'Интернет работает нормально.'}]}
        ref=references(call)[0]
        fact={'object':{'id':'s','type':'service','name':'интернет'},'attribute':'service_state',
              'value':'работает нормально','valid_from':ref['value'],'valid_to':call['started_at'],
              'source':{'turn_idx':8}}
        prediction={'facts':[fact],'date_pointer_evidence':[{'object_id':'s','attribute':'service_state',
            'turn_idx':8,'start':ref,'end':None,'closed_unknown':True}]}
        before=deepcopy(prediction)
        result=with_date_evidence(call,prediction)
        self.assertEqual(result['temporal_evidence'][0]['start']['sources'][0]['turn_idx'],2)
        self.assertEqual(result['temporal_evidence'][0]['end']['kind'],'unknown')
        self.assertEqual(prediction,before)

    def test_api_keeps_month_uncertainty_when_model_links_an_incompatible_day(self):
        from compactfacts.date_evidence import with_date_evidence
        call={'started_at':'2026-11-14T15:00:00','turns':[
            {'idx':2,'speaker':'operator','text':'Сегодня проверяю оборудование.'},
            {'idx':8,'speaker':'client','text':'В прошлом месяце интернет работал нормально.'}]}
        ref=references(call)[0]
        fact={'object':{'id':'s','type':'service','name':'интернет'},'attribute':'service_state',
            'value':'работал нормально','valid_from':ref['value'],'valid_to':None,'source':{'turn_idx':8}}
        prediction={'facts':[fact],'date_pointer_evidence':[{'object_id':'s','attribute':'service_state',
            'turn_idx':8,'start':ref,'end':None,'closed_unknown':False}]}
        self.assertEqual(with_date_evidence(call,prediction)['temporal_evidence'][0]['start']['kind'],'range')

    def test_api_distinguishes_two_dated_states_in_the_same_utterance(self):
        from compactfacts.date_evidence import with_date_evidence
        call=self.call()
        call['turns']=[{'idx':2,'speaker':'client','text':'Вчера интернет не работал, а сегодня работает нормально.'}]
        call['facts'][0]['value']='не работал'
        call['facts'][1]['source']['turn_idx']=2
        rows=target_rows(call,call['turns'])
        candidates,_=parse_rows(json.dumps(rows,ensure_ascii=False),call,call['turns'])
        prediction=assemble_facts(call,candidates,None,agreement_evidence=False)
        apply_date_pointers(call,prediction,candidates)
        result=with_date_evidence(call,prediction)
        self.assertEqual([r['start'].get('value') for r in result['temporal_evidence']],
                         ['2026-11-13T00:00:00','2026-11-14T00:00:00'])

    def test_call_date_is_not_presented_as_the_start_of_an_unrelated_state(self):
        from compactfacts.date_evidence import with_date_evidence
        call={'call_id':'test:unknown','started_at':'2026-11-14T15:00:00','turns':[
            {'idx':2,'speaker':'client','text':'Сегодня звоню, интернет работает нормально.'}]}
        row=[2,'service_state','работает нормально','service','интернет',1,0,0]
        candidates,_=parse_rows(json.dumps([row],ensure_ascii=False),call,call['turns'])
        prediction=assemble_facts(call,candidates,None,agreement_evidence=False)
        apply_date_pointers(call,prediction,candidates,'full')
        result=with_date_evidence(call,prediction)
        self.assertEqual(result['temporal_evidence'][0]['start']['kind'],'unknown')
        self.assertEqual(result['temporal_evidence'][0]['start']['reason'],'start_not_resolved')

    def test_full_ends_can_preserve_or_refine_start_independently(self):
        call=self.call();row=target_rows(call,call['turns'])[0];row[7]=0
        candidates,_=parse_rows(json.dumps([row],ensure_ascii=False),call,call['turns'])
        for mode,expected_start in [('ends','2026-11-10T00:00:00'),('open_ends','2026-11-13T00:00:00')]:
            prediction=assemble_facts(call,candidates,None,agreement_evidence=False)
            prediction['facts'][0].update(valid_from='2026-11-10T00:00:00',valid_to=call['started_at'])
            apply_date_pointers(call,prediction,candidates,mode)
            self.assertEqual(prediction['facts'][0]['valid_from'],expected_start)
            self.assertIsNone(prediction['facts'][0]['valid_to'])
            evidence=prediction['date_pointer_evidence'][0]
            self.assertEqual(evidence['start'] is None,mode=='ends')

    def test_ends_only_does_not_close_intervals_from_unused_start_references(self):
        call=self.call();rows=target_rows(call,call['turns'])
        for row in rows:row[7]=0
        candidates,_=parse_rows(json.dumps(rows,ensure_ascii=False),call,call['turns'])
        prediction=assemble_facts(call,candidates,None,agreement_evidence=False)
        apply_date_pointers(call,prediction,candidates,'ends')
        self.assertTrue(all(f['valid_to'] is None for f in prediction['facts']))

    def test_spoken_visit_window_refines_unknown_pointer_metadata(self):
        call={'call_id':'test:visit','started_at':'2026-11-14T10:00:00','turns':[
            {'idx':2,'speaker':'operator','text':'Предлагаю выезд специалиста сегодня после 14:00 до 18:00.'},
            {'idx':8,'speaker':'client','text':'Да, выезд специалиста подходит.'}]}
        row=[8,'agreement','выезд специалиста','client','клиент',1,0,-1]
        candidates,_=parse_rows(json.dumps([row],ensure_ascii=False),call,call['turns'])
        prediction=assemble_facts(call,candidates,None,agreement_evidence=False)
        apply_date_pointers(call,prediction,candidates,'full')
        self.assertEqual(prediction['facts'][0]['valid_from'],'2026-11-14T14:00:00')
        self.assertEqual(prediction['facts'][0]['valid_to'],'2026-11-14T18:00:00')
        evidence=prediction['date_pointer_evidence'][0]
        self.assertFalse(evidence['start_unknown']);self.assertFalse(evidence['closed_unknown'])


if __name__=='__main__':unittest.main()
