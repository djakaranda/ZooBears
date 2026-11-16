from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.temporal_verifier import prompt_batches,apply_reply


class TemporalVerifierTests(unittest.TestCase):
    def example(self):
        call={'call_id':'v:1','started_at':'2026-11-14T15:00:00','turns':[
            {'idx':2,'speaker':'client','text':'Со вчера домашний интернет не работает.'}]}
        p={'call_id':'v:1','facts':[{'object':{'id':'s','type':'service','name':'домашний интернет'},
            'attribute':'service_state','value':'не работает','polarity':1,'speaker':'client',
            'source':{'turn_idx':2,'span':[26,37]},'valid_from':call['started_at'],'valid_to':None}]}
        return call,p

    def test_input_fact_dates_and_labels_are_never_sent_to_verifier(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.messages=deepcopy(messages);return [1]*300
        call,p=self.example();tokenizer=Tokenizer();list(prompt_batches(tokenizer,call,p));before=tokenizer.messages
        p['facts'][0].update(valid_from='1900-01-01T00:00:00',valid_to='1900-02-01T00:00:00')
        call['facts']=[{'secret':'gold label'}]
        list(prompt_batches(tokenizer,call,p));self.assertEqual(tokenizer.messages,before)

    def test_date_refinement_keeps_fact_identity_value_and_source(self):
        call,p=self.example();before=deepcopy(p)
        result,n=apply_reply(call,p,'[[0,1,0]]',[(0,p['facts'][0])])
        self.assertEqual(n,1);self.assertEqual(result['facts'][0]['valid_from'],'2026-11-13T00:00:00')
        for key in ['object','attribute','value','polarity','speaker','source']:
            self.assertEqual(result['facts'][0][key],before['facts'][0][key])
        self.assertEqual(p,before)

    def test_invalid_references_preserve_the_original_prediction(self):
        call,p=self.example();result,n=apply_reply(call,p,'[[0,999999,0]]',[(0,p['facts'][0])])
        self.assertEqual(n,0);self.assertEqual(result,p)


if __name__=='__main__':unittest.main()
