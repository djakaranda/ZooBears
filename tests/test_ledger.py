import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.ledger import TimelineLedger


def call(account,index):return {'call_id':account+':call'+str(index),'series':{'account_id':account}}
def fact(value,date,oid='object'):
    return {'object':{'id':oid,'name':'заявка 5009','type':'ticket'},'attribute':'ticket_status','value':value,
            'valid_from':date+'T00:00:00','valid_to':None,'polarity':1,'source':{'turn_idx':7}}


class LedgerTests(unittest.TestCase):
    def test_transition_across_calls_keeps_provenance(self):
        ledger=TimelineLedger();first=fact('принята','2026-11-13');second=fact('в работе','2026-11-14')
        ledger.ingest(call('a',1),{'facts':[first]});ledger.ingest(call('a',2),{'facts':[second]})
        state=ledger.snapshot()['accounts']['a']['object']
        self.assertEqual(state['current']['ticket_status']['value'],'в работе')
        self.assertEqual(state['history'][0]['valid_to'],'2026-11-14T00:00:00')
        self.assertEqual(state['history'][0]['source']['call_id'],'a:call1')
        self.assertIsNone(first['valid_to'])

    def test_accounts_are_isolated_even_if_object_id_repeats(self):
        ledger=TimelineLedger();ledger.ingest(call('a',1),{'facts':[fact('принята','2026-11-13')]})
        ledger.ingest(call('b',1),{'facts':[fact('в работе','2026-11-14')]})
        state=ledger.snapshot()['accounts'];self.assertEqual(state['a']['object']['current']['ticket_status']['value'],'принята')
        self.assertIsNone(state['a']['object']['history'][0]['valid_to'])

    def test_older_event_does_not_replace_newer_state(self):
        ledger=TimelineLedger();ledger.ingest(call('a',1),{'facts':[fact('в работе','2026-11-14')]})
        ledger.ingest(call('a',2),{'facts':[fact('принята','2026-11-10')]})
        self.assertEqual(ledger.snapshot()['accounts']['a']['object']['current']['ticket_status']['value'],'в работе')
