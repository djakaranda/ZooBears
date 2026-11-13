from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from make_stress import corrupt


class StressTests(unittest.TestCase):
    def test_seed_and_labels_are_preserved(self):
        calls=[{'call_id':'c','turns':[{'idx':7,'speaker':'client','text':'Интернет пропал вчера, заявка 5009'}],
                'facts':[{'value':'нет связи','valid_from':'2026-11-13T00:00:00','source':{'turn_idx':7,'span':[0,8],'quote':'Интернет'}}]}]
        a=corrupt(calls,1.,42);b=corrupt(calls,1.,42)
        self.assertEqual(a,b);self.assertNotEqual(a[0]['turns'][0]['text'],calls[0]['turns'][0]['text'])
        self.assertIn('5009',a[0]['turns'][0]['text'])
        self.assertEqual(a[0]['facts'][0]['value'],'нет связи');self.assertIsNone(a[0]['facts'][0]['source']['span'])
        self.assertEqual(calls[0]['facts'][0]['source']['quote'],'Интернет')
