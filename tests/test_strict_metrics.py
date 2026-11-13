from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_strict import pairs,evaluate


def fact(name,value,oid='o'):
    return {'object':{'name':name,'id':oid},'attribute':'service_state','value':value,'polarity':1,
            'valid_from':'2026-11-14T00:00:00','valid_to':None,'source':{'turn_idx':0}}


class StrictMetricTests(unittest.TestCase):
    def test_different_names_do_not_match_even_same_first_letter(self):
        self.assertEqual(pairs([fact('домашний интернет','нет связи')],[fact('домофон','нет связи')]),[])

    def test_one_prediction_cannot_cover_two_objects(self):
        gold=[{'call_id':'c','facts':[fact('интернет','нет связи','o1'),fact('телевизор','работает нормально','o2')]}]
        pred=[{'call_id':'c','facts':[fact('интернет','нет связи')]}]
        result=evaluate(gold,pred)
        self.assertEqual(result['state_accuracy_per_object'],.5)
        self.assertEqual(result['fact_recall'],.5)


if __name__=='__main__':unittest.main()
