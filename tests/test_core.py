from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.temporal import normalize_date
from compactfacts.api import validate_call


class DateTests(unittest.TestCase):
    anchor='2026-11-14T15:00:00'

    def test_day_offsets_and_weekday_direction(self):
        for text,future,expected in [
            ('со вторника',False,'2026-11-10T00:00:00'),
            ('в четверг после двух',True,'2026-11-19T14:00:00'),
            ('вчера',False,'2026-11-13T00:00:00'),
            ('пять дней назад',False,'2026-11-09T00:00:00'),
            ('через две недели',True,'2026-11-28T00:00:00'),
            ('послезавтра',True,'2026-11-16T00:00:00')]:
            with self.subTest(text=text): self.assertEqual(normalize_date(text,self.anchor,future),(expected,True))

    def test_month_year_boundary(self):
        self.assertEqual(normalize_date('месяц назад','2026-01-31T11:00:00')[0],'2025-12-31T00:00:00')
        self.assertEqual(normalize_date('месяц назад','2026-03-31T11:00:00')[0],'2026-02-28T00:00:00')

    def test_no_evidence_is_reported(self):
        self.assertFalse(normalize_date('роутер не работает',self.anchor)[1])


class ContractTests(unittest.TestCase):
    def test_labels_removed_and_nonconsecutive_indices_preserved(self):
        call={'call_id':'demo','started_at':'2026-11-14T12:00:00','turns':[{'idx':7,'speaker':'client','text':'Вчера пропал интернет'}],'facts':[{'value':'secret'}]}
        result=validate_call(call)
        self.assertNotIn('facts',result)
        self.assertEqual(result['turns'][0]['idx'],7)
        self.assertIn('facts',call)

    def test_invalid_input_rejected(self):
        for call in [{},[],{'call_id':'x','started_at':'nonsense','turns':[]}]:
            with self.subTest(call=call), self.assertRaises((ValueError,TypeError)): validate_call(call)


if __name__=='__main__': unittest.main()
