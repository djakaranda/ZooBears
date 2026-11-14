import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from compactfacts.asr import augment_calls
from compactfacts.temporal_v2 import find_dates


class ASRTrainingTests(unittest.TestCase):
    def test_augmentation_preserves_evidence_and_semantics(self):
        text = 'Я больше не пользуюсь интернетом со вторника после двух.'
        quote = 'больше не пользуюсь интернетом'
        start = text.index(quote)
        call = {
            'call_id': 'test:1',
            'started_at': '2026-11-14T15:00:00',
            'turns': [{'idx': 7, 'speaker': 'client', 'text': text}],
            'facts': [{'value': 'домашний интернет', 'polarity': -1,
                       'valid_from': '2026-11-10T14:00:00', 'valid_to': None,
                       'source': {'turn_idx': 7, 'span': [start, start + len(quote)], 'quote': quote}}],
        }
        original = copy.deepcopy(call)
        augmented = augment_calls([call], 1., 1., 42)
        self.assertEqual(augmented, augment_calls([call], 1., 1., 42))
        clean = augment_calls([call], 1., 1., 42, clean_control=True)
        self.assertEqual([c['call_id'] for c in augmented], [c['call_id'] for c in clean])
        self.assertEqual(clean, [original])
        self.assertEqual(call, original)
        self.assertEqual(len(augmented), 1)
        noisy = augmented[0]['turns'][0]['text']
        self.assertNotEqual(noisy, text)
        self.assertIn('больше не', noisy)
        self.assertEqual(find_dates(noisy, call['started_at']), find_dates(text, call['started_at']))
        fact = augmented[0]['facts'][0]
        self.assertEqual({k: v for k, v in fact.items() if k != 'source'},
                         {k: v for k, v in call['facts'][0].items() if k != 'source'})
        source = fact['source']
        self.assertEqual(source['turn_idx'], 7)
        lo, hi = source['span']
        self.assertEqual(source['quote'], noisy[lo:hi])


if __name__ == '__main__':
    unittest.main()
