import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from compactfacts.qwen_format import messages, parse_rows, target_rows, windows


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        # Deterministic stand-in: exercise window coverage without model weights.
        return list(range(sum(len(m['content']) for m in messages)))


class QwenFormatTests(unittest.TestCase):
    def call(self):
        return {'call_id': 'account:1', 'started_at': '2026-11-14T15:00:00',
                'turns': [{'idx': 7, 'speaker': 'client', 'text': 'У меня тариф Гранд.'},
                          {'idx': 12, 'speaker': 'operator', 'text': 'Могу предложить выезд завтра?'}]}

    def test_input_allowlist_excludes_labels_and_future_metadata(self):
        call = self.call(); call['facts'] = [{'secret': 'gold'}]; call['future'] = 'hidden'
        payload = json.loads(messages(call, call['turns'])[1]['content'])
        self.assertEqual(set(payload), {'started_at', 'turns'})
        self.assertEqual(payload['turns'][0][0], 7)

    def test_valid_rows_ground_to_real_indices_and_quotes(self):
        call = self.call()
        rows = [[7, 'tariff', 'Гранд', 'client', 'клиент', 1]]
        candidates, invalid = parse_rows(json.dumps(rows), call, call['turns'])
        self.assertEqual(invalid, 0)
        self.assertEqual(candidates[0]['index'], 0)
        lo, hi = candidates[0]['span']
        self.assertEqual(call['turns'][0]['text'][lo:hi], 'Гранд')

    def test_invalid_indices_types_and_fields_are_rejected(self):
        call = self.call()
        rows = [[999, 'tariff', 'Гранд', 'client', 'клиент', 1],
                [True, 'tariff', 'Гранд', 'client', 'клиент', 1],
                [7, [], 'Гранд', 'client', 'клиент', 1],
                [7, 'tariff', '', 'client', 'клиент', 1],
                [7, 'tariff', 'Гранд', 'client', 'клиент', True]]
        candidates, invalid = parse_rows(json.dumps(rows), call, call['turns'])
        self.assertEqual(candidates, []); self.assertEqual(invalid, len(rows))

    def test_source_must_belong_to_current_window(self):
        call = self.call()
        candidates, invalid = parse_rows('[[7,"tariff","Гранд","client","клиент",1]]', call, [call['turns'][1]])
        self.assertEqual((candidates, invalid), ([], 1))

    def test_invalid_json_never_becomes_partial_facts(self):
        with self.assertRaises(ValueError):
            parse_rows('[[7,"tariff"', self.call(), self.call()['turns'])
        with self.assertRaises(ValueError):
            parse_rows('{"facts":[]}', self.call(), self.call()['turns'])

    def test_windows_cover_every_turn_without_silent_truncation(self):
        call = self.call()
        call['turns'] = [{'idx': i * 3 + 7, 'speaker': 'client', 'text': 'а' * 80} for i in range(12)]
        tokenizer = TinyTokenizer()
        budget = len(tokenizer.apply_chat_template(messages(call, call['turns'][:3]), enable_thinking=False))
        chunks = list(windows(call, tokenizer, budget))
        self.assertGreater(len(chunks), 1)
        self.assertEqual({t['idx'] for ts, _ in chunks for t in ts}, {t['idx'] for t in call['turns']})
        self.assertTrue(all(len(ids) <= budget for _, ids in chunks))

    def test_target_uses_only_visible_sources_and_omits_hidden_dates(self):
        call = self.call()
        fact = {'source': {'turn_idx': 7}, 'attribute': 'tariff', 'value': 'Гранд',
                'object': {'type': 'client', 'name': 'клиент'}, 'valid_from': '2000-01-01'}
        call['facts'] = [fact, fact]
        self.assertEqual(target_rows(call, call['turns']), [[7, 'tariff', 'Гранд', 'client', 'клиент', 1]])
        self.assertEqual(target_rows(call, [call['turns'][1]]), [])

    def test_malformed_other_labels_do_not_teach_hidden_account_ids(self):
        call = self.call()
        call['facts'] = [{'source': {'turn_idx': 7}, 'attribute': 'ticket_status', 'value': 'принята',
                          'object': {'id': 'u101_00020:ticket0', 'type': 'other', 'name': 'u101_00020:ticket0'}}]
        row = target_rows(call, call['turns'])[0]
        self.assertEqual(row[3:5], ['ticket', 'заявка'])
        self.assertNotIn('u101', json.dumps(row))


if __name__ == '__main__':
    unittest.main()
