"""Uncertain starts must not become invented validity intervals or exact days."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from compactfacts.date_evidence import with_date_evidence


class DateEvidenceTests(unittest.TestCase):
    def example(self, text, value='Гранд', anchor='2026-11-14T15:00:00', attribute='tariff'):
        call = {'call_id': 'demo:uncertain', 'started_at': anchor,
                'turns': [{'idx': 7, 'speaker': 'client', 'text': text}]}
        prediction = {'call_id': call['call_id'], 'facts': [{
            'object': {'id': 'c', 'name': 'клиент', 'type': 'client'},
            'attribute': attribute, 'value': value, 'polarity': 1,
            'speaker': 'client', 'valid_from': anchor, 'valid_to': None,
            'source': {'turn_idx': 7, 'span': [0, len(text)], 'quote': text}}]}
        return call, prediction

    def start(self, call, prediction):
        return with_date_evidence(call, prediction)['temporal_evidence'][0]['start']

    def test_previous_month_and_competition_output_are_preserved(self):
        call, prediction = self.example('В прошлом месяце перешёл на тариф Гранд.')
        original = deepcopy(prediction)
        result = with_date_evidence(call, prediction)
        start = result['temporal_evidence'][0]['start']
        self.assertEqual((start['kind'], start['precision']), ('range', 'month'))
        self.assertEqual((start['earliest'], start['latest_exclusive']),
                         ('2026-10-01T00:00:00', '2026-11-01T00:00:00'))
        self.assertEqual(start['sources'][0]['quote'], 'В прошлом месяце')
        self.assertEqual(prediction, original)
        self.assertEqual(result['facts'], original['facts'])
        self.assertIsNone(result['facts'][0]['valid_to'])

    def test_month_boundaries_include_new_year_and_leap_year(self):
        for anchor, lower, upper in [
            ('2026-01-13T18:00:00', '2025-12-01', '2026-01-01'),
            ('2024-03-31T18:00:00', '2024-02-01', '2024-03-01'),
        ]:
            call, prediction = self.example('В прошлом месяце включили Гранд.', anchor=anchor)
            start = self.start(call, prediction)
            self.assertEqual(start['earliest'][:10], lower)
            self.assertEqual(start['latest_exclusive'][:10], upper)

    def test_last_week_is_a_calendar_period(self):
        call, prediction = self.example('На прошлой неделе включили Гранд.')
        start = self.start(call, prediction)
        self.assertEqual(start['precision'], 'week')
        self.assertEqual((start['earliest'][:10], start['latest_exclusive'][:10]),
                         ('2026-11-02', '2026-11-09'))

    def test_missing_start_is_not_observation_time(self):
        call, prediction = self.example('У меня тариф Гранд.')
        start = self.start(call, prediction)
        self.assertEqual(start['kind'], 'unknown')
        self.assertEqual(start['observed_at'], call['started_at'])
        self.assertNotIn('value', start)

    def test_spoken_day_refines_month(self):
        call, prediction = self.example('В прошлом месяце, 19 октября, включили Гранд.')
        start = self.start(call, prediction)
        self.assertEqual((start['kind'], start['value'], start['precision']),
                         ('exact', '2026-10-19T00:00:00', 'day'))

    def test_other_event_date_is_not_borrowed(self):
        call, prediction = self.example('У меня тариф Гранд. Вчера перезагрузил роутер.')
        self.assertEqual(self.start(call, prediction)['kind'], 'unknown')
        call, prediction = self.example('Гранд. В прошлом месяце перезагрузил роутер.')
        self.assertEqual(self.start(call, prediction)['kind'], 'unknown')

    def test_conflicting_dates_are_not_presented_as_exact(self):
        call, prediction = self.example('Гранд включили вчера или позавчера.')
        self.assertEqual(self.start(call, prediction)['kind'], 'unknown')

    def test_time_without_a_day_does_not_invent_today(self):
        call, prediction = self.example('Выезд специалиста после двух.', 'выезд специалиста', attribute='agreement')
        self.assertEqual(self.start(call, prediction)['kind'], 'unknown')

    def test_accepted_offer_can_supply_period_with_nonconsecutive_indices(self):
        call, prediction = self.example('Да, выезд специалиста подходит.', 'выезд специалиста', attribute='agreement')
        call['turns'].insert(0, {'idx': 2, 'speaker': 'operator',
                                 'text': 'Выезд специалиста на прошлой неделе согласовали.'})
        start = self.start(call, prediction)
        self.assertEqual(start['kind'], 'range')
        self.assertEqual(start['sources'][0]['turn_idx'], 2)

    def test_reference_labels_are_not_used(self):
        call, prediction = self.example('У меня тариф Гранд.')
        call['facts'] = [{'valid_from': '2000-01-01T00:00:00'}]
        self.assertEqual(self.start(call, prediction)['kind'], 'unknown')


if __name__ == '__main__':
    unittest.main()
