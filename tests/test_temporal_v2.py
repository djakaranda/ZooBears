import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.temporal_v2 import find_dates,choose_date
from compactfacts.pipeline import assemble_facts


class TimeEvidenceTests(unittest.TestCase):
    anchor='2026-11-14T15:00:00'

    def test_two_events_have_separate_dates(self):
        text='Вчера роутер перезагрузил. Завтра в 16:30 приедет мастер.'
        dates=find_dates(text,self.anchor,True)
        self.assertEqual([d.value for d in dates],['2026-11-13T00:00:00','2026-11-15T16:30:00'])
        first=choose_date(text,self.anchor,'action',[6,24])[0]
        second=choose_date(text,self.anchor,'agreement',[40,54])[0]
        self.assertEqual(first[:10],'2026-11-13');self.assertEqual(second[:10],'2026-11-15')

    def test_until_is_not_used_as_start(self):
        start,end,found=choose_date('Заявка была в работе до пятницы.',self.anchor,'ticket_status')
        self.assertIsNone(start);self.assertFalse(found);self.assertEqual(end,'2026-11-13T00:00:00')

    def test_calendar_and_relative_boundaries(self):
        self.assertEqual(find_dates('месяц назад','2026-03-31T12:00:00')[0].value,'2026-02-28T00:00:00')
        self.assertEqual(find_dates('через два месяца','2026-11-30T12:00:00')[0].value,'2027-01-30T00:00:00')
        self.assertEqual(find_dates('20 ноября',self.anchor)[0].value,'2026-11-20T00:00:00')

    def test_ambiguous_period_retains_interval(self):
        mention=find_dates('в прошлом месяце',self.anchor)[0]
        self.assertIsNone(mention.value);self.assertEqual(mention.interval,('2026-10-01T00:00:00','2026-11-01T00:00:00'))

    def test_no_false_weekday_or_clock(self):
        self.assertEqual(find_dates('Средства списаны в 2 раза больше. Заявка 5009.',self.anchor),[])
        self.assertEqual(find_dates('в 29:70 придут',self.anchor),[])

    def test_asr_alias_keeps_original_offsets(self):
        text='Зафтра после двух приедет мастер'
        mention=find_dates(text,self.anchor,True)[0]
        self.assertEqual(mention.value,'2026-11-15T14:00:00')
        self.assertEqual(text[mention.start:mention.end],'Зафтра после двух')
        self.assertEqual(find_dates('со фторника',self.anchor)[0].value,'2026-11-10T00:00:00')

    def test_agreement_window_keeps_both_times(self):
        start, end, _ = choose_date('В четверг после двух до 18:00', self.anchor, 'agreement')
        self.assertEqual((start, end), ('2026-11-19T14:00:00', '2026-11-19T18:00:00'))
        self.assertEqual(find_dates('Я не вчера родился.', self.anchor), [])

    def test_date_stays_with_the_matching_event(self):
        text = 'Заявка принята сегодня. Подтверждение по смс придёт через пять дний.'
        start = text.index('Подтверждение')
        end = text.index(' придёт', start)
        date, _, _ = choose_date(text, self.anchor, 'agreement', [start, end])
        self.assertEqual(date, '2026-11-19T00:00:00')


class TransitionTests(unittest.TestCase):
    def call(self,texts):
        return {'call_id':'test:1','started_at':'2026-11-14T15:00:00',
            'turns':[{'idx':i*3+7,'speaker':speaker,'text':text} for i,(speaker,text) in enumerate(texts)]}

    def test_distinct_objects_do_not_close_each_other(self):
        call=self.call([('client','Вчера интернет пропал'),('client','Сегодня телевизор работает')])
        candidates=[{'index':i,'attribute':'service_state','value':v,'object':{'id':oid,'type':'service','name':oid}}
                    for i,v,oid in [(0,'нет связи','internet'),(1,'работает нормально','tv')]]
        result=assemble_facts(call,candidates,None)
        self.assertTrue(all(f['valid_to'] is None for f in result['facts']))

    def test_successor_closes_history(self):
        call=self.call([('operator','Вчера заявка принята'),('operator','Сегодня заявка в работе')])
        obj={'id':'ticket1','type':'ticket','name':'заявка 5009'}
        candidates=[{'index':i,'attribute':'ticket_status','value':value,'object':obj} for i,value in enumerate(['принята','в работе'])]
        result=assemble_facts(call,candidates,None)['facts']
        self.assertEqual(result[0]['valid_to'],'2026-11-14T00:00:00');self.assertIsNone(result[1]['valid_to'])

    def test_rejected_operator_offer_is_removed(self):
        call=self.call([('operator','Могу предложить выезд мастера завтра?'),('client','Нет, не надо')])
        candidate={'index':0,'attribute':'agreement','value':'выезд специалиста','object':{'id':'c','type':'client','name':'клиент'}}
        self.assertEqual(assemble_facts(call,[candidate, {**candidate, 'index': 1}],None)['facts'],[])

    def test_confirmation_recovers_time_even_if_operator_turn_was_not_selected(self):
        call = self.call([
            ('operator', 'Предлагаю выезд в четверг после двух до 18:00.'),
            ('client', 'Да, в четверг подходит.'),
        ])
        candidate = {'index': 1, 'attribute': 'agreement', 'value': 'выезд специалиста',
                     'object': {'id': 'c', 'type': 'client', 'name': 'клиент'}}
        result = assemble_facts(call, [candidate], None)['facts'][0]
        self.assertEqual(result['valid_from'], '2026-11-19T14:00:00')
        self.assertEqual(result['valid_to'], '2026-11-19T18:00:00')

    def test_repeated_confirmation_preserves_window_and_grounding(self):
        call = self.call([
            ('operator', 'Выезд мастера в четверг после двух до 18:00.'),
            ('client', 'Да, выезд в четверг подходит.'),
        ])
        obj = {'id': 'c', 'type': 'client', 'name': 'клиент'}
        candidates = [
            {'index': i, 'attribute': 'agreement', 'value': value, 'object': obj}
            for i, value in enumerate(['выезд специалиста', 'выезд мастера назначен на четверг'])
        ]
        facts = assemble_facts(call, candidates, None)['facts']
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]['valid_from'], '2026-11-19T14:00:00')
        self.assertEqual(facts[0]['valid_to'], '2026-11-19T18:00:00')
        self.assertEqual(facts[0]['source']['turn_idx'], 10)

    def test_separate_visits_are_kept(self):
        call = self.call([
            ('operator', 'Выезд мастера в четверг.'),
            ('client', 'В пятницу второй выезд тоже подходит.'),
        ])
        obj = {'id': 'c', 'type': 'client', 'name': 'клиент'}
        candidates = [{'index': i, 'attribute': 'agreement', 'value': 'выезд мастера',
                       'object': obj} for i in range(2)]
        self.assertEqual(len(assemble_facts(call, candidates, None)['facts']), 2)


if __name__=='__main__':unittest.main()
