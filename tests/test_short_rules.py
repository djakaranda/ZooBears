import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.short_rules import candidates_for_short_call
from compactfacts.pipeline import assemble_facts
from compactfacts.date_evidence import with_date_evidence


class ShortRulesTests(unittest.TestCase):
    catalog={'tariffs':['Гранд','Гранд-Люкс','Мой онлайн','Всё включено XL']}

    def call(self,text):
        return {'call_id':'short:1','started_at':'2026-11-14T15:00:00',
                'turns':[{'idx':7,'speaker':'client','text':text}]}

    def test_catalog_tariff_with_unknown_day_and_original_span(self):
        call=self.call('В прошлом месяце мне подключили тариф Гранд-Люкс.')
        rows=candidates_for_short_call(call,self.catalog)
        self.assertEqual([r['value'] for r in rows],['Гранд-Люкс'])
        lo,hi=rows[0]['span'];self.assertEqual(call['turns'][0]['text'][lo:hi],'Гранд-Люкс')
        result=with_date_evidence(call,assemble_facts(call,rows,None))
        self.assertEqual(result['temporal_evidence'][0]['start']['kind'],'range')

    def test_question_hypothesis_and_refusal_are_not_tariff_facts(self):
        for text in ['У меня тариф Гранд?', 'Мне не подключили тариф Гранд.', 'Мне нужен тариф Гранд.']:
            self.assertEqual(candidates_for_short_call(self.call(text),self.catalog),[])

    def test_acceptance_preserves_window_and_refusal_stays_empty(self):
        for name,expected in [('appointment',1),('refusal',0)]:
            call=json.loads((ROOT/'reports/defense'/f'{name}.json').read_text(encoding='utf-8'))
            rows=candidates_for_short_call(call,self.catalog)
            result=assemble_facts(call,rows,None)
            self.assertEqual(len(result['facts']),expected)
            if expected:
                self.assertEqual(result['facts'][0]['valid_from'],'2026-11-19T14:00:00')
                self.assertEqual(result['facts'][0]['valid_to'],'2026-11-19T18:00:00')

    def test_refusal_description_is_not_an_agreement(self):
        call=self.call('Нет-нет, я пока без этого пакета.')
        rows=[{'index':0,'attribute':'agreement','value':'клиент отказался от пакета',
               'object':{'id':'c','type':'client','name':'клиент'}}]
        self.assertEqual(assemble_facts(call,rows,None)['facts'],[])


if __name__=='__main__':unittest.main()
