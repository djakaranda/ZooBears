from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compactfacts.calendar_constraints import word_date_mentions,apply_calendar_constraints
from compactfacts.date_evidence import with_date_evidence
from compactfacts.temporal_v2 import find_dates


def prediction(call,value='Плюс',attr='tariff',source=2,ref=None):
    fact={'object':{'id':'c','type':'client','name':'клиент'},'attribute':attr,'value':value,'polarity':1,
          'valid_from':call['started_at'],'valid_to':None,'source':{'turn_idx':source}}
    return {'facts':[fact],'date_pointer_evidence':[{'object_id':'c','attribute':attr,'value':value,'polarity':1,
        'turn_idx':source,'start':ref,'end':None,'closed_unknown':False}]}


class CalendarConstraintTests(unittest.TestCase):
    def test_appointment_pointer_uses_future_weekday_and_preserves_clock_window(self):
        text='Давайте оформим выезд специалиста в четверг после двух до 18:00.'
        call={'started_at':'2026-11-14T15:00:00','turns':[{'idx':2,'speaker':'operator','text':text}]}
        mentions=find_dates(text,call['started_at'],False)
        first=next(m for m in mentions if m.value and m.relation!='until')
        last=next(m for m in mentions if m.value and m.relation=='until')
        ref={'turn_idx':2,'span':[first.start,first.end],'quote':text[first.start:first.end],'value':first.value}
        pred=prediction(call,'выезд специалиста','agreement',2,ref)
        pred['facts'][0].update(valid_from=first.value,valid_to=last.value)
        pred['date_pointer_evidence'][0]['end']={'value':last.value,'precision':'time'}
        apply_calendar_constraints(call,pred)
        self.assertEqual(pred['facts'][0]['valid_from'],'2026-11-19T14:00:00')
        self.assertEqual(pred['facts'][0]['valid_to'],'2026-11-19T18:00:00')

    def test_spelled_day_with_month_retains_real_offsets(self):
        text='Тариф подключили пятого октября.'
        m=word_date_mentions(text,'2026-11-14T15:00:00')[0]
        self.assertEqual(m.value,'2026-10-05T00:00:00')
        self.assertEqual(text[m.start:m.end],'пятого октября')

    def test_month_refinement_is_an_exact_spoken_day(self):
        call={'started_at':'2026-11-14T15:00:00','turns':[{'idx':2,'speaker':'client',
            'text':'В прошлом месяце, а точнее пятого октября, подключил тариф Плюс.'}]}
        pred=prediction(call);apply_calendar_constraints(call,pred)
        self.assertEqual(pred['facts'][0]['valid_from'],'2026-10-05T00:00:00')
        e=with_date_evidence(call,pred)['temporal_evidence'][0]['start']
        self.assertEqual(e['kind'],'exact');self.assertEqual(len(e['sources']),2)

    def test_next_week_disambiguates_monday_on_a_monday(self):
        text='На следующей неделе в понедельник можем назначить выезд специалиста.'
        call={'started_at':'2026-11-16T15:00:00','turns':[{'idx':2,'speaker':'operator','text':text},
            {'idx':8,'speaker':'client','text':'Да, выезд специалиста в понедельник подходит.'}]}
        m=next(m for m in find_dates(text,call['started_at'],True) if m.value)
        ref={'turn_idx':2,'span':[m.start,m.end],'quote':text[m.start:m.end],'value':m.value}
        pred=prediction(call,'выезд специалиста','agreement',8,ref);apply_calendar_constraints(call,pred)
        self.assertEqual(pred['facts'][0]['valid_from'],'2026-11-23T00:00:00')

    def test_next_month_bare_day_and_clock_window_cross_year(self):
        text='В следующем месяце первого числа после 14:00 до 18:00 предлагаю выезд специалиста.'
        call={'started_at':'2026-12-28T15:00:00','turns':[{'idx':2,'speaker':'operator','text':text},
            {'idx':8,'speaker':'client','text':'Да, выезд специалиста подходит.'}]}
        m=next(m for m in find_dates(text,call['started_at'],True) if m.value and m.relation!='until')
        ref={'turn_idx':2,'span':[m.start,m.end],'quote':text[m.start:m.end],'value':m.value}
        pred=prediction(call,'выезд специалиста','agreement',8,ref)
        pred['facts'][0].update(valid_from='2026-12-28T14:00:00',valid_to='2026-12-28T18:00:00')
        apply_calendar_constraints(call,pred)
        self.assertEqual(pred['facts'][0]['valid_from'],'2027-01-01T14:00:00')
        self.assertEqual(pred['facts'][0]['valid_to'],'2027-01-01T18:00:00')

    def test_another_state_does_not_inherit_a_calendar_date(self):
        call={'started_at':'2026-11-14T15:00:00','turns':[{'idx':2,'speaker':'operator',
            'text':'Заявка принята пятого октября, потом стала в работе.'}]}
        pred=prediction(call,'в работе','ticket_status');apply_calendar_constraints(call,pred)
        self.assertEqual(pred['facts'][0]['valid_from'],call['started_at'])

    def test_explicit_conflicting_year_is_not_silently_rewritten(self):
        call={'started_at':'2026-12-28T15:00:00','turns':[{'idx':2,'speaker':'operator',
            'text':'В следующем месяце 5 января 2026 года будет выезд специалиста.'}]}
        pred=prediction(call,'выезд специалиста','agreement');apply_calendar_constraints(call,pred)
        self.assertEqual(pred['facts'][0]['valid_from'],call['started_at'])


if __name__=='__main__':unittest.main()
