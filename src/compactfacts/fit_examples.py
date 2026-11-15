"""Small deterministic training supplement; catalog only, no held-out labels."""
from datetime import datetime, timedelta
import random


def targeted_examples(catalog, seed=42):
    rng = random.Random(seed)
    tariffs = list(catalog['tariffs'])
    output = []
    def make(index, turns, facts):
        call = {'call_id':f'fit_supplement:{index}', 'started_at':'2026-11-14T15:00:00',
                'turns':[{'idx':i*3+2,'speaker':s,'text':t} for i,(s,t) in enumerate(turns)], 'facts':[]}
        for pos,attr,value,kind,name,polarity,start,end in facts:
            turn=call['turns'][pos];lo=turn['text'].lower().find(value.lower())
            span=[lo,lo+len(value)] if lo>=0 else [0,len(turn['text'])]
            call['facts'].append({'object':{'id':name,'type':kind,'name':name},'attribute':attr,
                 'value':value,'polarity':polarity,'speaker':turn['speaker'],
                 'valid_from':start,'valid_to':end,'source':{'turn_idx':turn['idx'],
                 'span':span,'quote':turn['text'][span[0]:span[1]]}})
        output.append(call)
    anchor='2026-11-14T15:00:00';today='2026-11-14T00:00:00';yesterday='2026-11-13T00:00:00'
    for i in range(32):
        old,new=rng.sample(tariffs,2)
        make(len(output),[('operator','Какой у вас сейчас тариф?'),
             ('client',f'Семь дней назад подключили тариф {old}, а вчера поменяли на {new}. Сейчас пользуюсь им.')],
             [(1,'tariff',old,'client','клиент',1,'2026-11-07T00:00:00',yesterday),
              (1,'tariff',new,'client','клиент',1,yesterday,None)])
        make(len(output),[('operator',f'Могу предложить тариф {new}. Подключить?'),
             ('client',rng.choice(['Нет, не надо.','Пока не хочу, спасибо.','Не нужно, оставьте как есть.']))],[])
    for i in range(16):
        ticket=str(6100+i)
        make(len(output),[('operator',f'Заявка {ticket} принята вчера.'),
             ('client','Я понял, спасибо.'),('operator','Сейчас проверю новое сообщение.'),
             ('client','Хорошо, подожду.'),('operator',f'Сегодня заявка {ticket} уже в работе.')],
             [(0,'ticket_status','принята','ticket','заявка '+ticket,1,yesterday,today),
              (4,'ticket_status','в работе','ticket','заявка '+ticket,1,today,None)])
        make(len(output),[('operator','Предлагаю выезд специалиста завтра после 14:00 до 18:00.'),
             ('client',rng.choice(['Да, завтра подходит.','Согласен, буду дома.','Хорошо, жду специалиста.']))],
             [(1,'agreement','выезд специалиста','client','клиент',1,'2026-11-15T14:00:00','2026-11-15T18:00:00')])
        make(len(output),[('client','Первый роутер работает нормально, а второй роутер не работает с утра.')],
             [(0,'service_state','работает нормально','device','первый роутер',1,anchor,None),
              (0,'service_state','не работает','device','второй роутер',1,anchor,None)])
    return output
