"""Build a bounded fit-only continuation set; keep the 43-case suite out of fit."""
import copy
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from compactfacts.io import read_jsonl,write_json,write_jsonl
from compactfacts.qwen_date_format import references,windows,target_rows,weighted_target
from compactfacts.qwen_format import training_object
from transformers import AutoTokenizer


def supplements(catalog,seed=73):
    rng=random.Random(seed);out=[]
    anchors=['2026-01-01','2026-02-01','2026-03-01','2026-07-17','2026-09-12','2026-12-31']
    def make(anchor,turns,labels,family):
        account=f'sprint_synthetic_{len(out):05d}'
        call={'call_id':account+':call0','started_at':anchor,'series':{'account_id':account},
              'turns':[{'idx':2+j*6,'speaker':speaker,'text':text} for j,(speaker,text) in enumerate(turns)],'facts':[]}
        for position,attr,value,kind,name,start,end in labels:
            turn=call['turns'][position];at=turn['text'].lower().find(value.lower())
            lo,hi=(at,at+len(value)) if at>=0 else (0,len(turn['text']))
            call['facts'].append({'object':{'id':account+':'+kind+':'+name,'type':kind,'name':name},
                'attribute':attr,'value':value,'polarity':1,'speaker':turn['speaker'],
                'valid_from':start,'valid_to':end,'source':{'turn_idx':turn['idx'],'span':[lo,hi],'quote':turn['text'][lo:hi]}})
        call['_training_family']=family
        out.append(call)
    for i in range(40):
        anchor=rng.choice(anchors)+'T15:00:00';day=datetime.fromisoformat(anchor)
        date=lambda delta:(day+timedelta(days=delta)).replace(hour=0).isoformat(timespec='seconds')
        old,new=rng.sample(catalog['tariffs'],2);ticket=str(7000+i*13)
        svc=rng.choice(['домашний интернет','мобильный интернет','цифровое телевидение'])
        dev=rng.choice(['роутер','модем','приставка'])
        genitive={'роутер':'роутера','модем':'модема','приставка':'приставки'}[dev]
        value=rng.choice(['не работает','работает нормально','низкая скорость'])
        make(anchor,[('client',rng.choice([f'{svc.capitalize()} {value}.',f'Здравствуйте. {svc.capitalize()}: {value}.',f'Проверил: {svc} {value}.']))],
             [(0,'service_state',value,'service',svc,anchor,None)],'short_service')
        make(anchor,[('client',f'Сейчас тариф {new}.')],[(0,'tariff',new,'client','клиент',anchor,None)],'short_tariff')
        make(anchor,[('operator',f'Вчера заявка {ticket} перешла в работу. Сейчас статус — в работе.')],
             [(0,'ticket_status','в работе','ticket','заявка '+ticket,date(-1),None)],'short_ticket')
        make(anchor,[('client',f'Вчера я выполнил перезагрузку {genitive}.')],
             [(0,'action','перезагрузка '+genitive,'client','клиент',date(-1),None)],'short_action')
        past='работало' if 'телевидение' in svc else 'работал'
        make(anchor,[('client',f'Пять дней назад {svc} {past} нормально.'),
                     ('operator','Уточните, что изменилось?'),('client',f'А вчера {svc} перестал работать, теперь не работает.')],
             [(0,'service_state','работает нормально','service',svc,date(-5),date(-1)),
              (2,'service_state','не работает','service',svc,date(-1),None)],'service_chain')
        make(anchor,[('operator',f'Заявка {ticket} принята пять дней назад.'),('client','Продолжайте, слушаю.'),
                     ('operator',f'Три дня назад заявка {ticket} перешла в работу.'),
                     ('operator',f'А вчера заявка {ticket} закрыта.')],
             [(0,'ticket_status','принята','ticket','заявка '+ticket,date(-5),date(-3)),
              (2,'ticket_status','в работе','ticket','заявка '+ticket,date(-3),date(-1)),
              (3,'ticket_status','закрыта','ticket','заявка '+ticket,date(-1),None)],'three_ticket_states')
        make(anchor,[('client',f'Пять дней назад подключили тариф {old}. Вчера сменили на тариф {new}.')],
             [(0,'tariff',old,'client','клиент',date(-5),date(-1)),(0,'tariff',new,'client','клиент',date(-1),None)],'tariff_chain')
        make(anchor,[('client',f'Первый {dev} работает нормально, а второй {dev} не работает.')],
             [(0,'service_state','работает нормально','device','первый '+dev,anchor,None),
              (0,'service_state','не работает','device','второй '+dev,anchor,None)],'distinct_devices')
        start=(day+timedelta(days=2)).replace(hour=14).isoformat(timespec='seconds')
        end=(day+timedelta(days=2)).replace(hour=18).isoformat(timespec='seconds')
        make(anchor,[('operator','Предлагаю выезд специалиста послезавтра после 14:00 до 18:00.'),
                     ('client',rng.choice(['Да, послезавтра подходит.','Согласен на выезд в это время.','Хорошо, буду ждать специалиста.']))],
             [(1,'agreement','выезд специалиста','client','клиент',start,end)],'accepted_window')
        make(anchor,[('operator','Можно назначить выезд специалиста послезавтра после 14:00 до 18:00.'),
                     ('client','Да, согласен, буду дома.'),('operator','Фиксирую выезд специалиста послезавтра после 14:00 до 18:00.')],
             [(2,'agreement','выезд специалиста','client','клиент',start,end)],'operator_confirmation')
        make(anchor,[('client',f'Вчера перезагрузил {dev}, но {svc} по-прежнему не работает.')],
             [(0,'action','перезагрузка '+genitive,'client','клиент',date(-1),None),
              (0,'service_state','не работает','service',svc,anchor,None)],'action_date_not_state_date')
        make(anchor,[('operator',f'Предлагаю тариф {new}. Подключать?'),
                     ('client',rng.choice(['Нет, не надо.','Не хочу менять тариф.','Подумаю, пока ничего не меняйте.']))],[],'rejected_tariff')
        make(anchor,[('operator','Могу предложить выезд специалиста завтра.'),('client','Я понял, спасибо за информацию.')],[],'neutral_not_acceptance')
        make(anchor,[('client',f'Наверное, позже попробую перезагрузить {dev}. Пока ничего не делал.')],[],'intention_not_action')
        make(anchor,[('client',f'Со вчера {svc} не работает.')],
             [(0,'service_state','не работает','service',svc,date(-1),None)],'short_dated_service')
        make(anchor,[('operator',f'Заявка {ticket}: пять дней назад принята, а вчера назначен выезд специалиста.')],
             [(0,'ticket_status','принята','ticket','заявка '+ticket,date(-5),date(-1)),
              (0,'ticket_status','назначен выезд специалиста','ticket','заявка '+ticket,date(-1),None)],'same_turn_ticket_chain')
    return out


def main():
    rng=random.Random(73)
    baseline=Path('artifacts/qwen_dates')
    backup=Path('reports/sprint_baseline_hashes.json')
    if not backup.exists():write_json(backup,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.iterdir() if p.is_file()})
    fit=read_jsonl('tmp/splits/fit.jsonl');dev=read_jsonl('tmp/splits/dev.jsonl')
    # Weighted sampling without replacement, using fit labels only.
    def priority(call):
        attrs={f['attribute'] for f in call['facts']}
        return 1+2*bool(attrs&{'service_state','tariff'})+2*(len(call['facts'])>=5)+2*(len(call['turns'])<=8)
    ranked=sorted(fit,key=lambda c:rng.random()**(1/priority(c)),reverse=True)
    real=ranked[:800]
    catalog=json.loads(Path('data/raw/catalog.json').read_text(encoding='utf8'))
    synthetic=supplements(catalog)
    # An independently supplied regression suite is never copied into fit.
    signature=lambda c:tuple((t['speaker'],t['text'].casefold()) for t in c['turns'])
    heldout=[]
    for p in Path('tmp/new_suite_v21').glob('*.input.jsonl'):heldout.extend(read_jsonl(p))
    if not heldout:
        with zipfile.ZipFile('data/regression/ZooBears_tests_v2_1.zip') as archive:
            for name in archive.namelist():
                if name.endswith('.input.jsonl'):
                    heldout.extend(json.loads(s) for s in archive.read(name).decode('utf8').splitlines() if s.strip())
    forbidden={signature(c) for c in heldout+dev}
    synthetic=[c for c in synthetic if signature(c) not in forbidden]
    for call in synthetic:
        bank={r['value'] for r in references(call)}
        for fact in call['facts']:
            assert fact['valid_from']==call['started_at'] or fact['valid_from'] in bank,(call['call_id'],fact)
            assert fact['valid_to'] is None or fact['valid_to'] in bank,(call['call_id'],fact)
    tokenizer=AutoTokenizer.from_pretrained('artifacts/qwen_base',local_files_only=True)
    def cost(call):
        total=0
        for turns,ids in windows(call,tokenizer,3072):
            rows=target_rows(call,turns);target,_=weighted_target(tokenizer,rows,3.)
            assert len(target)<=512
            total+=len(ids)+len(target)
        return total
    synthetic_tokens=sum(cost(c) for c in synthetic)
    costs=[cost(c) for c in real]
    while sum(costs)+synthetic_tokens>1_850_000 and len(real)>400:
        index=max(range(len(costs)),key=costs.__getitem__);real.pop(index);costs.pop(index)
    accounts=lambda calls:{(c.get('series') or {}).get('account_id') or c['call_id'].split(':')[0] for c in calls}
    assert not accounts(real+synthetic)&accounts(dev)
    assert not {signature(c) for c in real+synthetic}&{signature(c) for c in heldout}
    # Reuse the original, frozen table numbering and format.
    total_tokens=sum(costs)+synthetic_tokens
    write_jsonl('tmp/sprint_fit.jsonl',real+synthetic)
    write_jsonl('data/derived/sprint_fit.jsonl',real+synthetic)
    write_json('reports/sprint_data_audit.json',{'seed':73,'real_fit_calls':len(real),'synthetic_calls':len(synthetic),
        'tokens_before_asr':total_tokens,'real_tokens':sum(costs),'synthetic_tokens':synthetic_tokens,
        'account_overlap':0,'suite_exact_dialogue_overlap':0,'new_suite_calls':len(heldout),
        'source':'original fit plus newly generated templates; no dev/val/suite labels in training',
        'fit_sha256':hashlib.sha256(Path('tmp/sprint_fit.jsonl').read_bytes()).hexdigest()})
    out=Path('artifacts/qwen_sprint');out.mkdir(parents=True,exist_ok=True)
    (out/'inference_config.json').write_bytes((baseline/'inference_config.json').read_bytes())
    print(json.loads(Path('reports/sprint_data_audit.json').read_text(encoding='utf8')),flush=True)


if __name__=='__main__':main()
