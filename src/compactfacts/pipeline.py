"""Ground facts in input turns and resolve timelines per object, never globally."""
from collections import defaultdict
from datetime import datetime
import hashlib
import re

from .temporal_v2 import choose_date

CONFIRMED = re.compile(r'\b(?:подход\w*|соглас\w*|договорились|устраива\w*|хорошо|фиксирую|подтверждаю|тогда оформ\w*)\b|^\s*да[.!]?\s*$', re.I)
REFUSED = re.compile(r'\b(?:не\s+(?:надо|нужно|хочу|сейчас|подходит|устраивает|согласен|согласна)|подумаю)\b|^\s*нет\s*[.!]?\s*$', re.I)
AGREEMENT_KINDS = re.compile(r'выезд\w*|перенастро\w*|возврат\w*|замен\w*|перезвон\w*|подключ\w*|отключ\w*|достав\w*|провер\w*|перерасчет\w*')


def merge_agreements(facts, turns):
    """Collapse adjacent confirmations of the same event and keep the best stated date."""
    output = []
    for fact in facts:
        if fact['attribute'] != 'agreement' or fact['polarity'] != 1:
            output.append(fact)
            continue
        text = turns[fact['_index']]['text']
        kinds = {m.group()[:6] for m in AGREEMENT_KINDS.finditer(fact['value'].lower().replace('ё', 'е'))}
        for index in range(len(output) - 1, -1, -1):
            prior = output[index]
            if (prior['attribute'] != 'agreement' or prior['polarity'] != 1
                    or prior['object']['id'] != fact['object']['id']
                    or fact['_index'] - prior['_index'] not in (1, 2)
                    or prior['speaker'] == fact['speaker']):
                continue
            prior_text = turns[prior['_index']]['text']
            if REFUSED.search(text) or REFUSED.search(prior_text):
                continue
            prior_kinds = {m.group()[:6] for m in AGREEMENT_KINDS.finditer(prior['value'].lower().replace('ё', 'е'))}
            if not (fact['value'] == prior['value'] or kinds & prior_kinds):
                continue
            if not (CONFIRMED.search(text) or CONFIRMED.search(prior_text)):
                continue
            if prior['_explicit'] and fact['_explicit']:
                old_start, new_start = prior['valid_from'], fact['valid_from']
                if old_start[:10] != new_start[:10]:
                    continue
                if old_start[11:19] != '00:00:00' and new_start[11:19] != '00:00:00' and old_start != new_start:
                    continue
            if prior['valid_to'] and fact['valid_to'] and prior['valid_to'] != fact['valid_to']:
                continue
            dated = max((prior, fact), key=date_precision)
            kept = max((prior, fact), key=lambda f: (f['speaker'] == 'client', date_precision(f)))
            output[index] = {**kept, 'valid_from': dated['valid_from'],
                             'valid_to': dated['valid_to'] or prior['valid_to'] or fact['valid_to'],
                             '_explicit': dated['_explicit']}
            break
        else:
            output.append(fact)
    return output


def date_precision(fact):
    return (fact['_explicit'], fact['valid_from'][11:19] != '00:00:00')


def assemble_facts(call,candidates,normalizer,temporal=True,confirmation=True, *, agreement_evidence=True):
    from .grounding import align_quote,date_options
    facts=[];cache={};resolver_cache={}
    for item in sorted(candidates, key=lambda candidate: candidate['index']):
        i=item['index'];turn=call['turns'][i];attr=item['attribute'];text=turn['text'];value=item['value']
        if not text.strip():continue
        after=call['turns'][i+1] if i+1<len(call['turns']) else None
        if confirmation and attr == 'agreement' and turn['speaker'] == 'client' and REFUSED.search(text):
            continue
        if confirmation and attr in {'agreement','tariff'} and turn['speaker']=='operator' and re.search(r'можем|могу|предлага|\?',text,re.I):
            if after and after['speaker']=='client':
                if agreement_evidence:
                    declined = REFUSED.search(after['text']) or (
                        re.match(r'\s*нет[, ]', after['text'], re.I) and not CONFIRMED.search(after['text']))
                else:
                    declined = re.search(r'не надо|не нужно|не хочу|не сейчас|подумаю|потом|нет[, ]', after['text'], re.I)
                if declined:
                    continue
        polarity=item.get('polarity',1)
        if confirmation and polarity<0 and not (turn['speaker']=='client' and re.search(r'больше не|уже не|не пользую|не использу|отказал',text,re.I)):
            polarity=1
        kind=item.get('kind','client')
        if item.get('object'):
            obj=item['object'].copy()
        else:
            key=(i,kind) if kind!='client' else ('client',)
            if key not in cache:cache[key]=normalizer.resolve_object(call,i,kind,resolver_cache)
            name,kind=cache[key]
            account=(call.get('series') or {}).get('account_id') or call['call_id'].split(':')[0]
            obj={'id':account+':'+kind+':'+hashlib.sha256(name.encode()).hexdigest()[:8],'type':kind,'name':name}
        span=item.get('span')
        if not span:
            alignment=align_quote(text,value)
            span=list(alignment[:2]) if alignment else [0,len(text)]
        valid_from=call['started_at'];valid_to=None;explicit=False
        if temporal:
            valid_from,valid_to,explicit=choose_date(text,call['started_at'],attr,span)
            # Neural policies are learned only where a deterministic candidate
            # matches a gold date. No date labels or offsets are copied at inference.
            if item.get('date_policy') is not None and not (agreement_evidence and attr == 'agreement' and explicit):
                options=date_options(call,i,attr);policy=item['date_policy']
                if options[policy]:valid_from=options[policy];explicit=policy>0
            elif not agreement_evidence and not explicit and attr=='agreement' and i and len(text)<120 and re.search(r'да|подходит|хорошо|соглас',text,re.I):
                inherited,_,found=choose_date(call['turns'][i-1]['text'],call['started_at'],attr)
                if found:valid_from=inherited;explicit=True
            if agreement_evidence and attr=='agreement' and i and turn['speaker']=='client' and CONFIRMED.search(text) and not REFUSED.search(text):
                previous = call['turns'][i-1]
                if previous['speaker'] == 'operator':
                    previous_match = align_quote(previous['text'], value)
                    inherited, end, found = choose_date(
                        previous['text'], call['started_at'], attr,
                        previous_match[:2] if previous_match else None,
                    )
                    if found and (not explicit or valid_from[:10] == inherited[:10]):
                        if not explicit or valid_from[11:19] == '00:00:00':
                            valid_from, explicit = inherited, True
                        valid_to = valid_to or end
            if not valid_from:valid_from=call['started_at']
            if valid_to and valid_to<valid_from:valid_to=None
        facts.append({'object':obj,'attribute':attr,'value':value,'polarity':polarity,'speaker':turn['speaker'],
            'valid_from':valid_from,'valid_to':valid_to,'source':{'turn_idx':turn['idx'],'span':span,'quote':text[span[0]:span[1]]},
            '_confidence':item.get('confidence',0.),'_index':i,'_explicit':explicit,
            '_historical':bool(re.search(r'раньше|был[аои]?\b|до этого|прежде|потом|статус сменился',text,re.I))})
    if temporal and agreement_evidence:
        facts = merge_agreements(facts, call['turns'])
    unique={}
    for f in facts:
        key=(f['object']['id'],f['attribute'],f['value'],f['polarity'])
        if agreement_evidence and f['attribute']=='agreement':
            key += (f['valid_from'], f['valid_to'])
        if key not in unique or f['_confidence']>unique[key]['_confidence']:unique[key]=f
    facts=list(unique.values());timelines=defaultdict(list)
    for f in facts:
        if f['attribute'] in {'service_state','ticket_status','tariff'} and f['polarity']>0:
            timelines[(f['object']['id'],f['attribute'])].append(f)
    if temporal:
        for timeline in timelines.values():
            timeline.sort(key=lambda f:f['_index'])
            for old,new in zip(timeline,timeline[1:]):
                # A stated historical value is closed by the observed successor.
                # Unknown starts use observation time; don't infer hidden months.
                if old['value']!=new['value'] and old['valid_to'] is None:
                    boundary=new['valid_from']
                    if boundary<old['valid_from']:boundary=call['started_at']
                    old['valid_to']=max(old['valid_from'],boundary)
    for f in facts:
        for key in list(f):
            if key.startswith('_'):f.pop(key)
    return {'call_id':call['call_id'],'facts':sorted(facts,key=lambda f:f['source']['turn_idx'])}


class TemporalExtractor:
    def __init__(self,path):
        from .model import Extractor
        self.normalizer=Extractor.load(path)

    def predict(self,call,mode='balanced'):
        base=self.normalizer.predict(call,mode)
        return self.transform(call,base,self.normalizer)

    @staticmethod
    def transform(call,base,normalizer=None, *, agreement_evidence=True):
        indices={t['idx']:i for i,t in enumerate(call['turns'])}
        candidates=[{'index':indices[f['source']['turn_idx']],'attribute':f['attribute'],'value':f['value'],
            'polarity':f['polarity'],'kind':f['object']['type'],'object':f['object'],
            'span':f['source'].get('span') if agreement_evidence and f['attribute']=='agreement' else None} for f in base['facts']]
        return assemble_facts(call,candidates,normalizer,confirmation=False,agreement_evidence=agreement_evidence)
