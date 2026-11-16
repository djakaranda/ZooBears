"""Exact dates entailed by spoken ordinal days and calendar constraints.

This supplements postprocessing without changing the student's frozen ref IDs.
"""
from datetime import datetime,timedelta
import re
from .temporal_v2 import MONTHS,NUMBERS,DateMention,find_dates,_shift_month
from .grounding import align_quote
from .date_ranker import words

ORDINALS=['первого','второго','третьего','четвертого','пятого','шестого','седьмого','восьмого',
          'девятого','десятого','одиннадцатого','двенадцатого','тринадцатого','четырнадцатого',
          'пятнадцатого','шестнадцатого','семнадцатого','восемнадцатого','девятнадцатого','двадцатого']
ORDINALS += ['двадцать '+x for x in ORDINALS[:9]]+['тридцатого','тридцать первого']
DAY_WORDS={word:i+1 for i,word in enumerate(ORDINALS)}
for word,value in list(DAY_WORDS.items()):
    DAY_WORDS[word[:-3]+('ое' if word.endswith('ого') else 'е')]=value
DAY_WORDS.update({k:v for k,v in NUMBERS.items() if isinstance(v,int) and 1<=v<=31})
DAY_PATTERN='|'.join(re.escape(k).replace(r'\ ',r'\s+') for k in sorted(DAY_WORDS,key=len,reverse=True))
WORD_DATE=re.compile(r'\b('+DAY_PATTERN+r')\s+('+'|'.join(MONTHS)+r')(?:\s+(20\d{2}))?\b',re.I)
BARE_DAY=re.compile(r'\b(?:(\d{1,2})\s*(?:числа|число|[-–]го)|('+DAY_PATTERN+r')\s+числа)\b',re.I)
WEEKDAY=re.compile(r'понедель|вторник|сред[ауы]|четверг|пятниц|суббот|воскрес',re.I)
HARD_BREAK=re.compile(r'[!?;\n]|\.(?!\d)|\b(?:потом|затем|теперь)\b|,\s*(?:но\b|а\s+(?!точнее\b|именно\b))',re.I)


def calendar_mentions(text,started_at,future=False):
    result=find_dates(text,started_at,future)
    anchor=datetime.fromisoformat(started_at).replace(hour=0,minute=0,second=0,microsecond=0)
    lower=text.lower().replace('ё','е')
    for match in re.finditer(r'\b(?:на следующей неделе|в следующем месяце|на позапрошлой неделе)\b',lower):
        if 'месяце' in match.group():
            first=_shift_month(anchor.replace(day=1),1);last=_shift_month(first,1)
        else:
            monday=anchor-timedelta(days=anchor.weekday())
            first=monday+timedelta(days=-14 if 'позапрош' in match.group() else 7);last=first+timedelta(days=7)
        result.append(DateMention(match.start(),match.end(),None,'at','interval',
                      (first.isoformat(timespec='seconds'),last.isoformat(timespec='seconds'))))
    return result


def word_date_mentions(text,started_at):
    lower=text.lower().replace('ё','е');anchor=datetime.fromisoformat(started_at)
    intervals=[m for m in calendar_mentions(text,started_at) if m.interval]
    result=[]
    for match in WORD_DATE.finditer(lower):
        day=DAY_WORDS[re.sub(r'\s+',' ',match[1])];month=MONTHS[match[2]]
        try:value=anchor.replace(year=int(match[3] or anchor.year),month=month,day=day,hour=0,minute=0,second=0,microsecond=0)
        except ValueError:continue
        if not match[3]:
            compatible=[]
            for period in intervals:
                if min(abs(match.start()-period.end),abs(period.start-match.end()))>70:continue
                for year in [anchor.year-1,anchor.year,anchor.year+1]:
                    try:date=value.replace(year=year)
                    except ValueError:continue
                    if period.interval[0]<=date.isoformat(timespec='seconds')<period.interval[1]:compatible.append(date)
            if len(set(compatible))==1:value=compatible[0]
        prefix=lower[max(0,match.start()-5):match.start()]
        relation='until' if re.search(r'\b(?:до|по)\s*$',prefix) else 'from' if re.search(r'\b(?:с|со)\s*$',prefix) else 'at'
        result.append(DateMention(match.start(),match.end(),value.isoformat(timespec='seconds'),relation,'day'))
    return result


def apply_calendar_constraints(call,prediction):
    indices={t['idx']:i for i,t in enumerate(call['turns'])}
    pointers={(p['object_id'],p['attribute'],p['turn_idx'],p.get('value'),p.get('polarity',1)):p
              for p in prediction.get('date_pointer_evidence',[])}
    changed=set()
    for fact in prediction['facts']:
        if fact['attribute']=='action':continue
        pointer=pointers.get((fact['object']['id'],fact['attribute'],fact['source']['turn_idx'],fact['value'],fact['polarity']))
        source_indices=[indices[fact['source']['turn_idx']]]
        if pointer and pointer.get('start'):
            other=indices.get(pointer['start']['turn_idx'])
            if other is not None and other not in source_indices:source_indices.append(other)
        for index in source_indices:
            turn=call['turns'][index];text=turn['text'];lower=text.lower().replace('ё','е')
            if index!=source_indices[0] and not words(fact['value'])&words(text):continue
            aligned=align_quote(text,fact['value']);span=aligned[:2] if aligned else [0,len(text)]
            center=sum(span)/2;lo=0;hi=len(text)
            for cut in HARD_BREAK.finditer(text):
                if cut.end()<=center:lo=cut.end()
                elif cut.start()>center:hi=cut.start();break
            parsed=[m for m in calendar_mentions(text,call['started_at'],fact['attribute']=='agreement') if lo<=m.start and m.end<=hi]
            extra=[m for m in word_date_mentions(text,call['started_at']) if lo<=m.start and m.end<=hi]
            periods=[m for m in parsed if m.interval]
            choices=[];supports=[]
            if len({m.interval for m in periods})==1:
                period=periods[0];first=datetime.fromisoformat(period.interval[0]);last=datetime.fromisoformat(period.interval[1])
                if (last-first).days>35:continue
                supports=[{'turn_idx':turn['idx'],'span':[period.start,period.end],'quote':text[period.start:period.end]}]
                for mention in parsed+extra:
                    if not mention.value or mention.relation=='until':continue
                    value=datetime.fromisoformat(mention.value);quote=lower[mention.start:mention.end]
                    candidates=[value]
                    if WEEKDAY.search(quote):
                        candidates=[first+timedelta(days=d) for d in range((last-first).days) if (first+timedelta(days=d)).weekday()==value.weekday()]
                        candidates=[d.replace(hour=value.hour,minute=value.minute) for d in candidates]
                    elif not re.search(r'20\d{2}',quote):
                        candidates=[]
                        for y in [first.year,last.year]:
                            try:candidates.append(value.replace(year=y))
                            except ValueError:pass
                    for value in set(candidates):
                        if first<=value<last:choices.append((value,mention))
                for match in BARE_DAY.finditer(lower,lo,hi):
                    number=int(match[1]) if match[1] else DAY_WORDS[re.sub(r'\s+',' ',match[2])]
                    dates=[first+timedelta(days=d) for d in range((last-first).days) if (first+timedelta(days=d)).day==number]
                    for value in dates:choices.append((value,DateMention(match.start(),match.end(),value.isoformat(timespec='seconds'),'at','day')))
            elif not periods:
                choices=[(datetime.fromisoformat(m.value),m) for m in extra if m.relation!='until']
                # A date of an upcoming visit is not the onset of its ticket status.
                if fact['attribute']=='ticket_status' and 'назнач' in fact['value'].lower():
                    choices=[(v,m) for v,m in choices if m.relation=='from']
            if len({v.date() for v,m in choices})!=1:continue
            timed=[x for x in choices if x[1].precision=='time']
            if timed and len({v for v,m in timed})!=1:continue
            value,mention=(timed or choices)[0]
            clocks=[m for m in parsed if m.value and m.precision=='time' and m.relation!='until'
                    and re.match(r'^(?:после|в)\s',lower[m.start:m.end])
                    and min(abs(m.start-mention.end),abs(mention.start-m.end))<=45]
            if not timed and len({datetime.fromisoformat(m.value).time() for m in clocks})==1:
                clock=clocks[0];clock_value=datetime.fromisoformat(clock.value)
                value=value.replace(hour=clock_value.hour,minute=clock_value.minute)
                supports.append({'turn_idx':turn['idx'],'span':[clock.start,clock.end],'quote':text[clock.start:clock.end]})
            end_value=None;end_mention=None
            ends=[m for m in parsed if m.value and m.precision=='time' and m.relation=='until'
                  and re.match(r'^до\s',lower[m.start:m.end])]
            if fact['attribute']=='agreement' and len({datetime.fromisoformat(m.value).time() for m in ends})==1:
                end_mention=ends[0];end_clock=datetime.fromisoformat(end_mention.value)
                candidate=value.replace(hour=end_clock.hour,minute=end_clock.minute)
                if candidate>value:end_value=candidate.isoformat(timespec='seconds')
            date=value.isoformat(timespec='seconds')
            precision='time' if timed or clocks else mention.precision
            if fact['valid_to'] and date>fact['valid_to'] and not end_value:continue
            if fact['valid_from']!=date:changed.add(id(fact));fact['valid_from']=date
            if pointer:
                pointer.pop('range_estimate',None)
                pointer.update(start={'turn_idx':turn['idx'],'span':[mention.start,mention.end],
                    'quote':text[mention.start:mention.end],'value':date,'precision':precision,
                    'relation':'at','selection_method':'calendar_constraints','supporting_sources':supports},start_unknown=False)
            if end_value:
                fact['valid_to']=end_value
                if pointer:pointer.update(end={'turn_idx':turn['idx'],'span':[end_mention.start,end_mention.end],
                    'quote':text[end_mention.start:end_mention.end],'value':end_value,'precision':'time',
                    'relation':'until','selection_method':'calendar_constraints','supporting_sources':supports},closed_unknown=False)
            break
    groups={}
    for f in prediction['facts']:
        if f['attribute'] in {'service_state','ticket_status','tariff'} and f['polarity']>0:
            groups.setdefault((f['object']['id'],f['attribute']),[]).append(f)
    for group in groups.values():
        group.sort(key=lambda f:(f['source']['turn_idx'],(f['source'].get('span') or [0])[0]))
        for old,new in zip(group,group[1:]):
            if not {id(old),id(new)}&changed or old['value']==new['value'] or new['valid_from']<old['valid_from']:continue
            p=pointers.get((old['object']['id'],old['attribute'],old['source']['turn_idx'],old['value'],old['polarity']))
            if p and p.get('end'):continue
            old['valid_to']=new['valid_from']
            if p:
                following=pointers.get((new['object']['id'],new['attribute'],new['source']['turn_idx'],new['value'],new['polarity']))
                p.update(end=following.get('start') if following else None,
                         closed_unknown=not bool(following and following.get('start')))
    return prediction
