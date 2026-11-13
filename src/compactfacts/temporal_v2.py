"""Date mentions retain character offsets, direction and uncertainty.

The initial parser in temporal.py is kept intact for reproducible v1 results.
Ambiguous calendar periods are represented as intervals, not invented exact days.
"""
import calendar
from dataclasses import dataclass
from datetime import datetime,timedelta
import re

from .temporal import NUMBERS,DAYS

MONTHS={'января':1,'февраля':2,'марта':3,'апреля':4,'мая':5,'июня':6,
        'июля':7,'августа':8,'сентября':9,'октября':10,'ноября':11,'декабря':12}
NUMBER=r'(?:\d+|'+ '|'.join(sorted(NUMBERS,key=len,reverse=True))+')'
WEEKDAY_FORMS={
    'понедельник':r'понедельник(?:а|у|ом|е)?',
    'вторник':r'вторник(?:а|у|ом|е)?',
    'сред':r'сред(?:а|у|ы|е|ой)',
    'четверг':r'четверг(?:а|у|ом|е)?',
    'пятниц':r'пятниц(?:а|у|ы|е|ей)',
    'суббот':r'суббот(?:а|у|ы|е|ой)',
    'воскресень':r'воскресень(?:е|я|ю|ем|и)',
}
_PHONETIC=str.maketrans({'а':'о','и':'е','д':'т','б':'п','з':'с','ф':'в'})
_DATE_WORDS=['сегодня','вчера','завтра','позавчера','послезавтра',
 'понедельник','понедельника','понедельнику','понедельнике',
 'вторник','вторника','вторнику','вторнике','среда','среду','среды','среде','средой',
 'четверг','четверга','четвергу','четверге','пятница','пятницу','пятницы','пятнице',
 'суббота','субботу','субботы','субботе','воскресенье','воскресенья','воскресенью']
_DATE_ALIASES={word.translate(_PHONETIC):word for word in _DATE_WORDS}


@dataclass
class DateMention:
    start:int
    end:int
    value:str|None
    relation:str='at'
    precision:str='day'
    interval:tuple[str,str]|None=None


def _iso(value):return value.isoformat(timespec='seconds')


def _shift_month(day,n):
    index=day.year*12+day.month-1+n
    year,month=divmod(index,12);month+=1
    return day.replace(year=year,month=month,day=min(day.day,calendar.monthrange(year,month)[1]))


def find_dates(text,started_at,future=False,asr=True):
    anchor=datetime.fromisoformat(started_at);midnight=anchor.replace(hour=0,minute=0,second=0,microsecond=0)
    lower=text.lower().replace('ё','е');mentions=[]
    def add(match,value,precision='day',interval=None):
        if any(match.start()<m.end and match.end()>m.start for m in mentions):return
        before=lower[max(0,match.start()-4):match.start()]
        relation='until' if re.search(r'\bдо\s*$',before) or match.group().startswith('до ') else 'since' if re.search(r'\bсо?\s*$',before) else 'at'
        mentions.append(DateMention(match.start(),match.end(),_iso(value) if value else None,relation,precision,interval))
    for match in re.finditer(r'\b(20\d{2})-(\d{2})-(\d{2})\b',lower):
        try:add(match,midnight.replace(year=int(match[1]),month=int(match[2]),day=int(match[3])))
        except ValueError:pass
    for match in re.finditer(r'\b(\d{1,2})[./](\d{1,2})(?:[./](20\d{2}))?\b',lower):
        try:add(match,midnight.replace(year=int(match[3] or anchor.year),month=int(match[2]),day=int(match[1])))
        except ValueError:pass
    for match in re.finditer(r'\b(\d{1,2})\s+('+'|'.join(MONTHS)+r')(?:\s+(20\d{2}))?\b',lower):
        try:add(match,midnight.replace(year=int(match[3] or anchor.year),month=MONTHS[match[2]],day=int(match[1])))
        except ValueError:pass
    for match in re.finditer(r'\b(?:(через)\s+)?('+NUMBER+r')\s+(день|дня|дней|недел\w*|месяц\w*|час\w*)\s*(назад)?\b',lower):
        if not match[1] and not match[4]:continue
        raw=match[2];n=int(raw) if raw.isdigit() else NUMBERS[raw];direction=-1 if match[4] else 1
        unit=match[3]
        if unit.startswith('месяц'):value=_shift_month(midnight,direction*n)
        elif unit.startswith('недел'):value=midnight+timedelta(weeks=direction*n)
        elif unit.startswith('час'):value=anchor+timedelta(hours=direction*n)
        else:value=midnight+timedelta(days=direction*n)
        add(match,value)
    for phrase,delta in [('через неделю',7),('неделю назад',-7)]:
        for match in re.finditer(r'\b'+phrase+r'\b',lower):add(match,midnight+timedelta(days=delta))
    for phrase,n in [('месяц назад',-1),('через месяц',1)]:
        for match in re.finditer(r'\b'+phrase+r'\b',lower):add(match,_shift_month(midnight,n))
    for match in re.finditer(r'\b(?:в прошлом месяце|на прошлой неделе)\b',lower):
        if 'месяце' in match.group():
            start=_shift_month(midnight.replace(day=1),-1);end=midnight.replace(day=1)
        else:
            end=midnight-timedelta(days=midnight.weekday());start=end-timedelta(days=7)
        add(match,None,'interval',(_iso(start),_iso(end)))
    specials={'послезавтра':2,'позавчера':-2,'завтра':1,'вчера':-1,'сегодня':0,
              'вчерашнего дня':-1,'сегодняшнего дня':0}
    for phrase,delta in specials.items():
        for match in re.finditer(r'\b'+phrase+r'\b',lower):add(match,midnight+timedelta(days=delta))
    for stem,weekday in DAYS.items():
        pattern=r'\b(?:(со?|в|на|до)\s+)?(?:(следующ\w+|прошл\w+)\s+)?'+WEEKDAY_FORMS[stem]+r'\b'
        for match in re.finditer(pattern,lower):
            prep=match[1] or '';modifier=match[2] or ''
            ahead=(future or modifier.startswith('следующ')) and prep not in ['с','со'] and not modifier.startswith('прошл')
            offset=(weekday-anchor.weekday())%7 if ahead else -((anchor.weekday()-weekday)%7)
            if modifier.startswith('следующ') and offset==0:offset=7
            if modifier.startswith('прошл') and offset==0:offset=-7
            add(match,midnight+timedelta(days=offset))
    # Clock expressions attach only to a nearby date, not to every event in a turn.
    clock_pattern=r'\b(?:после|в)\s+(\d{1,2})(?::(\d{2}))?\s*(час\w*|вечера|утра)?\b|\bпосле\s+(двух|трех|четырех|пяти|шести)\b'
    for match in re.finditer(clock_pattern,lower):
        if not match.group().startswith('после') and not match[2] and not match[3]:continue
        hour=NUMBERS[match[4]] if match[4] else int(match[1]);minute=int(match[2] or 0)
        if ('после' in match.group() or match[3]=='вечера') and hour<12:hour+=12
        if hour>=24 or minute>=60:continue
        nearby=[m for m in mentions if m.value and abs(match.start()-m.end)<=45 and m.precision=='day']
        if nearby:
            m=min(nearby,key=lambda m:abs(match.start()-m.end))
            m.value=_iso(datetime.fromisoformat(m.value).replace(hour=hour,minute=minute));m.precision='time';m.end=max(m.end,match.end())
        else:add(match,midnight.replace(hour=hour,minute=minute),'time')
    if asr:
        for token in re.finditer(r'[а-я]+',lower):
            word=token.group();canonical=_DATE_ALIASES.get(word.translate(_PHONETIC))
            if not canonical or word==canonical or any(m.start<=token.start()<m.end for m in mentions):continue
            # Only known homophone substitutions are accepted. Keep original
            # offsets by mapping the one replacement back into the source.
            corrected=lower[:token.start()]+canonical+lower[token.end():]
            for mention in find_dates(corrected,started_at,future,asr=False):
                if not mention.start<=token.start()<mention.end:continue
                delta=len(word)-len(canonical)
                mention.end+=delta
                overlaps=[m for m in mentions if mention.start<m.end and mention.end>m.start]
                if overlaps and all(m.precision=='time' and mention.start<=m.start and mention.end>=m.end for m in overlaps):
                    mentions=[m for m in mentions if m not in overlaps];mentions.append(mention)
                elif not overlaps:mentions.append(mention)
    return sorted(mentions,key=lambda m:m.start)


def choose_date(text,started_at,attribute,event_span=None):
    mentions=find_dates(text,started_at,future=attribute=='agreement')
    centre=(event_span[0]+event_span[1])/2 if event_span else len(text)/2
    starts=[m for m in mentions if m.value and m.relation!='until']
    ends=[m for m in mentions if m.value and m.relation=='until']
    distance=lambda m:0 if m.start<=centre<=m.end else min(abs(centre-m.start),abs(centre-m.end))
    first=min(starts,key=distance).value if starts else None
    last=min(ends,key=distance).value if ends else None
    return first,last,bool(starts)
