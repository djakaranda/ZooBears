"""A disclosed point estimate inside a spoken range, fitted on fit only.

The estimate is never represented as a spoken exact day. The API keeps the
range and the estimate's provenance separate from its required point field.
"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from .grounding import align_quote
from .temporal_v2 import find_dates
from .scoped_dates import BREAK


def local_range(call,fact):
    turn=next(t for t in call['turns'] if t['idx']==fact['source']['turn_idx'])
    text=turn['text'];span=fact['source'].get('span')
    if not span or span==[0,len(text)]:
        alignment=align_quote(text,fact['value']);span=alignment[:2] if alignment else None
    if not span:return None
    center=sum(span)/2;lo=0;hi=len(text)
    for cut in BREAK.finditer(text):
        if cut.end()<=center:lo=cut.end()
        elif cut.start()>center:hi=cut.start();break
    ranges=[m for m in find_dates(text,call['started_at'],False) if m.interval and lo<=m.start and m.end<=hi]
    if len({m.interval for m in ranges})!=1:return None
    m=ranges[0];quote=text[m.start:m.end].lower()
    if 'позапрош' in quote:direction='previous2'
    elif 'прошл' in quote:direction='previous'
    elif 'следующ' in quote:direction='next'
    else:return None
    unit='month' if 'месяц' in quote else 'week'
    return turn,m,unit+'/'+direction


class RangeEstimator:
    def fit(self,calls):
        stats=defaultdict(lambda:{'total':0,'inside':0,'delta':Counter(),'offset':Counter()})
        for call in calls:
            anchor=datetime.fromisoformat(call['started_at']).date()
            for fact in call['facts']:
                if fact['attribute'] not in {'service_state','ticket_status','tariff'}:continue
                located=local_range(call,fact)
                if not located:continue
                _,m,key=located;lower,upper=m.interval;target=fact['valid_from']
                for group in [key,key+'/'+fact['attribute']]:
                    item=stats[group];item['total']+=1
                    if lower[:10]<=target[:10]<upper[:10]:
                        item['inside']+=1;day=datetime.fromisoformat(target).date()
                        item['delta'][(day-anchor).days]+=1
                        item['offset'][(day-datetime.fromisoformat(lower).date()).days]+=1
        self.stats={}
        for key,item in stats.items():
            if item['inside']:
                self.stats[key]={'total':item['total'],'inside':item['inside'],
                    'delta_mode':item['delta'].most_common(1)[0][0],
                    'offset_mode':item['offset'].most_common(1)[0][0],
                    'delta_mode_count':item['delta'].most_common(1)[0][1]}
        return self

    def apply(self,call,prediction,*,method='call_offset'):
        changed=set();pointers={(p['object_id'],p['attribute'],p['turn_idx'],p.get('value'),p.get('polarity',1)):p
            for p in prediction.get('date_pointer_evidence',[])}
        for fact in prediction['facts']:
            if fact['attribute'] not in {'service_state','ticket_status','tariff'}:continue
            located=local_range(call,fact)
            if not located:continue
            turn,m,key=located;lower,upper=m.interval
            if lower[:10]<=fact['valid_from'][:10]<upper[:10]:continue
            stats=self.stats.get(key+'/'+fact['attribute']) or self.stats.get(key)
            if not stats or stats['total']<12 or stats['inside']/stats['total']<.65:continue
            anchor=datetime.fromisoformat(call['started_at']).replace(hour=0,minute=0,second=0,microsecond=0)
            estimate=anchor+timedelta(days=stats['delta_mode'])
            first=datetime.fromisoformat(lower);last=datetime.fromisoformat(upper)-timedelta(days=1)
            if method=='interval_offset':estimate=first+timedelta(days=stats['offset_mode'])
            estimate=max(first,min(last,estimate)).isoformat(timespec='seconds')
            if fact['valid_to'] and estimate>fact['valid_to']:continue
            fact['valid_from']=estimate;changed.add(id(fact))
            pointer=pointers.get((fact['object']['id'],fact['attribute'],turn['idx'],fact['value'],fact['polarity']))
            if pointer:
                pointer.update(start=None,start_unknown=True,range_estimate={
                    'value':estimate,'earliest':lower,'latest_exclusive':upper,
                    'method':'fit_modal_interval_offset' if method=='interval_offset' else 'fit_modal_offset','precision':'month' if key.startswith('month') else 'week',
                    'sources':[{'turn_idx':turn['idx'],'span':[m.start,m.end],'quote':turn['text'][m.start:m.end]}]})
        groups=defaultdict(list)
        for f in prediction['facts']:
            if f['attribute'] in {'service_state','ticket_status','tariff'} and f['polarity']>0:groups[(f['object']['id'],f['attribute'])].append(f)
        for group in groups.values():
            group.sort(key=lambda f:(f['source']['turn_idx'],(f['source'].get('span') or [0])[0]))
            for old,new in zip(group,group[1:]):
                if id(old) not in changed or old['value']==new['value'] or new['valid_from']<old['valid_from']:continue
                following=pointers.get((new['object']['id'],new['attribute'],new['source']['turn_idx'],new['value'],new['polarity']))
                previous=pointers.get((old['object']['id'],old['attribute'],old['source']['turn_idx'],old['value'],old['polarity']))
                if previous and previous.get('end'):continue
                if following and following.get('start'):
                    old['valid_to']=new['valid_from']
                    if previous:previous.update(end=following['start'],closed_unknown=False)
        return prediction
