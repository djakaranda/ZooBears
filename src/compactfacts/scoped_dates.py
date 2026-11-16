"""Bind a spoken date to its event clause; never borrow another clause's date."""
from collections import defaultdict
import re
from .grounding import align_quote
from .temporal_v2 import find_dates

BREAK=re.compile(r'[!?;\n]|\.(?!\d)|,\s*(?:а|но|затем|потом)\b|\b(?:потом|затем|теперь)\b',re.I)


def apply_scoped_dates(call,prediction,*,chains=False,attributes=None):
    attributes=set(attributes or ['service_state','ticket_status','tariff','agreement'])
    turns={t['idx']:t for t in call['turns']};changed=set();groups=defaultdict(list)
    pointers={(p['object_id'],p['attribute'],p['turn_idx'],p.get('value'),p.get('polarity',1)):p
              for p in prediction.get('date_pointer_evidence',[])}
    for fact in prediction['facts']:
        if fact['attribute'] not in attributes:continue
        turn=turns[fact['source']['turn_idx']];text=turn['text']
        span=fact['source'].get('span')
        if not span or span==[0,len(text)]:
            alignment=align_quote(text,fact['value']);span=alignment[:2] if alignment else None
        if not span or span==[0,len(text)]:continue
        center=sum(span)/2;lo=0;hi=len(text)
        for cut in BREAK.finditer(text):
            if cut.end()<=center:lo=cut.end()
            elif cut.start()>center:hi=cut.start();break
        mentions=[m for m in find_dates(text,call['started_at'],fact['attribute']=='agreement')
                  if m.value and m.relation!='until' and lo<=m.start and m.end<=hi]
        if len({m.value[:10] for m in mentions})!=1:continue
        mention=min(mentions,key=lambda m:(abs((m.start+m.end)/2-center),m.precision!='time'))
        if fact['valid_to'] and mention.value>fact['valid_to']:continue
        if fact['valid_from']==mention.value:continue
        fact['valid_from']=mention.value;changed.add(id(fact))
        key=(fact['object']['id'],fact['attribute'],turn['idx'],fact['value'],fact.get('polarity',1))
        pointer=pointers.get(key)
        if pointer:
            pointer.update(start={'turn_idx':turn['idx'],'span':[mention.start,mention.end],
                'quote':text[mention.start:mention.end],'value':mention.value,'precision':mention.precision,
                'relation':mention.relation,'selection_method':'event_clause'},start_unknown=False)
    if chains:
        for fact in prediction['facts']:
            if fact['attribute'] in {'service_state','ticket_status','tariff'} and fact['polarity']>0:
                groups[(fact['object']['id'],fact['attribute'])].append(fact)
        for group in groups.values():
            group.sort(key=lambda f:(f['source']['turn_idx'],(f['source'].get('span') or [0])[0]))
            for old,new in zip(group,group[1:]):
                if not ({id(old),id(new)}&changed) or old['value']==new['value']:continue
                if new['valid_from']<old['valid_from']:continue
                key=(old['object']['id'],old['attribute'],old['source']['turn_idx'],old['value'],old['polarity'])
                pointer=pointers.get(key)
                if pointer and pointer.get('end'):continue
                old['valid_to']=new['valid_from']
                if pointer:
                    following=pointers.get((new['object']['id'],new['attribute'],new['source']['turn_idx'],new['value'],new['polarity']))
                    pointer.update(end=following.get('start') if following else None,closed_unknown=False)
    return prediction


def apply_sprint_policy(call,prediction,policy,model=None,ranges=None):
    if policy.get('model'):
        if model is None:raise ValueError('Boundary model is required')
        model.apply(call,prediction,mode=policy['mode'],margin=policy['margin'])
    if policy.get('scope'):
        apply_scoped_dates(call,prediction,chains=True,attributes=['ticket_status'])
    if policy.get('ranges'):
        if ranges is None:raise ValueError('Range estimator is required')
        ranges.apply(call,prediction,method=policy.get('range_method','call_offset'))
    if policy.get('constraints'):
        from .calendar_constraints import apply_calendar_constraints
        apply_calendar_constraints(call,prediction)
    from .date_refinement import refine_date_evidence
    refine_date_evidence(call,prediction)
    return prediction
