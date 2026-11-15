"""Apply student-selected references, retaining legacy fallback and provenance."""
from collections import defaultdict


def apply_date_pointers(call,prediction,candidates,mode='conservative'):
    if mode not in {'off','starts','conservative','full','ends','open_ends'}:raise ValueError(mode)
    if mode=='off':return prediction
    source_indices={t['idx']:i for i,t in enumerate(call['turns'])}
    lookup={}
    for c in candidates:
        key=(c['index'],c['attribute'],c['value'],c['object']['id'],c['polarity'])
        lookup[key]=c
    provenance=[];evidence_by_fact={};applied={};groups=defaultdict(list)
    for fact in prediction['facts']:
        key=(source_indices[fact['source']['turn_idx']],fact['attribute'],fact['value'],fact['object']['id'],fact['polarity'])
        candidate=lookup.get(key,{})
        if fact['attribute']=='action' or 'date_from_ref' not in candidate:continue
        start_evidence=candidate.get('date_from_evidence');end_evidence=candidate.get('date_to_evidence')
        if mode=='ends':start_evidence=None
        start=start_evidence['value'] if start_evidence else call['started_at'] if mode=='full' else fact['valid_from']
        end=fact['valid_to'];end_ref=candidate['date_to_ref']
        if mode!='starts':
            if end_evidence:end=end_evidence['value']
            elif end_ref==-1:end=call['started_at']
            elif end_ref==0 and mode in {'full','ends','open_ends'}:end=None
        if end is not None and end<start:
            # A model reference never makes an invalid interval. Keep the existing dates.
            continue
        fact.update(valid_from=start,valid_to=end)
        applied[id(fact)]=candidate
        provenance.append({'object_id':fact['object']['id'],'attribute':fact['attribute'],
            'value':fact['value'],'polarity':fact['polarity'],
            'turn_idx':fact['source']['turn_idx'],'start':start_evidence,
            'start_unknown':candidate['date_from_ref']==0 and mode=='full',
            'end':end_evidence,'closed_unknown':end_ref==-1 and mode!='starts'})
        evidence_by_fact[id(fact)]=provenance[-1]
        if fact['attribute'] in {'tariff','service_state','ticket_status'} and fact['polarity']>0:
            groups[(fact['object']['id'],fact['attribute'])].append(fact)
    for facts in groups.values():
        facts.sort(key=lambda f:(f['source']['turn_idx'],(f['source'].get('span') or [0])[0]))
        for old,new in zip(facts,facts[1:]):
            previous=applied[id(old)];following=applied[id(new)]
            if (mode!='ends' and old['value']!=new['value'] and following.get('date_from_evidence')
                and not previous.get('date_to_evidence') and new['valid_from']>=old['valid_from']):
                old['valid_to']=new['valid_from']
                evidence_by_fact[id(old)].update(end=following['date_from_evidence'],closed_unknown=False)
    from .date_ranker import preserve_visit_window
    preserve_visit_window(call,prediction)
    for fact in prediction['facts']:
        evidence=evidence_by_fact.get(id(fact))
        if evidence:
            if fact['valid_from']!=call['started_at']:evidence['start_unknown']=False
            if fact['valid_to']!=call['started_at']:evidence['closed_unknown']=False
    prediction['date_pointer_evidence']=provenance
    from .date_refinement import refine_date_evidence
    refine_date_evidence(call,prediction)
    return prediction
