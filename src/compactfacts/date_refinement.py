"""Recover confirmed visit windows and explain unresolved multi-state starts.

Uses only the frozen model's references and the current transcript. No labels,
new date candidates, or changes to the date-reference numbering are involved.
"""
from collections import defaultdict
import re

VISIT = re.compile(r'\b(?:выезд|визит|мастер)\w*', re.I)


def refine_date_evidence(call, prediction):
    from .pipeline import CONFIRMED, REFUSED
    from .temporal_v2 import choose_date, find_dates

    evidence = prediction.get('date_pointer_evidence', [])
    pointers = {(p['object_id'], p['attribute'], p['turn_idx'], p.get('value'),
                 p.get('polarity', 1)): p for p in evidence}
    indices = {t['idx']: i for i, t in enumerate(call['turns'])}
    for fact in prediction['facts']:
        if (fact['attribute'] != 'agreement' or fact['polarity'] != 1
                or not VISIT.search(fact['value'])):
            continue
        pointer = pointers.get((fact['object']['id'], fact['attribute'],
                                fact['source']['turn_idx'], fact['value'], fact['polarity']))
        ref = pointer.get('start') if pointer else None
        if not ref or ref['value'] != fact['valid_from']:
            continue
        offer_index = indices.get(ref['turn_idx'])
        source_index = indices.get(fact['source']['turn_idx'])
        if (offer_index is None or source_index is None
                or not 0 < source_index - offer_index <= 6):
            continue
        offer = call['turns'][offer_index]
        lo, hi = ref['span']
        if (offer['speaker'] != 'operator' or not VISIT.search(offer['text'])
                or not 0 <= lo < hi <= len(offer['text'])
                or offer['text'][lo:hi] != ref['quote']):
            continue
        start, end, found = choose_date(offer['text'], call['started_at'], 'agreement')
        if (not found or not end or start != fact['valid_from'] or not start < end
                or start[:10] != end[:10] or start[11:19] == '00:00:00'):
            continue
        mentions = [m for m in find_dates(offer['text'], call['started_at'], True)
                    if m.relation == 'until' and m.value]
        if {m.value for m in mentions} != {end}:
            continue
        between = call['turns'][offer_index + 1:source_index + 1]
        if (any(REFUSED.search(t['text']) for t in between)
                or not any(t['speaker'] == 'client' and CONFIRMED.search(t['text'])
                           for t in between)):
            continue
        conflict = False
        for turn in between:
            if not VISIT.search(turn['text']):
                continue
            other_start, other_end, dated = choose_date(turn['text'], call['started_at'], 'agreement')
            if dated and ((other_start != start and
                           (other_start[:10] != start[:10] or other_start[11:19] != '00:00:00'))
                          or other_end is not None and other_end != end):
                conflict = True
        if conflict:
            continue
        fact['valid_to'] = end
        mention = mentions[0]
        pointer.update(end={'turn_idx': offer['idx'], 'span': [mention.start, mention.end],
                            'quote': offer['text'][mention.start:mention.end], 'value': end,
                            'precision': mention.precision, 'relation': mention.relation},
                       closed_unknown=False, confirmed_visit_window_recovered=True)

    # A date explicitly selected for one state in an utterance does not prove
    # the onset of its other states when the model selected no reference.
    groups = defaultdict(list)
    for pointer in evidence:
        if (pointer['attribute'] in {'service_state', 'ticket_status', 'tariff'}
                and pointer.get('polarity', 1) > 0):
            groups[(pointer['object_id'], pointer['attribute'], pointer['turn_idx'])].append(pointer)
    for group in groups.values():
        if len({p.get('value') for p in group}) > 1 and any(p.get('start') for p in group):
            for pointer in group:
                if not pointer.get('start'):
                    pointer.update(start_unknown=True,
                                   start_unknown_reason='unresolved_in_multi_state_turn')
    return prediction
