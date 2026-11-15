"""Explain uncertain starts without changing the competition's point dates.

Range bounds describe possible *start dates*, not the duration of a fact.
Only input turns are consulted; reference facts and hidden history are ignored.
"""
import re

from .grounding import align_quote
from .temporal_v2 import NUMBER, find_dates


def _event_span(text, fact):
    span = fact['source'].get('span')
    if span and 0 <= span[0] < span[1] <= len(text) and list(span) != [0, len(text)]:
        return span
    alignment = align_quote(text, fact['value'])
    return alignment[:2] if alignment else None


def _local_mentions(call, turn, fact):
    text = turn['text']
    if turn['idx'] == fact['source']['turn_idx']:
        span = _event_span(text, fact)
    else:
        alignment = align_quote(text, fact['value'])
        span = alignment[:2] if alignment else None
    mentions = [m for m in find_dates(text, call['started_at'], fact['attribute'] == 'agreement')
                if m.relation != 'until']
    if span:
        center = sum(span) / 2
        # A date across a sentence boundary can belong to a different event.
        mentions = [m for m in mentions if not re.search(
            r'[!?;\n]|\.(?!\d)',
            text[int(min(center, m.end)):int(max(center, m.start))],
        )]
    elif len(re.findall(r'[!?;\n]|\.(?!\d)(?=\s|$)', text.strip().rstrip('.!?'))) > 0:
        # No localized source: don't borrow a date from a different sentence.
        return []
    return mentions


def _source(turn, mention):
    return {'turn_idx': turn['idx'], 'span': [mention.start, mention.end],
            'quote': turn['text'][mention.start:mention.end]}


def _start_evidence(call, fact):
    turns = call['turns']
    index = next((i for i, t in enumerate(turns) if t['idx'] == fact['source']['turn_idx']), None)
    result = {'kind': 'unknown', 'observed_at': call['started_at'], 'sources': [],
              'reason': 'start_not_stated'}
    if index is None:
        return result
    turn = turns[index]
    mentions = _local_mentions(call, turn, fact)
    located = [(turn, m) for m in mentions]
    # A client's acceptance can inherit the date of the immediately preceding
    # offer. Do not search arbitrary neighbouring events or future calls.
    if fact['attribute'] == 'agreement' and turn['speaker'] == 'client' and index:
        from .pipeline import CONFIRMED, REFUSED
        if CONFIRMED.search(turn['text']) and not REFUSED.search(turn['text']):
            previous = turns[index - 1]
            if previous['speaker'] == 'operator' and align_quote(previous['text'], fact['value']):
                located.extend((previous, m) for m in _local_mentions(call, previous, fact))

    intervals = [(t, m) for t, m in located if m.interval]
    points = [(t, m) for t, m in located if m.value and not (
        m.precision == 'time' and re.match(
            r'^(?:после|до|в)\s+' + NUMBER + r'\b', t['text'][m.start:m.end], re.I))]
    # A spoken day may refine a month/week, but must lie inside every range.
    compatible = [(t, m) for t, m in points if all(
        lower <= m.value < upper for _, interval in intervals
        for lower, upper in [interval.interval])]
    values = {m.value for _, m in compatible}
    if len({v[:10] for v in values}) == 1:
        # Keep time precision if spoken; a day mention does not establish an
        # exact hour. Competing explicit times remain ambiguous.
        timed = [(t, m) for t, m in compatible if m.precision == 'time']
        selected = timed or compatible
        if len({m.value for _, m in selected}) == 1:
            t, mention = selected[0]
            return {**result, 'kind': 'exact', 'reason': 'spoken_date',
                    'value': mention.value, 'precision': mention.precision,
                    'sources': [_source(t, m) for t, m in located
                                if m.value == mention.value]}
    if intervals and not points:
        lower = max(m.interval[0] for _, m in intervals)
        upper = min(m.interval[1] for _, m in intervals)
        if lower < upper:
            quotes = [t['text'][m.start:m.end] for t, m in intervals]
            precision = 'month' if all('месяц' in q.lower() for q in quotes) else 'week'
            return {**result, 'kind': 'range', 'reason': 'spoken_period',
                    'earliest': lower, 'latest_exclusive': upper, 'precision': precision,
                    'sources': [_source(t, m) for t, m in intervals]}
    if located:
        result.update(reason='ambiguous_dates', sources=[_source(t, m) for t, m in located])
    return result


def with_date_evidence(call, prediction):
    """Return API/UI metadata; the original facts and their dates stay intact."""
    pointers={(p['object_id'],p['attribute'],p['turn_idx'],p.get('value'),p.get('polarity')):p for p in prediction.get('date_pointer_evidence',[])}
    turns={t['idx']:t for t in call['turns']}
    def reference_source(ref):
        if not ref or ref['turn_idx'] not in turns:return None
        lo,hi=ref['span'];text=turns[ref['turn_idx']]['text']
        if not 0<=lo<=hi<=len(text) or text[lo:hi]!=ref['quote']:return None
        return {'turn_idx':ref['turn_idx'],'span':ref['span'],'quote':ref['quote']}
    evidence=[]
    for i,fact in enumerate(prediction['facts']):
        item={'fact_index':i,'start':_start_evidence(call,fact)}
        key=(fact['object'].get('id'),fact['attribute'],fact['source']['turn_idx'])
        pointer=pointers.get((*key,fact['value'],fact.get('polarity',1))) or pointers.get((*key,None,None))
        if pointer:
            if pointer.get('start_unknown') and item['start']['kind']!='range':
                item['start']={'kind':'unknown','observed_at':call['started_at'],'reason':'start_not_resolved','sources':[]}
            for field,key in [('start','valid_from'),('end','valid_to')]:
                ref=pointer.get(field);source=reference_source(ref)
                if source and ref['value']==fact[key]:
                    local=item.get(field,{})
                    if local.get('kind')=='range' and not local['earliest']<=ref['value']<local['latest_exclusive']:
                        continue
                    item[field]={'kind':'exact','value':ref['value'],'precision':ref.get('precision','day'),
                        'observed_at':call['started_at'],'reason':'model_linked_spoken_date','sources':[source]}
            if pointer.get('closed_unknown') and not item.get('end'):
                item['end']={'kind':'unknown','observed_at':call['started_at'],'reason':'end_not_resolved','sources':[]}
        evidence.append(item)
    return {**prediction,'temporal_evidence':evidence}
