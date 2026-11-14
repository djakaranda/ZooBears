"""Conservative fallback for explicit short fragments; no extra neural weights."""
import hashlib
import re
from .pipeline import CONFIRMED, REFUSED


def candidates_for_short_call(call, catalog):
    if len(call['turns']) > 5:
        return []
    result = []
    names = sorted(catalog.get('tariffs', []), key=len, reverse=True)
    negative = re.compile(r'\b(?:не|нет)\b|\bотказ\w*|\?', re.I)
    account = (call.get('series') or {}).get('account_id') or call['call_id'].split(':')[0]
    obj = {'id': account + ':client:' + hashlib.sha256('клиент'.encode()).hexdigest()[:8],
           'type': 'client', 'name': 'клиент'}
    for i, turn in enumerate(call['turns']):
        text = turn['text']
        if turn['speaker'] != 'client' or negative.search(text):
            continue
        if re.search(r'\b(?:у\s+меня|мой\s+тариф|подключ\w*|переш[её]л\w*|пользуюсь|действует|сейчас)\b', text, re.I):
            for name in names:
                normalized_name = name.replace('ё', 'е').replace('Ё', 'Е')
                pattern = r'\bтариф\w*\s*(?:план\w*\s*)?[:—-]?\s*[«"„“]?\s*(' + re.escape(normalized_name).replace(r'\ ', r'\s+') + r')(?!\w|[+\-])'
                match = re.search(pattern, text.replace('ё', 'е').replace('Ё', 'Е'), re.I)
                if match:
                    result.append({'index': i, 'attribute': 'tariff', 'value': name, 'object': obj,
                                   'polarity': 1, 'span': [match.start(1), match.end(1)], 'confidence': 1.})
                    break
        if CONFIRMED.search(text) and not REFUSED.search(text):
            previous = call['turns'][i-1] if i else None
            evidence = text + ' ' + (previous['text'] if previous and previous['speaker']=='operator' else '')
            if (re.search(r'\b(?:выезд|мастер|визит)\w*', evidence, re.I)
                    and not negative.search(evidence.replace('?', ''))):
                result.append({'index': i, 'attribute': 'agreement', 'value': 'выезд специалиста',
                               'object': obj, 'polarity': 1, 'confidence': 1.})
    return result
