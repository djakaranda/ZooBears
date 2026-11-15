"""Conservative within-call coreference after extraction; preserve distinct devices."""
from copy import deepcopy
import hashlib
import re


def tokens(name):
    value = name.lower().replace('ё', 'е')
    value = re.sub(r'\b(?:маршрутизатор|роутера|роутере)\b', 'роутер', value)
    value = re.sub(r'\bинет\b', 'интернет', value)
    return tuple(w for w in re.findall(r'[а-яa-z0-9]+', value)
                 if w not in {'мой', 'моя', 'мое', 'моего', 'этот', 'эта', 'это', 'ваш', 'ваша', 'тариф'})


def canonicalize_objects(candidates, account):
    output = deepcopy(candidates)
    objects = {}
    for item in output:
        obj = item['object']
        key = (obj['type'], tokens(obj['name']))
        objects.setdefault(key, obj)
    generic = {'client': {('клиент',)}, 'device': {('устройство',)},
               'service': {('услуга',)}, 'ticket': {('заявка',)}}
    mapping = {}
    for key, obj in objects.items():
        kind, words = key
        alternatives = []
        for other, target in objects.items():
            if other == key or other[0] != kind:
                continue
            larger = other[1]
            if len(larger) <= len(words):
                continue
            # A generic mention must never connect two explicitly numbered objects.
            if any(w.isdigit() or w in {'первый', 'второй', 'третий', 'другой'} for w in words):
                continue
            if words in generic.get(kind, set()) or set(words) < set(larger):
                alternatives.append(other)
        # Resolve only a unique maximal explicit referent, avoiding generic bridges.
        maximal = [a for a in alternatives if not any(set(a[1]) < set(b[1]) for b in alternatives)]
        mapping[key] = maximal[0] if len(maximal) == 1 else key
    for item in output:
        obj = item['object']; key = (obj['type'], tokens(obj['name']))
        target = objects[mapping[key]]
        name = target['name']
        normalized = ' '.join(tokens(name))
        obj.update(name=name, id=account + ':' + obj['type'] + ':' + hashlib.sha256(normalized.encode()).hexdigest()[:8])
    return output
