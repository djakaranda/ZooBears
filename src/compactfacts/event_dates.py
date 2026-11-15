"""Fit-only event/date ranking, followed by coherent per-object timelines."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .date_ranker import words
from .temporal_v2 import find_dates

SINGLE = {'tariff', 'service_state', 'ticket_status'}


@dataclass(frozen=True)
class EventOption:
    date: str | None
    index: int
    text: str
    relation: str
    distance: float


class EventDateModel:
    def __init__(self):
        self.vectorizers = {}
        self.models = {}
        self.audit = {}

    def mentions(self, call, future):
        return [(i, m) for i, t in enumerate(call['turns'])
                for m in find_dates(t['text'], call['started_at'], future=future) if m.value]

    def choices(self, call, fact, kind, mentions):
        indices = {t['idx']: i for i, t in enumerate(call['turns'])}
        index = indices[fact['source']['turn_idx']]
        options = [EventOption(call['started_at'] if kind == 'start' else None, index, '', 'default', 0.)]
        event_words = words(fact['value']) | words(fact['object']['name'])
        source = call['turns'][index]['text']
        span = fact['source'].get('span')
        center = sum(span)/2 if span else len(source)/2
        for i, mention in mentions:
            if kind == 'start' and mention.relation == 'until':
                continue
            text = call['turns'][i]['text']
            # Wider context is admitted only with lexical evidence for the event/object.
            if abs(i-index) > 4 and not event_words & words(text):
                continue
            lo, hi = max(0, mention.start-85), min(len(text), mention.end+85)
            distance = max(mention.start-center, center-mention.end, 0) if i == index else 0
            options.append(EventOption(mention.value, i, text[lo:hi], mention.relation, distance))
        return options

    def features(self, call, fact, option, kind):
        indices = {t['idx']: i for i, t in enumerate(call['turns'])}
        index = indices[fact['source']['turn_idx']]
        offset = option.index-index
        attr = fact['attribute']
        value, obj, context = words(fact['value']), words(fact['object']['name']), words(option.text)
        source_text = call['turns'][index]['text'].lower()
        delta = (datetime.fromisoformat(option.date).date()-datetime.fromisoformat(call['started_at']).date()).days if option.date else 0
        bucket = max(-5, min(5, offset)) if option.relation != 'default' else 'default'
        f = {'attribute': attr, 'attribute_source': f'{attr}/{bucket}',
             'relation': option.relation, 'attribute_relation': f'{attr}/{option.relation}',
             'default': int(option.relation == 'default'), 'same_turn': int(offset == 0),
             'turn_distance': min(abs(offset), 12)/4, 'character_distance': min(option.distance, 240)/80,
             'value_overlap': len(value & context), 'object_overlap': len(obj & context),
             'direction': 'past' if delta < 0 else 'future' if delta > 0 else 'same_day',
             'attribute_direction': f'{attr}/{"past" if delta<0 else "future" if delta>0 else "same"}',
             'historical': int(bool(re.search(r'раньше|прежде|до этого|был[аои]?\b', source_text))),
             'current': int(bool(re.search(r'сейчас|теперь|уже|текущ', source_text))),
             'source_speaker': call['turns'][index]['speaker'],
             'mention_speaker': call['turns'][option.index]['speaker']}
        for word in sorted(value):
            f[f'value/{word}/{bucket}'] = 1
        for word in sorted(context):
            f[f'context/{word}/{attr}/{kind}'] = 1
        for word in sorted(value & context):
            f[f'overlap/{word}/{attr}'] = 1
        return f

    def fit(self, calls):
        xs = {'start': [], 'end': []}; ys = {'start': [], 'end': []}
        counts = defaultdict(int)
        for call in calls:
            parsed = {future: self.mentions(call, future) for future in [False, True]}
            for fact in call['facts']:
                if fact['attribute'] == 'action':
                    continue
                for kind, field in [('start', 'valid_from'), ('end', 'valid_to')]:
                    if kind == 'end' and fact['attribute'] not in SINGLE:
                        continue
                    opts = self.choices(call, fact, kind, parsed[fact['attribute'] == 'agreement'])
                    target = fact.get(field)
                    labels = [o.date[:10] == target[:10] if o.date and target else o.date is target for o in opts]
                    if not any(labels):
                        # An unobservable gold date must not teach a fabricated offset.
                        labels[0] = True
                        counts[f'{kind}_unobservable_uses_default'] += 1
                    if len(opts) == 1:
                        continue
                    xs[kind].extend(self.features(call, fact, o, kind) for o in opts)
                    ys[kind].extend(labels)
                    counts[f'{kind}_facts'] += 1
        for kind in xs:
            vec = DictVectorizer()
            model = LogisticRegression(C=.3, max_iter=250, class_weight='balanced')
            model.fit(vec.fit_transform(xs[kind]), ys[kind])
            self.vectorizers[kind] = vec; self.models[kind] = model
            counts[f'{kind}_features'] = len(vec.feature_names_)
        self.audit = dict(counts)
        return self

    def apply(self, call, prediction, *, timeline=True, start_only=False, end_only=False, margin=0.):
        if start_only and end_only:
            raise ValueError('Choose either start_only or end_only')
        parsed = {future: self.mentions(call, future) for future in [False, True]}
        explicit_ends = set()
        for kind, field in [('start', 'valid_from'), ('end', 'valid_to')]:
            if end_only and kind=='start':
                continue
            if start_only and kind == 'end':
                continue
            entries = []; features = []
            for fact in prediction['facts']:
                if fact['attribute'] == 'action' or (kind == 'end' and fact['attribute'] not in SINGLE):
                    continue
                opts = self.choices(call, fact, kind, parsed[fact['attribute'] == 'agreement'])
                entries.append((fact, opts))
                features.extend(self.features(call, fact, o, kind) for o in opts)
            if not features:
                continue
            scores = iter(self.models[kind].decision_function(self.vectorizers[kind].transform(features)))
            for fact, opts in entries:
                ranked = [(o, next(scores)) for o in opts]
                chosen, score = max(ranked, key=lambda pair: pair[1])
                if end_only:
                    # Do not reopen historical states or create an end before the known start.
                    if chosen.date is None or chosen.date < fact['valid_from'] or score-ranked[0][1] < margin:
                        continue
                if start_only:
                    if chosen.relation == 'default' or score-ranked[0][1] < margin:
                        continue
                    if fact['valid_to'] and chosen.date > fact['valid_to']:
                        continue
                fact[field] = chosen.date
                if kind == 'end' and chosen.date and chosen.relation == 'until':
                    explicit_ends.add(id(fact))
        if timeline:
            groups = defaultdict(list)
            for fact in prediction['facts']:
                if fact['attribute'] in SINGLE and fact['polarity'] > 0:
                    groups[(fact['object']['id'], fact['attribute'])].append(fact)
            for facts in groups.values():
                facts.sort(key=lambda f: (f['valid_from'], f['source']['turn_idx']))
                for old, new in zip(facts, facts[1:]):
                    if old['value'] != new['value'] and id(old) not in explicit_ends:
                        old['valid_to'] = new['valid_from']
                # The final observed state is open unless its end was explicitly stated.
                if facts and id(facts[-1]) not in explicit_ends:
                    facts[-1]['valid_to'] = None
        for fact in prediction['facts']:
            if fact['valid_to'] and fact['valid_to'] < fact['valid_from']:
                fact['valid_to'] = None
        from .date_ranker import preserve_visit_window
        preserve_visit_window(call, prediction)
        return prediction
