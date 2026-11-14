"""Rank date mentions observed in a fact’s source turn and adjacent turns."""
from datetime import datetime
import re
from collections import Counter
from typing import NamedTuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .grounding import align_quote
from .temporal_v2 import find_dates

class Option(NamedTuple):
    date: str
    offset: int
    text: str
    distance: int
    relation: str
    precision: str

WORDS = re.compile(r'[а-яё]+')
STOP = {'этот', 'этого', 'данный', 'свою', 'свой', 'того', 'сейчас'}
def words(s):
    return {word[:5] for word in WORDS.findall(s.lower().replace('ё','е')) if len(word)>=4 and word not in STOP}

class DateModel:
    def __init__(self):
        self.vectorizer = DictVectorizer()
        self.model = LogisticRegression(C=.5, max_iter=200, class_weight='balanced')

    def choices(self, call, index, fact):
        anchor = call['started_at']
        turn = call['turns'][index]
        source = fact['source']
        span = source.get('span')
        if not span or span == [0, len(turn['text'])]:
            alignment = align_quote(turn['text'], fact['value'])
            span = alignment[:2] if alignment else None
        center = sum(span) // 2 if span else len(turn['text']) // 2
        options = [Option(anchor, 0, '', 0, 'anchor', 'observation')]
        for j in (index, index-1, index+1):
            if not 0 <= j < len(call['turns']):
                continue
            text = call['turns'][j]['text']
            for m in find_dates(text, anchor, future=fact['attribute']=='agreement'):
                if m.value and m.relation!='until':
                    lo = max(0,m.start-65)
                    hi = min(len(text),m.end+65)
                    options.append(Option(m.value, j-index, text[lo:hi], max(m.start-center, center-m.end, 0), m.relation, m.precision))
        return options

    def features(self, call, index, fact, option):
        attr = fact['attribute']
        turn = call['turns'][index]
        fact_words = words(fact['value'])
        overlap = fact_words & words(option.text)
        bucket = option.offset if option.relation != 'anchor' else 'anchor'
        delta = (datetime.fromisoformat(option.date).date() - datetime.fromisoformat(call['started_at']).date()).days
        speaker = turn['speaker']
        f = {
            'attribute':attr, 'source':str(bucket), 'relation': option.relation, 'precision':option.precision,
            'attribute_source': f'{attr}/{bucket}', 'attribute_relation': f'{attr}/{option.relation}',
            'speaker_source': f'{speaker}/{bucket}', 'attribute_speaker_source': f'{attr}/{speaker}/{bucket}',
            'overlap':len(overlap), 'distance':min(option.distance,160)/40 if option.offset==0 else 0,
            'direction': 'past' if delta<0 else 'future' if delta>0 else 'same_day',
            'relative_turn':index/max(len(call['turns']),1),
        }
        for word in sorted(fact_words):
            f[f'value/{word}/{bucket}']=1
        for word in sorted(overlap):
            f[f'overlap/{word}/{bucket}']=1
        for word in sorted(words(option.text)):
            f[f'context/{word}/{attr}']=1
        return f

    def fit(self, calls):
        xs, ys = [], []
        for call in calls:
            turns = {turn['idx']:i for i,turn in enumerate(call['turns'])}
            for fact in call['facts']:
                if fact['attribute'] not in {'agreement','action','tariff','service_state','ticket_status'}:continue
                i = turns[fact['source']['turn_idx']]
                options = self.choices(call,i,fact)
                gold = fact['valid_from'][:10]
                labels = [o.date[:10]==gold for o in options]
                if len(set(labels)) <= 1:continue
                xs.extend(self.features(call,i,fact,o) for o in options)
                ys.extend(labels)
        self.model.fit(self.vectorizer.fit_transform(xs),ys)
        return self

    def proposals(self, call, base):
        turns = {turn['idx']:i for i,turn in enumerate(call['turns'])}
        features, counts = [],[]
        for fact in base['facts']:
            index = turns[fact['source']['turn_idx']]
            opts = self.choices(call, index, fact)
            features.extend(self.features(call,index,fact,o) for o in opts)
            counts.append(opts)
        scores = iter(self.model.decision_function(self.vectorizer.transform(features))) if features else iter(())
        return [max(zip(opts, (next(scores) for _ in opts)), key=lambda pair:pair[1]) for opts in counts]

def apply_date_model(call, prediction, model, attrs):
    single_valued = {'tariff', 'service_state', 'ticket_status'}
    timeline_sizes = Counter(
        (f['object']['id'], f['attribute']) for f in prediction['facts']
        if f['attribute'] in single_valued and f['polarity'] > 0
    )
    proposals = model.proposals(call, prediction)
    for fact, (option, score) in zip(prediction['facts'], proposals):
        if fact['attribute'] not in attrs or score < 0:
            continue
        if timeline_sizes[(fact['object']['id'], fact['attribute'])] > 1:
            continue
        if not fact['valid_to'] or option.date <= fact['valid_to']:
            fact['valid_from'] = option.date
    return prediction


def preserve_visit_window(call, prediction):
    from .pipeline import CONFIRMED, REFUSED
    from .temporal_v2 import choose_date

    indices = {turn['idx']: index for index, turn in enumerate(call['turns'])}
    for fact in prediction['facts']:
        if (fact['attribute'] != 'agreement' or fact['polarity'] != 1
                or not re.search(r'\b(?:выезд|визит|мастер)\w*', fact['value'], re.I)):
            continue
        index = indices[fact['source']['turn_idx']]
        turn = call['turns'][index]
        if (not index or turn['speaker'] != 'client' or not CONFIRMED.search(turn['text'])
                or REFUSED.search(turn['text'])):
            continue
        proposal = call['turns'][index - 1]
        if proposal['speaker'] != 'operator' or not re.search(r'\b(?:выезд|визит|мастер)\w*', proposal['text'], re.I):
            continue
        start, end, found = choose_date(proposal['text'], call['started_at'], 'agreement')
        spoken_date, _, spoken = choose_date(turn['text'], call['started_at'], 'agreement')
        if spoken and found and spoken_date[:10] != start[:10]:
            continue
        if found and end and start[:10] == end[:10] and start < end and start[11:19] != '00:00:00':
            fact['valid_from'], fact['valid_to'] = start, end
    return prediction
