"""Fit-only extension of the legacy date ranker, with optional state chains."""
from collections import defaultdict
import re

from .date_ranker import DateModel, Option, words, preserve_visit_window
from .temporal_v2 import find_dates


class WideDateModel(DateModel):
    def choices(self, call, index, fact):
        options = super().choices(call, index, fact)
        event = words(fact['value'])
        obj = words(fact['object']['name'])
        identifiers=set(re.findall(r'\d+',fact['object']['name']))
        for j, turn in enumerate(call['turns']):
            if abs(j-index) <= 1:
                continue
            tokens = words(turn['text'])
            if identifiers and not identifiers <= set(re.findall(r'\d+',turn['text'])):
                continue
            # Distant dates require evidence for this event and its object.
            if not event & tokens or (abs(j-index) > 4 and not obj & tokens):
                continue
            for mention in find_dates(turn['text'], call['started_at'], future=fact['attribute']=='agreement'):
                if mention.value and mention.relation != 'until':
                    text = turn['text'][max(0,mention.start-65):mention.end+65]
                    options.append(Option(mention.value,j-index,text,0,mention.relation,mention.precision))
        return options

    def features(self, call, index, fact, option):
        features = super().features(call,index,fact,option)
        features.update(object_overlap=len(words(fact['object']['name']) & words(option.text)),
                        turn_distance=min(abs(option.offset),12)/4,
                        default=int(option.relation=='anchor'))
        return features


def apply_wide_dates(call, prediction, model, *, chains=False):
    groups = defaultdict(list)
    for fact, (option, score) in zip(prediction['facts'], model.proposals(call,prediction)):
        if fact['attribute']=='action':
            continue
        # Keep the existing answer when the classifier has no positive evidence.
        if score >= 0 and (not fact['valid_to'] or option.date <= fact['valid_to']):
            fact['valid_from'] = option.date
        if fact['attribute'] in {'tariff','service_state','ticket_status'} and fact['polarity']>0:
            groups[(fact['object']['id'],fact['attribute'])].append(fact)
    if chains:
        turns={t['idx']:t for t in call['turns']}
        for facts in groups.values():
            facts.sort(key=lambda f:(f['source']['turn_idx'],(f['source'].get('span') or [0])[0]))
            for old,new in zip(facts,facts[1:]):
                if old['value']==new['value'] or new['valid_from']<old['valid_from']:
                    continue
                text=turns[old['source']['turn_idx']]['text']
                explicit_end=any(m.value and m.relation=='until' for m in find_dates(text,call['started_at']))
                if not explicit_end:
                    old['valid_to']=new['valid_from']
            # Never reopen a state whose end was stated explicitly.
    preserve_visit_window(call,prediction)
    return prediction
