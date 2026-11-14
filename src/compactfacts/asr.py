"""Reproducible, label-preserving ASR substitutions for training calls."""

from collections import defaultdict
from copy import deepcopy
import random
import re

from .grounding import align_quote
from .temporal import DAYS, NUMBERS, normalize_date
from .temporal_v2 import find_dates

CONFUSIONS = dict(zip('оаеитдпбсзвф', 'аоиедтбпзсфв'))
WORDS = re.compile(r'[а-яё]+', re.I)
PROTECTED = re.compile(
    r'не|ни|нет|да|без|до|после|уже|ещ[её]|больше|никогда|отказ\w*|соглас\w*|'
    r'подход\w*|против|отмен\w*|перенес\w*|мож\w*|буд\w*|'
    r'\w*(?:вчера|завтра|сегодня)\w*|прошл\w*|следующ\w*|'
    r'час\w*|минут\w*|дн\w*|день|недел\w*|месяц\w*|'
    + '|'.join(stem + r'\w*' for stem in DAYS)
)


def _substitute(text, rate, rng, protected):
    chars = list(text)
    for token in WORDS.finditer(text):
        word = token.group().lower()
        if len(word) < 5 or word in NUMBERS or PROTECTED.fullmatch(word):
            continue
        if any(token.start() < end and start < token.end() for start, end in protected):
            continue
        for pos in range(token.start() + 1, token.end() - 1):
            replacement = CONFUSIONS.get(text[pos].lower())
            if replacement and rng.random() < rate:
                chars[pos] = replacement.upper() if text[pos].isupper() else replacement
                break
    return ''.join(chars)


def augment_calls(calls, fraction, rate, seed, *, clean_control=False):
    """Return noisy copies from fit only. Keep semantic labels and turn IDs intact."""
    rng = random.Random(seed)
    output = []
    for index in sorted(rng.sample(range(len(calls)), round(len(calls) * fraction))):
        call = deepcopy(calls[index])
        anchor = call['started_at']
        facts_by_turn = defaultdict(list)
        for fact in call['facts']:
            facts_by_turn[fact['source']['turn_idx']].append(fact)
        changed = False
        for turn in call['turns']:
            text = turn['text']
            dates = find_dates(text, anchor)
            noisy = _substitute(text, rate, rng, [(m.start, m.end) for m in dates])
            if noisy == text or find_dates(noisy, anchor) != dates:
                continue
            if any(normalize_date(text, anchor, future) != normalize_date(noisy, anchor, future)
                   for future in (False, True)):
                continue
            for fact in facts_by_turn[turn['idx']]:
                source = fact['source']
                alignment = align_quote(text, source.get('quote'))
                if alignment:
                    start, end, _ = alignment
                    source.update(span=[start, end], quote=noisy[start:end])
                else:
                    source.update(span=None, quote=None)
            turn['text'] = noisy
            changed = True
        if changed:
            output.append(calls[index] if clean_control else call)
    return output
