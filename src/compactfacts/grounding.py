"""Shared grounding/date candidates without a neural-runtime dependency."""
from rapidfuzz.fuzz import partial_ratio_alignment
from .temporal import normalize_date


def align_quote(text,quote):
    if not quote or not quote.strip():return None
    pos=text.find(quote)
    if pos>=0:return (pos,pos+len(quote),1.)
    match=partial_ratio_alignment(quote.lower().replace('ё','е'),text.lower().replace('ё','е'))
    if match and match.score>=78 and match.dest_end>match.dest_start:
        return (match.dest_start,match.dest_end,match.score/100)
    return None


def date_options(call,i,attr):
    options=[call['started_at']]
    for j in [i,i-1,i+1]:
        if 0<=j<len(call['turns']):
            value,explicit=normalize_date(call['turns'][j]['text'],call['started_at'],future=attr=='agreement')
            options.append(value if explicit else None)
        else:options.append(None)
    return options
