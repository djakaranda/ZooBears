"""Load the official scorer and its catalog; explicit local repair for its v2.1 crash."""
from contextlib import contextmanager
from datetime import timedelta
import importlib.util
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('compactfacts_official_metrics', ROOT / 'data/raw/case_metrics.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


@contextmanager
def scorer_context(gold_path=None, repair_known_errors=False):
    if hasattr(official, 'load_catalog'):
        official.load_catalog(str(gold_path or ROOT / 'data/raw/train.jsonl'))
    original = getattr(official, 'recover_dates', None)
    repairs = {'recover_dates_type_errors': 0}
    if repair_known_errors and getattr(official, 'VERSION', None) == '2.1':
        def recover(text, day):
            try:
                return original(text, day)
            except TypeError as exc:
                low = text.lower()
                two = official._TWO_WEEKS.search(low)
                past = official._DAY_BEFORE_YESTERDAY.search(low)
                if not (two or past) or 'not iterable' not in str(exc):
                    raise
                cleaned = official._DAY_BEFORE_YESTERDAY.sub('', official._TWO_WEEKS.sub('', text.lower()))
                dates = original(cleaned, day)
                if two:
                    dates.update(day + timedelta(days=n) for n in (13, 14, 15))
                if past:
                    dates.update(day - timedelta(days=n) for n in (3, 4))
                repairs['recover_dates_type_errors'] += 1
                return dates
        official.recover_dates = recover
    if original is not None:
        resolver=official.recover_dates
        @lru_cache(maxsize=32768)
        def cached_dates(text, day):
            return frozenset(resolver(text,day))
        def recover_cached(text,day):
            # Return a fresh set so callers cannot mutate a cached answer.
            return set(cached_dates(text,day))
        official.recover_dates=recover_cached
    try:
        yield repairs
    finally:
        if original is not None:
            official.recover_dates = original
