"""Explicit, conservative Russian relative-date rules anchored to the call."""
import calendar
import re
from datetime import datetime, timedelta

NUMBERS = {"один": 1, "одну": 1, "два": 2, "две": 2, "двух": 2,
           "три": 3, "трех": 3, "четыре": 4, "четырех": 4,
           "пять": 5, "пяти": 5, "шесть": 6, "шести": 6,
           "семь": 7, "семи": 7, "восемь": 8, "девять": 9, "десять": 10}
DAYS = {"понедельник": 0, "вторник": 1, "сред": 2, "четверг": 3,
        "пятниц": 4, "суббот": 5, "воскресень": 6}


def normalize_date(text, started_at, future=False):
    anchor = datetime.fromisoformat(started_at)
    day = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    text = text.lower().replace("ё", "е")
    explicit = False
    if "послезавтра" in text:
        day += timedelta(days=2); explicit = True
    elif "позавчера" in text:
        day -= timedelta(days=2); explicit = True
    elif "завтра" in text:
        day += timedelta(days=1); explicit = True
    elif "вчера" in text:
        day -= timedelta(days=1); explicit = True
    else:
        number = r"(\d+|" + "|".join(NUMBERS) + r")"
        found = re.search(r"(?:через\s+)?" + number + r"\s+(дн\w*|день|недел\w*|час\w*)\s*(назад)?", text)
        if found:
            raw, unit, ago = found.groups()
            n = int(raw) if raw.isdigit() else NUMBERS[raw]
            delta = timedelta(weeks=n) if unit.startswith("нед") else timedelta(hours=n) if unit.startswith("час") else timedelta(days=n)
            if ago or "через" in text[max(0, found.start()-8):found.end()]:
                day = (anchor if unit.startswith("час") else day) + (-delta if ago else delta)
                explicit = True
        elif "неделю назад" in text or "через неделю" in text:
            day += timedelta(days=-7 if "назад" in text else 7); explicit = True
        elif "прошлом месяце" in text or "месяц назад" in text:
            month = anchor.month-1 or 12
            year = anchor.year - (anchor.month == 1)
            day = day.replace(year=year, month=month, day=1 if "прошлом месяце" in text else min(day.day, calendar.monthrange(year, month)[1]))
            explicit = True
        else:
            for stem, weekday in DAYS.items():
                match = re.search(r"\b(?:со?|в|на|до)\s+(?:следующ\w+\s+|прошл\w+\s+)?" + stem + r"\w*", text)
                if match:
                    phrase = match.group()
                    ahead = future and not phrase.startswith("с") and "прошл" not in phrase
                    offset = (weekday-anchor.weekday()) % 7 if ahead else -((anchor.weekday()-weekday) % 7)
                    if "следующ" in phrase and offset == 0: offset = 7
                    if "прошл" in phrase and offset == 0: offset = -7
                    day += timedelta(days=offset); explicit = True
                    break
    clock = re.search(r"(?:после|в)\s+(\d{1,2})(?::(\d{2}))?\s*(?:час\w*|вечера|утра)?\b", text)
    word_clock = re.search(r"после\s+(двух|трех|четырех|пяти|шести)\b", text)
    if clock:
        hour = int(clock[1])
        if ("после" in clock[0] or "вечера" in clock[0]) and hour < 12: hour += 12
        if hour < 24:
            day = day.replace(hour=hour, minute=int(clock[2] or 0)); explicit = True
    elif word_clock:
        day = day.replace(hour=NUMBERS[word_clock[1]]+12); explicit = True
    return day.isoformat(timespec="seconds"), explicit
