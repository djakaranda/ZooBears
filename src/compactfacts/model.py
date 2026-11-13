import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime

import joblib
import numpy as np
from rapidfuzz import fuzz,process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

from .temporal import normalize_date

ATTRIBUTES = ("service_state", "tariff", "ticket_status", "action", "agreement")


def normalize(text):
    return " ".join(re.findall(r"[а-яa-z0-9+]+", text.lower().replace("ё", "е")))


def turn_text(call, i):
    turns = call["turns"]
    current = turns[i]
    before = turns[i-1]["text"] if i else ""
    after = turns[i+1]["text"] if i+1 < len(turns) else ""
    return f"роль {current['speaker']} " + (current["text"] + " ")*3 + before + " " + after


def make_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           max_features=100_000, sublinear_tf=True, dtype=np.float32)


class Extractor:
    def fit(self, calls, seed=42):
        texts, groups, targets, examples = [], [], [], defaultdict(list)
        names = defaultdict(set)
        for call in calls:
            by_turn = defaultdict(list)
            for fact in call["facts"]:
                by_turn[fact["source"]["turn_idx"]].append(fact)
                names[fact["object"]["type"]].add(fact["object"]["name"])
            for i, turn in enumerate(call["turns"]):
                texts.append(turn_text(call, i))
                groups.append((call.get("series") or {}).get("account_id") or call["call_id"].split(":")[0])
                facts = by_turn.get(turn["idx"], [])
                targets.append([any(f["attribute"] == a for f in facts) for a in ATTRIBUTES])
                for fact in facts:
                    examples[fact["attribute"]].append((len(texts)-1, {
                        "value": fact["value"], "polarity": fact.get("polarity", 1),
                        "object_type": fact["object"]["type"]}))
        y = np.asarray(targets, dtype=np.int8)
        train, dev = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed).split(texts, y, groups))
        print(f"Internal grouped split: {len(train)} train turns / {len(dev)} dev turns", flush=True)
        calibration_vectorizer = make_vectorizer()
        train_x = calibration_vectorizer.fit_transform([texts[i] for i in train])
        dev_x = calibration_vectorizer.transform([texts[i] for i in dev])
        self.thresholds, self.calibration = {}, {}
        for j, attribute in enumerate(ATTRIBUTES):
            classifier = LinearSVC(C=1.0, random_state=seed, max_iter=3000)
            classifier.fit(train_x, y[train, j])
            scores = classifier.decision_function(dev_x)
            candidates = np.arange(-.8, .31, .1)
            threshold = max(candidates, key=lambda t: fbeta_score(y[dev, j], scores >= t, beta=.8, zero_division=0))
            self.thresholds[attribute] = float(threshold)
            self.calibration[attribute] = {"threshold": float(threshold), "turn_f0.8": float(fbeta_score(y[dev, j], scores >= threshold, beta=.8))}
            print(attribute, self.calibration[attribute], flush=True)
        del train_x, dev_x, calibration_vectorizer
        self.vectorizer = make_vectorizer()
        x = self.vectorizer.fit_transform(texts)
        self.classifiers, self.memories = {}, {}
        for j, attribute in enumerate(ATTRIBUTES):
            classifier = LinearSVC(C=1.0, random_state=seed, max_iter=3000)
            classifier.fit(x, y[:, j])
            self.classifiers[attribute] = classifier
            indices, metadata = zip(*examples[attribute])
            self.memories[attribute] = (x[list(indices)].tocsr(), list(metadata))
        self.names = {kind: sorted(v) for kind, v in names.items()}
        self.training = {"seed": seed, "calls": len(calls), "turns": len(texts), "facts": sum(map(len, examples.values())),
                         "features": len(self.vectorizer.vocabulary_), "calibration": self.calibration,
                         "call_ids_sha256": hashlib.sha256("\n".join(sorted(c['call_id'] for c in calls)).encode()).hexdigest()}
        return self

    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def resolve_object(self, call, i, kind, cache=None):
        turns = call["turns"]
        if kind == "ticket":
            for distance in sorted(range(len(turns)), key=lambda n: abs(n-i)):
                match = re.search(r"(?:заявк\w*|номер|обращени\w*)\s*(?:номер\s*|[№#]\s*)?(\d{3,})", turns[distance]["text"], re.I)
                if match: return "заявка " + match[1], "ticket"
            return "заявка", "ticket"
        if kind not in {"client", "device", "service"}: kind = "client"
        vocabulary = self.names.get(kind, [])
        if not vocabulary: return "клиент", "client"
        # Match names to the input; never reuse a training account's object IDs.
        if cache is None:cache={}
        key=('name_scores',kind)
        if key not in cache:
            names=[normalize(name) for name in vocabulary]
            texts=[normalize(turn['text']) for turn in turns]
            matrix=process.cdist(names,texts,scorer=fuzz.partial_ratio,dtype=np.float64)/100
            cache[key]=(names,matrix)
        names,matrix=cache[key]
        adjusted=matrix.copy()
        if kind!='client':adjusted-=np.minimum(np.abs(np.arange(len(turns))-i),20)[None,:]*.003
        else:adjusted+=np.array([.015 if t['speaker']=='client' else 0 for t in turns])[None,:]
        bests=np.maximum(adjusted.max(axis=1),0)
        scored=[(float(score),len(normalized),name) for score,normalized,name in zip(bests,names,vocabulary)]
        score, _, name = max(scored)
        if score < .68: return ("клиент", "client") if kind == "client" else (turns[i]["text"][:80], kind)
        return name, kind

    def predict(self, call, mode="balanced"):
        if not call.get("turns"):
            return {"call_id": call["call_id"], "facts": []}
        x = self.vectorizer.transform([turn_text(call, i) for i in range(len(call["turns"]))])
        facts, object_cache, resolver_cache = [], {}, {}
        delta = {"precise": .2, "balanced": 0., "recall": -.2}[mode]
        for attr in ATTRIBUTES:
            scores = self.classifiers[attr].decision_function(x)
            selected = np.flatnonzero(scores >= self.thresholds[attr]+delta)
            if not len(selected): continue
            memory, metadata = self.memories[attr]
            similarities = (x[selected] @ memory.T).toarray()
            for k, i in enumerate(selected):
                order = np.argsort(similarities[k])[-5:][::-1]
                votes = defaultdict(float)
                for rank, pos in enumerate(order):
                    key = (metadata[pos]["value"], metadata[pos]["polarity"], metadata[pos]["object_type"])
                    votes[key] += float(similarities[k, pos])**4 / (rank+1)
                value, polarity, kind = max(votes, key=votes.get)
                turn = call["turns"][int(i)]
                cache_key = (int(i), kind)
                if cache_key not in object_cache:
                    object_cache[cache_key] = self.resolve_object(call, int(i), kind,resolver_cache)
                name, kind = object_cache[cache_key]
                date_text = turn["text"]
                valid_from, explicit = normalize_date(date_text, call["started_at"], future=attr == "agreement")
                if not explicit and attr == "agreement" and i and len(turn["text"]) < 120:
                    valid_from, explicit = normalize_date(call["turns"][int(i)-1]["text"], call["started_at"], future=True)
                if not explicit and attr in {"tariff", "agreement"}: valid_from = call["started_at"]
                account = (call.get("series") or {}).get("account_id") or call["call_id"].split(":")[0]
                object_id = account+":"+kind+":"+hashlib.sha256(name.encode()).hexdigest()[:8]
                facts.append({"object": {"id": object_id, "type": kind, "name": name},
                    "attribute": attr, "value": value, "polarity": polarity, "speaker": turn["speaker"],
                    "valid_from": valid_from, "valid_to": None,
                    "source": {"turn_idx": turn["idx"], "span": [0, len(turn["text"])], "quote": turn["text"]},
                    "_confidence": float(scores[i])})
        # Collapse repeated mentions, preserving the strongest grounded occurrence.
        unique = {}
        for fact in facts:
            key = (fact["object"]["id"], fact["attribute"], fact["value"], fact["polarity"])
            if key not in unique or fact["_confidence"] > unique[key]["_confidence"]: unique[key] = fact
        facts = list(unique.values())
        timelines = defaultdict(list)
        for fact in facts:
            if fact["attribute"] in {"service_state", "tariff", "ticket_status"} and fact["polarity"] > 0:
                timelines[(fact["object"]["id"], fact["attribute"])].append(fact)
        for timeline in timelines.values():
            timeline.sort(key=lambda f: (f["valid_from"], f["source"]["turn_idx"]))
            for earlier, later in zip(timeline, timeline[1:]): earlier["valid_to"] = later["valid_from"]
        for fact in facts: fact.pop("_confidence")
        return {"call_id": call["call_id"], "facts": sorted(facts, key=lambda f: f["source"]["turn_idx"])}
