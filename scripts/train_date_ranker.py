"""Train the date ranker on labelled fit calls."""
import argparse
import hashlib
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from compactfacts.date_ranker import DateModel
from compactfacts.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=ROOT / 'tmp/splits/fit.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/date_ranker.joblib')
    args = parser.parse_args()
    calls = read_jsonl(args.train)
    model = DateModel().fit(calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)
    write_json(args.output.with_suffix('.json'), {
        'fit_calls': len(calls),
        'train_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
        'features': len(model.vectorizer.feature_names_),
    })


if __name__ == '__main__':
    main()
