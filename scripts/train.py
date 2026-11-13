import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from compactfacts.io import read_jsonl, write_json
from compactfacts.model import Extractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/train.jsonl")
    parser.add_argument("--output", default="artifacts/cpu_baseline.joblib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = time.perf_counter()
    model = Extractor().fit(read_jsonl(args.train), seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    model.training.update(seconds=time.perf_counter()-start, model_bytes=Path(args.output).stat().st_size,
                          train_sha256=hashlib.sha256(Path(args.train).read_bytes()).hexdigest())
    write_json(Path(args.output).with_suffix(".json"), model.training)
    print(model.training)
