import argparse
from pathlib import Path
import sys
import time
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from compactfacts.io import read_jsonl, write_json, write_jsonl
from compactfacts.model import Extractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="artifacts/cpu_baseline.joblib")
    parser.add_argument("--mode", choices=["precise", "balanced", "recall"], default="balanced")
    args = parser.parse_args()
    model = Extractor.load(args.model)
    calls = read_jsonl(args.input)
    predictions, times, memory = [], [], []
    for i, call in enumerate(calls):
        start = time.perf_counter()
        # Labels are explicitly removed at the inference boundary.
        predictions.append(model.predict({k:v for k,v in call.items() if k != "facts"}, mode=args.mode))
        times.append(time.perf_counter()-start)
        memory.append(psutil.Process().memory_info().rss)
        if (i+1)%50 == 0: print(f"{i+1}/{len(calls)}", flush=True)
    write_jsonl(args.output, predictions)
    write_json(Path(args.output).with_suffix(".resources.json"), {"mode":args.mode, "calls":len(calls),
        "device":"CPU", "latency_mean_ms":float(np.mean(times)*1000), "latency_p50_ms":float(np.percentile(times,50)*1000),
        "latency_p95_ms":float(np.percentile(times,95)*1000), "calls_per_hour":3600/float(np.mean(times)),
        "max_sampled_rss_mb":max(memory)/1024**2, "model_bytes":Path(args.model).stat().st_size,
        "notes":"Sequential batch=1; includes first call, excludes model load and JSON IO. RSS sampled after calls, not continuous peak."})
