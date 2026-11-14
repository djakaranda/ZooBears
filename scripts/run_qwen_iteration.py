"""Train, freeze on internal dev, evaluate val once, benchmark common inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from compactfacts.io import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--evaluation-batch-size', type=int, default=4)
    args = parser.parse_args()
    logs = ROOT / 'tmp/qwen_logs'; logs.mkdir(parents=True, exist_ok=True)
    history = []; offset = 0.
    if args.skip_training and Path('reports/qwen_pipeline_status.json').exists():
        previous=json.loads(Path('reports/qwen_pipeline_status.json').read_text(encoding='utf-8'))
        history=[r for r in previous if r['stage']=='train' and r['status']=='complete']
        offset=history[-1]['elapsed_seconds'] if history else 0.
    started = time.perf_counter()-offset
    def run(name, command):
        record = {'stage': name, 'status': 'running', 'command': command, 'elapsed_seconds': time.perf_counter() - started}
        history.append(record); write_json('reports/qwen_pipeline_status.json', history)
        print(f'START {name}', flush=True)
        with (logs / f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(status='complete' if result.returncode == 0 else 'failed',
                      exit_code=result.returncode, elapsed_seconds=time.perf_counter() - started)
        write_json('reports/qwen_pipeline_status.json', history)
        if result.returncode:
            print((logs / f'{name}.log').read_text(encoding='utf-8')[-6000:], flush=True)
            raise RuntimeError(f'{name} failed; see {logs / (name + ".log")}')
        print(f'DONE {name}', flush=True)
    if not args.skip_training:
        run('train', ['scripts/train_qwen.py', '--epochs', str(args.epochs)])
    common = ['--engine', 'qwen', '--model', 'artifacts/qwen', '--base-model', 'artifacts/qwen_base']
    run('dev_predict', ['scripts/predict_experiment.py', *common, '--input', 'tmp/splits/dev.jsonl',
                        '--output', 'tmp/qwen_dev_raw_pred.jsonl', '--candidate-output', 'tmp/qwen_dev_candidates.jsonl',
                        '--device', 'cuda', '--batch-size',str(args.evaluation_batch_size)])
    run('dev_select_dates', ['scripts/select_qwen_dates.py'])
    selection = json.loads(Path('reports/qwen_dev_selection.json').read_text(encoding='utf-8'))
    ranker = ['--date-ranker', selection['date_ranker']] if selection['date_ranker'] else []
    run('dev_score', ['scripts/evaluate.py', 'tmp/splits/dev.jsonl', 'reports/qwen_dev_pred.jsonl', '--output', 'reports/qwen_dev_metrics.json'])
    run('dev_schema', ['scripts/check_predictions.py', 'tmp/splits/dev.jsonl', 'reports/qwen_dev_pred.jsonl', '--output', 'reports/qwen_dev_schema.json'])
    run('val_predict', ['scripts/predict_experiment.py', *common, *ranker, '--input', 'data/raw/val.jsonl',
                        '--output', 'reports/qwen_val_pred.jsonl', '--device', 'cuda',
                        '--batch-size',str(args.evaluation_batch_size)])
    run('val_score', ['scripts/evaluate.py', 'data/raw/val.jsonl', 'reports/qwen_val_pred.jsonl', '--output', 'reports/qwen_val_metrics.json'])
    run('val_schema', ['scripts/check_predictions.py', 'data/raw/val.jsonl', 'reports/qwen_val_pred.jsonl', '--output', 'reports/qwen_val_schema.json'])
    # Same deterministic 24 train-dev calls across devices; no cherry-picking by quality.
    sample = read_jsonl('tmp/splits/dev.jsonl')[:24]
    write_jsonl('tmp/qwen_benchmark_calls.jsonl', sample)
    for name, device, extra in [('cuda', 'cuda', []), ('cuda_batch4', 'cuda', ['--batch-size','4']),
                                ('cpu', 'cpu', []), ('int8', 'cpu', ['--quantized'])]:
        path = f'reports/qwen_benchmark_{name}_pred.jsonl'
        run(f'benchmark_{name}', ['scripts/predict_experiment.py', *common, *ranker, '--input', 'tmp/qwen_benchmark_calls.jsonl',
                                  '--output', path, '--device', device, *extra])
        run(f'benchmark_{name}_score', ['scripts/evaluate.py', 'tmp/qwen_benchmark_calls.jsonl', path,
                                       '--output', f'reports/qwen_benchmark_{name}_metrics.json'])
    # Rerun the current 94M reference on this machine, rather than reusing MPS timings.
    run('baseline_val', ['scripts/predict_experiment.py', '--engine', 'hybrid', '--model', 'artifacts/neural',
                         '--normalizer', 'artifacts/cpu_baseline.joblib', '--generator', 'artifacts/generator',
                         '--date-ranker', 'artifacts/date_ranker.joblib', '--input', 'data/raw/val.jsonl',
                         '--output', 'reports/qwen_reference_val_pred.jsonl', '--device', 'cuda'])
    run('baseline_val_score', ['scripts/evaluate.py', 'data/raw/val.jsonl', 'reports/qwen_reference_val_pred.jsonl',
                               '--output', 'reports/qwen_reference_val_metrics.json'])
    run('baseline_benchmark', ['scripts/predict_experiment.py', '--engine', 'hybrid', '--model', 'artifacts/neural',
                               '--normalizer', 'artifacts/cpu_baseline.joblib', '--generator', 'artifacts/generator',
                               '--date-ranker', 'artifacts/date_ranker.joblib', '--input', 'tmp/qwen_benchmark_calls.jsonl',
                               '--output', 'reports/qwen_reference_benchmark_pred.jsonl', '--device', 'cuda'])
    run('report', ['scripts/write_qwen_report.py'])
    print('Qwen iteration complete: reports/QWEN_REPORT.md', flush=True)


if __name__ == '__main__':
    main()
