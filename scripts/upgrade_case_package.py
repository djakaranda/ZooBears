"""Install the supplied v2.1 release after checksum verification; retain v1."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    with zipfile.ZipFile(archive) as outer:
        files = {name: outer.read(name) for name in ['train.jsonl', 'val.jsonl', 'schema.md',
                 'case_metrics.py', 'baseline_metrics.json', 'baseline_pred.jsonl', 'catalog.json',
                 'README.md', 'CHECKSUMS.txt']}
        with zipfile.ZipFile(io.BytesIO(outer.read('CompactFactExtraction_participants.zip'))) as inner:
            files['ANSWERS_v2.md'] = inner.read('01_public_participants/ANSWERS_v2.md')
    checks = {}
    for line in files['CHECKSUMS.txt'].decode().splitlines():
        expected, name = line.split(maxsplit=1)
        if name == 'CHECKSUMS.txt':
            continue  # A manifest cannot contain its own final hash.
        actual = hashlib.sha256(files[name]).hexdigest()
        if actual != expected:
            raise ValueError(f'Package checksum mismatch: {name}')
        checks[name] = actual
    assert json.loads(files['baseline_metrics.json'])['metrics_version'] == '2.1'
    raw = ROOT / 'data/raw'
    saved = ROOT / 'data/releases/v1'
    if not saved.exists():
        saved.mkdir(parents=True)
        for path in raw.iterdir():
            if path.is_file():
                shutil.copyfile(path, saved / path.name)
    for name, data in files.items():
        (raw / name).write_bytes(data)
    (raw / 'README (1).md').write_bytes(files['README.md'])
    report = {'metrics_version': '2.1', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'files_sha256': checks, 'previous_release': 'data/releases/v1',
              'official_scorer_modified': False,
              'manifest_self_hash': 'Not checked: the supplied self-reference is stale.'}
    (ROOT / 'reports/data_package_v21.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
