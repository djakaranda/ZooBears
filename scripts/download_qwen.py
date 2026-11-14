"""Download public, revision-pinned Qwen weights; never upload case data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from huggingface_hub import snapshot_download

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
WEIGHT_SHA256 = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
WEIGHT_BYTES = 1503300328


def modelscope_weights(folder):
    """Official Qwen mirror; accept only bytes matching the pinned HF SHA-256."""
    destination = folder / 'model.safetensors'
    if destination.exists() and hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest() == WEIGHT_SHA256:
        return
    cache = folder / '.cache' / 'modelscope-parts'; cache.mkdir(parents=True, exist_ok=True)
    part_size = 8 * 1024**2
    count = (WEIGHT_BYTES + part_size - 1) // part_size
    def fetch(index):
        lo = index * part_size; hi = min(WEIGHT_BYTES, lo + part_size) - 1
        part = cache / f'{index:04d}.part'
        if part.exists() and part.stat().st_size == hi - lo + 1:
            return index
        for attempt in range(5):
            try:
                url = f'https://modelscope.cn/models/{MODEL}/resolve/master/model.safetensors'
                with requests.get(url, headers={'Range': f'bytes={lo}-{hi}'}, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    assert response.headers.get('Content-Range') == f'bytes {lo}-{hi}/{WEIGHT_BYTES}'
                    pending = part.with_suffix('.pending')
                    with pending.open('wb') as stream:
                        for block in response.iter_content(1024**2):
                            stream.write(block)
                    assert pending.stat().st_size == hi - lo + 1
                    pending.replace(part)
                return index
            except (OSError, requests.RequestException, AssertionError):
                if attempt == 4:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, future in enumerate(as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
            future.result()
            if done % 5 == 0 or done == count:
                print(f'ModelScope verified ranges: {done}/{count}', flush=True)
    pending = destination.with_suffix('.download')
    digest = hashlib.sha256()
    with pending.open('wb') as stream:
        for index in range(count):
            with (cache / f'{index:04d}.part').open('rb') as part:
                while block := part.read(1024**2):
                    digest.update(block); stream.write(block)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError('ModelScope weights differ from the pinned upstream SHA-256; refusing to load')
    pending.replace(destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/qwen_base')
    parser.add_argument('--source', choices=['huggingface', 'modelscope'], default='huggingface')
    args = parser.parse_args()
    for attempt in range(1, 6):
        try:
            folder = Path(snapshot_download(MODEL, revision=REVISION, local_dir=args.output,
                                           allow_patterns=['*.json', '*.txt', '*.md', 'LICENSE'] +
                                           (['*.safetensors'] if args.source == 'huggingface' else []),
                                           max_workers=3))
            break
        except (OSError, ConnectionError) as exc:
            if attempt == 5:
                raise
            print(f'Download interrupted ({type(exc).__name__}); resuming attempt {attempt + 1}/5', flush=True)
            time.sleep(3)
    if args.source == 'modelscope':
        modelscope_weights(folder)
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != 'UPSTREAM_SOURCE.json':
            files[path.name] = {'bytes': path.stat().st_size,
                                'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
    manifest = {'model': MODEL, 'revision': REVISION, 'license': 'Apache-2.0', 'download_source': args.source,
                'card_url': f'https://huggingface.co/{MODEL}/blob/{REVISION}/README.md', 'files': files}
    (folder / 'UPSTREAM_SOURCE.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'downloaded': str(folder), 'revision': REVISION, 'files': len(files)}), flush=True)
