"""Check the deployed model and defense scenarios through the real HTTP API."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = Request(args.url + path, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=180) as response:
            return json.load(response)

    health = request('/health')
    results = {}
    for name in ['transition', 'appointment', 'refusal']:
        call = json.loads((ROOT / 'reports/defense' / f'{name}.json').read_text(encoding='utf-8'))
        start = time.perf_counter()
        results[name] = {'response': request('/extract', call), 'seconds': time.perf_counter() - start}
    for name, text in [('uncertain', 'В прошлом месяце мне подключили тариф Гранд.'),
                       ('unknown', 'У меня тариф Гранд.')]:
        call = {'call_id': f'check:{name}', 'started_at': '2026-11-14T15:00:00',
                'turns': [{'idx': 7, 'speaker': 'client', 'text': text}]}
        start = time.perf_counter()
        results[name] = {'response': request('/extract', call), 'seconds': time.perf_counter() - start}

    transition = [f for f in results['transition']['response']['facts'] if f['attribute'] == 'ticket_status']
    appointment = results['appointment']['response']['facts']
    checks = {
        'qwen_596m': health.get('model') == 'qwen' and health.get('base_parameters') == 596049920,
        'status_chain': any(f['valid_from'] == '2026-03-31T00:00:00'
                            and f['valid_to'] == '2026-04-04T00:00:00'
                            and f['source']['turn_idx'] == 8 for f in transition)
                        and any(f['valid_from'] == '2026-04-04T00:00:00'
                                and f['valid_to'] is None and f['source']['turn_idx'] == 16 for f in transition),
        'appointment_window': len(appointment) == 1
                              and appointment[0]['valid_from'] == '2026-11-19T14:00:00'
                              and appointment[0]['valid_to'] == '2026-11-19T18:00:00',
        'refusal_empty': results['refusal']['response']['facts'] == [],
        'no_refusal_as_agreement': not any(f['attribute'] == 'agreement' and 'отказ' in f['value'].lower()
                                          for f in results['transition']['response']['facts']),
    }
    for name, kind in [('uncertain', 'range'), ('unknown', 'unknown')]:
        response = results[name]['response']
        evidence = response.get('temporal_evidence', [])
        checks[name] = len(response['facts']) == 1 and len(evidence) == 1 and evidence[0]['start']['kind'] == kind
        if name == 'uncertain' and evidence:
            checks[name] &= evidence[0]['start'].get('earliest') == '2026-10-01T00:00:00'
            checks[name] &= evidence[0]['start'].get('latest_exclusive') == '2026-11-01T00:00:00'
    report = {'health': health, 'checks': checks, 'all_passed': all(checks.values()), 'cases': results}
    path = ROOT / 'reports/qwen_api_check.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'checks': checks, 'all_passed': report['all_passed']}, ensure_ascii=False))
    if not report['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
