"""Write a factual comparison from completed experiments, with explicit limits."""
import json
from pathlib import Path
import sys
import shutil
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from compactfacts.io import write_json, read_jsonl
sys.path.insert(0, str(ROOT / 'data/raw'))
from compactfacts.metrics import official
official_evaluate = official.evaluate


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


if __name__ == '__main__':
    if hasattr(official, 'load_catalog'):
        official.load_catalog('data/raw/val.jsonl')
    exp = read('artifacts/qwen/experiment.json')
    audit = read('artifacts/qwen/data_audit.json')
    selection = read('reports/qwen_dev_selection.json')
    dev = read('reports/qwen_dev_metrics.json')['official']
    val_detail = read('reports/qwen_val_metrics.json')
    if val_detail['official'].get('metrics_version', '1') != getattr(official, 'VERSION', '1'):
        raise RuntimeError('Archived report metrics use a different scorer. Run scripts/run_v21_iteration.py and write_v21_report.py for v2.1.')
    old_detail = read('reports/qwen_reference_val_metrics.json')
    val = val_detail['official']
    old = old_detail['official']
    resources = read('reports/qwen_val_pred.resources.json')
    request_resources = read('reports/qwen_benchmark_cuda_pred.resources.json')
    reference_resources = read('reports/qwen_reference_benchmark_pred.resources.json')
    schema = read('reports/qwen_val_schema.json')
    predictions = [json.loads(l) for l in Path('reports/qwen_val_pred.jsonl').open(encoding='utf-8')]
    gold_calls = read_jsonl('data/raw/val.jsonl')
    published_baseline = read('data/raw/baseline_metrics.json')
    recomputed_baseline = official_evaluate({c['call_id']: c for c in gold_calls},
                                           {c['call_id']: c for c in read_jsonl('data/raw/baseline_pred.jsonl')})
    quality_checks = {}
    for name, baseline in [('published', published_baseline), ('recomputed', recomputed_baseline)]:
        temporal_min = baseline['temporal_recall'] - .02
        state_min = baseline['state_accuracy'] - .02
        temporal_ok = val['temporal_recall'] is not None and val['temporal_recall'] + 1e-12 >= temporal_min
        state_ok = val['state_accuracy'] is not None and val['state_accuracy'] + 1e-12 >= state_min
        quality_checks[name] = {'baseline': baseline, 'temporal_min': round(temporal_min, 4),
                                'state_min': round(state_min, 4), 'temporal_ok': temporal_ok,
                                'state_ok': state_ok, 'quality_thresholds_met': temporal_ok and state_ok,
                                'cost_reduction_verified': False}
    write_json('reports/qwen_threshold_check.json', quality_checks)
    diagnostics = {k: sum(p.get('generation_diagnostics', {}).get(k, 0) for p in predictions)
                   for k in ['windows', 'invalid_json_windows', 'invalid_rows', 'truncated_windows', 'generated_tokens', 'short_rule_fallback']}
    dev_raw = read_jsonl('tmp/qwen_dev_raw_pred.jsonl')
    dev_diagnostics = {k: sum(p.get('generation_diagnostics', {}).get(k, 0) for p in dev_raw)
                       for k in diagnostics}
    write_json('reports/qwen_dev_generation.json', dev_diagnostics)
    shutil.copyfile('tmp/qwen_dev_raw_pred.generation_errors.jsonl', 'reports/qwen_dev_generation_errors.jsonl')
    by_noise = defaultdict(list)
    for call in gold_calls:
        by_noise[str(call.get('noise_level', 'unknown'))].append(call)
    prediction_map = {p['call_id']: p for p in predictions}
    noise_metrics = {}; noise_rows = []
    for level, calls in sorted(by_noise.items()):
        m = official_evaluate({c['call_id']: c for c in calls},
                              {c['call_id']: prediction_map[c['call_id']] for c in calls})
        noise_metrics[level] = {'calls': len(calls), 'official': m}
        noise_rows.append(f"| {level} | {len(calls)} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} |")
    write_json('reports/qwen_val_by_noise.json', noise_metrics)
    rows = []
    for name, m in [('Внутренний dev: Qwen', dev), ('Внешний val: Qwen', val),
                    ('Внешний val: прежний гибрид', old), ('Выданный LLM baseline: пересчёт', recomputed_baseline),
                    ('Выданный LLM baseline: файл метрик', published_baseline)]:
        rows.append(f"| {name} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} | {m['grounding_turn']} |")
    benchmark = []; backend_generation = {}
    for name in ['cuda', 'cuda_batch4', 'cpu', 'int8']:
        r = read(f'reports/qwen_benchmark_{name}_pred.resources.json')
        m = read(f'reports/qwen_benchmark_{name}_metrics.json')['official']
        ps = read_jsonl(f'reports/qwen_benchmark_{name}_pred.jsonl')
        backend_generation[name] = {k: sum(p.get('generation_diagnostics', {}).get(k, 0) for p in ps)
                                    for k in ['invalid_json_windows', 'invalid_rows', 'truncated_windows']}
        benchmark.append(f"| {name} | {r['calls']} | {r['latency_mean_ms']:.1f} | {r['latency_p95_ms']:.1f} | {r['calls_per_hour']:.1f} | {r['process_peak_rss_mb']:.1f} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} |")
    write_json('reports/qwen_backend_generation.json', backend_generation)
    deployment = read('artifacts/deployment.json')
    deployment['qwen'] = {'engine': 'qwen', 'model': 'artifacts/qwen', 'base_model': 'artifacts/qwen_base',
                          'device': 'cuda', 'date_ranker': selection['date_ranker'],
                          'agreement_evidence': selection['agreement_evidence'],
                          'base_parameters': exp['base_parameters'], 'selected_on': 'internal train dev'}
    write_json('artifacts/deployment.json', deployment)
    lines = [
        '# Qwen3-0.6B: обучение и проверка', '',
        f"Базовая модель `{exp['backbone']}`, revision `{exp['revision']}`. "
        f"Фактическое число базовых параметров: **{exp['base_parameters']:,}**. "
        f"LoRA: {exp['trainable_adapter_parameters']:,} обучаемых параметров. "
        'Диапазон менторов 500 млн–1,5 млрд соблюдён; это размер всей модели, не только адаптера.', '',
        f"Обучение: {audit['fit_calls']} fit-звонка, {audit['fit_windows']} окон, "
        f"{audit['fit_tokens']:,} токенов на эпоху. Dev: {audit['dev_calls']} звонка; пересечение аккаунтов — 0. "
        f"Запрошено эпох: {exp['epochs_requested']}, выбранная эпоха: {exp['best_epoch']}. "
        f"Выбор checkpoint: {exp['selection']}. Выбор дат на полном dev: `{selection['selected']}`.", '',
        'Модель генерирует компактный JSON фактов с источниками. Даты, спаны, ID объектов и цепочки '
        'состояний восстанавливаются общим постпроцессором. Длинные звонки разбиваются на окна с '
        'перекрытием; метки и будущие звонки на вход не передаются. Потери считаются только по ответу.', '',
        'Для фрагментов до пяти реплик при пустом ответе Qwen включается консервативный '
        'резерв правил для явно названного тарифа из каталога и подтверждённого выезда. '
        'Это не дополнительная нейросеть; применение отмечено в generation_diagnostics.short_rule_fallback '
        'и в демо. Общая обработка коротких фрагментов остаётся ограничением.', '',
        f"Исправлены некорректные обучающие имена/типы объектов: {audit['canonicalized_training_objects']}. "
        'Скрытые ID аккаунтов не используются как генерируемые имена.', '',
        '| Выборка / модель | Precision | Recall | Temporal | State | Grounding turn |',
        '|---|---:|---:|---:|---:|---:|', *rows, '',
        '## Разбор внешнего val по атрибутам', '',
        '| Атрибут | Precision | Recall | Temporal | State |',
        '|---|---:|---:|---:|---:|',
        *[f"| {attribute} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} |"
          for attribute, m in val_detail['per_attribute'].items()], '',
        'Приоритет следующей итерации — точность договорённостей и связь даты с конкретным '
        'состоянием/тарифом. Это вывод из итоговой оценки, а не изменение выбранной модели: '
        'текущие веса и настройки после просмотра val не подбирались заново.', '',
        'Официальная temporal_recall считается только среди сопоставленных фактов; '
        'её нужно читать вместе с fact_recall. Дополнительно, доля всех эталонных фактов, '
        'которые найдены с правильными датами: '
        f"Qwen {val_detail['diagnostics']['correct_temporal_matches_divided_by_ALL_gold_facts']:.4f}, "
        f"прежний гибрид {old_detail['diagnostics']['correct_temporal_matches_divided_by_ALL_gold_facts']:.4f}. "
        'Эта диагностика не заменяет официальный показатель.', '',
        '## Пороги относительно LLM-бейзлайна', '',
        *[f"- {name}: temporal ≥ {check['temporal_min']}, state ≥ {check['state_min']}. "
          f"Оба порога качества пройдены: {'да' if check['quality_thresholds_met'] else 'нет'}."
          for name, check in quality_checks.items()], '',
        'Расхождение опубликованных метрик и пересчёта сохраняется; показаны обе версии. '
        'Даже прохождение порогов качества не подтверждает критерий снижения затрат: '
        'сопоставимых замеров модели-учителя нет.', '',
        '## Качество по уровню ASR-шума', '',
        'Разрез исходного val по выданному noise_level; дополнительный шум в этом прогоне не накладывается. '
        'Этот разрез не использован для выбора модели или правил.', '',
        '| Уровень шума | Звонки | Precision | Recall | Temporal | State |',
        '|---|---:|---:|---:|---:|---:|', *noise_rows, '',
        '## Ресурсы', '',
        'Сравнение задержки систем на одинаковых 24 dev-звонках, CUDA, batch=1:', '',
        '| Система | Звонки | Среднее, мс | p95, мс | Пик RSS процесса, МБ | Пик CUDA, МБ |',
        '|---|---:|---:|---:|---:|---:|',
        *[f"| {name} | {r['calls']} | {r['latency_mean_ms']:.1f} | {r['latency_p95_ms']:.1f} | "
          f"{r['process_peak_rss_mb']:.1f} | {r['gpu_peak_allocated_mb']:.1f} |"
          for name, r in [('Qwen3-0.6B', request_resources), ('Прежний гибрид ~94M', reference_resources)]], '',
        'Одинаковые первые 24 звонка внутреннего dev, 6 CPU threads. '
        'В cuda_batch4 время/звонок — доля времени пакета, а не задержка онлайн-запроса. Остальные режимы — batch=1. '
        'Время включает генерацию и обработку дат; загрузка весов и файловый ввод-вывод исключены. '
        'Первый запрос включён. Пик RSS процесса измерен средствами ОС и включает загрузку модели; '
        'отдельные замеры RSS после звонков также сохранены в JSON.', '',
        'Питание от сети и состояние GPU записаны в hardware_start/hardware_end ресурсных JSON. '
        'Во время предварительной проверки ноутбук работал от батареи с ограничением GPU около 30 Вт; '
        'этот участок не используется для контрольного сравнения скорости.', '',
        '| Режим | Звонки | Среднее, мс | p95, мс | Звонков/час | Пик RSS, МБ | Precision | Recall | Temporal | State |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|', *benchmark, '',
        'INT8 — поканальная динамическая квантизация, экспериментальный режим. На одном dev-примере '
        'первого checkpoint поканальный вариант восстановил JSON после ошибки обычного INT8, '
        'но содержание оставалось хуже FP32. Протокол: `reports/qwen_int8_preflight.json`. '
        'Рабочие варианты — CUDA BF16 и CPU FP32; результаты INT8 не переносятся на них.', '',
        f'Ошибки генерации по режимам на этих 24 звонках: `{json.dumps(backend_generation)}`.', '',
        f"Внешний val CUDA, batch={resources['batch_size']}: среднее амортизированное время "
        f"{resources['latency_mean_ms']:.1f} мс/звонок (не онлайн-задержка), пиковая выделенная память CUDA "
        f"{resources['gpu_peak_allocated_mb']:.1f} МБ. Проверка схемы: `{schema['valid']}`.", '',
        f"Пропускная способность CUDA на val: {resources['calls_per_hour']:.1f} звонков/час; "
        f"{resources['wall_seconds_per_1000_calls']:.1f} секунд на 1000 звонков в пакетном режиме. "
        'Это синхронизированное время запроса на выделенной GPU, включая CPU-подготовку; '
        'это не сумма времени GPU-ядер по профайлеру.', '',
        f'Диагностика генерации на val: `{json.dumps(diagnostics)}`.', '',
        f'Диагностика исходной генерации на dev до выбора дат: `{json.dumps(dev_diagnostics)}`. '
        'Подробности некорректных ответов сохранены в `qwen_dev_generation_errors.jsonl`; '
        'повторная обработка дат не устраняет и не скрывает ошибки исходной генерации.', '',
        '## Границы выводов', '',
        '- Прежний гибрид обучен на полном выданном train; Qwen — только на fit, поскольку dev '
        'оставлен для независимого выбора. Сравнение отражает готовые системы, не изолированный эффект архитектуры.',
        '- Даты без точного дня сохраняются отдельно в `temporal_evidence` API; конкурсная метрика требует точечных дат.',
        '- Некорректный JSON не превращается в придуманные факты; такие окна учитываются в диагностике.',
        '- Размер, скорость и качество не взаимозаменяемы: соответствие диапазону параметров само по себе не доказывает превосходство.',
        '- Замеров LLM-учителя нет. Снижение затрат в 5/10 раз не подтверждено.', '',
        'Воспроизведение: `docs/QWEN.md`. Базовые веса скачиваются отдельно с проверкой revision и SHA-256; '
        'LoRA-адаптер, метаданные и скрипты сохранены в проекте.', '',
    ]
    Path('reports/QWEN_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    for name in ['LICENSE', 'UPSTREAM_SOURCE.json']:
        shutil.copyfile(Path('artifacts/qwen_base') / name, Path('artifacts/qwen') / name)
    shutil.copyfile('data/raw/catalog.json','artifacts/qwen/catalog.json')
    card = [
        '---', 'base_model: Qwen/Qwen3-0.6B', 'library_name: peft', 'license: apache-2.0',
        'language:', '- ru', 'tags:', '- lora', '- information-extraction', '---', '',
        '# ZooBears Qwen fact extractor', '',
        f"LoRA adapter for pinned base revision `{exp['revision']}`. "
        f"Base parameters: {exp['base_parameters']:,}; adapter trainable parameters: {exp['trainable_adapter_parameters']:,}.", '',
        'Download the base separately using `python scripts/download_qwen.py`. '
        'Load the adapter through `compactfacts.qwen.QwenExtractor`; it expands the compact '
        'JSON transport into the competition schema and grounds sources in the input.', '',
        f"Training: {audit['fit_calls']} fit calls; {audit['dev_calls']} account-disjoint dev calls. "
        'No external validation labels used for fitting or model selection.', '',
        f"External val official scores: `{json.dumps(val)}`.", '',
        'See `docs/QWEN.md`, `reports/QWEN_REPORT.md`, `experiment.json` and `data_audit.json` '
        'for reproduction, evaluation scope and limitations. Teacher speedup is not established.', '',
    ]
    Path('artifacts/qwen/README.md').write_text('\n'.join(card), encoding='utf-8')
    print({'parameters': exp['base_parameters'], 'val': val, 'generation': diagnostics}, flush=True)
