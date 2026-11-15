"""Publish completed v2.1 measurements, without claiming unverified quality gains."""
import json
from pathlib import Path
import shutil
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.date_diagnostics import boundary_metrics


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    selection=read('reports/v21_checkpoint_selection.json')
    backup=read('reports/v21_original_adapter_backup.json')
    assert all(hashlib.sha256((Path('artifacts/qwen')/name).read_bytes()).hexdigest()==digest
               for name,digest in backup['sha256'].items()), 'Original adapter changed'
    chosen=selection['candidates'][selection['selected']]
    dev=chosen['metrics'];val=read('reports/v21_val_metrics.json')['official']
    base=read('data/raw/baseline_metrics.json')
    assert base['metrics_version']==val['metrics_version']=='2.1'
    schema=read('reports/v21_val_schema.json')
    dev_gold=read_jsonl('tmp/splits/dev.jsonl')
    boundaries={
        'baseline_dev':boundary_metrics(dev_gold,read_jsonl('reports/v21_dev_baseline_selected_pred.jsonl'),'tmp/splits/dev.jsonl',True),
        'selected_dev':boundary_metrics(dev_gold,read_jsonl('reports/v21_dev_selected_pred.jsonl'),'tmp/splits/dev.jsonl',True),
        'selected_val':boundary_metrics(read_jsonl('data/raw/val.jsonl'),read_jsonl('reports/v21_val_pred.jsonl'),'data/raw/val.jsonl',False)}
    assert boundaries['selected_dev']['temporal_recall_recomputed']==dev['temporal_recall']
    assert boundaries['selected_val']['temporal_recall_recomputed']==val['temporal_recall']
    write_json('reports/v21_date_boundary_metrics.json',boundaries)
    checks={'temporal_min':round(base['temporal_recall']-.02,4),'state_min':round(base['state_accuracy']-.02,4),
            'temporal_passed':val['temporal_recall']>=base['temporal_recall']-.02,
            'state_passed':val['state_accuracy']>=base['state_accuracy']-.02,
            'resource_reduction_against_teacher_verified':False}
    write_json('reports/v21_threshold_check.json',checks)
    model=Path('artifacts/qwen_v21')
    for name in ['LICENSE','UPSTREAM_SOURCE.json']:
        shutil.copyfile(Path('artifacts/qwen_base')/name,model/name)
    shutil.copyfile('data/raw/catalog.json',model/'catalog.json')
    deployment=read('artifacts/deployment.json')
    deployment['qwen_v21']={'engine':'qwen','model':str(model).replace('\\','/'),'base_model':'artifacts/qwen_base',
                            'device':'cuda',**chosen['config']}
    write_json('artifacts/deployment.json',deployment)
    rows=[]
    measurements=[('LLM baseline v2.1',base)]
    if Path('reports/v21_baseline_val_metrics.json').exists():
        measurements.append(('Исходная Qwen: свежий val',read('reports/v21_baseline_val_metrics.json')['official']))
    measurements += [('Выбранная модель: dev',dev),('Выбранная модель: val',val)]
    for label,m in measurements:
        rows.append(f"| {label} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} | {m['grounding_turn']} |")
    candidates=[]
    for name,item in selection['candidates'].items():
        m=item['metrics'];candidates.append(f"| {name} | {item['variant']} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} |")
    resources=[]
    for name in ['cuda','cuda_batch4','cpu']:
        r=read(f'reports/v21_benchmark_{name}_pred.resources.json')
        m=read(f'reports/v21_benchmark_{name}_metrics.json')['official']
        resources.append(f"| {name} | {r['calls']} | {r['latency_mean_ms']:.1f} | {r['latency_p95_ms']:.1f} | {r['process_peak_rss_mb']:.1f} | {m['temporal_recall']} | {m['state_accuracy']} |")
    stress=read('reports/v21_stress_metrics.json')['official'];clean=read('reports/v21_stress_clean_metrics.json')['official']
    predictions=read_jsonl('reports/v21_val_pred.jsonl')
    diagnostics={k:sum(p.get('generation_diagnostics',{}).get(k,0) for p in predictions)
                 for k in ['windows','invalid_json_windows','invalid_rows','truncated_windows','short_rule_fallback']}
    text=['# ZooBears: итерация на пакете v2.1','',
          f"Основная модель: Qwen3-0.6B, 596 049 920 параметров. Выбран checkpoint `{selection['selected']}` на полном dev; внешний val не использован для выбора.",'',
          'Дополнительное обучение: продолжение LoRA на новом fit, 1 эпоха, LR=5e-5, rank=16. Добавлены защищённая ASR-аугментация 8% fit и 112 целевых коротких примеров из каталога: переходы тарифов/статусов, подтверждения, отказы и разные устройства. Примеры не используют val. Checkpoints сохраняются в середине и в конце эпохи.','',
          'Если выбран baseline, новые веса отвергнуты по dev: это явно показано в таблице, а успешность пробного обучения не выдаётся за прирост качества.','',
          '| Система/выборка | Precision | Recall | Temporal | State | Grounding turn |','|---|---:|---:|---:|---:|---:|',*rows,'',
          f"Пороги: temporal ≥ {checks['temporal_min']}, state ≥ {checks['state_min']}. Результат: temporal — {checks['temporal_passed']}, state — {checks['state_passed']}. Схема на val: {schema['valid']}.",'',
          '## Выбор checkpoint и обработчика дат','',
          'Полные генерации dev для каждого checkpoint. Цель: максимум меньшего нормированного temporal/state, с ограничением потери recall/F1. Все сравнения сохраняются, включая ухудшения.','',
          '| Checkpoint | Даты/объекты | Precision | Recall | Temporal | State |','|---|---|---:|---:|---:|---:|',*candidates,'',
          'Расширенный ранжировщик дат рассматривает контекст всего звонка с лексическим фильтром и отдельно обучает начало/окончание. Цепочки строятся по объекту. Консервативное связывание объектов не объединяет общий «роутер» с двумя разными нумерованными устройствами. Эти компоненты включаются только при улучшении на dev.','',
          'Дополнительно сравнены end-only варианты: корректируют только окончание, сохраняют начало, не открывают заново исторические состояния и не ставят окончание раньше начала. Для каждого checkpoint проверяется одинаковый набор вариантов.','',
          '## Ресурсы','',
          '| Режим | Звонки | Среднее, мс | p95, мс | Пик RSS, МБ | Temporal | State |','|---|---:|---:|---:|---:|---:|---:|',*resources,'',
          'Одинаковые первые 24 dev-звонка. cuda_batch4 показывает амортизированное время на звонок, остальные — время отдельного запроса. Загрузка исключена из задержки, включена в пик RSS. Питание/частоты сохранены в ресурсных JSON. INT8 ранее показал ошибки содержания и JSON и не используется.','',
          '## Дополнительный шум','',
          f"Одинаковые {stress['n_calls']} dev-звонков, защищённые ASR-подстановки, seed=43, rate=0.2. Это локальный стресс-тест, не скрытый тест жюри. Чистый контроль: `{json.dumps(clean)}`. Шум: `{json.dumps(stress)}`.",'',
          f"Диагностика генерации val: `{json.dumps(diagnostics)}`.",'',
          'Начало и окончание проверены отдельно: reports/v21_date_boundary_metrics.json. Совпадение обеих границ воспроизводит официальный temporal_recall; дополнительные показатели по каждой границе не заменяют конкурсную метрику.','',
          '## Методика и ограничения','',
          '- Пакет организаторов и исходник case_metrics.py сохранены без изменения. Каталог загружается перед вызовом evaluate.',
          '- На dev явно включён локальный обход двух TypeError в recover_dates v2.1; счётчики обходов записаны в каждом отчёте. На внешнем val обход не включён.',
          '- Выданный внешний val уже использовался в прежнем анализе ошибок. В этой итерации выбор checkpoint и конфигурации выполнен на dev; независимую итоговую проверку обобщения даст новый или скрытый набор.',
          '- Если приведён свежий val исходной Qwen, это контроль исходных весов и прежнего ranker в CUDA BF16 batch=4. Его результаты не используются для изменения выбора модели.',
          '- Исходные v1-данные сохранены в data/releases/v1; старые QWEN_REPORT.md и qwen_* относятся к прежнему эксперименту.',
          '- Исходный адаптер artifacts/qwen проверен по сохранённым SHA-256 и не изменён. Новая выбранная версия хранится отдельно.',
          '- Окружение сохранено в reports/v21_environment.json; предварительная проверка полного fit — reports/v21_training_preflight.json. Продолжение LoRA начинает новый оптимизатор и не является точным возобновлением старого optimizer state.',
          '- Диагностика dev: reports/v21_dev_date_observability.json. Оценщик включает факт при восстановимости одной границы, но проверяет обе; отсутствие даты во встроенном банке само по себе не доказывает невозможность её восстановления.',
          '- Неизвестные даты и периоды показываются в temporal_evidence; скрытые точные дни не подставляются.',
          '- Ресурсное превосходство относительно LLM-учителя не подтверждено. Критерий допускает либо ≥10× по параметрам, либо ≥5× по GPU-секундам.',
          '- Наличие Dockerfile не является подтверждением сборки Qwen-контейнера.','',
          'Полное повторение: после установки зависимостей, загрузки базы и prepare_dev.py запустить `python scripts/run_v21_iteration.py` без --resume. Флаг --resume применим только к локальному прогону с сохранёнными tmp и checkpoints.','']
    Path('reports/V21_ITERATION_REPORT.md').write_text('\n'.join(text),encoding='utf8')
    (model/'README.md').write_text('# ZooBears Qwen v2.1\n\nSelected on internal dev. See reports/V21_ITERATION_REPORT.md.\n\n'+json.dumps(val,indent=2),encoding='utf8')
    print({'selected':selection['selected'],'val':val,'checks':checks})


if __name__=='__main__':main()
