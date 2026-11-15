"""Report the completed date-reference experiment, including rejected candidates."""
import json
from pathlib import Path
import sys
import hashlib
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.date_diagnostics import boundary_metrics


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    selection=read('reports/date_pointer_selection.json')
    chosen=selection['candidates'][selection['selected']]
    val=read('reports/date_pointer_val_metrics.json')['official']
    previous=read('reports/v21_val_metrics.json')['official']
    teacher=read('data/raw/baseline_metrics.json')
    schema=read('reports/date_pointer_val_schema.json')
    thresholds={'temporal_min':round(teacher['temporal_recall']-.02,4),
        'state_min':round(teacher['state_accuracy']-.02,4),
        'temporal_passed':val['temporal_recall']>=teacher['temporal_recall']-.02,
        'state_passed':val['state_accuracy']>=teacher['state_accuracy']-.02,
        'temporal_exclusion_boundary':round(teacher['temporal_recall']-.05,4),
        'temporal_exclusion_passed':val['temporal_recall']>=teacher['temporal_recall']-.05,
        'additional_correct_dates_at_fixed_denominator':max(0,math.ceil((teacher['temporal_recall']-.02)*val['n_temporal'])-round(val['temporal_recall']*val['n_temporal'])),
        'additional_correct_states_at_fixed_denominator':max(0,math.ceil((teacher['state_accuracy']-.02)*val['n_state'])-round(val['state_accuracy']*val['n_state'])),
        'parameter_branch_supported':read('reports/teacher_parameter_comparison.json')['parameter_branch_passed'] if Path('reports/teacher_parameter_comparison.json').exists() else False,
        'teacher_gpu_seconds_reduction_measured':False}
    write_json('reports/date_pointer_threshold_check.json',thresholds)
    diagnostics={
        'dev':boundary_metrics(read_jsonl('tmp/splits/dev.jsonl'),read_jsonl('reports/date_pointer_dev_selected_pred.jsonl'),'tmp/splits/dev.jsonl',True),
        'val':boundary_metrics(read_jsonl('data/raw/val.jsonl'),read_jsonl('reports/date_pointer_val_pred.jsonl'),'data/raw/val.jsonl',False)}
    assert diagnostics['dev']['temporal_recall_recomputed']==chosen['metrics']['temporal_recall']
    assert diagnostics['val']['temporal_recall_recomputed']==val['temporal_recall']
    write_json('reports/date_pointer_boundaries.json',diagnostics)
    rows=[]
    for name,m in [('LLM baseline v2.1',teacher),('Qwen после первой итерации',previous),('Выбранная версия',val)]:
        rows.append(f"| {name} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} | {m['grounding_turn']} |")
    comparisons=[]
    for name,item in selection['candidates'].items():
        m=item['metrics'];comparisons.append(f"| {name} | {m['fact_precision']} | {m['fact_recall']} | {m['temporal_recall']} | {m['state_accuracy']} |")
    resources=[]
    for name in ['cuda','cuda_batch4','cpu']:
        r=read(f'reports/date_pointer_benchmark_{name}_pred.resources.json')
        resources.append(f"| {name} | {r['calls']} | {r['latency_mean_ms']:.1f} | {r['latency_p95_ms']:.1f} | {r['process_peak_rss_mb']:.1f} |")
    stress_text=[]
    if Path('reports/date_pointer_stress_metrics.json').exists():
        noisy=read('reports/date_pointer_stress_metrics.json')['official']
        clean=read('reports/date_pointer_stress_clean_metrics.json')['official']
        stress_text=['## Дополнительный ASR-шум','',
            f"Одинаковые {noisy['n_calls']} dev-звонков; seed43, rate0.2. Локальный стресс-тест с сохранением отрицаний, дат и источников, не скрытый тест жюри.",'',
            '| Контроль | Precision | Recall | Temporal | State |','|---|---:|---:|---:|---:|',
            f"| Чистый | {clean['fact_precision']} | {clean['fact_recall']} | {clean['temporal_recall']} | {clean['state_accuracy']} |",
            f"| Шум | {noisy['fact_precision']} | {noisy['fact_recall']} | {noisy['temporal_recall']} | {noisy['state_accuracy']} |",'']
    api_text=[]
    if Path('reports/date_pointer_api_check.json').exists():
        api=read('reports/date_pointer_api_check.json')
        api_text=['## HTTP API и контроль исправления','',
            f"Проверены {len(api['cases'])} HTTP-сценариев; успешны {sum(api['checks'].values())} из {len(api['checks'])} проверок. Все пройдены: {api['all_passed']}. Адаптер проверен по /health. Полные ответы: date_pointer_api_check.json.",'',
            'Исправлено сохранение согласованного окна при повторном подтверждении выезда. Неозвученное начало другого состояния в той же реплике отмечается как неизвестное. В старом тесте цепочки ошибочно требовалась дата, относящаяся к SMS; точные границы теперь проверяются на явно датированном разговоре, а исходный пример проверяет неопределённость. Первоначальный неуспешный отчёт сохранён в date_pointer_api_check_initial.json. Подробности: ../docs/API_DATE_FIX.md.','',
            'После выбора модели выполнена сверка постпроцессора: на dev, val, ASR-контроле и наборах ресурсов ни один конкурсный факт не изменился. Метрики и веса сохранены; обновлены только вспомогательные сведения о неопределённости. Время ресурсов измерено перед этой небольшой CPU-правкой; её накладные расходы измерены отдельно в date_pointer_api_refinement_audit.json.','']
    text=['# Выбор дат моделью по ссылкам на транскрипт','',
        f"Выбрана версия `{selection['selected']}`, модель `{selection['deployed_model']}`. Выбор выполнен на полном внутреннем dev; внешний val не используется для выбора.",'',
        'Предыдущая итерация улучшила полноту и источники, но не прошла temporal/state пороги. Здесь Qwen дополнительно учится выбирать конкретное выражение даты для каждого факта. Модель возвращает номера из таблицы дат текущего звонка; свободная генерация ISO-дат запрещена форматом. Неозвученные дни не становятся обучающими ответами.','',
        'Вход содержит только дату звонка, реплики и автоматически разобранные выражения дат. Эталоны используются только для обучающих ответов. Таблица включает допустимые прошлое/будущее прочтения неоднозначного дня недели. Интервалы вроде «в прошлом месяце» не превращаются в выдуманный точный день.','',
        'Продолжение существующей LoRA: rank=16, 10 092 544 обучаемых параметра, одна эпоха, LR=5e-5, seed=42, вход до 3072 токенов. База Qwen3-0.6B — 596 049 920 параметров. Токены наблюдаемых ссылок на даты получают вес 3, неизвестного завершения — 2, остальные — 1. Используются тот же fit, защищённая ASR-аугментация и 112 синтетических записей; часть записей повторяет короткие шаблоны для усиления этих случаев.','',
        'Проверяются исходная версия и два новых checkpoints; для каждого нового checkpoint — выключенные ссылки, только начала, консервативные границы, полные границы, только окончания и известные начала с полными окончаниями. Два последних режима добавлены после первого dev-сравнения эпохи и до нового val; оба checkpoint проверены на одинаковых шести режимах. Есть ограничения на потерю recall/F1 и сохранение достигнутого порога State на dev. Совпадение повторной обработки кеша с реальным inference проверено для консервативного режима.','',
        '| Модель, внешний val | Precision | Recall | Temporal | State | Grounding turn |',
        '|---|---:|---:|---:|---:|---:|',*rows,'',
        f"Пороги: temporal ≥ {thresholds['temporal_min']}, state ≥ {thresholds['state_min']}. Пройдены: temporal={thresholds['temporal_passed']}, state={thresholds['state_passed']}. Схема: {schema['valid']}.",'',
        f"Граница исключения temporal {thresholds['temporal_exclusion_boundary']}: выше или равен границе — {thresholds['temporal_exclusion_passed']}. При фиксированных нынешних знаменателях до целевых порогов не хватает {thresholds['additional_correct_dates_at_fixed_denominator']} правильно датированных совпадений и {thresholds['additional_correct_states_at_fixed_denominator']} текущих состояний. Это диагностический ориентир: при изменении предсказаний знаменатели тоже могут измениться.",'',
        '| Checkpoint / режим, dev | Precision | Recall | Temporal | State |','|---|---:|---:|---:|---:|',*comparisons,'',
        '| Режим | Звонки | Среднее, мс | p95, мс | Пик RSS, МБ |','|---|---:|---:|---:|---:|',*resources,'',
        'CUDA batch=4 показывает амортизированное время на звонок; остальные режимы — задержку отдельного запроса. Используются одинаковые 24 dev-звонка. Если выбран baseline, повторные измерения не выполняются: используются уже полученные замеры той же версии.','',
        *stress_text,
        *api_text,
        '## Ограничения','',
        '- Если выбран baseline, обучение новых ссылок не дало достаточного результата на dev, и новые веса не разворачиваются.',
        '- Исходные artifacts/qwen и резервная версия artifacts/qwen_v21 сохраняются. Новое обучение находится в artifacts/qwen_dates.',
        '- Начала и окончания отдельно проверены в reports/date_pointer_boundaries.json; это диагностика, не замена официальных метрик.',
        '- Выданный val уже использовался в предыдущем анализе ошибок; независимую проверку даст скрытый или новый набор.',
        '- Исходник официального оценщика не изменён. На dev явно включён локальный обход известного TypeError; на val он выключен.',
        '- Параметрическая ветка сравнения с опубликованной GLM-5.3 обоснована отдельно: reports/TEACHER_PARAMETER_COMPARISON.md. Серверная ревизия учителя неизвестна; снижение GPU-секунд и денег не измерено. Docker-сборка Qwen не проверена.',
        '- Эта итерация оценивает совместный эффект нового формата и дообучения, не отдельный вклад каждого вида аугментации.','',
        'Повторение: сначала завершить run_v21_iteration.py, затем run_date_pointer_iteration.py. --resume используется только при наличии локальных промежуточных файлов.','']
    Path('reports/DATE_POINTER_REPORT.md').write_text('\n'.join(text),encoding='utf8')
    deployment=read('artifacts/deployment.json')
    deployment['qwen_final']={'engine':'qwen','model':selection['deployed_model'],'base_model':'artifacts/qwen_base',
        'date_ranker':'artifacts/date_ranker_v21.joblib','device':'cuda','selected_on':'internal dev'}
    write_json('artifacts/deployment.json',deployment)
    if selection['selected']!='baseline':
        model=Path(selection['deployed_model'])
        metadata=read(model/'experiment.json')
        metadata['date_reference_code_sha256']={name:hashlib.sha256((ROOT/'src/compactfacts'/name).read_bytes()).hexdigest()
            for name in ['qwen_date_format.py','temporal_v2.py','temporal.py']}
        metadata['postprocessing_code_sha256']={name:hashlib.sha256((ROOT/'src/compactfacts'/name).read_bytes()).hexdigest()
            for name in ['qwen.py','date_pointers.py','date_refinement.py','date_evidence.py']}
        write_json(model/'experiment.json',metadata)
        (model/'README.md').write_text('# ZooBears: grounded date references\n\nSee reports/DATE_POINTER_REPORT.md.\n\n'+json.dumps(val,indent=2),encoding='utf8')
    else:
        candidate=Path('artifacts/qwen_dates')
        metadata=read(candidate/'experiment.json')
        metadata.update(selected_for_deployment=False,
                        selection='Rejected against the frozen v2.1 baseline on generated internal dev metrics')
        write_json(candidate/'experiment.json',metadata)
        (candidate/'README.md').write_text(
            '# Experimental adapter: not deployed\n\n'
            'The v2.1 baseline won the generated internal dev comparison. '
            'See reports/DATE_POINTER_REPORT.md and artifacts/deployment.json.\n',encoding='utf8')
    print({'selected':selection['selected'],'val':val,'thresholds':thresholds})


if __name__=='__main__':main()
