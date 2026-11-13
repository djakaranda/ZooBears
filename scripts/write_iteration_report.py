"""Build a sourced progress report from experiment outputs, without hand-entered scores."""
import hashlib,json,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import write_json


def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


if __name__=='__main__':
    source=load('reports/frozen_selection.json');base=load('reports/baseline_recomputed.json')
    previous=load('reports/val_metrics.json')['official'];models={}
    for name in ['encoder_cpu','encoder_int8','encoder_gpu','quality_gpu']:
        models[name]={'metrics':load(f'reports/val_{name}_metrics.json')['official'],
                      'strict':load(f'reports/val_{name}_strict.json'),
                      'resources':load(f'reports/val_{name}.resources.json')}
    primary=models['quality_gpu']['metrics'];encoder=load('artifacts/neural/experiment.json');generator=load('artifacts/generator/experiment.json')
    def table(labels,rows):
        return '| '+' | '.join(labels)+' |\n|'+'|'.join(['---']+['---:']*(len(labels)-1))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
    main_table=table(['Метрика','LLM база, пересчёт','CPU v1','Новый quality'],[
        [label,f'{base[k]:.4f}',f'{previous[k]:.4f}',f'{primary[k]:.4f}'] for k,label in [
            ('fact_precision','Точность фактов'),('fact_recall','Полнота фактов'),('temporal_recall','Даты среди совпавших'),
            ('state_accuracy','Текущее состояние'),('grounding_turn','Реплика-источник')]])
    resources_table=table(['Режим','Precision','Recall','Temporal','State','Среднее, мс','p95, мс','RSS, МиБ','CUDA, МиБ'],[
        [name,*[f"{m['metrics'][k]:.4f}" for k in ['fact_precision','fact_recall','temporal_recall','state_accuracy']],
         f"{m['resources']['latency_mean_ms']:.1f}",f"{m['resources']['latency_p95_ms']:.1f}",
         f"{m['resources']['max_sampled_rss_mb']:.1f}",
         f"{m['resources']['gpu_peak_allocated_mb']:.1f}" if m['resources']['gpu_peak_allocated_mb'] is not None else '—'] for name,m in models.items()])
    dev_table=table(['Внутренний эксперимент','Precision','Recall','Temporal','State'],[
        [name,*[f'{m[k]:.4f}' for k in ['fact_precision','fact_recall','temporal_recall','state_accuracy']]]
        for name,m in source['internal_results'].items()])
    all_files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ['artifacts/neural','artifacts/generator'] for p in (ROOT/folder).iterdir() if p.is_file()}
    write_json('reports/model_manifest.json',{'models':all_files,'selection':'reports/frozen_selection.json'})
    write_json('reports/iteration_comparison.json',{'baseline_recomputed':base,'previous_v1':previous,'models':models,
               'quality_targets':{'temporal':max(.4005,.3961),'state':max(.2994,.3283)},
               'quality_retention_met_against_both_supplied_and_recomputed_baselines':primary['temporal_recall']>=.4005 and primary['state_accuracy']>=.3283,
               'retention_against_supplied_baseline':primary['temporal_recall']>=.3961 and primary['state_accuracy']>=.3283,
               'retention_against_recomputed_baseline':primary['temporal_recall']>=.4005 and primary['state_accuracy']>=.2994,
               'cost_reduction_verified':False})
    verdict='Допуск удержания качества по датам и состоянию выполнен на открытом val относительно обеих версий базы.' if primary['temporal_recall']>=.4005 and primary['state_accuracy']>=.3283 else 'Относительно пересчитанной базы допуск удержания качества пока не выполнен.'
    stress=load('reports/val_stress_metrics.json')['official'] if (ROOT/'reports/val_stress_metrics.json').exists() else None
    stress_section=table(['Проверка quality','Precision','Recall','Temporal','State'],[
        [name,*[f'{m[k]:.4f}' for k in ['fact_precision','fact_recall','temporal_recall','state_accuracy']]]
        for name,m in [('Исходный val',primary),('Дополнительный шум 8%',stress)]]) if stress else 'Дополнительный стресс-тест ещё не завершён.'
    text=f'''# ZooBears: результаты следующей итерации

14 ноября 2026. Все числа ниже получены локальными запусками. Исходный оценщик жюри не изменён.

## Результат на внешнем val

{main_table}

{verdict} Это не оценка на скрытом тесте. Ускорение относительно LLM и снижение стоимости пока не доказаны: нет замеров учителя и его ресурсной конфигурации. Сам по себе малый размер нашей модели этого сравнения не заменяет.

Для приложенного baseline_metrics допустимый temporal начинается с 0.3961, для пересчитанной базы — с 0.4005. Quality получил {primary['temporal_recall']:.4f}: относительно опубликованного файла допуск удержания качества пройден, относительно пересчёта не пройден. Расхождение базы здесь влияет на формальный вывод и требует ответа организаторов.

Дополнительная диагностика по всем эталонным фактам: доля фактов с верными датами начала и окончания выросла с {load('reports/baseline_diagnostics.json')['diagnostics']['correct_temporal_matches_divided_by_ALL_gold_facts']:.4f} у базы до {load('reports/val_quality_gpu_metrics.json')['diagnostics']['correct_temporal_matches_divided_by_ALL_gold_facts']:.4f}. Это неофициальный показатель с другим знаменателем. Официальный temporal остаётся в таблице выше.

## Что изменилось

- Добавлен предобученный русский энкодер с {encoder['parameters']:,} параметрами. Он обучается выбирать реплики, атрибуты, частые значения, объект, полярность, границы фрагмента и кандидата даты.
- Добавлен seq2seq-нормализатор с {generator['parameters']:,} параметрами для открытых значений. В выбранном quality-режиме он работает только с тарифами. Для действий и состояний он ухудшил dev-результат, поэтому эти ветки сохранили прежнюю нормализацию.
- Временная логика различает несколько событий в одной реплике, явное окончание «до», прошедшие и будущие даты, календарные границы и типичные ASR-подмены. Интервалы вроде «в прошлом месяце» не превращаются в выдуманный точный день.
- Переходы закрываются отдельно для каждого объекта. Операторское предложение с последующим отказом клиента отбрасывается.
- Добавлен отдельный ledger истории нескольких звонков с источниками call_id/turn_idx и изоляцией аккаунтов. Он не подмешивает старые реплики в предсказания текущего звонка.
- Добавлены локальная HTML-демонстрация, вторичная строгая оценка объектов, стресс-тест ASR и проверки модели/API.

## Разделение данных и выбор компонентов

Внутри train зафиксировано 3334 fit-звонка и 592 dev-звонка без пересечения аккаунтов. Все решения о головах, порогах, генераторе и датах приняты на этом dev. Затем модели переобучены на полном train с фиксированными выбранными настройками. Внешний val используется только для итогового измерения.

{dev_table}

Сравнение вариантов и причины выбора записаны в `reports/frozen_selection.json`. Метаданные финального обучения содержат seed, revision базовых весов, хэш списка train call_id, историю loss и размер модели. Файлы зафиксированы хэшами в `reports/model_manifest.json`.

## Кривая ресурсов

{resources_table}

Windows, Core i7-12700H, RTX 3060 Laptop 6 ГБ. Запуски последовательные после обучения. Один звонок за запрос, до 64 реплик в нейросетевом батче внутри звонка. Первая обработка включена, загрузка весов и JSON I/O исключены. Для GPU добавлена синхронизация. RSS измерен после запросов и не является непрерывным пиком. CUDA — максимум выделенной PyTorch-памяти, не вся память драйвера. CPU-замеры используют установленный CUDA-wheel PyTorch, поэтому память CPU-only Docker может отличаться.

`encoder_cpu`, `encoder_gpu` — энкодер и правила. `encoder_int8` — динамическая INT8-квантизация линейных слоёв на CPU. `quality_gpu` дополнительно использует генератор тарифов. Все параметры режимов и сырые времена сохранены в `reports/val_*.resources.json`.

## Честная интерпретация качества

Официальная temporal_recall условна на совпавших фактах. В дополнительных `*_strict.json` полные имена объектов должны совпадать fuzzy >= 0.8, состояние считается по объекту, а полнота дат делится на все эталонные факты. Это отдельная диагностика и не подмена метрик жюри.

Старые даты, не заданные точным выражением, по-прежнему являются проблемой. При отсутствии однозначного начала выход использует время наблюдения — started_at звонка. Это fallback, а не подтверждённая дата начала события. Выравнивание quote с ASR-текстом используется только при достаточно высоком сходстве. Границы источника требуют дальнейшего улучшения.

Официальный grounding_span проверяет любое пересечение интервалов. При отсутствии уверенного выравнивания выход может ссылаться на всю реплику. Поэтому высокий grounding_span не означает точных границ аргумента.

Часть эталонных объектов имеет тип other или технический ID вместо естественного имени. Расхождение baseline_metrics с пересчётом также сохранено в отчётах. Вопросы организаторам подготовлены в `docs/QUESTIONS_FOR_ORGANIZERS.md`, сообщения никому не отправлялись.

## Проверки и ограничения поставки

- Unit-тесты дат, переходов, отрицательных предложений, строгих метрик и шума.
- Независимость инференса от facts, неизменность входа, пустой звонок, произвольные idx и буквальные цитаты.
- HTTP /health, /extract, HTML-демонстрация и отказы на некорректный JSON.
- Структура предсказаний проверена на всём val для каждого режима.
- Dockerfile для базового и нейросетевого CPU-режимов подготовлены. Docker/Podman в окружении отсутствует, поэтому образы не собраны и не проверены.
- Локальный дополнительный ASR-шум не равен скрытому adversarial-тесту жюри. Его результаты хранятся отдельно.

{stress_section}

Стресс-тест использует seed=43 и дополнительные фонетические подмены/удаления символов сверх исходного шума. Семантические значения и даты эталона остаются прежними, недействительные после искажений символьные границы очищаются. Падение качества показывает, что устойчивость к более сильному шуму ещё требует обучения с аугментациями.

## Воспроизведение

См. README и `docs/EXPERIMENTS.md`. Базовые модели: cointegrated/rubert-tiny2 и cointegrated/rut5-small, MIT согласно их карточкам. Неизменённые карточки и immutable revision сохранены рядом с весами. Транскрипты не отправлялись в API внешней модели.
'''
    (ROOT/'reports/ITERATION_2_REPORT.md').write_text(text,encoding='utf-8');print(text)
