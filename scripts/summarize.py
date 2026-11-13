"""Generate the first experiment report from saved measurements."""
import json
from pathlib import Path
import platform
import sys
import importlib.metadata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import write_json


def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))


if __name__=='__main__':
    audit=load('reports/data_audit.json'); train=load('artifacts/cpu_baseline.json')
    supplied=load('data/raw/baseline_metrics.json'); base=load('reports/baseline_recomputed.json')
    metrics=load('reports/val_metrics.json'); balanced=metrics['official']
    r=load('reports/val_pred.resources.json')
    modes={}
    for mode,suffix in [('balanced',''),('precise','_precise'),('recall','_recall')]:
        modes[mode]={'metrics':load(f'reports/val_metrics{suffix}.json')['official'],
                     'resources':load(f'reports/val_pred{suffix}.resources.json')}
    environment={'os':platform.platform(),'python':sys.version,'machine':platform.machine(),'cpu':platform.processor(),
                 'logical_cpus':__import__('os').cpu_count(),
                 'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','rapidfuzz','joblib','psutil']}}
    write_json(ROOT/'reports/environment.json',environment)
    write_json(ROOT/'reports/comparison.json',{'supplied_baseline':supplied,'recomputed_baseline':base,'modes':modes})
    labels={'fact_precision':'Точность фактов','fact_recall':'Полнота фактов','temporal_recall':'Даты среди совпавших фактов',
            'state_accuracy':'Текущее состояние','grounding_turn':'Реплика-источник'}
    table='\n'.join(f'| {label} | {supplied[key]:.4f} | {base[key]:.4f} | {balanced[key]:.4f} | {balanced[key]-base[key]:+.4f} |' for key,label in labels.items())
    modes_table='\n'.join(f"| {mode} | {v['metrics']['fact_precision']:.4f} | {v['metrics']['fact_recall']:.4f} | {v['metrics']['temporal_recall']:.4f} | {v['metrics']['state_accuracy']:.4f} | {v['resources']['latency_mean_ms']:.1f} |" for mode,v in modes.items())
    attr_table='\n'.join(f"| {attr} | {v['fact_precision']:.4f} | {v['fact_recall']:.4f} | {v['temporal_recall'] if v['temporal_recall'] is not None else '—'} | {v['grounding_turn'] if v['grounding_turn'] is not None else '—'} |" for attr,v in metrics['per_attribute'].items())
    content=f'''# Первый результат CompactFactExtraction

Дата: 14 ноября 2026. Это стартовый CPU-прототип, не финальная заявка.

## Что сделано

- Разобран комплект кейса, сохранены копии исходных данных и неизменённого оценщика.
- Проведён аудит train/val: структура, пересечение аккаунтов, цитаты, спаны, словарь.
- Обучены пять линейных классификаторов и индекс нормализации фактов на train.
- Пороги подобраны на внутреннем групповом срезе train; затем модель обучена на
  всём train. Seed=42. Метки val не используются для обучения и инференса.
- Получены JSONL-предсказания для всех 383 звонков, официальный пересчёт и разбор
  по типам фактов. Есть три заранее заданных режима порога, замеры CPU и памяти.
- Реализованы CLI, HTTP `/health` и `/extract`, правила дат, закрытие интервалов,
  проверки выходной схемы, unit- и HTTP-проверки. Подготовлен Dockerfile.

## Сравнение на val

| Метрика | В файле baseline_metrics | База, пересчёт | Наш balanced | Разница с пересчётом |
|---|---:|---:|---:|---:|
{table}

**Критерии кейса пока не выполнены:** даты и текущее состояние уступают базе.
Темпоральная метрика ниже базы более чем на 0.05. Рост precision/recall сам по себе
не компенсирует этот провал. Сравнение относится только к открытому val, не к тесту.
Даже при сравнении с исходным baseline_metrics вывод о невыполнении не меняется.

Приложенный baseline_metrics не воспроизводится на приложенных val и baseline_pred
тем же case_metrics. Для честного сравнения основная колонка базы пересчитана.
Оригинальный файл сохранён; оценщик не менялся. Различие нужно уточнить у организаторов.

## Ресурсы и режимы

Обучение заняло {train['seconds']:.1f} с. Веса и поисковый индекс занимают
{train['model_bytes']/1e6:.1f} МБ (десятичные). Линейные головы содержат
500 005 коэффициентов вместе с intercept; это не число параметров нейросети:
дополнительно хранятся TF-IDF, словарь и разреженный индекс примеров.

CPU: Intel Core i7-12700H, Windows, 20 логических потоков. Обработка последовательная,
один звонок за раз; учтён первый вызов, исключены загрузка модели и JSON I/O.
В balanced среднее {r['latency_mean_ms']:.1f} мс, p50 {r['latency_p50_ms']:.1f} мс,
p95 {r['latency_p95_ms']:.1f} мс. Максимальный RSS, измеренный **между** вызовами,
{r['max_sampled_rss_mb']:.1f} МиБ; это не непрерывно измеренный пик.

| Режим | Precision | Recall | Temporal | State | Среднее, мс |
|---|---:|---:|---:|---:|---:|
{modes_table}

Пороги меняют число кандидатов и работу retrieval, но это ограниченная проба
ресурсов, а не сравнение трёх разных моделей. Прогоны выполнены последовательно,
без параллельной нагрузки других наших режимов. GPU не использовалась; GPU-бенчмарк,
измерения учителя и доказательство ускорения/снижения стоимости пока отсутствуют.

## По типам фактов

| Атрибут | Precision | Recall | Temporal | Grounding turn |
|---|---:|---:|---:|---:|
{attr_table}

Нормализация выбирает значение из обучающих примеров. Она не генерирует новые
формулировки: {audit['val_facts_value_not_seen_verbatim_in_train']} из 1845 эталонных
фактов val имеют значения, не встречавшиеся в train дословно. Это не абсолютный
потолок fuzzy-метрики, но существенное ограничение архитектуры.
Даты вычисляются по явным выражениям, иначе используется день/время звонка.
Исторические даты без текстового указания такой подход не восстанавливает.
Переходы, несколько объектов и подтверждения договорённостей требуют улучшения.

`grounding_span = 1.0` не означает точные границы: наш span покрывает всю реплику,
а оценщик проверяет любое пересечение. Более полезен текущий grounding_turn.
Диагностика полных имён объектов: только {metrics['diagnostics']['full_name_agreement_among_official_matches']:.1%}
официально совпавших пар проходят similarity >= 0.8 по полному имени.
Это дополнительная проверка уже найденных пар, не самостоятельная строгая F1.

## Аудит и проверки

Train: {audit['train']['calls']} звонков, {audit['train']['turns']} реплики,
{audit['train']['facts']} фактов. Val: 383 звонка, 1845 фактов.
Пересечений call_id: {audit['overlapping_call_ids']}; аккаунтов: {audit['overlapping_accounts']}.
`catalog.json` отсутствует; словарь восстановлен исключительно из train.

В train {audit['train']['observations'].get('quote_not_exact_substring',0)} цитат
не являются точной подстрокой указанной реплики; у
{audit['train']['observations'].get('span_quote_disagreement',0)} спанов текст не
совпадает с quote. Встречается тип объекта `other`, которого нет в схеме.
Эти наблюдения не доказывают семантическую ошибку каждого факта: цитаты могли
быть сохранены до ASR-шума. Нужны выравнивание и разбор, а не безусловное удаление.

- `reports/schema_check.json`: все звонки присутствуют, обязательные поля,
  даты и ссылки на реальные реплики прошли проверку.
- `reports/runtime_check.json`: загрузка весов, независимость от gold facts,
  пустой диалог, непоследовательные idx, реальный HTTP и обработка плохого запроса.
- 5 unit-тестов: календарные границы, относительные даты, валидация запроса.
- Dockerfile подготовлен; Docker в окружении не найден, образ не собран и не проверен.

## Что делать дальше

Приоритет 1 — даты и текущее состояние: связать временные выражения с конкретными
событиями, научиться различать старое/новое состояние и подтверждённые переходы.
Приоритет 2 — обучаемый энкодер и открытые значения, отдельно договорённости.
Приоритет 3 — ASR-стресс-тест, CPU/GPU-кривая ресурсов, Docker и демонстрация.

Полный план и разбор ограничений официального оценщика: `docs/PLAN.md`.
Численные первоисточники: `reports/val_metrics*.json`, `reports/comparison.json`,
`reports/data_audit.json`, `reports/environment.json`, `artifacts/cpu_baseline.json`.
Команды воспроизведения: `README.md`.
'''
    (ROOT/'reports/START_REPORT.md').write_text(content,encoding='utf-8')
    print(content)
