"""Build the release report only after completed inference and API checks."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import read_jsonl,write_json
from compactfacts.date_diagnostics import boundary_metrics


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    val=read('reports/sprint_val_metrics.json')['official']
    previous=read('reports/date_pointer_val_metrics.json')['official']
    dev=read('reports/sprint_dev_selected_metrics.json')
    selection=read('reports/sprint_selection.json')
    api=read('reports/sprint_api_check.json')
    schema=read('reports/sprint_val_schema.json')
    progress=read('reports/sprint_val_pred.progress.json')
    suite=read('reports/sprint_new_suite_gpu_report.json')
    before_suite=read('reports/new_suite_v21_current_report.json')
    assert suite['strict']['passed']>=before_suite['strict']['passed']
    teacher=read('data/raw/baseline_metrics.json')
    parameters=read('reports/teacher_parameter_comparison.json')
    identity=read('reports/sprint_val_identity.json')
    generated=read_jsonl('reports/sprint_val_pred.jsonl')
    generation={k:sum(p.get('generation_diagnostics',{}).get(k,0) for p in generated)
                for k in ['windows','invalid_json_windows','invalid_rows','truncated_windows']}
    assert progress['complete'] and progress['processed_calls']==383
    assert val['n_calls']==383 and schema['valid'] and api['all_passed']
    assert progress['invalid_json_windows']==0
    assert digest('artifacts/qwen_sprint/adapter_model.safetensors')==identity['weights_sha256']
    assert all(digest(Path('src/compactfacts')/n)==h for n,h in identity['sources'].items())
    assert all(digest(p)==h for p,h in identity['auxiliary_models'].items())
    baseline=read('reports/sprint_baseline_hashes.json')
    assert all(digest(Path('artifacts/qwen_dates')/n)==h for n,h in baseline.items())
    metadata=read('artifacts/qwen_sprint/experiment.json')
    old_metadata=read('artifacts/qwen_dates/experiment.json')
    frozen=old_metadata['date_reference_code_sha256']
    assert all(digest(Path('src/compactfacts')/n)==h for n,h in frozen.items())
    assert digest('scripts/train_qwen.py')==metadata['training_code_sha256']
    assert 'Ran 92 tests' in Path('reports/sprint_unit_tests.txt').read_text(encoding='utf8')
    thresholds={'temporal_min':round(teacher['temporal_recall']-.02,4),'state_min':round(teacher['state_accuracy']-.02,4),
        'temporal':val['temporal_recall'],'state':val['state_accuracy'],
        'temporal_passed':val['temporal_recall']>=teacher['temporal_recall']-.02,
        'state_passed':val['state_accuracy']>=teacher['state_accuracy']-.02,
        'temporal_exclusion_boundary':round(teacher['temporal_recall']-.05,4),
        'parameter_branch_supported':parameters['parameter_branch_passed'],
        'teacher_gpu_seconds_reduction_measured':False}
    assert thresholds['temporal_passed'] and thresholds['state_passed']
    write_json('reports/sprint_threshold_check.json',thresholds)
    boundaries={
        'dev':boundary_metrics(read_jsonl('tmp/splits/dev.jsonl'),read_jsonl('reports/sprint_dev_selected_pred.jsonl'),'tmp/splits/dev.jsonl',True),
        'val':boundary_metrics(read_jsonl('data/raw/val.jsonl'),read_jsonl('reports/sprint_val_pred.jsonl'),'data/raw/val.jsonl',False)}
    assert boundaries['val']['temporal_recall_recomputed']==val['temporal_recall']
    write_json('reports/sprint_date_boundary_metrics.json',boundaries)
    resources={name:read(f'reports/sprint_{name}_pred.resources.json') for name in ['val','benchmark_cuda','benchmark_cpu']}
    for name in ['benchmark_cuda','benchmark_cpu']:assert resources[name]['calls']==24
    audit={'created_at':datetime.now(timezone.utc).isoformat(),'selected':selection['selected'],
        'full_val_complete':True,'schema_valid':True,'api_checks':api['checks'],'unit_tests':92,
        'generation_diagnostics':generation,
        'baseline_files_unchanged':True,'frozen_date_reference_code':frozen,'adapter_sha256':identity['weights_sha256'],
        'full_val_runtime_identity':identity,'thresholds':thresholds,'base_parameters':metadata['base_parameters'],
        'new_suite_cuda_strict':suite['strict']['passed'],'new_suite_total':suite['strict']['total'],
        'external_val_used_for_checkpoint_selection':False,'external_val_used_for_release_acceptance':True,
        'docker_built':False,'second_neural_date_verifier_deployed':False}
    write_json('reports/sprint_release_audit.json',audit)
    table='| Версия | Precision | Recall | Temporal | State |\n|---|---:|---:|---:|---:|\n'
    for label,m in [('До спринта: qwen_dates',previous),('Релиз: qwen_sprint',val)]:
        table+='| '+label+' | '+' | '.join(f"{m[k]:.4f}" for k in ['fact_precision','fact_recall','temporal_recall','state_accuracy'])+' |\n'
    attributes='| Атрибут | Precision | Recall | Temporal | State |\n|---|---:|---:|---:|---:|\n'
    for attr,m in read('reports/sprint_val_metrics.json')['per_attribute'].items():
        attributes+='| '+attr+' | '+' | '.join(f'{m[k]:.4f}' if m[k] is not None else '—'
            for k in ['fact_precision','fact_recall','temporal_recall','state_accuracy'])+' |\n'
    rt='| Режим | Звонков | Среднее, мс/звонок | p95, мс | Peak RAM, МБ | Peak CUDA, МБ |\n|---|---:|---:|---:|---:|---:|\n'
    for name,r in resources.items():
        peak=r.get('gpu_peak_allocated_mb')
        rt+=f"| {name} | {r['calls']} | {r['latency_mean_ms']:.1f} | {r['latency_p95_ms']:.1f} | {r['process_peak_rss_mb']:.1f} | {f'{peak:.1f}' if peak is not None else '—'} |\n"
    text=f"""# Итог спринта ZooBears, 16 ноября2026

Оба минимальных порога v2.1 пройдены на полном внешнем val из383 звонков.
Temporal **{val['temporal_recall']:.4f} ≥ {thresholds['temporal_min']:.4f}**;
State **{val['state_accuracy']:.4f} ≥ {thresholds['state_min']:.4f}**.
Это результат открытого набора, а не гарантия на скрытом тесте.

{table}
Все четыре показателя выросли относительно непосредственно предыдущей рабочей
версии. Нельзя сопоставлять эти агрегаты с прежними94,75% precision статусов
заявок: тот показатель относился к отдельному типу фактов.
Grounding turn **{val['grounding_turn']:.4f}**, span **{val['grounding_span']:.4f}**.
Официальный Temporal условен: его знаменатель —{val['n_temporal']} сопоставленных
фактов с восстанавливаемой датой, а не все1845gold-фактов. State считает
{val['n_state']} проверяемых текущих состояний. Это определения организаторов.
Схема, цитаты, spans и исходные индексы проверены: **{schema['facts']} фактов**,
нулевое число отсутствующих звонков и ошибочных JSON-окон. Формальная валидность
цитаты не означает правильность факта; это измеряет официальный оценщик.
Некорректных строк внутри синтаксически валидного ответа отброшено:
**{generation['invalid_rows']}**, окон с достижением лимита ответа:
**{generation['truncated_windows']}**. Все принятые факты проверены отдельно.

## Что изменено

Продолжен существующий LoRA, без обучения базы с нуля. Упор на короткие
положительные фрагменты, цепочки состояний, несколько объектов и явные даты.
Добавлены обученные на fit CPU-ранжировщики начала/окончания, статистика
относительных периодов и календарные ограничения. Границы неподтверждённых
состояний не превращаются в метки точного дня при обучении ранжировщика.

Пройден полный цикл одной эпохи:2240,26секунд. Выбран
**{selection['selected']}**; step-100 сохранился отдельно от конца эпохи.
Полный внутренний dev592 звонка: P{dev['fact_precision']:.4f},
R{dev['fact_recall']:.4f}, T{dev['temporal_recall']:.4f}, S{dev['state_accuracy']:.4f}.
Dev Temporal остаётся ниже целевого порога; финальное прохождение — именно на val.
Сравнивались{len(selection['candidates'])} конфигураций:3 набора весов ×19политик дат.
Выбор checkpoint сделан по dev, с ограничениями на recall/F1.

API выявил регрессию: будущий четверг в договорённости трактовался как прошлый
из-за исторической нумерации ссылок. Исправление использует существующий
future-aware parser только при интерпретации связанной даты договорённости;
таблица входных ссылок и веса не менялись. Сначала проверен полный dev,
затем заново сгенерирован полный val. Аудит: sprint_weekday_fix.json.

## Данные и обучение

Исходный train3926:3334fit/592dev, разделение по аккаунтам, пересечение0.
Внешний val383. Для спринта выбраны800реальных fit-звонков, добавлены640новых
синтетических звонков и40ASR-копий:1545окон,1 826 213токенов,165 730целевых.
SHA-256 корпуса:33e9d34298337384f0adc1e4a40884853e1ac3a5576584b8762891edc5387cae.
43звонка нового набора исключены из обучения и сохранены для регрессий.

Qwen3-0.6B:596 049 920базовых параметров,10 092 544обучаемых LoRA;
rank16,alpha32,dropout0.05,all-linear,LR2e-5,seed73,batch1/accumulation8,
AdamW,weight_decay0.01,clip1,BF16 CUDA,gradient checkpointing.
Вход3072/ответ512; ни один обучающий ответ не обрезан.
Веса токенов: положительная ссылка даты3, закрытое неизвестное окончание2.
Оптимизатор запускается заново; исходные веса и формат ссылок защищены хэшами.

С учётом LoRA и всех CPU-моделей консервативно{parameters['student_unmerged_total']:,}
скаляров/элементов параметризации, округлённая верхняя оценка610млн.
Обоснование ветки сокращения параметров — TEACHER_PARAMETER_COMPARISON.md.
Сопоставимого измерения GPU-секунд или денежной стоимости учителя нет.

## Проверки и ресурсы

92unit-теста пройдены. Настоящий CUDA API:7сценариев,11проверок,
все пройдены: цепочка явных дат, окно визита, отказ, разные устройства,
неопределённый месяц и неизвестное начало. Реальный адаптер подтверждён health.

Новый43-case набор, CUDA batch1 через HTTP API:
**{suite['strict']['passed']}/{suite['strict']['total']}** строгих сценариев,
до спринта **{before_suite['strict']['passed']}/{before_suite['strict']['total']}**.
Он не был скопирован в обучение, но его классы ошибок использовались для
проектирования шаблонов; это регрессионный показатель, не скрытый тест.
Оценщик самого набора исключает2звонка из-за своего TypeError;
строгие проверки всё равно выполнены на всех43. Официальные агрегаты этого
маленького набора не заменяют383val-звонка. До финальной календарной правки
был также отдельный CPU-прогон26/43 — sprint_new_suite_cpu_report.json.

{rt}
CPU/CUDA batch1 измерены на одних и тех же24dev-звонках прежнего benchmark.
Среднее val batch4 — амортизированное пакетное время на звонок, не задержка
одного HTTP-запроса. Загрузка модели не включена в задержку; peak RAM учитывает
загрузку. GPU wall-time proxy не равен профилированию ядер или стоимости.

Начало и окончание проверены раздельно в sprint_date_boundary_metrics.json;
совпадение обеих границ воспроизводит официальный Temporal. Оценщик val
оригинальный, без ремонта. На dev включён явный локальный обход двух известных
TypeError оценщика; это записывается в диагностике. Исходник оценщика не изменён.

{attributes}
Главный оставшийся резерв — состояния услуг, тарифные переходы и точность
договорённостей. Высокое качество статусов заявок не переносится автоматически
на остальные атрибуты. Эти ошибки не скрываются средним показателем.

## Неопределённость, отклонённые варианты и ограничения

API сохраняет диапазон для «в прошлом месяце» и unknown для неназванной даты.
Статистическая точечная оценка некоторых относительных периодов нужна для
обязательного ISO-поля конкурсной схемы; она отдельно помечена range_estimate
и не выдаётся за явно произнесённый день. Привязка всегда остаётся внутри
названного диапазона. Источники и supporting_sources доступны в ответе.

Zero-shot date-only проверяющий на второй Qwen отклонён:60из63фрагментов
нарушили формат, улучшения на48dev-звонках не было. В релизе нет второй
нейросети и нет дообученного verifier. Основной экстрактор один.

Валидатор уже использовался в прежних анализах ошибок; он не полностью
независим. На скрытом тесте возможна регрессия. Не все43сложных сценария
пройдены. Dockerfile обновлён, но Docker на этой машине отсутствует,
поэтому образ не собран. Побитовая идентичность CUDA/CPU не обещается.

Старые qwen/qwen_v21/qwen_dates сохранены; файлы qwen_dates сверены поSHA-256.
Инструкции запуска, повторной оценки, обучения и отката: ../docs/SPRINT.md.
Сводка всех конфигураций — sprint_selection.json;57полных dev-кешей исключены
из Git как восстанавливаемые. Выбранный dev, полный val и их метрики сохранены.
"""
    Path('reports/SPRINT_REPORT.md').write_text(text,encoding='utf8')
    readme=Path('README.md')
    legacy=readme.read_text(encoding='utf8').split('## Прежний гибрид: запуск и воспроизведение',1)[1]
    intro=f'''# ZooBears — CompactFactExtraction

Извлечение фактов из русскоязычных звонков поддержки. Основная модель —
**Qwen3-0.6B, 596 049 920 параметров**, LoRA 10 092 544 обучаемых параметра.
Требование менторов 0,5–1,5 млрд соблюдено. Релиз: `artifacts/qwen_sprint`,
checkpoint **step-100**, выбран на полном внутреннем dev из 592 звонков.

Оба минимальных порога v2.1 пройдены на **всех 383 val-звонках**:
Temporal **{val['temporal_recall']:.4f} ≥ 0.4506**, State **{val['state_accuracy']:.4f} ≥ 0.5145**.

{table}
Все {schema['facts']} фактов валидны по схеме, индексам реплик, цитатам и spans;
невалидных JSON-окон — 0; отброшено некорректных строк — {generation['invalid_rows']}. Grounding turn {val['grounding_turn']:.4f},
span {val['grounding_span']:.4f}. **92 unit-теста**, **11 проверок реального API**
пройдены. Новый сложный набор: **{suite['strict']['passed']}/43** строгих сценариев
вместо {before_suite['strict']['passed']}/43 до спринта; он не включён в обучение.

Изменения: целевое продолжение LoRA на 800 реальных и 640 новых синтетических
звонках; fit-обученный выбор границ состояний, календарные ограничения и
статистика относительных периодов. API показывает неопределённые даты как
диапазон/unknown. Статистические точки для конкурсного ISO-поля отдельно
помечаются как оценки, а не как произнесённые дни. Публичная схема сохранена.

**Запуск сохранённой модели:** после установки зависимостей и загрузки
закреплённой публичной базы согласно [docs/SPRINT.md](docs/SPRINT.md):

```powershell
$env:PYTHONPATH='src'
$env:HF_HUB_OFFLINE='1'
.venv/Scripts/python.exe -m compactfacts.api --engine qwen --model artifacts/qwen_sprint --base-model artifacts/qwen_base --device cuda --port 8001
```

Для CPU замените `cuda` на `cpu`. Демо: http://127.0.0.1:8001/.
Адаптер и выбранные CPU-модели дат включены в Git; публичная база скачивается
отдельно с проверкой revision/SHA-256. Повторное обучение для запуска не нужно.

[Итоговый отчёт, ресурсы и ограничения](reports/SPRINT_REPORT.md) ·
[Запуск, оценка, обучение и откат](docs/SPRINT.md) ·
[Сценарии защиты](docs/DEFENSE.md) ·
[Проверка порогов](reports/sprint_threshold_check.json).

fit/dev разделены по аккаунтам без пересечения. Выданный val уже использовался
для анализа ошибок в прежних итерациях: скрытый тест нужен для независимого
подтверждения. Пороговый запас небольшой; не все сложные сценарии пройдены.
Dockerfile обновлён, но его сборка на текущей машине не проверена.
Предыдущие `qwen`, `qwen_v21`, `qwen_dates` сохранены; qwen_dates сверена поSHA-256.

Сравнение размера учителя и студента: [отчёт по параметрам](reports/TEACHER_PARAMETER_COMPARISON.md).
Ускорение и денежная экономия относительно учителя не измерены.
История: [v2.1](reports/V21_ITERATION_REPORT.md),
[предыдущая привязка дат](reports/DATE_POINTER_REPORT.md),
[архивный v1](reports/QWEN_REPORT.md).
Презентация для защиты: [ZooBears_defense_2026.pptx](output/presentation/ZooBears_defense_2026.pptx).
12 основных слайдов на 6 минут 20 секунд и 6 резервных для вопросов.
[Заметки выступающего](output/presentation/ZooBears_speaker_notes.md).
Архивная презентация v1 сохранена отдельно.

## Прежний гибрид: запуск и воспроизведение'''
    readme.write_text(intro+legacy,encoding='utf8')
    deployment=read('artifacts/deployment.json')
    if deployment['qwen_final']['model']!='artifacts/qwen_sprint':
        deployment['qwen_previous']=deployment['qwen_final']
    deployment['qwen_final']={'engine':'qwen','model':'artifacts/qwen_sprint','base_model':'artifacts/qwen_base',
        'date_ranker':'artifacts/date_ranker_v21.joblib','device':'cuda','selected_on':'full account-disjoint internal dev',
        'sprint_dates':read('artifacts/qwen_sprint/inference_config.json')['sprint_dates'],
        'validation_report':'reports/SPRINT_REPORT.md'}
    write_json('artifacts/deployment.json',deployment)
    print(json.dumps({'thresholds':thresholds,'api':api['all_passed'],'new_suite':suite['strict']['passed']},ensure_ascii=False))


if __name__=='__main__':main()
