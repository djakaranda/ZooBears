# ZooBears — CompactFactExtraction

Извлечение фактов из русскоязычных звонков поддержки. Основная модель —
**Qwen3-0.6B, 596 049 920 параметров**, LoRA 10 092 544 обучаемых параметра.
Требование менторов 0,5–1,5 млрд соблюдено. Релиз: `artifacts/qwen_sprint`,
checkpoint **step-100**, выбран на полном внутреннем dev из 592 звонков.

Оба минимальных порога v2.1 пройдены на **всех 383 val-звонках**:
Temporal **0.4589 ≥ 0.4506**, State **0.5297 ≥ 0.5145**.

| Версия | Precision | Recall | Temporal | State |
|---|---:|---:|---:|---:|
| До спринта: qwen_dates | 0.5980 | 0.5523 | 0.4336 | 0.5104 |
| Релиз: qwen_sprint | 0.6022 | 0.5637 | 0.4589 | 0.5297 |

Все 1727 фактов валидны по схеме, индексам реплик, цитатам и spans;
невалидных JSON-окон — 0; отброшено некорректных строк — 3. Grounding turn 0.8048,
span 0.9699. **92 unit-теста**, **11 проверок реального API**
пройдены. Новый сложный набор: **27/43** строгих сценариев
вместо 16/43 до спринта; он не включён в обучение.

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
Старая презентация v1 сохранена в output/presentation; её цифры не относятся
к текущему релизу. Для защиты используйте новый итоговый отчёт.

## Прежний гибрид: запуск и воспроизведение

Малый русский энкодер выделяет факты, затем правила восстанавливают даты и
переходы. В quality-режиме дополнительная модель нормализует тарифы, включая
формулировки вне обучающего словаря. Выбор компонентов зафиксирован на внутреннем
dev до итоговой проверки внешнего val. Подробности — [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

```powershell
# Для RTX 3060 используется официальный CUDA 12.6 wheel.
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
$env:PYTHONPATH = "$PWD/src"
.\.venv\Scripts\python.exe -m compactfacts.api --engine hybrid --model artifacts/neural
```

Демонстрация открывается по адресу `http://127.0.0.1:8000/`. JSON API:
`POST /extract`, состояние сервера: `GET /health`.

Quality-режим на GPU:

```powershell
.\.venv\Scripts\python.exe -m compactfacts.api --engine hybrid --model artifacts/neural --generator artifacts/generator --device cuda
```

Указывать оба запуска одновременно на одном порту не нужно: выберите один режим.
Для CPU без NVIDIA установите torch из официального CPU-индекса вместо CUDA.
Скачивать исходные модели при запуске API не требуется: веса и токенизаторы в репозитории.

## Запуск на Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe scripts/train.py
.\.venv\Scripts\python.exe scripts/predict.py --input data/raw/val.jsonl --output reports/val_pred.jsonl
.\.venv\Scripts\python.exe scripts/evaluate.py data/raw/val.jsonl reports/val_pred.jsonl --output reports/val_metrics.json
.\.venv\Scripts\python.exe scripts/check_predictions.py data/raw/val.jsonl reports/val_pred.jsonl
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--mode precise|balanced|recall` меняет порог выделения кандидатов. Выбор порога
обучения сделан на внутренней групповой валидации train, seed=42. Внешний val
не используется в fit, словаре или подборе порога. Все три режима заданы заранее.
Инференс читает только входные реплики и метаданные; `facts` отбрасывается.

Оригинальные файлы скопированы из Telegram Desktop в `data/raw`; PDF — в `docs`.
Исходные файлы пользователя сохранены. `artifacts/train_vocabulary.json` содержит
словарь, полученный только из train, и **не является официальным catalog.json**.
Данные, обученные веса и отчёты включены в репозиторий для воспроизводимого запуска.
Виртуальное окружение и временные файлы исключены через `.gitignore`.

## HTTP

```powershell
$env:PYTHONPATH = "$PWD/src"
.\.venv\Scripts\python.exe -m compactfacts.api
```

`GET /health`, `POST /extract`, адрес `http://127.0.0.1:8000`.
Тело POST — один звонок в формате исходного JSONL: `call_id`, `started_at`,
`turns` с `idx`, `speaker`, `text`; `series` необязателен. Ответ — `call_id`, `facts`.
Некорректный вход возвращает HTTP 400; ограничение размера тела — 2 МБ.
Сервер однопоточный, предназначен для локальной демонстрации.

```powershell
curl.exe -H "Content-Type: application/json" --data-binary "@reports/demo_input.json" http://127.0.0.1:8000/extract
```

## Docker

Команды этого раздела относятся к прежним CPU-моделям. Для Qwen подготовлен
`Dockerfile.qwen`; его сборка и запуск пока не подтверждены.

```text
docker build -t compactfacts .
docker run --rm -p 8000:8000 compactfacts
```

Для нейросетевого CPU-режима: `docker build -f Dockerfile.neural -t zoobears-neural .`
и `docker run --rm -p 8000:8000 zoobears-neural`.

Образ включает исходники, зависимости и обученные веса CPU-режима. Нейросетевой
CPU-образ собран на Docker Desktop (Apple Silicon). Ответы на 24 запроса совпали
с локальным CPU API. [Результат проверки](reports/docker_parity.json) ·
[Сценарии для защиты](docs/DEFENSE.md).

Проверенный CPU-режим с ранжированием дат:

```bash
docker run --rm -p 127.0.0.1:8000:8000 zoobears-neural python -m compactfacts.api --engine hybrid --model artifacts/neural --date-ranker artifacts/date_ranker.joblib --host 0.0.0.0
```

Для конфигурации `quality` из отчёта добавьте `--date-ranker artifacts/date_ranker.joblib`
к локальному запуску API с генератором тарифов. Внешний val для этой конфигурации
проверен на Apple GPU с `--device mps`.

## Воспроизводимость и ограничения

API и демо поддерживают [неточную дату начала](docs/UNCERTAIN_DATES.md):
«в прошлом месяце» отображается как месяц без выдуманного дня, неозвученное
начало — как неизвестное. Метаданные `temporal_evidence` отделены от конкурсных
точечных дат. В демо есть кнопка «Пример с неточной датой».
После выбора модели дополнительно проверены семь HTTP-сценариев: все 11 проверок
прошли. Исправление окна выезда и честная неопределённость неозвученных переходов
описаны в [API_DATE_FIX.md](docs/API_DATE_FIX.md). Конкурсные факты dev/val при
этой правке не изменились; первоначальный неуспешный API-отчёт сохранён.

Хэши исходников пакета — `reports/data_audit.json`, параметры и хэш train —
`artifacts/cpu_baseline.json`. Официальный скрипт метрик сохранён без изменений.
Веса `joblib` загружайте только из доверенных источников.

Дополнительные проверки и отчёт:

```powershell
.\.venv\Scripts\python.exe scripts/audit.py
.\.venv\Scripts\python.exe scripts/verify_runtime.py
.\.venv\Scripts\python.exe scripts/predict.py --input data/raw/val.jsonl --output reports/val_pred_precise.jsonl --mode precise
.\.venv\Scripts\python.exe scripts/predict.py --input data/raw/val.jsonl --output reports/val_pred_recall.jsonl --mode recall
.\.venv\Scripts\python.exe data/raw/case_metrics.py data/raw/val.jsonl data/raw/baseline_pred.jsonl --json reports/baseline_recomputed.json
.\.venv\Scripts\python.exe scripts/evaluate.py data/raw/val.jsonl reports/val_pred_precise.jsonl --output reports/val_metrics_precise.json
.\.venv\Scripts\python.exe scripts/evaluate.py data/raw/val.jsonl reports/val_pred_recall.jsonl --output reports/val_metrics_recall.json
.\.venv\Scripts\python.exe scripts/summarize.py
```

Базовый CPU-прототип выбирает значения из train и ограниченно обобщает на новые
формулировки. Словарь имён, отрицания и переходы пока требуют улучшения; старые
даты без явного текстового указания не восстанавливаются. Это стартовая точка
с измеренными результатами, а не готовое конкурсное решение.
