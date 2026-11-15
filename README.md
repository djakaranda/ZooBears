# ZooBears — CompactFactExtraction

Извлечение фактов из русскоязычных звонков поддержки. Основная модель —
**Qwen3-0.6B, 596 049 920 параметров**, с LoRA после двух эпох обучения.
Диапазон 0,5–1,5 млрд параметров, уточнённый менторами, соблюдён.
Обучение и выбор дат проводятся на fit/dev с разделением по аккаунтам;
внешний val используется для итоговой оценки.

Подключён новый официальный пакет и оценщик **v2.1**. Аудит существующей модели:
precision **0,6060**, recall **0,5436**, temporal **0,3740**, state **0,4928**.
Пороги v2.1 temporal ≥ **0,4506** и state ≥ **0,5145** пока не пройдены.
Аудит использует 378 сохранённых ответов на неизменившиеся входы и 5 свежих;
это не новый единый полный прогон. Подробности: [сверка пакета](reports/CASE_PACKAGE_COMPARISON_V21.md).
Сокращение затрат относительно учителя пока не подтверждено.

Продолжение LoRA на исправленных данных выполняется отдельно, без замены
`artifacts/qwen`. Команда и методика: [итерация v2.1](docs/V21_ITERATION.md).
Финальные результаты новой итерации публикуются только после её завершения.
Архивный отчёт v1 с CUDA/CPU/INT8 и сравнением с прежним гибридом:
[QWEN_REPORT.md](reports/QWEN_REPORT.md).

Установка, загрузка публичной базы и методика: [docs/QWEN.md](docs/QWEN.md).
LoRA-адаптер включён в репозиторий; базовые веса скачиваются отдельно
скриптом с проверкой закреплённой версии и SHA-256.

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m compactfacts.api --engine qwen --model artifacts/qwen --base-model artifacts/qwen_base --date-ranker artifacts/date_ranker.joblib --device cuda
```

Для CPU замените `cuda` на `cpu`. Демо: `http://127.0.0.1:8000/`.
Короткие положительные фрагменты имеют ограниченный резерв правил,
явно отмеченный в ответе API и демо. Экспериментальный INT8 пока не рекомендован.

План дальнейшего решения и ограничения метрик — [docs/PLAN.md](docs/PLAN.md).
Последняя презентация прежней модели — [ZooBears_defense.pptx](output/presentation/ZooBears_defense.pptx).

Прежний гибрид сохранён для сравнения. Его отчёты:
[проверки дат, ASR и Docker](reports/ITERATION_4_REPORT.md),
[нейросетевая версия](reports/ITERATION_2_REPORT.md),
[первый CPU-прототип](reports/START_REPORT.md).

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
