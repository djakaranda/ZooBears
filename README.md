# ZooBears — CompactFactExtraction

Рабочая CPU-цепочка извлечения фактов из русскоязычных звонков поддержки.
Код, обученные веса, валидационные предсказания и отчёты находятся в этом проекте.
Подробный результат первого запуска — [reports/START_REPORT.md](reports/START_REPORT.md).
План дальнейшего решения и ограничения метрик — [docs/PLAN.md](docs/PLAN.md).

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

```text
docker build -t compactfacts .
docker run --rm -p 8000:8000 compactfacts
```

Образ включает только исходники, зависимости и обученные веса. Dockerfile подготовлен;
факт сборки и запуска нужно проверять отдельно на машине с Docker.

## Воспроизводимость и ограничения

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

Этот прототип выбирает значения из train и ограниченно обобщает на новые
формулировки. Словарь имён, отрицания и переходы пока требуют улучшения; старые
даты без явного текстового указания не восстанавливаются. Это стартовая точка
с измеренными результатами, а не готовое конкурсное решение.
