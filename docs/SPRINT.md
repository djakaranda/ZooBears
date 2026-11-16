# Финальная итерация ZooBears: запуск и воспроизведение

Основная версия: `artifacts/qwen_sprint`. Предыдущая рабочая
`artifacts/qwen_dates` сохранена без изменения файлов; backup-ветка
`backup/pre-sprint-20261116-0237` указывает на77c4a2c.
Итоговые цифры и ограничения: [SPRINT_REPORT.md](../reports/SPRINT_REPORT.md).

## Запуск готовой модели

Обучать повторно для запуска не требуется. Из корня репозитория:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-qwen.txt
# Для NVIDIA установите PyTorch CUDA согласно docs/EXPERIMENTS.md.
.venv/Scripts/python.exe scripts/download_qwen.py
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
$env:HF_HUB_OFFLINE='1'
.venv/Scripts/python.exe -m compactfacts.api --engine qwen --model artifacts/qwen_sprint --base-model artifacts/qwen_base --device cuda --port 8001
```

Демо: http://127.0.0.1:8001/; API: `POST /extract`, `GET /health`.
На компьютере без NVIDIA замените `cuda` на `cpu`.
База Qwen закреплена на revision `c1899de289a04d12100db370d81485cdf75e47ca`;
загрузчик проверяет SHA-256. Публичные базовые веса исключены из Git.
LoRA, токенизатор, каталоги, все выбранные CPU-модели дат включены в Git.

## Независимая проверка сохранённого решения

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/check_qwen_api.py --url http://127.0.0.1:8001 --expected-adapter artifacts/qwen_sprint --extended --output tmp/api_recheck.json
.venv/Scripts/python.exe scripts/predict_experiment.py --engine qwen --model artifacts/qwen_sprint --base-model artifacts/qwen_base --device cuda --batch-size 4 --input data/raw/val.jsonl --output tmp/val_recheck.jsonl
.venv/Scripts/python.exe scripts/evaluate.py data/raw/val.jsonl tmp/val_recheck.jsonl --output tmp/val_recheck_metrics.json
.venv/Scripts/python.exe scripts/check_predictions.py data/raw/val.jsonl tmp/val_recheck.jsonl --output tmp/val_recheck_schema.json
```

Оценщик v2.1 и catalog неизменны. Для val локальный ремонт оценщика не включён.
Greedy-генерация повторяется с сохранёнными весами; CPU FP32 и CUDA BF16,
а также разные размеры пакетов могут дать разные ответы. Поэтому отчёт фиксирует
устройство и batch. Побитовая идентичность переобучения между GPU не обещается.

## Повторение эксперимента обучения

Делайте это в отдельной копии репозитория: команды создают заново
`artifacts/qwen_sprint` и отчёты. Сохранённая релизная версия нужна для сравнения.

```powershell
.venv/Scripts/python.exe scripts/prepare_dev.py
.venv/Scripts/python.exe scripts/prepare_sprint.py
.venv/Scripts/python.exe scripts/fit_sprint_dates.py --kind all
.venv/Scripts/python.exe scripts/run_sprint.py
```

Разбиение: GroupShuffleSplit seed42 по аккаунтам,3334fit/592dev. Затем seed73
выбирает800 реальных fit-звонков и создаёт640 новых синтетических примеров.
Готовый корпус сохранён в `data/derived/sprint_fit.jsonl`; его SHA-256 и отсутствие
пересечений — в `reports/sprint_data_audit.json` и `sprint_training_data_audit.json`.
43 разговора нового пользовательского набора не добавлены в обучение.
Шаблоны дополняют категории ошибок, поэтому этот набор — регрессионный,
а не независимое свидетельство обобщения.

Продолжение LoRA: одна полная эпоха, rank16/alpha32/dropout0.05/all-linear;
LR2e-5, AdamW weight_decay0.01, seed73, batch1/accumulation8, clip1,
BF16 CUDA, gradient checkpointing. Вход3072, ответ512; потери только по ответу.
Вес положительных ссылок дат3, закрытого состояния с неизвестной датой2.
ASR-копии3%:40 разговоров,1545 окон,1 826 213 токенов,165 730 целевых.
Обучаются10 092 544 параметра. База596 049 920 заморожена;
состояние оптимизатора предыдущего обучения не переносится.

В сохранённом запуске эпоха заняла2240,26секунд; checkpoint step100 выбран
по полному dev, затем проверен весь val. До val сравнивались3 набора весов ×19
политик дат. Критерий: минимум отношений temporal/state к порогам, с ограничениями
на recall/F1; предпочтение календарным ограничениям при равенстве.
После API-проверки исправлено направление будущего дня недели у договорённости:
его влияние сначала проверено на dev, затем выполнен новый полный val.
Эта поправка отражена отдельно в `reports/sprint_weekday_fix.json`.
Новый повтор эксперимента на актуальном коде может выбрать иной checkpoint:
для точного воспроизведения опубликованных метрик используйте сохранённые веса.

CPU-ранжировщик границ обучен на исходном fit и новой синтетике; наблюдаемые
даты-кандидаты берутся только из текущего звонка. Неозвученные эталонные границы
пропускаются при обучении этого ранжировщика. Статистика периодов обучена только
на исходном fit и применяется при достаточной поддержке/согласованности.
Календарные правила связывают период с названным днём и временем. Формат
обучающих ссылок и их нумерация не менялись при продолжении LoRA.

`--resume` продолжает прерванные стадии при неизменном коде; после правок
исходников нельзя считать старые стадии актуальными без нового расчёта.

## Неопределённые даты

«В прошлом месяце» в API показывает календарный диапазон, а неназванное
начало — unknown. В конкурсном обязательном ISO-поле для некоторых хорошо
поддержанных категорий используется статистическая точечная оценка внутри
диапазона. Она помечена в `date_pointer_evidence.range_estimate` и не выдаётся
за явно произнесённый день. `temporal_evidence` сохраняет диапазон и цитаты.

## Возврат к предыдущей версии

Остановите сервер, затем замените в команде запуска только модель:
`--model artifacts/qwen_dates`. Ранжировщик выбирается из её старого config.
Для прежней полностью зафиксированной сборки доступен commit77c4a2c.
Git reset и удаление новых файлов для отката демо не требуются.

## Контейнер

`docker build -f Dockerfile.qwen -t zoobears-qwen .`
и `docker run --rm -p 8000:8000 zoobears-qwen`.
Перед сборкой скачайте базу. В образ включены выбранные CPU-модели дат,
checkpoints исключены. Docker на данной Windows-машине отсутствует:
сборка этого Dockerfile не проверена; CPU и CUDA проверяются локально.
