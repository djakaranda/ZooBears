# Эксперименты и воспроизведение

## Окружение

Python 3.13, Windows, Core i7-12700H, RTX 3060 Laptop 6 ГБ. Основные версии:
torch 2.9.1+cu126, transformers 4.57.1. Полное установленное окружение —
requirements-neural-lock.txt. Для его воспроизведения нужен официальный индекс
`https://download.pytorch.org/whl/cu126` для CUDA-wheel torch.

`requirements.txt` достаточно для первого CPU-варианта. `requirements-neural.txt`
добавляет нейросетевые зависимости. Dockerfile.neural использует CPU-wheel,
не требует GPU и не скачивает веса при запуске.

## Внутренний dev

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dev.py
.\.venv\Scripts\python.exe scripts/train.py --train tmp/splits/fit.jsonl --output tmp/dev_baseline.joblib
.\.venv\Scripts\python.exe scripts/train_neural.py --output tmp/neural_dev --epochs 3
.\.venv\Scripts\python.exe scripts/train_generator.py --output tmp/generator_dev --epochs 3
```

Разделение по account_id, seed=42: 3334 fit, 592 dev. Манифест —
reports/dev_split.json. train/val не пересекаются по аккаунтам.

На dev проверены исходные даты, новые правила, энкодер, расширенная калибровка,
генератор всех значений и генератор только тарифов. Изменение порогов, которое
улучшало turn F-score, ухудшило полную цепочку и отклонено. Генератор действий и
состояний также отклонён. Выбор и все сравнения сохранены в
reports/frozen_selection.json, пороги для полного обучения — artifacts/encoder_calibration.json.

`scripts/calibrate_neural.py` воспроизводит **отклонённый** эксперимент с расширенным
диапазоном порогов и изменяет experiment.json переданной dev-модели. Для финального
полного обучения используйте сохранённый encoder_calibration.json.

## Полное обучение с выбранными настройками

```powershell
.\.venv\Scripts\python.exe scripts/train_neural.py --train data/raw/train.jsonl --dev none --calibration artifacts/encoder_calibration.json --output artifacts/neural --epochs 3
.\.venv\Scripts\python.exe scripts/train_generator.py --train data/raw/train.jsonl --dev none --output artifacts/generator --epochs 3
```

Данные остаются локально. Исходные публичные веса скачиваются с Hugging Face.
Revision и карточки моделей сохранены рядом с конечными весами, без выполнения
удалённого кода. Для Windows скрипт генератора использует ASCII-путь кеша публичного
токенизатора из-за ограничения SentencePiece на некоторых Unicode-путях. Конечный
fast-токенизатор работает из обычного каталога проекта.

## Проверки и внешний val

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_neural_contract.py
.\.venv\Scripts\python.exe scripts/verify_runtime.py --engine hybrid --model artifacts/neural --output reports/neural_runtime_check.json
.\.venv\Scripts\python.exe scripts/run_benchmarks.py
.\.venv\Scripts\python.exe scripts/write_iteration_report.py
```

Бенчмарки выполняйте после завершения обучения. Четыре режима запускаются
последовательно на одних и тех же звонках. Сценарий считает официальный результат,
дополнительные строгие показатели, проверку схемы, время и память. Первая обработка
входит в замер, загрузка модели и JSON I/O исключены. На GPU есть synchronize.

Финальная нейросетевая часть обучена на полном train; пороги и композиция выбраны
на внутреннем dev. Открытый val не участвует в fit, словаре, выборе компонентов
или подборе порогов. После первой итоговой оценки параметры под val не подгоняются.

## Дополнительный ASR-шум

```powershell
.\.venv\Scripts\python.exe scripts/make_stress.py --input data/raw/val.jsonl --output tmp/val_stress.jsonl --rate 0.08 --seed 43
.\.venv\Scripts\python.exe scripts/predict_experiment.py --engine hybrid --input tmp/val_stress.jsonl --output reports/val_stress_pred.jsonl --model artifacts/neural --normalizer artifacts/cpu_baseline.joblib --generator artifacts/generator --device cuda
.\.venv\Scripts\python.exe scripts/evaluate.py tmp/val_stress.jsonl reports/val_stress_pred.jsonl --output reports/val_stress_metrics.json
```

Символьные подмены и удаления накладываются сверх исходного шума. Значения и даты
эталона сохраняются, спаны и цитаты эталона очищаются, поскольку их смещения после
изменения текста недействительны. Это локальная проверка, не скрытый тест жюри.

## Что пока не подтверждено

Docker/Podman не установлен, поэтому образы здесь не проверялись. GPU-замеры
модели-учителя отсутствуют. Из малых размеров наших моделей нельзя выводить
конкретную экономию или стоимость звонка. Неоднозначные и отсутствующие точные
даты требуют уточнения правил эталона у организаторов.

## История нескольких звонков

`scripts/merge_timeline.py --input calls.jsonl --pred predictions.jsonl --output state.json`
строит состояние аккаунтов по последовательности звонков. Ledger сохраняет call_id
и turn_idx каждого источника, изолирует аккаунты и закрывает предыдущее значение
при подтверждённом более позднем переходе. Перезвон с более старым событием не
перезаписывает новое состояние.

Ledger хранится отдельно от JSONL-предсказаний звонков: схема отдельного звонка не
умеет ссылаться на реплику другого звонка. Этот слой не использует gold facts и не
улучшает искусственно метрики отдельного звонка. Пока он опирается на стабильные
ID объектов экстрактора, разрешение неоднозначных алиасов между звонками требует
официального каталога и отдельного эксперимента.
