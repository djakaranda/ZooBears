# Источники нейросетевого эксперимента

- Базовый энкодер: https://huggingface.co/cointegrated/rubert-tiny2
  Модельная карточка указывает MIT, 29.4M параметров и русский язык.
  Точный revision фиксируется в метаданных каждого эксперимента.
- Конфигурация: https://huggingface.co/cointegrated/rubert-tiny2/blob/main/config.json
- Установка PyTorch CUDA: https://pytorch.org/get-started/previous-versions/
  Для этого эксперимента выбран официальный wheel torch 2.9.1, CUDA 12.6.
- Загрузка AutoModel / AutoTokenizer:
  https://huggingface.co/docs/transformers/model_doc/auto
- Дополнительный генератор открытых значений:
  https://huggingface.co/cointegrated/rut5-small
  Карточка указывает MIT и около 65M параметров. Исходная модель имеет ограничения
  качества перефразирования, поэтому выбор зависит от нашего dev-эксперимента.
- Обучение seq2seq:
  https://huggingface.co/docs/transformers/tasks/summarization

Сетевые вызовы используются для скачивания публичных весов и документации.
Транскрипты не передаются внешней модели или API. `trust_remote_code` не используется.
