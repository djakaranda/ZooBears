---
base_model: Qwen/Qwen3-0.6B
library_name: peft
license: apache-2.0
language:
- ru
tags:
- lora
- information-extraction
---

# ZooBears Qwen fact extractor

LoRA adapter for pinned base revision `c1899de289a04d12100db370d81485cdf75e47ca`. Base parameters: 596,049,920; adapter trainable parameters: 10,092,544.

Download the base separately using `python scripts/download_qwen.py`. Load the adapter through `compactfacts.qwen.QwenExtractor`; it expands the compact JSON transport into the competition schema and grounds sources in the input.

Training: 3334 fit calls; 592 account-disjoint dev calls. No external validation labels used for fitting or model selection.

External val official scores: `{"n_calls": 383, "n_gold_facts": 1845, "n_pred_facts": 1655, "fact_precision": 0.5233, "fact_recall": 0.4694, "temporal_recall": 0.4065, "n_temporal": 866, "state_accuracy": 0.4398, "n_state": 623, "grounding_turn": 0.8349, "grounding_span": 0.9729, "n_grounded": 866}`.

See `docs/QWEN.md`, `reports/QWEN_REPORT.md`, `experiment.json` and `data_audit.json` for reproduction, evaluation scope and limitations. Teacher speedup is not established.
