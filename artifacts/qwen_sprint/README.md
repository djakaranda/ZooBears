---
base_model: Qwen/Qwen3-0.6B
library_name: peft
language: ru
license: apache-2.0
---

# ZooBears: sprint adapter

Russian support-call fact extraction with date references. Base: Qwen3-0.6B,
revision c1899de289a04d12100db370d81485cdf75e47ca, 596,049,920 parameters.
LoRA: rank16, alpha32, dropout0.05, all linear modules,
10,092,544 trainable parameters. Continued from artifacts/qwen_dates;
optimizer state was not resumed. Base weights are downloaded separately.

One full epoch completed (2240.26 seconds), LR2e-5, seed73, batch1,
accumulation8, BF16 CUDA, input3072/target512. Selected step100 by
full account-disjoint internal dev temporal/state metrics, with recall/F1 guards.
800 original fit calls plus640 new synthetic calls;40 protected ASR copies.
The supplied43-case regression suite was excluded from training.

Use this directory with inference_config.json and the referenced fit-trained
CPU date models; adapter-only text generation does not reproduce final facts.
Relative periods retain range/unknown metadata in the API. Some competition
point dates are explicit statistical estimates, not verbatim spoken dates.

See reports/SPRINT_REPORT.md and docs/SPRINT.md for validation, limitations,
and reproducible commands. Public fact schema is unchanged.
