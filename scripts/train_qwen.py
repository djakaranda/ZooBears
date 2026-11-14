"""Response-only LoRA SFT, account-disjoint dev, bounded-memory causal loss."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from compactfacts.io import read_jsonl, write_json
from compactfacts.qwen_format import BACKBONE, REVISION, target_rows, windows, training_object


def accounts(calls):
    return {(c.get('series') or {}).get('account_id') or c['call_id'].split(':')[0] for c in calls}


def dataset(calls, tokenizer, max_prompt, max_target):
    samples = []
    for call in calls:
        for turns, prompt in windows(call, tokenizer, max_prompt):
            target = json.dumps(target_rows(call, turns), ensure_ascii=False, separators=(',', ':'))
            target = tokenizer.encode(target + tokenizer.eos_token, add_special_tokens=False)
            if len(target) > max_target:
                raise ValueError(f"Target for {call['call_id']} has {len(target)} tokens, limit={max_target}; no truncation allowed")
            samples.append({'call_id': call['call_id'], 'input': prompt + target, 'prefix': len(prompt)})
    return samples


def loss_for(model, sample, device):
    ids = torch.tensor([sample['input']], dtype=torch.long, device=device)
    prefix = sample['prefix']
    # Compute the large vocabulary projection only for response positions.
    # Input tokens remain visible to attention but are never loss targets.
    positions = torch.arange(prefix - 1, ids.shape[1] - 1, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   logits_to_keep=positions, use_cache=False)
    return F.cross_entropy(output.logits[0].float(), ids[0, prefix:]), ids.shape[1], ids.shape[1] - prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='tmp/splits/fit.jsonl')
    parser.add_argument('--dev', default='tmp/splits/dev.jsonl')
    parser.add_argument('--base-model', default='artifacts/qwen_base')
    parser.add_argument('--output', default='artifacts/qwen')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--max-prompt-tokens', type=int, default=2560)
    parser.add_argument('--max-target-tokens', type=int, default=512)
    parser.add_argument('--limit', type=int, help='Explicit smoke run only; recorded in experiment')
    parser.add_argument('--dev-loss-limit', type=int, default=96, help='Fixed train-dev prefix used for checkpoint loss; full dev generation is evaluated separately')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    device = torch.device(args.device)
    fit = read_jsonl(args.train)
    dev = read_jsonl(args.dev) if args.dev != 'none' else []
    assert not accounts(fit) & accounts(dev), 'Train/dev account overlap'
    if args.limit:
        fit = fit[:args.limit]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    started = time.perf_counter()
    print('Preparing response-only training windows', flush=True)
    fit_data = dataset(fit, tokenizer, args.max_prompt_tokens, args.max_target_tokens)
    dev_data = dataset(dev[:args.dev_loss_limit], tokenizer, args.max_prompt_tokens, args.max_target_tokens)
    write_json(output / 'data_audit.json', {
        'fit_calls': len(fit), 'fit_windows': len(fit_data), 'dev_calls': len(dev), 'dev_loss_calls': min(len(dev), args.dev_loss_limit),
        'dev_loss_windows': len(dev_data), 'account_overlap': 0,
        'train_sha256': hashlib.file_digest(Path(args.train).open('rb'), 'sha256').hexdigest(),
        'dev_sha256': hashlib.file_digest(Path(args.dev).open('rb'), 'sha256').hexdigest() if dev else None,
        'fit_tokens': sum(len(s['input']) for s in fit_data),
        'fit_target_tokens': sum(len(s['input']) - s['prefix'] for s in fit_data),
        'max_sequence_tokens': max(len(s['input']) for s in fit_data),
        'fit_ids_sha256': hashlib.sha256('\n'.join(sorted(c['call_id'] for c in fit)).encode()).hexdigest(),
        'dev_ids_sha256': hashlib.sha256('\n'.join(sorted(c['call_id'] for c in dev)).encode()).hexdigest(),
        'truncated_training_examples': 0, 'smoke_limit': args.limit,
        'canonicalized_training_objects': sum(training_object(c, f) != (f['object']['type'], f['object']['name'])
                                             for c in fit for f in c['facts']),
    })
    print(json.dumps(json.loads((output / 'data_audit.json').read_text())), flush=True)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype,
                                               attn_implementation='sdpa', local_files_only=True)
    base_parameters = sum(p.numel() for p in base.parameters())
    assert 500_000_000 <= base_parameters <= 1_500_000_000, base_parameters
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.rank,
                                          lora_alpha=args.rank * 2, lora_dropout=.05,
                                          target_modules='all-linear', bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'base_parameters={base_parameters} trainable_adapter={trainable}', flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    steps_per_epoch = math.ceil(len(fit_data) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(.1, 1 - step / total_steps)))
    history = []; best = float('inf'); step = 0; tokens_seen = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        order = list(range(len(fit_data))); random.shuffle(order)
        epoch_loss = 0.; recent = []
        for i, index in enumerate(order):
            loss, tokens, _ = loss_for(model, fit_data[index], device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite Qwen training loss')
            group_size = min(args.accumulation, len(order) - (i // args.accumulation) * args.accumulation)
            (loss / group_size).backward()
            value = loss.detach().item(); epoch_loss += value; recent.append(value); tokens_seen += tokens
            if (i + 1) % args.accumulation == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                elapsed = time.perf_counter() - started
                progress = {'epoch': epoch, 'step': step, 'total_steps': total_steps,
                            'samples': i + 1, 'epoch_samples': len(order), 'loss': float(np.mean(recent)),
                            'elapsed_seconds': elapsed, 'eta_seconds': elapsed / step * (total_steps - step),
                            'tokens_per_second': tokens_seen / elapsed,
                            'gpu_peak_mb': torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else None}
                write_json(output / 'progress.json', progress)
                if step % 10 == 0 or i + 1 == len(order):
                    print(json.dumps(progress), flush=True); recent = []
                if device.type == 'cuda':
                    # Variable-length Windows/WDDM workloads otherwise retain
                    # large unused blocks and can exhaust the 6 GB device.
                    torch.cuda.empty_cache()
        model.eval(); numerator = denominator = 0
        with torch.inference_mode():
            for sample in dev_data:
                loss, _, targets = loss_for(model, sample, device)
                numerator += loss.item() * targets; denominator += targets
        score = numerator / denominator if denominator else epoch_loss / len(order)
        record = {'epoch': epoch, 'train_loss_per_window': epoch_loss / len(order),
                  'dev_response_token_loss': score if dev_data else None, 'seconds': time.perf_counter() - started}
        history.append(record); print(json.dumps(record), flush=True)
        if score < best:
            best = score
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
            write_json(output / 'experiment.json', {
                'backbone': BACKBONE, 'revision': REVISION, 'license': 'Apache-2.0',
                'transport_format_version': 1,
                'format_code_sha256': hashlib.sha256((ROOT / 'src/compactfacts/qwen_format.py').read_bytes()).hexdigest(),
                'training_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'base_parameters': base_parameters, 'trainable_adapter_parameters': trainable,
                'total_parameters_with_adapter': sum(p.numel() for p in model.parameters()),
                'size_requirement_satisfied': True, 'fit_calls': len(fit), 'dev_calls': len(dev),
                'fit_windows': len(fit_data), 'epochs_requested': args.epochs, 'best_epoch': epoch,
                'max_prompt_tokens': args.max_prompt_tokens, 'max_target_tokens': args.max_target_tokens,
                'lora_rank': args.rank, 'learning_rate': args.lr, 'accumulation': args.accumulation,
                'seed': args.seed, 'training_seconds': time.perf_counter() - started,
                'selection': f'response token loss on first {min(len(dev), args.dev_loss_limit)} internal dev calls; no external val selection',
                'history': history, 'smoke_limit': args.limit,
                'cuda_peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else None,
            })
        write_json(output / 'history.json', history)
    print(f'Training complete. Best adapter: {output}; dev_response_loss={best:.4f}', flush=True)


if __name__ == '__main__':
    main()
