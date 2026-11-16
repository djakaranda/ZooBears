"""Locally trained Qwen3-0.6B + grounded date/timeline postprocessing."""
import json
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .pipeline import assemble_facts
from .qwen_format import parse_rows, windows


class QwenExtractor:
    def __init__(self, directory, base_model='artifacts/qwen_base', device='cuda', date_ranker=None,
                 max_new_tokens=None, max_prompt_tokens=None, merge=True, quantized=False):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
        self.format_windows=windows
        self.format_parser=parse_rows
        if self.metadata.get('transport_format_version',1)==2:
            formatter=Path(__file__).with_name('qwen_date_format.py')
            expected=self.metadata.get('format_code_sha256')
            if expected and hashlib.sha256(formatter.read_bytes()).hexdigest()!=expected:
                raise ValueError('Date-reference format differs from training; use the recorded source version')
            reference_hashes=self.metadata.get('date_reference_code_sha256',{})
            for name in ['qwen_date_format.py','temporal_v2.py','temporal.py']:
                if name in reference_hashes and hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=reference_hashes[name]:
                    raise ValueError('Date-reference parser differs from the saved model: '+name)
            from .qwen_date_format import windows as dated_windows, parse_rows as dated_parser
            self.format_windows,self.format_parser=dated_windows,dated_parser
        catalog_path = directory / 'catalog.json'
        if not catalog_path.exists():
            catalog_path = Path(__file__).resolve().parents[2] / 'data/raw/catalog.json'
        self.catalog = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.exists() else {}
        inference_path = directory / 'inference_config.json'
        inference = json.loads(inference_path.read_text(encoding='utf-8')) if inference_path.exists() else {}
        self.sprint_dates_policy=inference.get('sprint_dates') or {}
        self.sprint_dates_model=None
        self.sprint_ranges_model=None
        if self.sprint_dates_policy.get('model'):
            from joblib import load
            self.sprint_dates_model=load(self.sprint_dates_policy['model'])
        if self.sprint_dates_policy.get('ranges'):
            from joblib import load
            self.sprint_ranges_model=load(self.sprint_dates_policy['ranges'])
        self.date_pointer_mode=inference.get('date_pointer_mode','conservative') if self.metadata.get('transport_format_version',1)==2 else 'off'
        if date_ranker is None:
            date_ranker = inference.get('date_ranker')
        self.device = torch.device(device)
        self.max_prompt_tokens = max_prompt_tokens or self.metadata['max_prompt_tokens']
        self.max_new_tokens = max_new_tokens or self.metadata['max_target_tokens']
        torch.set_num_threads(6)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype,
                                                   attn_implementation='sdpa', local_files_only=True)
        self.base_parameters = sum(p.numel() for p in base.parameters())
        if not 500_000_000 <= self.base_parameters <= 1_500_000_000:
            raise ValueError(f'Model outside mentor size requirement: {self.base_parameters}')
        model = PeftModel.from_pretrained(base, directory, local_files_only=True)
        self.model = (model.merge_and_unload() if merge else model).to(self.device).eval()
        if quantized:
            if self.device.type != 'cpu':
                raise ValueError('Dynamic INT8 Qwen requires CPU')
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear: torch.ao.quantization.per_channel_dynamic_qconfig},
                dtype=torch.qint8, inplace=True)
        self.date_ranker = None
        self.canonical_objects = inference.get('canonical_objects', False)
        self.event_date_options = inference.get('event_date_options', {})
        self.event_dates = None
        self.wide_dates = None
        self.wide_date_chains = inference.get('wide_date_chains', False)
        if inference.get('wide_date_model'):
            from joblib import load
            self.wide_dates = load(inference['wide_date_model'])
        if inference.get('event_date_model'):
            from joblib import load
            self.event_dates = load(inference['event_date_model'])
        self.agreement_evidence = inference.get('agreement_evidence', date_ranker is None)
        if date_ranker:
            from joblib import load
            self.date_ranker = load(date_ranker)

    @torch.inference_mode()
    def predict(self, call, mode='balanced'):
        return self.predict_batch([call], mode, batch_size=1)[0]

    @torch.inference_mode()
    def predict_batch(self, calls, mode='balanced', batch_size=4):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        candidates = [[] for _ in calls]
        errors = [[] for _ in calls]
        diagnostics = [{'windows': 0, 'invalid_json_windows': 0, 'invalid_rows': 0,
                        'truncated_windows': 0, 'generated_tokens': 0} for _ in calls]
        entries = []
        for index, call in enumerate(calls):
            for turns, ids in self.format_windows(call, self.tokenizer, self.max_prompt_tokens):
                entries.append((index, turns, ids))
        self.tokenizer.padding_side = 'left'
        for begin in range(0, len(entries), batch_size):
            chunk = entries[begin:begin + batch_size]
            length = max(len(ids) for _, _, ids in chunk)
            inputs = torch.full((len(chunk), length), self.tokenizer.eos_token_id,
                                dtype=torch.long, device=self.device)
            mask = torch.zeros_like(inputs)
            for row, (_, _, ids) in enumerate(chunk):
                inputs[row, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=self.device)
                mask[row, -len(ids):] = 1
            output = self.model.generate(input_ids=inputs, attention_mask=mask,
                                         do_sample=False, max_new_tokens=self.max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id,
                                         use_cache=True)
            for row, (index, turns, _) in enumerate(chunk):
                generated = output[row, length:].tolist()
                if self.tokenizer.eos_token_id in generated:
                    generated = generated[:generated.index(self.tokenizer.eos_token_id) + 1]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                diagnostic = diagnostics[index]
                diagnostic['windows'] += 1
                diagnostic['generated_tokens'] += len(generated)
                diagnostic['truncated_windows'] += int(len(generated) == self.max_new_tokens
                                                        and generated[-1] != self.tokenizer.eos_token_id)
                try:
                    rows, invalid = self.format_parser(text, calls[index], turns)
                    candidates[index].extend(rows)
                    diagnostic['invalid_rows'] += invalid
                    if invalid:
                        errors[index].append({'turn_indices': [t['idx'] for t in turns],
                                              'reason': 'invalid_rows', 'text': text})
                except (ValueError, TypeError) as exc:
                    diagnostic['invalid_json_windows'] += 1
                    errors[index].append({'turn_indices': [t['idx'] for t in turns],
                                          'reason': str(exc), 'text': text})
        results = []
        self.last_candidates_by_call = {}; self.last_generation_errors_by_call = {}
        for index, call in enumerate(calls):
            if not candidates[index] and len(call['turns']) <= 5:
                from .short_rules import candidates_for_short_call
                candidates[index] = candidates_for_short_call(call, self.catalog)
                diagnostics[index]['short_rule_fallback'] = int(bool(candidates[index]))
            unique = {(c['index'], c['attribute'], c['value'], c['object']['id'], c['polarity']): c
                      for c in candidates[index]}
            selected = list(unique.values())
            self.last_candidates_by_call[call['call_id']] = selected
            self.last_generation_errors_by_call[call['call_id']] = errors[index]
            results.append(self._assemble(call, selected, diagnostics[index]))
        if len(calls) == 1:
            self.last_candidates = self.last_candidates_by_call[calls[0]['call_id']]
            self.last_generation_errors = errors[0]
        return results

    def _assemble(self, call, candidates, diagnostics):
        if self.canonical_objects:
            from .entities import canonicalize_objects
            account = (call.get('series') or {}).get('account_id') or call['call_id'].split(':')[0]
            candidates = canonicalize_objects(candidates, account)
        # Repeated context windows should not count as repeated events.
        result = assemble_facts(call, candidates, None, agreement_evidence=self.agreement_evidence)
        if self.date_ranker is not None:
            from .date_ranker import apply_date_model, preserve_visit_window
            apply_date_model(call, result, self.date_ranker,
                             {'action', 'agreement', 'tariff', 'service_state', 'ticket_status'})
            preserve_visit_window(call, result)
        if self.wide_dates is not None:
            from .wide_dates import apply_wide_dates
            apply_wide_dates(call,result,self.wide_dates,chains=self.wide_date_chains)
        if self.event_dates is not None:
            self.event_dates.apply(call, result, **self.event_date_options)
        if self.date_pointer_mode!='off':
            from .date_pointers import apply_date_pointers
            apply_date_pointers(call,result,candidates,self.date_pointer_mode)
        result['generation_diagnostics'] = diagnostics
        if self.sprint_dates_policy:
            from .scoped_dates import apply_sprint_policy
            apply_sprint_policy(call,result,self.sprint_dates_policy,self.sprint_dates_model,self.sprint_ranges_model)
        return result
