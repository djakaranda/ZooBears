"""Optional 65M seq2seq normalizer for values outside the training dictionary."""
from pathlib import Path
import json,random,time,hashlib
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq
from .io import write_json

GENERATOR='cointegrated/rut5-small'
GENERATOR_REVISION='8d74f734a90cfb55f71d9792bfd446b0e9b5267b'


def prompt(call,index,attribute):
    turns=call['turns'];t=turns[index]
    centre=f"Атрибут: {attribute}. Реплика {t['speaker']}: {t['text']}"
    context=' '.join(f"{turns[j]['speaker']}: {turns[j]['text']}" for j in [index-1,index+1] if 0<=j<len(turns))
    return centre+' Контекст: '+context


class ValueDataset(Dataset):
    def __init__(self,calls,tokenizer):
        texts=[];values=[]
        for c in calls:
            indices={t['idx']:i for i,t in enumerate(c['turns'])}
            for f in c['facts']:
                texts.append(prompt(c,indices[f['source']['turn_idx']],f['attribute']));values.append(f['value'])
        self.inputs=tokenizer(texts,truncation=True,max_length=256,padding=False)
        self.labels=tokenizer(text_target=values,truncation=True,max_length=64,padding=False)['input_ids']

    def __len__(self):return len(self.labels)
    def __getitem__(self,i):return {**{k:v[i] for k,v in self.inputs.items()},'labels':self.labels[i]}


def train_generator(fit,dev,output,epochs=3,batch_size=16,seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(6)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # SentencePiece on Windows cannot open some Unicode filenames. Convert once
    # from a public-weight cache with an ASCII path; exported fast tokenizer is
    # self-contained and works in the Cyrillic project directory.
    tokenizer=AutoTokenizer.from_pretrained(GENERATOR,revision=GENERATOR_REVISION,use_fast=True,cache_dir='C:/Users/Public/ZooBearsCache' if __import__('os').name=='nt' else None)
    model=AutoModelForSeq2SeqLM.from_pretrained(GENERATOR,revision=GENERATOR_REVISION).to(device)
    fit_data=ValueDataset(fit,tokenizer);dev_data=ValueDataset(dev,tokenizer) if dev else None
    collator=DataCollatorForSeq2Seq(tokenizer,model,padding=True)
    loader=DataLoader(fit_data,batch_size=batch_size,shuffle=True,collate_fn=collator,num_workers=0,generator=torch.Generator().manual_seed(seed))
    dev_loader=DataLoader(dev_data,batch_size=batch_size*2,collate_fn=collator,num_workers=0) if dev_data else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    total=epochs*len(loader);warmup=max(1,total//20)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warmup,max(.1,1-step/total)))
    best=float('inf');history=[];start=time.perf_counter();output=Path(output)
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0
        for step,raw in enumerate(loader):
            batch={k:v.to(device) for k,v in raw.items()};optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.bfloat16):loss=model(**batch).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite generator loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();loss_sum+=loss.item()
            if (step+1)%100==0:print(f'generator epoch={epoch} step={step+1}/{len(loader)} loss={loss_sum/(step+1):.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        model.eval();dev_losses=[]
        if dev_loader:
            with torch.inference_mode():
                for raw in dev_loader:
                    with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.bfloat16):
                        loss=model(**{k:v.to(device) for k,v in raw.items()}).loss
                    dev_losses.append(loss.item())
        score=float(np.mean(dev_losses)) if dev_losses else loss_sum/len(loader)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(loader),'dev_loss':score if dev_losses else None,'seconds':time.perf_counter()-start})
        print(history[-1],flush=True)
        if score<best:
            best=score;output.mkdir(parents=True,exist_ok=True)
            state={k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k,v in model.state_dict().items()}
            model.save_pretrained(output,state_dict=state,safe_serialization=True,max_shard_size='80MB');tokenizer.save_pretrained(output)
            write_json(output/'experiment.json',{'backbone':GENERATOR,'revision':getattr(model.config,'_commit_hash',None),'license':'MIT (upstream card)',
                'parameters':sum(p.numel() for p in model.parameters()),'fit_calls':len(fit),'dev_calls':len(dev),'fit_facts':len(fit_data),
                'seed':seed,'epochs_requested':epochs,'best_epoch':epoch,'history':history,'train_seconds':time.perf_counter()-start,
                'fit_ids_sha256':hashlib.sha256('\n'.join(sorted(c['call_id'] for c in fit)).encode()).hexdigest(),
                'cuda_peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None})
        write_json(output/'history.json',history)


class ValueGenerator:
    def __init__(self,directory,device='cuda'):
        self.device=torch.device(device)
        self.tokenizer=AutoTokenizer.from_pretrained(directory,use_fast=True,local_files_only=True)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(directory,local_files_only=True).to(self.device).eval()
        torch.set_num_threads(6)

    @torch.inference_mode()
    def generate(self,call,facts):
        if not facts:return []
        indices={t['idx']:i for i,t in enumerate(call['turns'])}
        return self.generate_prompts([prompt(call,indices[f['source']['turn_idx']],f['attribute']) for f in facts])

    @torch.inference_mode()
    def generate_prompts(self,texts):
        inputs=self.tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.autocast(device_type=self.device.type,enabled=self.device.type=='cuda',dtype=torch.bfloat16):
            output=self.model.generate(**{k:v.to(self.device) for k,v in inputs.items()},do_sample=False,num_beams=1,max_new_tokens=64)
        return [v.strip() for v in self.tokenizer.batch_decode(output,skip_special_tokens=True)]
