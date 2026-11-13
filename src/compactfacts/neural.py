"""A small pretrained Russian encoder with task heads and train-only normalization.

Inputs contain a centre turn and adjacent context. Gold labels are used exclusively
by dataset construction in training. Inference never reads call['facts'].
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from safetensors.torch import load_file, save_file
from sklearn.metrics import fbeta_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .model import ATTRIBUTES, Extractor, turn_text
from .grounding import align_quote,date_options
from .io import write_json

BACKBONE='cointegrated/rubert-tiny2'
KINDS=['client','service','device','ticket']


def context(call,i):
    turns=call['turns'];t=turns[i]
    centre=t['speaker']+': '+t['text']
    side=' '.join(turns[j]['speaker']+': '+turns[j]['text'] for j in [i-1,i+1] if 0<=j<len(turns))
    return centre,side,len(t['speaker'])+2


class TurnDataset(Dataset):
    def __init__(self,calls,tokenizer,vocabulary,max_length=256,labels=True):
        self.rows=[];self.calls=calls;self.vocabulary=vocabulary
        self.value_indices={a:{v:i+1 for i,v in enumerate(vocabulary[a])} for a in ATTRIBUTES}
        pairs=[]
        for ci,call in enumerate(calls):
            by_turn=defaultdict(list)
            if labels:
                for fact in call['facts']:by_turn[fact['source']['turn_idx']].append(fact)
            for i,t in enumerate(call['turns']):
                centre,side,prefix=context(call,i)
                pairs.append((centre,side))
                targets={a:[f for f in by_turn[t['idx']] if f['attribute']==a] for a in ATTRIBUTES}
                self.rows.append((ci,i,prefix,targets))
        print(f'Tokenizing {len(pairs)} turns, max_length={max_length}',flush=True)
        self.tokens=tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=False,
            truncation=True,max_length=max_length,return_offsets_mapping=True)
        self.sequence_ids=[self.tokens.sequence_ids(i) for i in range(len(pairs))]

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        ci,i,prefix,targets=self.rows[index];call=self.calls[ci]
        row={key:self.tokens[key][index] for key in ['input_ids','attention_mask','token_type_ids']}
        row['centre_mask']=[int(seq==0 and end>start) for seq,(start,end) in zip(self.sequence_ids[index],self.tokens['offset_mapping'][index])]
        gates=[];values=[];kinds=[];polarity=[];dates=[];starts=[];ends=[]
        for a in ATTRIBUTES:
            facts=targets[a];gates.append(float(bool(facts)))
            if not facts:
                for array in [values,kinds,polarity,dates,starts,ends]:array.append(-100)
                continue
            # A turn may contain several facts of the same attribute. The first
            # head target is deterministic; explicit ticket events are expanded
            # separately in postprocessing. This limitation is reported.
            fact=facts[0]
            values.append(self.value_indices[a].get(fact['value'],0))
            kinds.append(KINDS.index(fact['object']['type']) if fact['object']['type'] in KINDS else 0)
            polarity.append(int(fact.get('polarity',1)<0))
            options=date_options(call,i,a)
            matching=[j for j,value in enumerate(options) if value and value[:10]==fact['valid_from'][:10]]
            dates.append(matching[0] if matching else -100)
            match=align_quote(call['turns'][i]['text'],fact['source'].get('quote'))
            bounds=[]
            if match:
                lo,hi,_=match
                bounds=[j for j,((start,end),seq) in enumerate(zip(self.tokens['offset_mapping'][index],self.sequence_ids[index]))
                        if seq==0 and end>start and end>lo+prefix and start<hi+prefix]
            starts.append(bounds[0] if bounds else -100);ends.append(bounds[-1] if bounds else -100)
        row.update(gates=gates,values=values,kinds=kinds,polarity=polarity,dates=dates,starts=starts,ends=ends,index=index)
        return row


def collate(rows):
    length=max(len(row['input_ids']) for row in rows)
    batch={}
    for key in ['input_ids','attention_mask','token_type_ids','centre_mask']:
        batch[key]=torch.tensor([r[key]+[0]*(length-len(r[key])) for r in rows],dtype=torch.long)
    for key in ['gates','values','kinds','polarity','dates','starts','ends','index']:
        batch[key]=torch.tensor([r[key] for r in rows],dtype=torch.float32 if key=='gates' else torch.long)
    return batch


class FactEncoder(nn.Module):
    def __init__(self,config,value_counts,pretrained=None):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(pretrained,config=config,revision=getattr(config,'_commit_hash',None)) if pretrained else AutoModel.from_config(config)
        h=config.hidden_size
        self.shared=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),nn.Dropout(.1))
        self.gate=nn.Linear(h,5)
        self.values=nn.ModuleList(nn.Linear(h,n+1) for n in value_counts)
        self.kinds=nn.Linear(h,5*4);self.polarity=nn.Linear(h,5*2)
        self.dates=nn.Linear(h,5*4);self.spans=nn.Linear(h,5*2)

    def forward(self,input_ids,attention_mask,token_type_ids,centre_mask):
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).last_hidden_state
        mask=centre_mask.unsqueeze(-1)
        mean=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
        pooled=self.shared(torch.cat([hidden[:,0],mean],dim=-1))
        spans=self.spans(hidden).reshape(hidden.shape[0],hidden.shape[1],5,2)
        spans=spans.masked_fill(~centre_mask.bool()[:,:,None,None],-1e4)
        return {'gates':self.gate(pooled),'values':[head(pooled) for head in self.values],
                'kinds':self.kinds(pooled).reshape(-1,5,4),'polarity':self.polarity(pooled).reshape(-1,5,2),
                'dates':self.dates(pooled).reshape(-1,5,4),'spans':spans,'embedding':pooled}


def safe_ce(logits,targets):
    valid=targets!=-100
    if not valid.any():return logits.sum()*0
    return nn.functional.cross_entropy(logits[valid],targets[valid])


def loss_for(outputs,batch,pos_weight):
    loss=nn.functional.binary_cross_entropy_with_logits(outputs['gates'],batch['gates'],pos_weight=pos_weight)
    for j,head in enumerate(outputs['values']):loss=loss+.35*safe_ce(head,batch['values'][:,j])
    loss=loss+.15*safe_ce(outputs['kinds'].flatten(0,1),batch['kinds'].flatten())
    loss=loss+.1*safe_ce(outputs['polarity'].flatten(0,1),batch['polarity'].flatten())
    loss=loss+.25*safe_ce(outputs['dates'].flatten(0,1),batch['dates'].flatten())
    for j in range(5):
        loss=loss+.025*(safe_ce(outputs['spans'][:,:,j,0],batch['starts'][:,j])+safe_ce(outputs['spans'][:,:,j,1],batch['ends'][:,j]))
    return loss


def to_device(batch,device):return {k:v.to(device) for k,v in batch.items()}


def forward_inputs(batch):return {k:batch[k] for k in ['input_ids','attention_mask','token_type_ids','centre_mask']}


@torch.inference_mode()
def validate(model,loader,device):
    model.eval();scores=[];targets=[];value_correct=np.zeros(5);value_total=np.zeros(5)
    for raw in loader:
        batch=to_device(raw,device)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):output=model(**forward_inputs(batch))
        scores.append(output['gates'].float().cpu().numpy());targets.append(raw['gates'].numpy())
        for j,head in enumerate(output['values']):
            labels=batch['values'][:,j];mask=labels!=-100
            value_correct[j]+=(head.argmax(-1)[mask]==labels[mask]).sum().item();value_total[j]+=mask.sum().item()
    scores=np.concatenate(scores);targets=np.concatenate(targets)
    thresholds={};metrics={}
    for j,a in enumerate(ATTRIBUTES):
        options=np.arange(-4.,6.01,.2)
        threshold=max(options,key=lambda t:fbeta_score(targets[:,j],scores[:,j]>=t,beta=.8,zero_division=0))
        thresholds[a]=float(threshold)
        metrics[a]={'turn_f0.8':float(fbeta_score(targets[:,j],scores[:,j]>=threshold,beta=.8,zero_division=0)),
                    'value_accuracy_at_gold_turn':float(value_correct[j]/max(value_total[j],1))}
    return thresholds,metrics


def save_checkpoint(model,tokenizer,directory,vocabulary,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    # FP16 storage stays below GitHub's single-file limit; load into FP32 on CPU.
    state={k:v.detach().cpu().half().contiguous() if v.is_floating_point() else v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(state,str(directory/'model.safetensors'))
    model.encoder.config.save_pretrained(directory);tokenizer.save_pretrained(directory)
    write_json(directory/'experiment.json',{**metadata,'vocabulary':vocabulary})


def fit_neural(fit_calls,dev_calls,output,epochs=3,batch_size=32,max_length=256,seed=42,backbone=BACKBONE,calibration=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    revision=calibration.get('revision') if calibration else None
    tokenizer=AutoTokenizer.from_pretrained(backbone,revision=revision)
    config=AutoConfig.from_pretrained(backbone,revision=revision)
    vocabulary={}
    for a in ATTRIBUTES:
        counts=Counter(f['value'] for c in fit_calls for f in c['facts'] if f['attribute']==a)
        vocabulary[a]=sorted(v for v,n in counts.items() if n>=3)
    fit_data=TurnDataset(fit_calls,tokenizer,vocabulary,max_length)
    dev_data=TurnDataset(dev_calls,tokenizer,vocabulary,max_length) if dev_calls else None
    fit_loader=DataLoader(fit_data,batch_size=batch_size,shuffle=True,collate_fn=collate,num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
    dev_loader=DataLoader(dev_data,batch_size=batch_size*2,shuffle=False,collate_fn=collate,num_workers=0) if dev_data else None
    model=FactEncoder(config,[len(vocabulary[a]) for a in ATTRIBUTES],pretrained=backbone).to(device)
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':3e-5},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],'lr':3e-4}],weight_decay=.01)
    total_steps=epochs*len(fit_loader);warmup=max(1,total_steps//20)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warmup,max(.1,1-step/total_steps)))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    positives=np.array([sum(any(f['attribute']==a for f in row[3][a]) for row in fit_data.rows) for a in ATTRIBUTES])
    weights=torch.tensor(np.minimum((len(fit_data)-positives)/np.maximum(positives,1),10),dtype=torch.float32,device=device)
    metadata={'backbone':backbone,'revision':getattr(config,'_commit_hash',None),'license':'MIT (base model card)',
        'seed':seed,'fit_calls':len(fit_calls),'dev_calls':len(dev_calls),'fit_turns':len(fit_data),
        'batch_size':batch_size,'max_length':max_length,'epochs_requested':epochs,'device':str(device),
        'parameters':sum(p.numel() for p in model.parameters()),'value_counts':{a:len(vocabulary[a]) for a in ATTRIBUTES},
        'fit_ids_sha256':hashlib.sha256('\n'.join(sorted(c['call_id'] for c in fit_calls)).encode()).hexdigest()}
    history=[];best=-1;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();running=0;epoch_start=time.perf_counter()
        for step,raw in enumerate(fit_loader):
            batch=to_device(raw,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                out=model(**forward_inputs(batch));loss=loss_for(out,batch,weights)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();scheduler.step();running+=loss.item()
            if (step+1)%100==0:print(f'epoch={epoch} step={step+1}/{len(fit_loader)} loss={running/(step+1):.4f} elapsed={time.perf_counter()-epoch_start:.1f}s',flush=True)
        thresholds,metrics=validate(model,dev_loader,device) if dev_loader else (calibration['thresholds'] if calibration else {a:-.4 for a in ATTRIBUTES},{})
        score=float(np.mean([v['turn_f0.8'] for v in metrics.values()])) if metrics else epoch
        record={'epoch':epoch,'loss':running/len(fit_loader),'seconds':time.perf_counter()-epoch_start,'metrics':metrics,'score':score}
        history.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
        if score>best:
            best=score
            info={**metadata,'best_epoch':epoch,'thresholds':thresholds,'history':history,'train_seconds':time.perf_counter()-start,
                  'cuda_peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None}
            save_checkpoint(model,tokenizer,output,vocabulary,info)
        write_json(Path(output)/'history.json',history)
    return metadata


class NeuralExtractor:
    def __init__(self,directory,normalizer_path='artifacts/cpu_baseline.joblib',device='cpu',max_length=None,quantized=False):
        directory=Path(directory)
        self.metadata=json.loads((directory/'experiment.json').read_text(encoding='utf-8'))
        self.vocabulary=self.metadata['vocabulary'];self.max_length=max_length or self.metadata['max_length']
        self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        config=AutoConfig.from_pretrained(directory,local_files_only=True)
        self.model=FactEncoder(config,[len(self.vocabulary[a]) for a in ATTRIBUTES])
        self.model.load_state_dict(load_file(str(directory/'model.safetensors')))
        torch.set_num_threads(6)
        if quantized:
            if device!='cpu':raise ValueError('Dynamic quantization requires CPU')
            self.model=torch.ao.quantization.quantize_dynamic(self.model,{nn.Linear},dtype=torch.qint8)
        self.device=torch.device(device);self.model.to(self.device).eval()
        self.normalizer=Extractor.load(normalizer_path)

    @torch.inference_mode()
    def predict(self,call,mode='balanced'):
        from .pipeline import assemble_facts
        if not call.get('turns'):return {'call_id':call['call_id'],'facts':[]}
        inputs=[]
        for i in range(len(call['turns'])):
            centre,side,_=context(call,i);inputs.append((centre,side))
        enc=self.tokenizer([x[0] for x in inputs],[x[1] for x in inputs],padding=True,truncation=True,
            max_length=self.max_length,return_tensors='pt',return_offsets_mapping=True)
        offsets=enc.pop('offset_mapping').numpy();sequence_ids=[enc.sequence_ids(i) for i in range(len(inputs))]
        centre_mask=torch.tensor([[int(seq==0 and hi>lo) for seq,(lo,hi) in zip(sequence_ids[i],offsets[i])] for i in range(len(inputs))])
        enc['centre_mask']=centre_mask
        chunks=[]
        for start in range(0,len(inputs),64):
            with torch.autocast(device_type=self.device.type,enabled=self.device.type=='cuda',dtype=torch.float16):
                chunks.append(self.model(**{k:v[start:start+64].to(self.device) for k,v in enc.items()}))
        out={key:torch.cat([chunk[key] for chunk in chunks],dim=0) for key in chunks[0] if key!='values'}
        out['values']=[torch.cat([chunk['values'][j] for chunk in chunks],dim=0) for j in range(5)]
        scores=out['gates'].float().cpu().numpy()
        char_x=None;candidates=[]
        for j,a in enumerate(ATTRIBUTES):
            threshold=self.metadata['thresholds'][a]+{'balanced':0.,'precise':.3,'recall':-.3}[mode]
            selected=np.flatnonzero(scores[:,j]>=threshold)
            if not len(selected):continue
            probabilities=out['values'][j].float().softmax(-1).cpu().numpy()
            for i in selected:
                label=int(probabilities[i].argmax());confidence=float(probabilities[i,label])
                kind=KINDS[int(out['kinds'][i,j].argmax())]
                polarity=-1 if int(out['polarity'][i,j].argmax()) else 1
                if label and confidence>=.3:
                    value=self.vocabulary[a][label-1]
                else:
                    if char_x is None:char_x=self.normalizer.vectorizer.transform([turn_text(call,k) for k in range(len(call['turns']))])
                    memory,metadata=self.normalizer.memories[a]
                    similarity=(char_x[i]@memory.T).toarray()[0]
                    best=int(similarity.argmax());value=metadata[best]['value']
                    # Retrieval is a fallback for normalization only, never dates or IDs.
                start=int(out['spans'][i,:,j,0].argmax());end=int(out['spans'][i,:,j,1].argmax())
                prefix=inputs[i][0].find(': ')+2
                lo=int(offsets[i,start,0])-prefix;hi=int(offsets[i,end,1])-prefix
                text=call['turns'][int(i)]['text']
                span=[lo,hi] if 0<=lo<hi<=len(text) and hi-lo<350 else None
                candidates.append({'index':int(i),'attribute':a,'value':value,'polarity':polarity,
                    'kind':kind,'confidence':float(scores[i,j]),'date_policy':int(out['dates'][i,j].argmax()),'span':span})
        return assemble_facts(call,candidates,self.normalizer)
