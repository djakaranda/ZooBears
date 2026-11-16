"""Fit-only ranking with event-clause and chronological context features.

Unobservable gold endpoints are skipped, never relabelled as an open interval.
"""
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from .event_dates import EventDateModel,EventOption
from .grounding import align_quote
from .temporal_v2 import find_dates
from .qwen_date_format import references
from .date_ranker import words
from .scoped_dates import BREAK


@dataclass(frozen=True)
class BoundaryOption(EventOption):
    lo:int=0
    hi:int=0


class BoundaryRanker(EventDateModel):
    def classifier(self):
        return LogisticRegression(C=.2,max_iter=180,class_weight='balanced',solver='liblinear',random_state=73)

    def _prepare(self,call,facts):
        self._facts=facts;self._layouts={}
        indices={t['idx']:i for i,t in enumerate(call['turns'])}
        for fact in facts:
            index=indices[fact['source']['turn_idx']];text=call['turns'][index]['text']
            span=fact['source'].get('span')
            if not span or span==[0,len(text)]:
                alignment=align_quote(text,fact['value']);span=alignment[:2] if alignment else [0,len(text)]
            center=sum(span)/2;lo=0;hi=len(text)
            for cut in BREAK.finditer(text):
                if cut.end()<=center:lo=cut.end()
                elif cut.start()>center:hi=cut.start();break
            self._layouts[id(fact)]=(index,center,lo,hi,text[lo:hi])

    def bank(self,call):
        bank=[];seen=set()
        for i,t in enumerate(call['turns']):
            for future in (False,True):
                for m in find_dates(t['text'],call['started_at'],future):
                    key=(i,m.start,m.end,m.value)
                    if not m.value or key in seen:continue
                    seen.add(key);bank.append(BoundaryOption(m.value,i,t['text'][max(0,m.start-40):m.end+40],m.relation,0.,m.start,m.end))
        for ref in references(call):
            key=(ref['index'],*ref['span'],ref['value'])
            if key not in seen:
                seen.add(key);bank.append(BoundaryOption(ref['value'],ref['index'],ref['context'],ref['relation'],0.,*ref['span']))
        return bank

    def options(self,call,fact,kind,bank):
        i,center,lo,hi,_=self._layouts[id(fact)]
        result=[BoundaryOption(call['started_at'] if kind=='start' else None,i,'','default',0.)]
        if kind=='end':result.append(BoundaryOption(call['started_at'],i,'','observation',0.))
        event=words(fact['value'])|words(fact['object']['name'])
        numbers=set(re.findall(r'\d+',fact['object']['name']))
        for item in bank:
            if kind=='start' and item.relation=='until':continue
            if abs(item.index-i)>6 and not event&words(item.text):continue
            if fact['attribute']=='ticket_status' and numbers:
                other_numbers=set(re.findall(r'(?:заявк\w*|обращени\w*)\s*(?:номер\s*|[№#]\s*)?(\d{3,})',call['turns'][item.index]['text'],re.I))
                if other_numbers and not numbers&other_numbers:continue
            distance=max(item.lo-center,center-item.hi,0) if i==item.index else 0.
            result.append(BoundaryOption(item.date,item.index,item.text,item.relation,distance,item.lo,item.hi))
        return result

    def features(self,call,fact,option,kind):
        f=super().features(call,fact,option,kind)
        i,center,lo,hi,clause=self._layouts[id(fact)]
        same=option.index==i;same_clause=same and lo<=option.lo and option.hi<=hi and bool(option.text)
        attr=fact['attribute'];local=clause.lower();context=option.text.lower()
        siblings=[x for x in self._facts if x is not fact and x['attribute']==attr and x['object']['id']==fact['object']['id']]
        position=(i,center)
        before=sum((self._layouts[id(x)][0],self._layouts[id(x)][1])<position for x in siblings)
        after=len(siblings)-before
        next_events=[x for x in siblings if (self._layouts[id(x)][0],self._layouts[id(x)][1])>position]
        next_fact=min(next_events,key=lambda x:self._layouts[id(x)][:2]) if next_events else None
        near_next=False
        if next_fact:
            ni,nc,nlo,nhi,_=self._layouts[id(next_fact)]
            near_next=option.index==ni and nlo<=option.lo and option.hi<=nhi and bool(option.text)
        f.update(same_clause=int(same_clause),attribute_clause=f'{attr}/{same_clause}',
            date_before_event=int(same and option.hi<=center),signed_character_distance=max(-3,min(3,(option.lo-center)/80)) if same else 0.,
            earlier_states=min(before,3),later_states=min(after,3),last_state=int(not after),
            endpoint_last=f'{kind}/{attr}/{not after}',next_event_date=int(near_next),
            attribute_next_event=f'{attr}/{near_next}',
            local_historical=int(bool(re.search(r'раньше|прежде|был[аои]?\b|работал|пользовал',local))),
            local_current=int(bool(re.search(r'сейчас|теперь|на данный|текущ|по-прежнему',local))),
            local_negative=int(bool(re.search(r'\bне\b|\bнет\b',local))),
            context_negative=int(bool(re.search(r'\bне\b|\bнет\b',context))))
        return f

    def fit(self,calls):
        xs={'start':[],'end':[]};ys={'start':[],'end':[]};audit=Counter()
        for call in calls:
            self._prepare(call,call['facts']);bank=self.bank(call)
            for fact in call['facts']:
                if fact['attribute'] not in {'service_state','tariff','ticket_status'}:continue
                for kind,field in [('start','valid_from'),('end','valid_to')]:
                    options=self.options(call,fact,kind,bank);target=fact[field]
                    labels=[o.date[:10]==target[:10] if o.date and target else o.date is target for o in options]
                    if not any(labels):audit[kind+'_unobservable_skipped']+=1;continue
                    if len(set(labels))<2:continue
                    xs[kind].extend(self.features(call,fact,o,kind) for o in options);ys[kind].extend(labels)
                    audit[kind+'_facts']+=1
        for kind in xs:
            vec=DictVectorizer();matrix=vec.fit_transform(xs[kind])
            assert matrix.nnz<2**31
            matrix.indices=matrix.indices.astype(np.int32)
            matrix.indptr=matrix.indptr.astype(np.int32)
            model=self.classifier()
            model.fit(matrix,ys[kind]);self.vectorizers[kind]=vec;self.models[kind]=model
            audit[kind+'_rows']=len(ys[kind]);audit[kind+'_features']=len(vec.feature_names_)
        self._facts=[];self._layouts={};self.audit=dict(audit);return self

    def apply(self,call,prediction,*,mode='both',margin=1.):
        self._prepare(call,prediction['facts']);bank=self.bank(call)
        pointers={(p['object_id'],p['attribute'],p['turn_idx'],p.get('value'),p.get('polarity',1)):p
                  for p in prediction.get('date_pointer_evidence',[])}
        for kind,field in [('start','valid_from'),('end','valid_to')]:
            if mode not in {'both',kind}:continue
            entries=[];features=[]
            for fact in prediction['facts']:
                if fact['attribute'] not in {'service_state','tariff','ticket_status'}:continue
                options=self.options(call,fact,kind,bank)
                entries.append((fact,options));features.extend(self.features(call,fact,o,kind) for o in options)
            if not features:continue
            scores=iter(self.models[kind].decision_function(self.vectorizers[kind].transform(features)))
            for fact,options in entries:
                ranked=[(o,next(scores)) for o in options];chosen,score=max(ranked,key=lambda x:x[1]);current=fact[field]
                same=lambda v,w:v[:10]==w[:10] if v and w else v is w
                if same(chosen.date,current):continue
                old_scores=[s for o,s in ranked if same(o.date,current)]
                if not old_scores or score-max(old_scores)<margin:continue
                if kind=='start' and fact['valid_to'] and chosen.date>fact['valid_to']:continue
                if kind=='end' and chosen.date is not None and chosen.date<fact['valid_from']:continue
                fact[field]=chosen.date
                pointer=pointers.get((fact['object']['id'],fact['attribute'],fact['source']['turn_idx'],fact['value'],fact['polarity']))
                if pointer:
                    ref=None
                    if chosen.text:
                        text=call['turns'][chosen.index]['text']
                        ref={'turn_idx':call['turns'][chosen.index]['idx'],'span':[chosen.lo,chosen.hi],
                             'quote':text[chosen.lo:chosen.hi],'value':chosen.date,'precision':'day',
                             'relation':chosen.relation,'selection_method':'fit_boundary_ranker'}
                    pointer[kind]=ref
                    if kind=='start':pointer['start_unknown']=ref is None
                    else:pointer['closed_unknown']=chosen.date is not None and ref is None
        self._facts=[];self._layouts={}
        return prediction


from sklearn.ensemble import ExtraTreesClassifier


class ProbabilityTrees(ExtraTreesClassifier):
    def decision_function(self,matrix):
        p=np.clip(self.predict_proba(matrix)[:,1],1e-5,1-1e-5)
        return np.log(p/(1-p))


class NonlinearBoundaryRanker(BoundaryRanker):
    def classifier(self):
        return ProbabilityTrees(n_estimators=96,max_depth=14,min_samples_leaf=8,
            max_features=.7,n_jobs=2,class_weight='balanced',random_state=73)

    def features(self,call,fact,option,kind):
        features=super().features(call,fact,option,kind)
        features={k:v for k,v in features.items() if not k.startswith(('context/','value/','overlap/'))}
        delta=(datetime.fromisoformat(option.date).date()-datetime.fromisoformat(call['started_at']).date()).days if option.date else 0
        features['candidate_age_weeks']=max(-12,min(12,delta/7))
        features['value_class']=next((k for k in ['принят','работ','закры','назнач','отмен'] if k in fact['value'].lower()),'other')
        features['value_negative']=int(bool(re.search(r'\bне\b|\bнет\b|низк|слаб|пропад|обрыв|нестаб',fact['value'],re.I)))
        return features
