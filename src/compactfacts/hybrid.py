"""Frozen composition selected exclusively on the held-out training dev."""
from .neural import NeuralExtractor
from .pipeline import TemporalExtractor, assemble_facts


class HybridExtractor:
    def __init__(self,directory,normalizer='artifacts/cpu_baseline.joblib',device='cpu',generator=None,quantized=False,
                 date_ranker=None):
        self.encoder=NeuralExtractor(directory,normalizer,device,quantized=quantized)
        self.date_ranker = None
        if date_ranker:
            from joblib import load
            self.date_ranker = load(date_ranker)
        if generator:
            from .generator import ValueGenerator
            self.generator=ValueGenerator(generator,device)
        else:self.generator=None

    def predict(self,call,mode='balanced'):
        if self.date_ranker is None:
            result=self.encoder.predict(call,mode)
        else:
            result=assemble_facts(call,self.encoder.collect_candidates(call,mode),self.encoder.normalizer,
                                  agreement_evidence=False)
        if self.generator:
            selected=[f for f in result['facts'] if f['attribute']=='tariff']
            values=self.generator.generate(call,selected)
            for fact,value in zip(selected,values):
                if value:fact['value']=value
        result = TemporalExtractor.transform(call,result,agreement_evidence=self.date_ranker is None)
        if self.date_ranker is not None:
            from .date_ranker import apply_date_model, preserve_visit_window
            apply_date_model(call,result,self.date_ranker,{'action','agreement','tariff','service_state','ticket_status'})
            preserve_visit_window(call,result)
        return result
