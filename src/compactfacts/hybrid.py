"""Frozen composition selected exclusively on the held-out training dev."""
from .neural import NeuralExtractor
from .pipeline import TemporalExtractor


class HybridExtractor:
    def __init__(self,directory,normalizer='artifacts/cpu_baseline.joblib',device='cpu',generator=None,quantized=False):
        self.encoder=NeuralExtractor(directory,normalizer,device,quantized=quantized)
        if generator:
            from .generator import ValueGenerator
            self.generator=ValueGenerator(generator,device)
        else:self.generator=None

    def predict(self,call,mode='balanced'):
        result=self.encoder.predict(call,mode)
        if self.generator:
            selected=[f for f in result['facts'] if f['attribute']=='tariff']
            values=self.generator.generate(call,selected)
            for fact,value in zip(selected,values):
                if value:fact['value']=value
        return TemporalExtractor.transform(call,result)
