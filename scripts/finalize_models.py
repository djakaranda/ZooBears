"""Record provenance after fixed full-train refits; does not alter model tensors."""
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import write_json


if __name__=='__main__':
    selection=json.loads((ROOT/'reports/frozen_selection.json').read_text(encoding='utf-8'))
    code={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/compactfacts').glob('*.py')}
    selection['inference_code_sha256']=code
    write_json(ROOT/'reports/frozen_selection.json',selection)
    for name in ['neural','generator']:
        directory=ROOT/'artifacts'/name;info=json.loads((directory/'experiment.json').read_text(encoding='utf-8'))
        assert info['fit_calls']==3926 and info['dev_calls']==0 and info['best_epoch']==3
        info['stage']='fixed-epoch full-train refit after internal dev selection'
        info['selection_source']='reports/frozen_selection.json'
        if name=='generator':
            for record in info['history']:record['dev_loss']=None
            history=json.loads((directory/'history.json').read_text(encoding='utf-8'))
            for record in history:record['dev_loss']=None
            write_json(directory/'history.json',history)
        write_json(directory/'experiment.json',info)
        for p in directory.glob('*.safetensors'):
            assert p.stat().st_size<100*1024**2,(p,p.stat().st_size)
        dev=ROOT/'tmp'/('neural_dev' if name=='neural' else 'generator_dev')/'history.json'
        write_json(ROOT/f'reports/{name}_dev_training_history.json',json.loads(dev.read_text(encoding='utf-8')))
    write_json(ROOT/'artifacts/deployment.json',{'fast':{'engine':'hybrid','model':'artifacts/neural','normalizer':'artifacts/cpu_baseline.joblib','device':'cpu'},
        'quality':{'engine':'hybrid','model':'artifacts/neural','normalizer':'artifacts/cpu_baseline.joblib','generator':'artifacts/generator','device':'cuda'},
        'temporal_policy':'rules with ASR date aliases','generator_attributes':['tariff'],'selected_on':'internal account-disjoint train dev'})
    print('Final model metadata and frozen inference-code hashes recorded')
