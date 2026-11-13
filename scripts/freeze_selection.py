"""Freeze the complete-pipeline choice before running the external validation."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import write_json

if __name__=='__main__':
    experiments={}
    for name in ['v1','temporal','neural','neural_calibrated','neural_rules','generated','hybrid','hybrid_rules','hybrid_rules_asr']:
        path=ROOT/f'reports/dev_{name}_metrics.json'
        if path.exists():experiments[name]=json.loads(path.read_text(encoding='utf-8'))['official']
    info={'selection_data':'592 account-disjoint calls held out from train; no val labels',
          'encoder':'cointegrated/rubert-tiny2','encoder_epochs':3,'encoder_max_length':256,
          'encoder_thresholds':json.loads((ROOT/'tmp/neural_dev/experiment.json').read_text(encoding='utf-8'))['thresholds'],
          'generator':'cointegrated/rut5-small','generator_epochs':3,'generator_attributes':['tariff'],
          'temporal_policy':'rules after value normalization, with phonetic ASR date aliases','fast_mode':'encoder + rules',
          'quality_mode':'encoder + generator for tariff + rules',
          'rationale':'Generator improves tariff precision/recall and state on dev. It harms action/service normalization and conditional dates for agreements, so those routes keep encoder/retrieval. Expanded gate calibration worsened complete-pipeline recall/state.',
          'internal_results':experiments,'external_val_used_for_selection':False}
    write_json(ROOT/'reports/frozen_selection.json',info);print(info['rationale'])
    calibration=json.loads((ROOT/'tmp/neural_dev/experiment.json').read_text(encoding='utf-8'))
    write_json(ROOT/'artifacts/encoder_calibration.json',{'revision':calibration['revision'],'thresholds':calibration['thresholds'],
        'selected_epoch':3,'source':'held-out train dev; complete-pipeline comparison in frozen_selection.json'})
