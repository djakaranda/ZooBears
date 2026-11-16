"""Reproduce the parameter-count branch using official public teacher metadata."""
import argparse
import json
import math
from pathlib import Path
import sys
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compactfacts.io import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='artifacts/qwen_v21')
    args=parser.parse_args()
    teacher=json.loads(Path('reports/glm53_public_metadata.json').read_text(encoding='utf8'))
    student=json.loads((Path(args.model)/'experiment.json').read_text(encoding='utf8'))
    assert 'glm-5.3' in Path('data/raw/ANSWERS_v2.md').read_text(encoding='utf8').lower()
    ranker=joblib.load('artifacts/date_ranker_v21.joblib')
    classic=int(ranker.model.coef_.size+ranker.model.intercept_.size)
    config=json.loads((Path(args.model)/'inference_config.json').read_text(encoding='utf8'))
    auxiliary={}
    policy=config.get('sprint_dates',{})
    if policy.get('model'):
        boundary=joblib.load(policy['model'])
        # Include tree topology, sample counts and impurity as well as values:
        # this deliberately overcounts learned scalar parameters.
        for kind,model in boundary.models.items():
            if hasattr(model,'estimators_'):
                total=0
                for estimator in model.estimators_:
                    state=estimator.tree_.__getstate__()
                    total+=state['nodes'].size*len(state['nodes'].dtype.names)+state['values'].size
                auxiliary[kind+'_tree_scalars_and_topology']=int(total)
            else:auxiliary[kind+'_linear_parameters']=int(model.coef_.size+model.intercept_.size)
            auxiliary[kind+'_vocabulary_entries']=len(boundary.vectorizers[kind].vocabulary_)
    if policy.get('ranges'):
        ranges=joblib.load(policy['ranges'])
        auxiliary['range_statistics']=sum(len(v) for v in ranges.stats.values())
    actual=student['total_parameters_with_adapter']+classic+sum(auxiliary.values())
    bound=math.ceil(actual/10_000_000)*10_000_000
    minimum=sum(v for k,v in teacher['safetensors']['parameters'].items() if k.startswith('F8'))
    proof={'teacher_name_in_case':'glm-5.3','teacher_source':teacher['source'],'teacher_revision':teacher['sha'],
        'teacher_total_parameters':teacher['safetensors']['total'],'teacher_weight_lower_bound':minimum,
        'student_model':Path(args.model).as_posix(),
        'student_base_parameters':student['base_parameters'],'student_adapter_parameters':student['trainable_adapter_parameters'],
        'student_date_ranker_parameters':classic,'student_auxiliary_parameters_upper_estimate':auxiliary,
        'student_unmerged_total':actual,'student_conservative_upper_bound':bound,
        'reduction_lower_bound':minimum/bound,'parameter_branch_passed':minimum/bound>=10,
        'teacher_runtime_revision_verified':False,'gpu_seconds_reduction_measured':False,'monetary_cost_reduction_measured':False,
        'comparison':'Total stored model parameters, not active per-token MoE compute. Identity follows the teacher name in the case; runtime revision is unavailable.'}
    active_path=Path('reports/glm53_active_parameter_bound.json')
    active=json.loads(active_path.read_text(encoding='utf8')) if active_path.exists() else None
    if active:
        proof.update(active_one_projection_lower_bound=active['active_down_projection_weights_lower_bound'],
            active_projection_reduction_lower_bound=active['active_down_projection_weights_lower_bound']/bound)
        # Exclude auxiliary/MTP weights: count only main-network expert down projections.
        minimum=min(minimum,active['main_routed_down_projection_weights_lower_bound'])
        proof.update(teacher_weight_lower_bound=minimum,reduction_lower_bound=minimum/bound,
                     parameter_branch_passed=minimum/bound>=10)
    write_json('reports/teacher_parameter_comparison.json',proof)
    text=f'''# Сравнение числа параметров

Пакет организаторов (`data/raw/ANSWERS_v2.md`) называет учителем `glm-5.3`.
На [официальной странице Z.ai](https://huggingface.co/zai-org/GLM-5.3) опубликован
размер около 753 млрд параметров. [Метаданные Hugging Face](https://huggingface.co/api/models/zai-org/GLM-5.3)
сохранены в `reports/glm53_public_metadata.json` с revision `{teacher['sha']}`.

- Всего скаляров параметров учителя: **{teacher['safetensors']['total']:,}**.
- Консервативная нижняя оценка весов основной сети: **{minimum:,}**.
- Студент до слияния LoRA вместе со всеми выбранными CPU-моделями дат: **{actual:,}**.
- Округляем размер студента вверх до **{bound:,}**.
- Отношение: **{minimum/bound:.1f}×**, требуемая ветка по параметрам — ≥10×.

Это обоснование параметрической ветки критерия для опубликованной GLM-5.3,
совпадающей по имени с учителем из пакета. Серверная ревизия учителя не раскрыта.
Нижняя оценка учитывает только down-проекции экспертов основной сети:
веса вспомогательного MTP и прочие компоненты для неё не требуются.
У GLM-5.3 архитектура MoE: общий размер не равен активным вычислениям на токен.
Поэтому отношение нельзя выдавать за ускорение, сокращение GPU-секунд или денег.
Таких сопоставимых замеров мы не проводили. Основная Qwen имеет 596 049 920
параметров; LoRA при inference сливается с базой, поэтому выбранная выше оценка
размера студента завышает реальную параметрическую нагрузку.

Повторение: `python scripts/check_teacher_parameters.py --model {args.model}`.
'''
    if active:
        text+=f'''
## Дополнительная проверка с учётом MoE

По [закреплённому конфигу модели]({active['config_source']}) имеется
{active['sparse_layers']} MoE-слоёв, {active['active_experts_per_token']} активных экспертов на токен,
размеры {active['hidden_size']}×{active['expert_intermediate_size']}.
[Официальная реализация Transformers]({active['implementation_source']})
содержит отдельную down-проекцию такой формы у каждого эксперта каждого слоя.
Если учитывать только эту одну проекцию, получается нижняя оценка:

`{active['sparse_layers']} × {active['active_experts_per_token']} × {active['hidden_size']} × {active['expert_intermediate_size']} = {active['active_down_projection_weights_lower_bound']:,}` весов.

Даже эта часть активных весов больше консервативного размера студента в
**{active['active_down_projection_weights_lower_bound']/bound:.2f}×**. Две другие проекции,
attention, общие эксперты, dense-слои и embedding не учитываются.
Это вывод из опубликованной архитектуры, а не трассировка сервера организаторов
и не измерение времени или потребления GPU. Исходные числа и хеш реализации:
`reports/glm53_active_parameter_bound.json`.
'''
    Path('reports/TEACHER_PARAMETER_COMPARISON.md').write_text(text,encoding='utf8')
    print({'parameter_branch_passed':proof['parameter_branch_passed'],'ratio_lower_bound':proof['reduction_lower_bound']})


if __name__=='__main__':main()
