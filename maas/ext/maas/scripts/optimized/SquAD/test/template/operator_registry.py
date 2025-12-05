from maas.ext.maas.scripts.optimized.SQuAD.train.template.operator import (
    Generate,
    ScEnsemble,
    SelfRefine,
    EarlyStop
)

operator_mapping = {
    "Generate": Generate,
    "ScEnsemble": ScEnsemble,
    "SelfRefine": SelfRefine,
    "EarlyStop": EarlyStop,
}

operator_names = list(operator_mapping.keys())
