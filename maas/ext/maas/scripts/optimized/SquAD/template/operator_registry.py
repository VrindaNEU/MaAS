from maas.ext.maas.scripts.optimized.SquAD.train.template.operator_an import (
    Generate,
    ScEnsemble,
    SelfRefine,
    EarlyStop,
    Test,
)

operator_mapping = {
    "Generate": Generate,
    "ScEnsemble": ScEnsemble,
    "SelfRefine": SelfRefine,
    "EarlyStop": EarlyStop,
    "Test": Test,
}

operator_names = list(operator_mapping.keys())
