# operator_registry.py
"""
Robust operator registry that attempts to import operator.py from a few likely package paths.
Put this file in the same package folder where graph.py expects it (e.g. optimized.SquAD.train.template).
"""

import importlib
import logging

logger = logging.getLogger(__name__)

# Candidate import paths to try (order matters — put your preferred path first)
CANDIDATE_OPERATOR_MODULE_PATHS = [
    "maas.ext.maas.scripts.optimized.SquAD.train.template.operator"
]

_operator_module = None
_last_exception = None

for path in CANDIDATE_OPERATOR_MODULE_PATHS:
    try:
        _operator_module = importlib.import_module(path)
        logger.info("Imported operator module from %s", path)
        break
    except Exception as e:
        _last_exception = e
        logger.debug("Failed to import %s: %s", path, e)

if _operator_module is None:
    raise ImportError(
        "Failed to import any operator module. Tried the following paths: "
        f"{CANDIDATE_OPERATOR_MODULE_PATHS}. Last exception: {_last_exception}"
    )

# Names of operator classes we expect (some may be absent depending on dataset)
_EXPECTED_OP_CLASS_NAMES = [
    "Generate",
    "GenerateCoT",
    "MultiGenerateCoT",
    "ScEnsemble",
    "SelfRefine",
    "EarlyStop",
    "Programmer",  # optional; may be absent in QA-only setups
]

# Build mapping only from classes that actually exist in the imported module
operator_mapping = {}
for cls_name in _EXPECTED_OP_CLASS_NAMES:
    if hasattr(_operator_module, cls_name):
        operator_mapping[cls_name] = getattr(_operator_module, cls_name)

if not operator_mapping:
    raise ImportError(
        "No operator classes found in the imported operator module. "
        f"Make sure operator module {_operator_module.__name__} defines one of: {_EXPECTED_OP_CLASS_NAMES}"
    )

# Expose names in stable order
operator_names = list(operator_mapping.keys())

# Simple debug print when module is imported
logger.info("Operator mapping loaded. Available operators: %s", operator_names)
