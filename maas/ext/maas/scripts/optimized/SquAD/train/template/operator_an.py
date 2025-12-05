# operator_an.py
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, model_validator
import json
import logging

logger = logging.getLogger(__name__)


class GenerateOp(BaseModel):
    """
    Robust model for Generate operator outputs.
    Accepts either:
      - a dict with keys final_answer, start_char, end_char, analysis, confidence
      - OR a dict like {'GenerateOp': '<json string>'} (ActionNode sometimes returns this),
        which will be parsed into the expected dict before validation.
    """
    final_answer: str = Field("", description="Exact substring from context that answers the question")
    start_char: int = Field(-1, description="0-based start index in context (inclusive). -1 if none.")
    end_char: int = Field(-1, description="0-based end index in context (exclusive). -1 if none.")
    analysis: Optional[str] = Field(None, description="Optional brief chain-of-thought or analysis")
    confidence: Optional[float] = Field(None, description="Optional confidence score 0.0-1.0")

    # Allow extra fields so we are tolerant to minor variations
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    def unwrap_possible_wrapped_json(cls, v: Any) -> Any:
        """
        If v looks like {'GenerateOp': '<json string>'}, parse that string to a dict.
        If v is already a dict with expected keys, return as-is.
        """
        try:
            if isinstance(v, dict):
                # Common wrapper: {'GenerateOp': '<json string>'}
                if "GenerateOp" in v and isinstance(v["GenerateOp"], str):
                    raw = v["GenerateOp"]
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            return parsed
                        # if parsed is not a dict, fall through
                    except Exception as e:
                        logger.debug("Could not json.loads GenerateOp string: %s", e)
                        # Fall back to trying to parse a JSON-looking substring
                        try:
                            # remove any leading/trailing whitespace
                            cleaned = raw.strip()
                            parsed = json.loads(cleaned)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            pass
                # Another possible wrapper: {'GenerateOp': {"final_answer": ...}} -> return inner dict
                if "GenerateOp" in v and isinstance(v["GenerateOp"], dict):
                    return v["GenerateOp"]
            # otherwise return as-is (pydantic will validate fields)
            return v
        except Exception as e:
            # Be permissive — log and return original value to let validation step raise a helpful error
            logger.debug("unwrap_possible_wrapped_json encountered exception: %s", e)
            return v


class ScEnsembleOp(BaseModel):
    solution_letter: str = Field("", description="Letter (A/B/...) of the chosen candidate")
    solution: str = Field("", description="Exact substring from context corresponding to the chosen candidate")
    start_char: int = Field(-1, description="0-based start index in context (inclusive). -1 if unknown.")
    end_char: int = Field(-1, description="0-based end index in context (exclusive). -1 if unknown.")


class SelfRefineOp(BaseModel):
    refined_answer: str = Field("", description="Exact substring from context after refinement (or empty).")
    start_char: int = Field(-1, description="0-based start index in context (inclusive). -1 if none.")
    end_char: int = Field(-1, description="0-based end index in context (exclusive). -1 if none.")
    valid: bool = Field(False, description="True if refined_answer is valid / non-empty.")
    note: Optional[str] = Field(None, description="Optional note if valid is False (e.g. 'no extractive span found')")
