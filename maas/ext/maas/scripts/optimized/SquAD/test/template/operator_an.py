from pydantic import BaseModel, Field

class GenerateOp(BaseModel):
    answer: str = Field(default="", description="The predicted answer to the question.")

class SelfRefineOp(BaseModel):
    answer: str = Field(default="", description="A refined and improved answer to the question.")

class ScEnsembleOp(BaseModel):
    answer: str = Field(default="", description="The final selected or aggregated answer from multiple candidates.")
