from typing import List, Optional
from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="The customer review or feedback sentence to analyze.",
        json_schema_extra={
            "example": "The crust is thin and crunchy, but the staff is aloof."
        }
    )


class AspectSpan(BaseModel):
    term: str = Field(..., description="Extracted aspect entity.")
    sentiment: str = Field(..., description="Predicted polarity: positive, negative, or neutral.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction probability score.")
    span: List[int] = Field(..., description="Character offsets [start_char, end_char].")
    syntactic_modifiers: Optional[str] = Field(default="", description="Linguistic modifiers parsed from dependency tree.")
    negated: bool = Field(default=False, description="Flag indicating if a negation particle governs this aspect.")


class TextAnalysisResponse(BaseModel):
    text: str
    total_aspects: int
    aspects: List[AspectSpan]


class HealthCheckResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str