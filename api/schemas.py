from pydantic import BaseModel, Field
from typing import List

class AnalyzeRequest(BaseModel):
    text: str = Field(..., example="The pizza was delicious, but the cashier was rude.")

class AspectSentimentResult(BaseModel):
    aspect: str
    sentiment: str
    confidence: float

class AnalyzeResponse(BaseModel):
    text: str
    aspects: List[AspectSentimentResult]