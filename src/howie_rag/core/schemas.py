from typing import Optional

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: str
    confidence: float
    reasoning: Optional[str] = None


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)
