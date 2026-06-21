from typing import List, Optional

from pydantic import BaseModel, Field


class SourceDocumentRecord(BaseModel):
    doc_id: str
    text: str
    title: str = ""
    source_type: str
    dataset_name: str
    domain: str
    context_id: str
    metadata: dict = Field(default_factory=dict)


class BenchmarkQARecord(BaseModel):
    question_id: str
    question: str
    gold_answers: List[str]
    gold_doc_id: str
    gold_context_id: str
    domain: str
    dataset_name: str
    metadata: dict = Field(default_factory=dict)
