from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    original_query: str
    query_for_retrieval: str
    detected_intent: str
    retrieval_mode: str
    preferred_document_types: List[str] = Field(default_factory=list)
    preferred_chunk_types: List[str] = Field(default_factory=list)
    metadata_preferences: Dict[str, object] = Field(default_factory=dict)
    top_k: int = 5
    candidate_pool_size: int = 30
    explanation: str = ""
