from dataclasses import dataclass
from typing import Literal, NotRequired, Optional, TypedDict

from pydantic import BaseModel


class Message(TypedDict):
    text: str
    files: list[str]


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    metadata: NotRequired[Optional[dict]]


class Metadata(TypedDict):
    created_at: str


class Document(BaseModel):
    id: str
    source: str
    metadata: Metadata


class Chunk(BaseModel):
    id: str
    doc_id: str
    page: int
    text: str


class Scores(TypedDict):
    dense_score: float
    sparse_score: float
    colbert_score: float
    hybrid_score: float
    rerank_score: float


@dataclass
class RetrievedChunk:
    chunk: Chunk
    scores: Scores

    def __repl__(self) -> str:
        return (
            f"Source   {self.chunk.doc_id}\n"
            f"Dense:   {self.scores['dense_score']:.3f}, "
            f"Sparse:  {self.scores['sparse_score']:.3f}, "
            f"Colbert: {self.scores['colbert_score']:.3f} \n"
            f"Hybrid:  {self.scores['hybrid_score']:.3f} \n"
            f"Rerank:  {self.scores['rerank_score']:.3f} \n"
            f"Text:\n  {self.chunk.text[:300]} ... "
        )
