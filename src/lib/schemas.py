import time
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


class Message(TypedDict):
    text: str
    files: list[str]


class ChatMetadata(TypedDict):
    title: str
    duration: NotRequired[float]
    status: NotRequired[Literal["pending", "done"]]


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    metadata: NotRequired[ChatMetadata]


class Metadata(TypedDict):
    created_at: str


class Document(BaseModel):
    id: str
    title: str = Field(default="unknown")
    language: str = Field(default="en")
    source: str
    metadata: Metadata = Field(default={"created_at": f"{time.time()}"})


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    original_text: str


class Scores(TypedDict):
    dense_score: float
    sparse_score: float
    colbert_score: float
    hybrid_score: float
    rerank_score: float | None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    scores: Scores

    def __hash__(self):
        return hash(self.chunk.id)

    def __str__(self) -> str:
        dense = self.scores.get("dense_score", 0.0)
        sparse = self.scores.get("sparse_score", 0.0)
        colbert = self.scores.get("colbert_score", 0.0)
        hybrid = self.scores.get("hybrid_score", 0.0)
        rerank = self.scores.get("rerank_score", 0.0)
        text = self.chunk.text[:500]
        original_text = self.chunk.original_text[:500]

        return (
            f"Dense: {dense}\nSparse: {sparse}\nColbert: {colbert:.3f}\n"
            f"Hybrid: {hybrid:.3f} - Rerank:  {rerank:.3f}\n\n"
            f"{text}"
            " ======== "
            f"{original_text}"
        )


class GenerationParams(TypedDict):
    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    stop: NotRequired[list[str]]
