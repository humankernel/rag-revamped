from dataclasses import dataclass
import time
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
    page: int
    text: str


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


class GenerationParams(TypedDict):
    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
