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


def system_msg(title: str, content: str = "") -> ChatMessage:
    return ChatMessage(
        role="assistant", content=content, metadata={"title": title}
    )


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


class QueryPlan(BaseModel):
    query: str = Field(
        default="",
        description="Transform of the user's query into a standardized, clear version maintaining original intent.",
        examples=[
            "What is the exact chemical process of ATP synthesis?",
            "How does electron transport chain contribute to energy production?",
            "What enzymes regulate cellular respiration phases?",
        ],
    )
    sub_queries: list[str] = Field(
        default=[],
        description="Break down of the original question into several simpler, more specific subquestions. "
        "Each subquestion should focus on a different aspect needed to fully answer the original query.",
        examples=[
            "¿Cuáles son los efectos del cambio climático en la agricultura?",
            "¿Cómo varían estos efectos según la región o el tipo de cultivo?",
            "¿Qué soluciones se están implementando actualmente para mitigar estos efectos?",
        ],
    )

    def __str__(self) -> str:
        q_str = f"Query: {self.query}\n"
        sb_str = "\n".join(f"[{i}] {q}" for i, q in enumerate(self.sub_queries))
        return q_str + sb_str


class GenerationState(BaseModel):
    answer: str = Field(
        default="",
        description="Answer with inline citations [n] and reference list\n"
        "Format:\nText [1]. More text [2].\n\n[1] Source text\n[2] Source text",
        examples=[
            "Paris hosts the Eiffel Tower [1]. Construction completed in 1889 [2].\n\n"
            "[1] Document 3: 'Major Paris landmarks...'\n"
            "[2] Archive 5: '1889 World Fair records...'"
        ],
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Unanswered sub-queries or missing citations (max 3)",
        examples=[
            "Sub-query 2 not addressed in documents",
            "No source for maintenance cost claims",
        ],
    )


class GenerationParams(TypedDict):
    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
