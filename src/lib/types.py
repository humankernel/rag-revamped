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


class QueryPlan(BaseModel):
    query: str = Field(
        default="",
        description=(
            "Versión normalizada y clara de la consulta original, manteniendo la intención y el idioma del usuario."
        ),
        examples=[
            "¿Cuál es el proceso químico exacto de la síntesis de ATP?",
            "¿Cómo contribuye la cadena de transporte de electrones a la producción de energía?",
            "¿Qué enzimas regulan las fases de la respiración celular?",
        ],
    )
    sub_queries: list[str] = Field(
        default=[],
        description=(
            "Lista de subpreguntas más pequeñas, cada una enfocada en "
            "un aspecto diferente de la consulta para cubrirla completamente."
        ),
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
        description=(
            "Respuesta completa con citaciones en línea [0], [2], … y sección de referencias al final.\n"
            "Formato:\n"
            "Texto con hechos y citas [0]. Más texto explicativo [2].\n\n"
            "[0] descripción de la referencia.\n"
            "[2] descripción de la referencia."
        ),
        examples=[
            "La Torre Eiffel se completó en 1889 [1]. Es el monumento más visitado de París [2].\n\n"
            "[1] Archivo Histórico de Francia: “Construcción de la Torre Eiffel”\n"
            "[2] Ministerio de Turismo de Francia: “Estadísticas 2023”"
        ],
    )
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Lista (máx. 3) de subpreguntas no respondidas o afirmaciones sin cita.\n"
            "Ejemplos:\n"
            "- “No se aborda la variación regional de los efectos climáticos.”\n"
            "- “Falta fuente para el dato de crecimiento poblacional.”"
        ),
        examples=[
            "No se aborda la variación regional de los efectos climáticos.",
            "Falta fuente para el dato de crecimiento poblacional.",
        ],
    )


class GenerationParams(TypedDict):
    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
