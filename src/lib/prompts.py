import logging
from typing import Final, TypedDict

from lib.helpers import count_tokens
from lib.types import ChatMessage, RetrievedChunk
from settings import settings


class Prompt(TypedDict):
    query_plan: str
    generate_answer: str
    validate_answer: str


log = logging.getLogger("rag")

# TODO: add deepseek recomendation if math
# Please reason step by step, and put your final answer within \boxed{}.
# we recommend enforcing the model to initiate its response with "<think>\n" at the beginning of every output
PROMPT: Final[Prompt] = {
    "query_plan": (
        "Normaliza la consulta y descompónla en sub-preguntas:\n"
        "Consulta:\n{query}\n\n"
        "Considera:\n"
        "1. Diferentes aspectos individuales entre sí de la pregunta.\n"
        "2. Conceptos relacionados.\n"
        "3. Mantén el idioma original.\n"
        "Ejemplos:\n"
        "Consulta:\n"
        "cuales son los efectos del cambio climatico  en la agricultura. y hay alguna solucion actualmente?\n"
        "{{ 'query': '¿Cuáles son los efectos del cambio climático en la agricultura y qué soluciones se están implementando?', "
        "'sub_queries': [ '¿Cuáles son los efectos del cambio climático en la agricultura?', "
        "'¿Cómo varían estos efectos según la región o el tipo de cultivo?', "
        "'¿Qué soluciones se están implementando actualmente para mitigar estos efectos?' ] }}"
    ),
    "generate_answer": (
        "Genera una respuesta con citaciones numéricas EN EL IDIOMA DEL USUARIO usando este formato:\n"
        "1. Cada afirmación relevante lleva [n] al final\n"
        "2. Lista de referencias al final con texto completo\n\n"
        "**Instrucciones**:\n"
        "1. Usar formato: Frase [1]. Otra frase [2].\n"
        "2. Numeración consecutiva en toda la respuesta\n"
        "3. Al final:\n"
        "   [1] Texto completo del fragmento citado\n"
        "   [2] Siguiente fragmento citado\n"
        "4. Conservar el idioma original de la consulta\n"
        "5. Máximo 5 citaciones por respuesta\n\n"
        "**Ejemplo Español**:\n"
        "Consulta: '¿Qué beneficios tiene X?'\n"
        "Respuesta:\n"
        "'X mejora la eficiencia operativa [1]. Además, reduce costos según estudios recientes [2].\n\n"
        "[1] 'La tecnología X aumenta un 40% la productividad...'\n"
        "[2] 'Estudio de 2023 muestra ahorros promedio de $2M...'\n\n"
        "**English Example**:\n"
        "Query: 'Technical advantages of Y'\n"
        "Answer:\n"
        "'Y demonstrates superior thermal resistance [1]. Its modular design allows quick deployment [2].\n\n"
        "[1] 'Testing results: Y withstands 500°C for...'\n"
        "[2] 'Assembly manual section 3.2: modular components...'\n\n"
        "**Contexto**:\n {context}\n\n"
        "**Consulta**:\n {query}\n\n"
    ),
    "validate_answer": (
        "Verificar:\n"
        "1. Si la respuesta responde completamente la consulta.\n"
        "2. Coincidencia numérica entre citaciones y referencias.\n"
        "3. Formato bilingüe correcto.\n\n"
        "Consulta: {query}\n"
        "Respuesta: {answer}\n\n"
        "Identifica la informacion faltante o inseguridad de la respuesta. Lista hasta 3 fallas si las hay."
    ),
}


def create_prompt(
    query: str,
    history: list[ChatMessage] | None = None,
    chunks: list[RetrievedChunk] | None = None,
    max_tokens: int = settings.CTX_WINDOW,
) -> str:
    """Prompt = Chat_History + Chunks + Query"""
    history = history or []
    chunks = chunks or []

    assert isinstance(query, str) and query.strip()

    h_str = "\n".join(
        f"<|{h['role']}|> {h['content']}"
        for h in history
        if isinstance(h["content"], str)
    )
    c_str = "\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(chunks))

    h_tokens = count_tokens(h_str)
    c_tokens = count_tokens(c_str)
    q_tokens = count_tokens(query)
    t_tokens = h_tokens + c_tokens + q_tokens

    if t_tokens > max_tokens:
        if h_tokens > t_tokens * 0.3:
            h_str[: t_tokens * 0.3]
        if c_tokens > t_tokens * 0.7:
            c_str[: t_tokens * 0.7]

    prompt = (
        f"{h_str}"
        f"<|User|>\n{query}"
        f"Context To Answer:\n{c_str}"
        f"<|Assistant|>\n<think>"
    )

    final_tokens = count_tokens(prompt)
    log.debug(
        "Tokens - History: %d, Chunks: %d, Query: %d, Total: %d",
        h_tokens,
        c_tokens,
        q_tokens,
        final_tokens,
    )

    assert count_tokens(prompt) < max_tokens
    return prompt
