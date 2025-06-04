import logging
from typing import Final, TypedDict

from lib.helpers import count_tokens
from lib.settings import settings
from lib.types import ChatMessage, RetrievedChunk


class Prompt(TypedDict):
    query_plan: str
    generate_answer: str
    validate_answer: str


log = logging.getLogger("app")


PROMPT: Final[Prompt] = {
    "query_plan": (
        "<prompt>\n"
        "  <instructions>\n"
        "    <item>Normalizar la consulta (gramática, puntuación) sin cambiar el idioma.</item>\n"
        "    <item>Descomponerla en subpreguntas específicas que cubran todos los ángulos.</item>\n"
        "    <format>\n"
        "      {{\n"
        "        'query': '<consulta normalizada>',\n"
        "        'sub_queries': [\n"
        "          '<subpregunta 1>',\n"
        "          '<subpregunta 2>',\n"
        "          '<subpregunta 3>'\n"
        "        ]\n"
        "      }}\n"
        "    </format>\n"
        "  </instructions>\n"
        "  <example>\n"
        "    <query>cuales son los efectos del cambio climatico  en la agricultura. y hay alguna solucion actualmente?</query>\n"
        "    <response>\n"
        "      {{\n"
        "        'query': '¿Cuáles son los efectos del cambio climático en la agricultura y qué soluciones existen actualmente?',\n"
        "        'sub_queries': [\n"
        "          '¿Cuáles son los efectos del cambio climático en la agricultura?',\n"
        "          '¿Cómo varían estos efectos según la región o el tipo de cultivo?',\n"
        "          '¿Qué soluciones se están implementando actualmente para mitigar estos efectos?'\n"
        "        ]\n"
        "      }}\n"
        "    </response>\n"
        "  </example>\n"
        "  <input>\n"
        "    <query>{query}</query>\n"
        "  </input>\n"
        "</prompt>\n"
        "/nothink"
    ),
    "generate_answer": (
        "<prompt>\n"
        "  <instructions>\n"
        "    <item>Genera una respuesta en el idioma de la consulta.</item>\n"
        "    <item>Cada afirmación relevante termina con [n], basado en los chunks que se hallan utilizados en la respuesta. </item>\n"
        "    <item>Si no se provee de contexto o no es suficiente, NO INVENTES LA RESPUESTA, solo di que no hay contexto suficiente.</item>\n"
        "    <format>\n"
        "      Frase relevante [1]. Otra afirmación [2].\n"
        "    </format>\n"
        "  </instructions>\n\n"
        "  <examples>\n"
        '    <example lang="es">\n'
        "      <context>[1] España, política económica reciente</context>\n"
        "      <query>¿Qué beneficios tiene X?</query>\n"
        "      <answer>\n"
        "        X mejora la eficiencia operativa [1]. Además, reduce costos según estudios recientes [2].\n"
        "      </answer>\n"
        "    </example>\n\n"
        '    <example lang="en">\n'
        "      <context>Global manufacturing standards</context>\n"
        "      <query>Technical advantages of Y</query>\n"
        "      <answer>\n"
        "        Y demonstrates superior thermal resistance [1]. Its modular design allows quick deployment [2].\n"
        "      </answer>\n"
        "    </example>\n"
        "  </examples>\n\n"
        "  <input>\n"
        "    <context>\n"
        "      {context}\n"
        "    </context>\n"
        "    <query>\n"
        "      {query}\n"
        "    </query>\n"
        "  </input>\n"
        "</prompt>\n"
        "/nothink"
    ),
    "validate_answer": (
        "<prompt>\n"
        "  <instructions>\n"
        "    <item>Comprobar que la respuesta cubre completamente la consulta del usuario.</item>\n"
        "    <item>Verificar que el número de citaciones en el texto coincide con la lista de referencias.</item>\n"
        "    <item>Asegurar que el formato bilingüe (español/inglés) esté correcto.</item>\n"
        "  </instructions>\n\n"
        "  <format>\n"
        "     {{\n"
        "       'answer': '<Texto con citaciones en línea [1], [2], … y sección de referencias al final>',\n"
        "       'gaps': [\n"
        "         '<gap 1>',\n"
        "         '<gap 2>',\n"
        "         '<gap 3>'\n"
        "       ]\n"
        "     }}\n"
        "  </format>\n\n"
        "  <example>\n"
        "    <query>¿Cuál es la población actual de Tokio y cómo ha cambiado en la última década?</query>\n"
        "    <answer>\n"
        "      La población de Tokio en 2025 es de 14 000 000 de habitantes [1]. En 2015 eran 13 500 000, "
        "lo que supone un crecimiento del 3,7 % [2].\n\n"
        "      [1] Oficina de Estadísticas de Japón: Informe 2025\n"
        "      [2] Oficina de Estadísticas de Japón: Informe 2015\n"
        "    </answer>\n"
        "    <gaps>[]</gaps>\n"
        "  </example>\n\n"
        "  <input>\n"
        "    <query>{query}</query>\n"
        "    <answer>{answer}</answer>\n"
        "  </input>\n"
        "</prompt>\n"
        "/nothink"
    ),
}


def create_prompt(
    query: str,
    history: list[ChatMessage] | None = None,
    chunks: list[RetrievedChunk] | None = None,
    max_tokens: int = settings.CTX_WINDOW,
) -> list[ChatMessage]:
    assert isinstance(query, str) and query.strip()

    history = history or []
    chunks = chunks or []
    context = (
        "\n".join(
            f"<chunk{i}>{c.chunk.text}</chunk{i}>" for i, c in enumerate(chunks)
        )
        if chunks
        else "NO CONTEXT"
    )

    # Add history + context + user query
    messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": PROMPT["generate_answer"].format(
                context=context, query=query
            ),
        }
    ]
    messages.extend(history)
    messages.reverse()
    log.debug(
        "Total Tokens: %d",
        count_tokens("\n".join(p["content"] for p in messages)),
    )

    # Trim Messages to fit the CTX_WINDOW
    total_tokens = 0
    prompt: list[ChatMessage] = []
    for msg in reversed(messages):
        tokens = count_tokens(msg["content"])
        if total_tokens + tokens > max_tokens:
            break
        prompt.append(msg)
        total_tokens += tokens

    assert total_tokens < max_tokens
    log.debug("Response without RAG prompt:\n%s", prompt)
    return prompt


def create_prompt_non_instruct(
    query: str,
    history: list[ChatMessage] | None = None,
    chunks: list[RetrievedChunk] | None = None,
    max_tokens: int = settings.CTX_WINDOW,
) -> str:
    assert isinstance(query, str) and query.strip()

    history = history or []
    chunks = chunks or []

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
