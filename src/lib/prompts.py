import logging
from typing import Final, TypedDict

from lib.helpers import count_tokens, normalize_text
from lib.settings import settings
from lib.schemas import ChatMessage, RetrievedChunk

log = logging.getLogger("app")


class Prompt(TypedDict):
    generation: str
    contextualize: str
    expand_query: str
    decompose_query: str


PROMPT: Final[Prompt] = {
    "generation": """
Tu tarea es responder la pregunta del usuario usando únicamente el contexto proporcionado.
<reglas>
    - Si el contexto no contiene información relevante suficiente para responder, di que no hay información relevante suficiente.
    - !IMPORTANTE: Siempre RESPONDE EN EL MISMO IDIOMA que la pregunta del usuario.
    - Referencia el chunk original usado para responder con [n], donde n es el número del chunk (ej. [1] para chunk <1>)
</reglas>
<contexto>
    {context}
</contexto>
<pregunta>
    {query}
</pregunta>
/no_think
""",
    "contextualize": """
<documento>
{context}
</documento>
Aquí hay un fragmento extraído del documento:
<chunk>
{chunk}
</chunk>
Proporciona un contexto conciso que sitúe este fragmento dentro del documento general. Enfócate en su función, relevancia y conexión con el resto del contenido. El contexto debe ser breve, preciso y diseñado para mejorar la recuperación en búsquedas. Evita redundancias y no repitas ideas. Responde solo con el contexto y nada más.
/no_think
""",
    "expand_query": """
Transforma esta pregunta del usuario en una versión más detallada para búsqueda documental.
<reglas>
1. Proporciona SOLO UNA pregunta expandida.
2. Nunca respondas la pregunta original.
3. Añade términos técnicos y contexto relevantes.
4. Mantén la intención original.
5. Longitud: 1-2 oraciones.
6. MANTENER EL IDIOMA ORIGINAL.
</reglas>
<ejemplo>
Original: "Qué causa la diabetes"
Expandida: "¿Cuáles son los factores fisiológicos, genéticos y ambientales que contribuyen al desarrollo de diabetes mellitus tipo 1 y 2?"
</ejemplo>
<ejemplo>
Original: "Cómo prevenir infartos"
Expandida: "¿Qué estrategias de prevención primaria y secundaria son efectivas para reducir el riesgo de infarto agudo de miocardio, considerando dieta, ejercicio y control de hipertensión?"
</ejemplo>
Original: "{query}"
Expandida: 
/no_think
""",
    "decompose_query": """
Descompón esta consulta en 2-3 sub-preguntas independientes que puedan responderse por separado.
<reglas>
1. Proporciona SOLO las sub-preguntas separadas por "|"
2. NO incluyas numeración como "Subpregunta 1"
3. NO añadas comentarios o explicaciones
4. Cada sub-pregunta debe ser completa y clara por sí misma
5. Mantén todos los términos técnicos del original
6. MANTENER EL IDIOMA ORIGINAL
</reglas>
<ejemplo>
Input: "Cuáles son las causas y tratamientos de la diabetes"
Output: "¿Cuáles son las principales causas de la diabetes?|¿Cuáles son los tratamientos más efectivos para la diabetes?"
</ejemplo>
<ejemplo>
Input: "Cómo funcionan las cachés CPU y por qué son importantes"
Output: "¿Cuál es el mecanismo de funcionamiento de las cachés CPU?|¿Por qué son importantes las cachés CPU para el rendimiento?"
</ejemplo>
Input: "{query}"
Output:
/no_think
""",
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
            f"<{i}>"
            f"<chunk_info> {normalize_text(c.chunk.text)} </chunk_info>"
            f"<chunk_text> {normalize_text(c.chunk.original_text)} </chunk_text>"
            f"</{i}>"
            for i, c in enumerate(chunks)
        )
        if chunks
        else "NO CONTEXT"
    )
    prompt_str = PROMPT["generation"].format(context=context, query=query)

    messages: list[ChatMessage] = [{"role": "user", "content": prompt_str}]
    messages.extend(filter(lambda m: len(m["content"]) > 0, history))
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
