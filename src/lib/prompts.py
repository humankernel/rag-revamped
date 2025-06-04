import logging

from lib.helpers import count_tokens
from lib.settings import settings
from lib.types import ChatMessage, RetrievedChunk

log = logging.getLogger("app")


PROMPT = """
<prompt>
  <instructions>
    <item>Genera una respuesta en el idioma de la consulta.</item>
    <item>Cita cada afirmación con [n], n hace referencia al numero del chunk utilizado para responder. </item>
    <item>Si no se provee de contexto o no es suficiente, NO INVENTES LA RESPUESTA, solo di que no hay contexto suficiente.</item>
  </instructions>
  <input>
    <context>
      {context}
    </context>
    <query>
      {query}
    </query>
  </input>
</prompt>
/nothink
"""


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
            "content": PROMPT.format(context=context, query=query),
        }
    ]
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
