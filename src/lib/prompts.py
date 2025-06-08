import logging
from typing import Final, TypedDict

from lib.helpers import count_tokens, normalize_text
from lib.settings import settings
from lib.schemas import ChatMessage, RetrievedChunk

log = logging.getLogger("app")


class Prompt(TypedDict):
    generation: str
    contextualize: str


PROMPT: Final[Prompt] = {
    "generation": """
Your task is to answer the user's question using only the provided context. If the answer can be found within the context, respond accordingly. If the context does not contain enough relevant information to answer, reply with there is not enough relevant information. Always respond in the same language as the use's question.
<context>
    {context}
</context>
<question>
    {query}
</question>
/no_think
""",
    "contextualize": """
<document>
{context}
</document>
Here is a chunk extracted from the document:
<chunk>
{chunk}
</chunk>
Provide a concise context that places this chunk within the overall document. Focus on its role, relevance, and connection to the rest of the content. The context should be brief, precise, and tailored to improve search retrieval. Avoid redundancy, and refrain from repeating ideas. Answer with only the context and nothing else.
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
