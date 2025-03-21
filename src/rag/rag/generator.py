from typing import Generator

from rag.models.llm import LLMModel
from rag.settings import settings
from rag.utils.types import ChatMessage, RetrievedChunk


def compress_history(
    history: list[ChatMessage], max_tokens: int = settings.CTX_WINDOW
) -> list[ChatMessage]:
    # TODO: Enhance to keep only relevant previous messages
    relevant_history = []
    total_tokens = 0
    for h in reversed(history):
        h_tokens = len(h["content"]) // 4
        if h_tokens + total_tokens > max_tokens:
            break
        relevant_history.append(h)
        total_tokens += h_tokens
    return list(reversed(relevant_history))


def compress_context(
    chunks: list[RetrievedChunk], max_tokens: int = settings.CTX_WINDOW
) -> str:
    # TODO: Use LLM to refine context instead of simple truncation
    formatted = [
        f"[Doc {i + 1}] {chunk.chunk.text}" for i, chunk in enumerate(chunks)
    ]
    context = "\n\n".join(formatted)
    return context[: max_tokens * 4]
    # todo: instead use LLM to clean the context


def create_prompt(query: str, history: list[ChatMessage], context: str) -> str:
    def format_chat_messages(messages: list[ChatMessage]) -> str:
        prompt = ""
        for msg in messages:
            if msg["role"] == "system" or msg["role"] == "assistant":
                prompt += f"<｜Assistant｜> {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"<｜User｜> {msg['content']}\n\n"
        return prompt

    prompt = format_chat_messages(history)
    prompt += f"<｜User｜> {query}\n\n"

    if context:
        prompt += (
            f"<｜Assistant｜> Given the context: {context},"
            f"let’s reason step-by-step about how to respond to your query: {query}. "
            " <think>\n"
        )
    else:
        # TODO: logic to avoid hallucinations
        prompt += "<｜Assistant｜> <think>\n"

    # 'Provide step-by-step reasoning enclosed in <think> </think> tags, followed by the final answer enclosed in \boxed{} tags.' \
    # If its math
    # "Please reason step by step, and put your final answer within \boxed{}"
    return prompt


def generate_answer(
    query: str,
    history: list[ChatMessage],
    chunks: list[RetrievedChunk],
    model: LLMModel,
    model_params: dict,
) -> Generator[str, None, None]:
    assert all(isinstance(msg["content"], str) for msg in history), (
        f"History content must be strings: {history}"
    )

    relevant_history = compress_history(history)
    relevant_context = compress_context(chunks)
    prompt = create_prompt(
        query=query, history=relevant_history, context=relevant_context
    )

    for response in model.generate(prompt, model_params):
        yield response
