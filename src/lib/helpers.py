from functools import lru_cache

import tiktoken

from lib.types import ChatMessage, Message


@lru_cache(1000)
def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    return len(tokens)


def extract_message_content(message: Message) -> tuple[str | None, list[str]]:
    if isinstance(message, dict):
        return message.get("text", ""), message.get("files", [])
    return message, []


def parse_history(history: list[ChatMessage]) -> list[ChatMessage]:
    # TODO: this step only need to be run in the last history input
    # TODO: history could be empty
    # this is because gradio can make a chat content a tuple instead of a str
    # Ensure all the content is strings
    for h in history:
        if isinstance(h["content"], tuple):
            h["content"] = h["content"][1] if len(h["content"]) > 1 else ""
    return history
