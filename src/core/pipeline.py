import logging
from pathlib import Path
from typing import Generator

from core.indexing import process_pdf
from lib.helpers import (
    extract_message_content,
    parse_history,
)
from lib.models.embedding import EmbeddingModel
from lib.models.llm import GenerationParams, OpenAIClient, vLLMClient
from lib.models.rerank import RerankerModel
from lib.prompts import create_prompt
from lib.settings import settings
from lib.types import (
    ChatMessage,
    Message,
    RetrievedChunk,
)
from lib.vectordb import KnowledgeBase

log = logging.getLogger("app")

ChatMessageAndChunk = tuple[ChatMessage | str, list[RetrievedChunk]]
MAX_RETRIES = 3
TOP_K = 10
TOP_R = 4
THRESHOLD = 0.5

# Models -----------------------------------------------------------------------

llm = vLLMClient() if settings.ENVIRONMENT == "prod" else OpenAIClient()
embeddings = EmbeddingModel()
# reranker = RerankerModel()
reranker = None


# Logic ------------------------------------------------------------------------


def ask(
    message: Message,
    history: list[ChatMessage],
    db: KnowledgeBase,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> Generator[ChatMessageAndChunk, None, None]:
    for _ in range(MAX_RETRIES):
        try:
            query, files = extract_message_content(message)
            history = parse_history(history)
            params: GenerationParams = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
            log.info("Query: %s, Files: %s", query, files)

            # 1. Indexing
            for file in files:
                yield (
                    ChatMessage(
                        role="assistant",
                        content=f"Indexing file: {file}",
                        metadata={"title": "Indexing", "status": "pending"},
                    ),
                    [],
                )
                doc, doc_chunks = process_pdf(path=Path(file), model=llm)
                db.insert([doc], doc_chunks, embeddings, batch_size=32)
                yield (
                    ChatMessage(
                        role="assistant",
                        content=f"Indexed Finished: (w/ {len(doc_chunks)} chunks)",
                        metadata={
                            "title": "Indexing",
                            "status": "done",
                        },
                    ),
                    [],
                )

            if not query:
                log.info("No query provided.")
                return

            # 2. Retrieval
            chunks: list[RetrievedChunk] = []
            if not db.is_empty:
                chunks = db.search(
                    query,
                    embedding_model=embeddings,
                    reranker_model=reranker,
                    top_k=TOP_K,
                    top_r=TOP_R,
                    threshold=THRESHOLD,
                )
                log.debug("Retrieved %d chunks", len(chunks))

            # 3. Generation
            prompt = create_prompt(query, history, chunks)
            log.debug("Generate Answer prompt:\n%s", prompt)

            buffer = ""
            for token in llm.generate_stream(prompt, params):
                buffer += token
                yield buffer, chunks

            return

        except Exception as e:
            log.exception("Pipeline failed: %s", str(e), exc_info=True)
            yield "Ocurrio un error durante el proceso.", []


# Como DMA reduce el uso de la CPU?
