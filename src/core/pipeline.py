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
from lib.prompts import PROMPT, create_prompt
from lib.schemas import (
    ChatMessage,
    Message,
    RetrievedChunk,
)
from lib.settings import settings
from lib.vectordb import VectorDB

log = logging.getLogger("app")

ChatMessageAndChunk = tuple[ChatMessage | str, list[RetrievedChunk]]

# Models -----------------------------------------------------------------------

llm = vLLMClient() if settings.ENVIRONMENT == "prod" else OpenAIClient()
embeddings = EmbeddingModel()
reranker = RerankerModel()


# Logic ------------------------------------------------------------------------


def ask(
    message: Message,
    history: list[ChatMessage],
    db: VectorDB,
    # generation params
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    # indexing
    advanced_indexing: bool,
    chunk_size: int,
    # retrieval
    top_k: int,
    top_r: int,
    threshold: float,
    # advanced options
    use_query_expansion: bool = True,
    use_query_decomposition: bool = True,
) -> Generator[ChatMessageAndChunk, None, None]:
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
            doc, doc_chunks = process_pdf(
                path=Path(file),
                chunk_size=chunk_size,
                chunk_overlap=200,
                model=llm,
                extend_chunks=advanced_indexing,
            )
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

        if db.is_empty:
            yield (
                ChatMessage(
                    role="assistant",
                    content="No hay contexto para responder la pregunta",
                ),
                [],
            )
            return

        final_query = query
        chunks: list[RetrievedChunk] = []

        # Query Expansion
        if use_query_expansion:
            yield (
                ChatMessage(
                    role="assistant",
                    content="",
                    metadata={"title": "Expanding Query", "status": "pending"},
                ),
                [],
            )
            final_query = expand_query(query, llm)
            log.debug("Consulta expandida: %s", final_query)
            yield (
                ChatMessage(
                    role="assistant",
                    content=final_query,
                    metadata={"title": "Expanding Query"},
                ),
                [],
            )

        # 2. Query Decomposition & Retrieval
        if use_query_decomposition and len(final_query) > 40:
            yield (
                ChatMessage(
                    role="assistant",
                    content="",
                    metadata={
                        "title": "Decomposing Query",
                        "status": "pending",
                    },
                ),
                [],
            )
            sub_queries = decompose_query(final_query, llm)
            log.debug(f"Sub-queries: {sub_queries}")
            yield (
                ChatMessage(
                    role="assistant",
                    content="\n".join(sub_queries),
                    metadata={"title": "Decomposing Query", "status": "done"},
                ),
                [],
            )

            for i, subq in enumerate(sub_queries):
                yield (
                    ChatMessage(
                        role="assistant",
                        content=f"Query: {subq}",
                        metadata={"title": "Retrieving..."},
                    ),
                    chunks,
                )
                subq_chunks = db.search(
                    subq,
                    embedding_model=embeddings,
                    reranker_model=reranker,
                    top_k=top_k // 2,
                    top_r=top_r,
                    threshold=threshold * 0.8,
                )
                log.debug("Retrieved %d subq_chunks", len(subq_chunks))
                chunks.extend(subq_chunks)

            # Desduplicar chunks manteniendo los mejores scores
            chunk_ids = set()
            unique_chunks = []
            for chunk in sorted(
                chunks,
                key=lambda x: x.scores["rerank_score"]
                or x.scores["hybrid_score"],
                reverse=True,
            ):
                if chunk.chunk.id not in chunk_ids:
                    chunk_ids.add(chunk.chunk.id)
                    unique_chunks.append(chunk)
            chunks = unique_chunks[:top_k]
        else:
            chunks = db.search(
                final_query,
                embedding_model=embeddings,
                reranker_model=reranker,
                top_k=top_k,
                top_r=top_r,
                threshold=threshold,
            )
        log.debug("Retrieved %d chunks", len(chunks))

        # 3. Generation
        prompt = create_prompt(final_query, history, chunks)
        log.debug("Generate Answer prompt:\n%s", prompt)

        buffer = ""
        for token in llm.generate_stream(messages=prompt, params=params):
            buffer += token
            yield buffer, chunks

        return

    except Exception as e:
        log.exception("Pipeline failed: %s", str(e), exc_info=True)
        log.info("Retrying ...")
        yield "Ocurrio un error durante el proceso.", []


def expand_query(query: str, llm: OpenAIClient | vLLMClient) -> str:
    log.info("Expanding query: %s", query)

    prompt = PROMPT["expand_query"].format(query=query)
    log.debug("Prompt: %s", prompt)

    response = llm.generate(
        [prompt],
        params={
            "max_tokens": 100,
            "temperature": 0.4,
        },
    )
    log.debug("Response: %s", response)

    return response[0]


def decompose_query(query: str, llm: OpenAIClient | vLLMClient) -> list[str]:
    log.info("Decompose query: %s", query)

    prompt = PROMPT["decompose_query"].format(query=query)
    log.debug("Prompt: %s", prompt)

    response = llm.generate(
        [prompt], params={"max_tokens": 100, "temperature": 0.4}
    )
    log.debug("Response: %s", response)

    return response[0].split("|")
