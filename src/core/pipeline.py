import logging
import time
from contextlib import contextmanager
from typing import Generator

from core.indexing import load_documents
from lib.helpers import (
    extract_message_content,
    parse_history,
)
from lib.models.embedding import EmbeddingModel
from lib.models.llm import GenerationParams, OpenAIClient, vLLMClient
from lib.models.rerank import RerankerModel
from lib.prompts import PROMPT, create_prompt
from lib.settings import settings
from lib.types import (
    ChatMessage,
    GenerationState,
    Message,
    QueryPlan,
    RetrievedChunk,
)
from lib.vectordb import KnowledgeBase

log = logging.getLogger("app")

ChatMessageAndChunk = tuple[ChatMessage | str, list[RetrievedChunk]]
MAX_RETRIES = 3

# Models -----------------------------------------------------------------------

llm = OpenAIClient() if settings.ENVIRONMENT == "dev" else vLLMClient()
embeddings = EmbeddingModel()
# reranker = RerankerModel()


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
    max_iterations: int = 3,
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

            if files:
                yield (
                    ChatMessage(
                        role="assistant",
                        content=f"Indexing {len(files)} files...",
                        metadata={"title": "Indexing", "status": "pending"},
                    ),
                    [],
                )
                with log_action(f"Indexing {len(files)} file(s)"):
                    docs, chunks = load_documents(files)
                    db.insert(docs, chunks, embeddings, batch_size=32)
                yield (
                    ChatMessage(
                        role="assistant",
                        content="Indexed Finished :)",
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
                with log_action("db empty, answering without context"):
                    history.append({"role": "user", "content": query})
                    buffer = ""
                    for token in llm.generate_stream(history, params):
                        buffer += token
                        yield buffer, []
                return

            # 1. Generate Query Plan
            with log_action("Generating initial Query Plan"):
                yield (
                    ChatMessage(
                        role="assistant",
                        content=f"Generating Query Plan for: {query}",
                        metadata={
                            "title": "Generating Query Plan",
                            "status": "pending",
                        },
                    ),
                    [],
                )
                prompt = PROMPT["query_plan"].format(query=query)
                log.debug("Query Plan prompt:\n%s", prompt)

                response = llm.generate(prompt, QueryPlan, params)
                log.debug("Query Plan response:\n%s", response)

                plan = QueryPlan.model_validate_json(response)
                yield (
                    ChatMessage(
                        role="assistant",
                        content=str(plan),
                        metadata={
                            "title": "Generating Query Plan",
                            "status": "done",
                        },
                    ),
                    [],
                )

            chunks: list[RetrievedChunk] = []
            state = GenerationState()
            seen_chunk_ids: set[str] = set()

            for iteration in range(max_iterations):
                log.info("Iteration %d/%d", iteration + 1, max_iterations)

                # 2. Retrieve
                for subquery in plan.sub_queries:
                    with log_action(
                        f"Retrieving Chunks for sub-query: {subquery}"
                    ):
                        new_chunks = db.search(
                            subquery,
                            embedding_model=embeddings,
                            reranker_model=reranker,
                            top_k=10,
                            top_r=3,
                            threshold=0.6,
                        )
                        log.debug("Retrieved %d chunks", len(new_chunks))

                        for chunk in new_chunks:
                            if chunk.chunk.id not in seen_chunk_ids:
                                seen_chunk_ids.add(chunk.chunk.id)
                                chunks.append(chunk)
                        yield "", chunks

                # 3. Generate Answer
                with log_action("Generating Answer"):
                    prompt = create_prompt(query, history, chunks)
                    log.debug("Generate Answer prompt:\n%s", prompt)

                    state.answer = ""
                    for token in llm.generate_stream(prompt, params):
                        state.answer += token
                        yield (
                            ChatMessage(
                                role="assistant",
                                content=state.answer,
                                metadata={
                                    "title": "Thinking",
                                    "status": "pending",
                                },
                            ),
                            chunks,
                        )

                    yield (
                        ChatMessage(
                            role="assistant",
                            content=state.answer,
                            metadata={
                                "title": "Thinking",
                                "status": "done",
                            },
                        ),
                        chunks,
                    )

                # 4. Validate Answer
                with log_action("Validating answer completeness"):
                    yield (
                        ChatMessage(
                            role="assistant",
                            content=f"Validating answer: {state.answer}",
                            metadata={
                                "title": "Validating answer",
                                "status": "pending",
                            },
                        ),
                        chunks,
                    )
                    prompt = PROMPT["validate_answer"].format(
                        query=plan.query, answer=state.answer
                    )
                    log.debug("Validating answer prompt:\n%s", prompt)

                    response = llm.generate(prompt, GenerationState, params)
                    log.debug("Validating answer response:\n%s", response)

                    state = GenerationState.model_validate_json(response)
                    yield (
                        ChatMessage(
                            role="assistant",
                            content=f"Answer Gaps: {state.gaps}",
                            metadata={
                                "title": "Validating answer",
                                "status": "done",
                            },
                        ),
                        chunks,
                    )

                if not state.gaps:
                    log.info(
                        "Answer complete (at iteration %d/%d)",
                        iteration,
                        max_iterations,
                    )
                    break

                # 5. Refine query plan
                with log_action("Answer has gaps. Refining query plan"):
                    yield (
                        ChatMessage(
                            role="assistant",
                            content=f"Creating new sub-queries using the gaps:\n{state.gaps}",
                            metadata={
                                "title": "Refining query plan",
                                "status": "pending",
                            },
                        ),
                        chunks,
                    )
                    prompt = PROMPT["query_plan"].format(
                        query=".\n".join(state.gaps)
                    )
                    log.debug("Refining query plan prompt:\n%s", prompt)

                    response = llm.generate(prompt, QueryPlan, params)
                    log.debug("Refining query plan response:\n%s", response)

                    plan = QueryPlan.model_validate_json(response)
                    log.debug(
                        "Refining query plan [sub-queries]:\n%s",
                        ".\n".join(plan.sub_queries),
                    )
                    yield (
                        ChatMessage(
                            role="assistant",
                            content=f"New sub-queries:\n{'.\n'.join(plan.sub_queries)}",
                            metadata={
                                "title": "Refining query plan",
                                "status": "done",
                            },
                        ),
                        chunks,
                    )

            yield state.answer, chunks

        except Exception as e:
            log.exception("Pipeline failed: %s", str(e), exc_info=True)
            yield "Ocurrio un error durante el proceso.", []


@contextmanager
def log_action(message: str, level=logging.DEBUG):
    start_time = time.time()
    log.info(message)
    try:
        yield
    finally:
        duration = time.time() - start_time
        log.log(level, "%s (took %.2f seconds)", message, duration)
