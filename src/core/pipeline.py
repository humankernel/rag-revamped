import logging
import time
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
from lib.types import (
    ChatMessage,
    GenerationState,
    Message,
    QueryPlan,
    RetrievedChunk,
    system_msg,
)
from lib.vectordb import KnowledgeBase
from settings import settings

log = logging.getLogger("rag")

ChatMessageAndChunk = tuple[ChatMessage | str, list[RetrievedChunk]]

# Models -----------------------------------------------------------------------

llm_model = OpenAIClient() if settings.ENVIRONMENT == "dev" else vLLMClient()
embedding_model = EmbeddingModel()
reranker_model = RerankerModel()


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
        log.info("Query: %s, Files: %s, History: %s", query, files, history)

        if files:
            start_time = time.time()
            log.info("Indexing %d file(s)", len(files))
            yield system_msg(title=f"Indexing {len(files)} files"), []

            docs, chunks = load_documents(files)
            db.insert(docs, chunks, embedding_model, batch_size=32)
            duration = time.time() - start_time

            log.debug("Indexed files in %.2f seconds", duration)
            yield system_msg(title=f"Indexed {len(files)} files"), []

        if not query:
            log.info("No query provided.")
            return

        if db.is_empty:
            log.info("db is empty, generating answer without context")
            prompt = create_prompt(query, history)
            buffer = ""
            for token in llm_model.generate_stream(prompt, params):
                buffer += token
                yield buffer, []
            return

        # 1. Generate Query Plan
        log.info("Generating initial query plan")
        yield system_msg("Query Plan", "Generating Query Plan"), []
        prompt = PROMPT["query_plan"].format(query=query)
        response = llm_model.generate(prompt, QueryPlan, params)
        plan = QueryPlan.model_validate_json(response)
        yield system_msg("Query Plan", str(plan)), []

        chunks: list[RetrievedChunk] = []
        state = GenerationState()
        seen_chunk_ids: set[str] = set()

        for iter in range(max_iterations):
            log.info("Iteration %d/%d", iter + 1, max_iterations)

            # 2. Retrieve
            for query in plan.sub_queries:
                log.info("Retrieving Chunks for query: %s", query)
                new_chunks = db.search(
                    query,
                    embedding_model=embedding_model,
                    reranker_model=reranker_model,
                    top_k=10,
                    top_r=3,
                    threshold=0.6,
                )
                log.debug("Chunks %s", new_chunks)
                for chunk in new_chunks:
                    if chunk.chunk.id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.chunk.id)
                        chunks.append(chunk)

            # 3. Generate Answer
            log.info("Generating Answer")
            prompt = create_prompt(query, history, chunks)
            state.answer = ""
            for token in llm_model.generate_stream(prompt, params):
                state.answer += token
                yield system_msg(title="Thinking", content=state.answer), chunks
            log.debug("Answer: %s", state.answer)

            # 4. Validate Answer
            log.info("Validating answer completeness")
            prompt = PROMPT["validate_answer"].format(
                query=plan.query, answer=state.answer
            )
            response = llm_model.generate(prompt, GenerationState)
            state = GenerationState.model_validate_json(response)
            log.debug("Gaps: %s", state.gaps)

            if not state.gaps:
                log.info("Answer is complete")
                break

            # 5. Refine query plan
            log.info("Answer has gaps. Refining query plan")
            prompt = PROMPT["query_plan"].format(query=". ".join(state.gaps))
            response = llm_model.generate(prompt, QueryPlan)
            plan = QueryPlan.model_validate_json(response)

        yield state.answer, chunks

    except Exception as e:
        log.exception("Pipeline failed: %s", str(e), exc_info=True)
        yield system_msg("An unexpected error occurred during processing."), []
