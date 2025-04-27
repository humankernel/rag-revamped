import time
from typing import Generator

from pydantic import BaseModel, Field

from core.generation import generate_answer
from core.indexing import load_documents
from lib.helpers import extract_message_content, parse_history
from lib.models.embedding import EmbeddingModel
from lib.models.llm import OpenAIClient, vLLMClient
from lib.prompts import PROMPT
from lib.types import ChatMessage, Message, RetrievedChunk
from lib.vectordb import KnowledgeBase
from settings import settings

ChatMessageAndChunk = tuple[ChatMessage | str, list[RetrievedChunk]]


# Schemas ----------------------------------------------------------------------


class QueryPlan(BaseModel):
    query: str = Field(
        default="",
        description="Normalized query with key aspects separated by hyphens\n"
        "Format:\n- aspect 1\n- aspect 2\n- aspect 3",
        examples=[
            "- smartphone comparison\n- 2023 models\n- technical specifications",
            "- cellular respiration\n- biological processes\n- energy production",
            "- REST API\n- error handling\n- best practices",
        ],
    )
    sub_queries: list[str] = Field(
        default=[],
        description="Hypothetical document sections that would contain answers\n"
        "Format for each entry:\n- doc section 1\n- doc section 2",
        examples=[
            [
                "- iPhone 15 Pro specs\n- materials\n- processor\n- camera specs",
                "- Battery comparison\n- charging speed\n- endurance tests",
                "- Display technology\n- brightness\n- refresh rates",
            ],
            [
                "- Mitochondria structure\n- electron transport chain\n- ATP synthesis",
                "- Enzymatic reactions\n- NADH oxidation\n- oxygen role",
            ],
        ],
    )
    language: str = Field(
        default="english",
        description="Language of the original query",
        examples=["english", "spanish"],
    )

    def __str__(self) -> str:  # TODO: improve this
        base = "Structured Query Aspects:\n" + self.query.replace("- ", "• ")
        if self.sub_queries:
            doc_str = "\n\nHypothetical Document Structures:"
            for i, sq in enumerate(self.sub_queries, 1):
                doc_str += f"\n\nDocument {i}:\n" + sq.replace("- ", "  ▸ ")
            return base + doc_str
        return base


class GenerationState(BaseModel):
    answer: str = Field(
        default="",
        description="The generated answer so far. May be partial or complete.",
        examples=[
            "The Eiffel Tower is located in Paris and was built in 1889."
        ],
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="List of missing pieces of information or unclear aspects. "
        "These can guide future queries.",
        examples=[
            "When was the Eiffel Tower renovated?",
            "Who funded the construction of the tower?",
        ],
    )
    complete: bool = Field(
        default=False,
        description="Whether the answer is considered complete and does not need further refinement.",
        examples=[True, False],
    )


# Models -----------------------------------------------------------------------


llm_model = OpenAIClient() if settings.ENVIRONMENT == "dev" else vLLMClient()
embedding_model = EmbeddingModel()
reranker_model = None


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
    query, files = extract_message_content(message)
    history = parse_history(history)
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }

    # Indexing files
    if files:
        yield from insert_files(files, db)

    if not query:
        return

    use_rag = not db.is_empty
    if not use_rag:
        answer = ""
        for token in generate_answer(
            query=query,
            history=history,
            chunks=None,
            model=llm_model,
            params=params,
        ):
            answer += token
            yield answer, []
    else:
        yield (
            ChatMessage(
                role="assistant",
                content="",
                metadata={
                    "title": "🔥 Answering using documents",
                    "status": "pending",
                },
            ),
            [],
        )

        plan = QueryPlan()
        state = GenerationState()
        chunks: list[RetrievedChunk] = []

        yield from create_query_plan(query, plan)

        iterations = 0
        while not state.complete and iterations < max_iterations:
            yield from iterative_retrieval(plan, db, chunks)

            # Generate Answer
            state.answer = ""
            for token in generate_answer(
                query=query,
                history=history,
                chunks=None,
                model=llm_model,
                params=params,
            ):
                state.answer += token
                yield (
                    ChatMessage(
                        role="assistant",
                        content=state.answer,
                        metadata={"title": "Thinking", "status": "pending"},
                    ),
                    chunks,
                )

            # Validate Answer
            yield from validate_answer(plan, state)

            if not state.complete:
                yield from refine_query_plan(plan, state)

            iterations += 1

        yield state.answer, chunks


def insert_files(
    files: list[str], db: KnowledgeBase
) -> Generator[ChatMessageAndChunk, None, None]:
    start_time = time.time()
    yield (
        ChatMessage(
            role="assistant",
            content=f"📥 Received {len(files)} Files(s). Starting indexing...",
            metadata={"title": "🔎 Indexing...", "status": "pending"},
        ),
        [],
    )
    docs, chunks = load_documents(files)
    yield (
        ChatMessage(
            role="assistant",
            content=f"Loaded {len(docs)} docs & {len(chunks)} chunks",
            metadata={
                "title": "🔎 Indexing...",
                "status": "pending",
            },
        ),
        [],
    )
    db.insert(docs, chunks, embedding_model, batch_size=32)
    yield (
        ChatMessage(
            role="assistant",
            content=f"Indexed {len(files)} files",
            metadata={
                "title": "🔎 Indexing...",
                "duration": time.time() - start_time,
                "status": "done",
            },
        ),
        [],
    )


def create_query_plan(
    query: str, plan: QueryPlan
) -> Generator[ChatMessageAndChunk, None, None]:
    """Generate initial query plan using LLM"""
    start_time = time.time()
    yield (
        ChatMessage(
            role="assistant",
            content=f"Creating Query Plan for {query}",
            metadata={
                "title": "✍ Creating Query Plan",
                "status": "pending",
            },
        ),
        [],
    )

    prompt = PROMPT["query_plan"].format(query=query)
    response = llm_model.generate(
        prompt, output_format=QueryPlan, params={"max_tokens": 500}
    )
    plan.query = query
    plan.sub_queries = response.sub_queries
    yield (
        ChatMessage(
            role="assistant",
            content=str(plan),
            metadata={
                "title": "✍ Creating Query Plan",
                "duration": time.time() - start_time,
                "status": "done",
            },
        ),
        [],
    )


def iterative_retrieval(
    plan: QueryPlan, db: KnowledgeBase, chunks: list[RetrievedChunk]
) -> Generator[ChatMessageAndChunk, None, None]:
    """Multi-step retrieval with query plan"""
    unique_chunks = set(chunks)
    for query in plan.sub_queries:
        new_chunks = db.search(
            query,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            top_k=10,
            top_r=3,
            threshold=0.5,
        )
        yield "", new_chunks

        unique_chunks.update(new_chunks)

    chunks = list(unique_chunks)
    yield "", chunks


def validate_answer(
    plan: QueryPlan,
    state: GenerationState,
) -> Generator[ChatMessageAndChunk, None, None]:
    """Validate completeness"""
    yield (
        ChatMessage(
            role="assistant",
            content=(
                "Validating if the current answer has any gaps:\n"
                f"Question: {plan.query}"
                f"Answer: {state.answer}"
            ),
            metadata={
                "title": "🔎 Validating Answer",
                "status": "pending",
            },
        ),
        [],
    )
    prompt = PROMPT["validate_answer"].format(
        query=plan.query, answer=state.answer
    )
    response = llm_model.generate(prompt, output_format=GenerationState)

    state.gaps = response.gaps
    state.complete = response.complete
    yield (
        ChatMessage(
            role="assistant",
            content=(
                "Answer is complete"
                if state.complete
                else f"Answer has gaps...\nGaps:\n{'\n-'.join(state.gaps)}"
            ),
            metadata={
                "title": "🔎 Validating Answer",
                "status": "pending",
            },
        ),
        [],
    )


def refine_query_plan(
    plan: QueryPlan, state: GenerationState
) -> Generator[ChatMessageAndChunk, None, None]:
    """Refine query plan based on missing information"""
    yield (
        ChatMessage(
            role="assistant",
            content="Refining...",
            metadata={
                "title": "✍ Refining Query Plan",
                "status": "pending",
            },
        ),
        [],
    )
    prompt = PROMPT["query_plan"].format(query=". ".join(state.gaps))
    response = llm_model.generate(prompt, output_format=QueryPlan)
    plan.sub_queries = response.sub_queries
    yield (
        ChatMessage(
            role="assistant",
            content="New sub-queries\n" + "\n-".join(plan.sub_queries),
            metadata={
                "title": "✍ Refining Query Plan",
                "status": "pending",
            },
        ),
        [],
    )
