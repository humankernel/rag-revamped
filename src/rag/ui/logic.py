import time
from typing import Generator

import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter  # TODO: temp

from rag.models.embedding import EmbeddingModel
from rag.models.llm import LLMModel
from rag.models.rerank import RerankerVLLMModel
from rag.rag.generator import generate_answer
from rag.rag.indexing import load_documents
from rag.rag.vectordb import KnowledgeBase
from rag.utils.types import ChatMessage, Message, RetrievedChunk


# class MockedModel:
#     def generate(self, prompt, model_params):
#         for response in ["Hola", "esto", "es", "una", "prueba"]:
#             time.sleep(1)
#             yield response


model = LLMModel()
# model = MockedModel()
embedding_model = EmbeddingModel()
reranker_model = RerankerVLLMModel()


def ask(
    message: Message,
    history: list[ChatMessage],
    db: KnowledgeBase,
    temperature,
    max_tokens,
    top_p,
    top_k,
    frequency_penalty,
    presence_penalty,
) -> Generator[tuple[ChatMessage | str, list[RetrievedChunk]], None, None]:
    start_time = time.time()
    query = message.get("text", "") if isinstance(message, dict) else message
    files = message.get("files", []) if isinstance(message, dict) else []
    chunks: list[RetrievedChunk] = []

    # Ensure all the content is strings
    for h in history:
        if isinstance(h["content"], tuple):
            h["content"] = h["content"][1] if len(h["content"]) > 1 else ""

    # Indexing files
    if files:
        if any(not f.endswith(".pdf") for f in files):
            gr.Warning("The system only supports PDF files currently.")
        else:
            yield (
                {
                    "role": "assistant",
                    "content": f"📥 Received {len(files)} Files(s). Starting indexing...",
                    "metadata": {
                        "title": "🔎 Indexing...",
                        "status": "pending",
                    },
                },
                chunks,  # TODO: this return should be optional
            )

            # TODO: impl Loading files and chunking
            f_docs, f_chunks = load_documents(
                files,
                splitter=RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                ),
            )

            yield (
                {
                    "role": "assistant",
                    "content": f"Loaded {len(f_docs)} docs & {len(f_chunks)} chunks",
                    "metadata": {
                        "title": "📥 Finished Loading & Chunking",
                        "duration": time.time() - start_time,
                        "status": "done",
                    },
                },
                chunks,
            )

            db.insert(f_docs, f_chunks, embedding_model, batch_size=32)

            yield (
                {
                    "role": "assistant",
                    "content": f"Indexed {len(files)} files",
                    "metadata": {
                        "title": "📥 Finished Indexing",
                        "duration": time.time() - start_time,
                        "status": "done",
                    },  # TODO: make time for each step
                },
                chunks,
            )

    if not query:
        return

    # Retrieval
    if not db.is_empty:
        yield (
            {
                "role": "assistant",
                "content": "Searching relevant sources",
                "metadata": {
                    "title": "🔎 Searching...",
                    "duration": time.time() - start_time,
                    "status": "pending",
                },
            },
            chunks,
        )

        # TODO: inform ui of everything is happening inside
        chunks = db.search(
            query,
            top_k=20,
            top_r=5,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            threshold=1,
        )

        yield (
            {
                "role": "assistant",
                "content": f"Retrieved {len(chunks)} relevant sources",
                "metadata": {
                    "title": f"🔎 Found {len(chunks)} relevant chunks",
                    "duration": time.time() - start_time,
                    "status": "done",
                },
            },
            chunks,
        )

    model_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": True,
    }

    # Generation
    try:
        answer_buffer = ""
        for stream in generate_answer(
            query, history, chunks, model, model_params
        ):
            answer_buffer += stream
            yield answer_buffer, chunks
    except Exception as e:
        print(f"Error {e}")
        return "An error occurred while processing your request.", []
