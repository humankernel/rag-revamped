from functools import cache
import logging
import time
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.models.llm import OpenAIClient, vLLMClient
from lib.prompts import PROMPT
from lib.schemas import Chunk, Document

log = logging.getLogger("app")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
)


class ContextualizedChunk(TypedDict):
    page_text: str
    chunk_text: str


def normalize_text(text: str) -> str:
    return text.strip().replace("\n", " ")

@cache
def process_pdf(
    path: Path,
    model: vLLMClient | OpenAIClient | None = None,
) -> tuple[Document, list[Chunk]]:
    log.info("Process PDF: %s", path)
    if path.suffix != ".pdf":
        raise ValueError("Input file must be a PDF")

    chunks_with_context: list[ContextualizedChunk] = []

    # 1. Split PDF content into chunks (with context)
    with fitz.open(path) as doc:
        for page in doc:
            page_text: str = normalize_text(page.get_text())
            page_chunks = splitter.split_text(page_text)
            chunks_with_context.extend([
                {
                    "page_text": page_text,
                    "chunk_text": normalize_text(chunk_text),
                }
                for chunk_text in page_chunks
            ])

    # 2. Extend chunks with contextual info (w/ LLM)
    if model:
        log.info("Extending chunks with context")
        # Generate prompts for augmentation
        prompts = [
            PROMPT["contextualize"].format(
                context=chunk["page_text"],
                chunk=chunk["chunk_text"],
            )
            for chunk in chunks_with_context
        ]

        # Get augmented text from LLM
        start_time = time.time()
        augmented_texts = model.generate(
            prompts,
            params={
                "temperature": 0.2,
                "max_tokens": 150,
                "top_p": 0.85,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.0,
                "stop": ["\n"],
            },
        )
        log.debug("Took: %d secs.", time.time() - start_time)
    else:
        log.info("Fallback to not extending chunks")
        # Use original chunk text if augmentation is disabled
        augmented_texts = [chunk["chunk_text"] for chunk in chunks_with_context]

    # create doc and chunk
    doc_id = str(uuid4())
    document = Document(id=doc_id, source=str(path))
    chunks = [
        Chunk(
            id=str(uuid4()),
            doc_id=doc_id,
            original_text=ctx_chunk["chunk_text"],
            text=aug_chunk,
        )
        for ctx_chunk, aug_chunk in zip(chunks_with_context, augmented_texts)
    ]
    log.debug("%d chunks created", len(chunks))
    return document, chunks


# llm = vLLMClient()
# process_pdf(path=Path("docs/attention-is-all-you-need.pdf"), model=llm)
# 87.57514214515686 secs
# 76.82038497924805 secs
