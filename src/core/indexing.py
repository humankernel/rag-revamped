import os
import time
from uuid import uuid4

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.types import Chunk, Document


def load_documents(
    paths: list[str],
    splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    ),
) -> tuple[list[Document], list[Chunk]]:
    assert all(os.path.isfile(path) for path in paths), (
        f"Every document path should point to an existing file: {paths}"
    )
    assert all(path.endswith(".pdf") for path in paths), (
        "This currently only supports pdf"
    )

    docs: list[Document] = []
    chunks: list[Chunk] = []

    for path in paths:
        doc_id = str(uuid4())
        docs.append(
            Document(
                id=doc_id,
                source=path,
                metadata={"created_at": str(time.time())},
            )
        )
        with fitz.open(path) as d:
            for idx, page in enumerate(d):
                page_text = page.get_text()
                page_chunks_texts: list[str] = splitter.split_text(page_text)
                page_chunks = (
                    Chunk(
                        id=str(uuid4()),
                        doc_id=doc_id,
                        page=idx,
                        text=text.strip(),
                    )
                    for text in page_chunks_texts
                    if text.strip()
                )
                chunks.extend(page_chunks)
    return docs, chunks
