import os
import time
from threading import Lock
from uuid import uuid4

import pymupdf  # temporal
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from settings import settings
from utils.helpers import SingletonMeta
from utils.types import Chunk, Document


class LoadDocuments(metaclass=SingletonMeta):
    _converter = None
    _chunker = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._converter = DocumentConverter(allowed_formats=["pdf"])
        self._chunker = HybridChunker(
            # tokenizer="BAAI/bge-small-en-v1.5",
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=settings.CTX_WINDOW,
        )

    def load(self, documents_path_or_url: list[str]) -> tuple[list, list]:
        docs = []
        chunks = []

        for d in documents_path_or_url:
            doc = self._convert(d)
            docs.append(doc)
            chunk = self._chunk(doc)
            chunks.append(chunk)

        return docs, chunks

    def _convert(self, document_path_or_url: str):
        assert self._converter, "DocumentConverter not initialized"
        with self._lock:
            document = self._converter.convert(document_path_or_url)
            return document

    def _chunk(self, document):
        assert self._chunker, "Chunker not initialized"
        with self._lock:
            chunks = self._chunker.chunk(document)
            return chunks


# # loader = LoadDocuments()
# # docs, chunks = loader.load(["docs/attention-is-all-you-need.pdf"])
# # print(docs)
# # print(chunks)

# converter = DocumentConverter()
# result = converter.convert("https://arxiv.org/pdf/2408.09869")
# print(result)


# TODO: temporal code until docling
def load_documents(
    paths: list[str], splitter
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
        with pymupdf.open(path) as d:
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


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs, chunks = load_documents(
#     ["docs/attention-is-all-you-need.pdf"],
#     splitter=RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=200,
#     ),
# )
# print(docs)
# print(chunks)
