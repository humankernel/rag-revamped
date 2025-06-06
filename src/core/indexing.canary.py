import logging
from pathlib import Path

import fitz
import vllm
from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument, DocTagsDocument
from PIL import Image

from lib.settings import settings
from lib.types import Chunk, Document

MODEL_PATH = "ds4sd/SmolDocling-256M-preview"
IMAGE_DIR = "img/"  # Place your page images here
OUTPUT_DIR = "out/"

BATCH_SIZE = 1

log = logging.getLogger("app")


class SmolDocling:
    def __init__(self) -> None:
        log.info("Loading SmolDocling")
        self.model = vllm.LLM(
            model=MODEL_PATH,
            limit_mm_per_prompt={"image": BATCH_SIZE},
            dtype=settings.DTYPE,
            device=settings.DEVICE,
            gpu_memory_utilization=0.55,
        )
        self.chunker = HybridChunker()

    def _get_images_from_pdf(self, pdf_path: Path) -> list[Image.Image]:
        log.info("Extracting images from pdf")
        doc = fitz.open(pdf_path)

        images: list[Image.Image] = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Convert page to image (pixmap)
            pix = page.get_pixmap()

            # Convert pixmap to PIL image
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(image)

        log.debug("Extracted %d images", len(images))
        return images

    def process_pdf(self, pdf_path: Path) -> tuple[Document, list[Chunk]]:
        log.info("Processing PDF: %s", pdf_path)
        images = self._get_images_from_pdf(pdf_path)

        chat_template = "<|im_start|>User:<image>Convert page to Docling.<end_of_utterance>\nassistant:"
        prompts: list[vllm.PromptType] = [
            {
                "prompt": chat_template,
                "multi_modal_data": {"image": image},
            }
            for image in images
        ]

        log.info("OCR using SmolDocling model")
        output = self.model.generate(
            prompts,
            sampling_params=vllm.SamplingParams(
                temperature=0.0, max_tokens=4096
            ),
        )
        log.info("OCR using SmolDocling model finished")

        doctags = [o.outputs[0].text for o in output]
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            doctags, images
        )
        doc = DoclingDocument.load_from_doctags(
            doctags_doc, document_name="Document"
        )

        # Save markdown
        markdown_path = pdf_path.with_suffix(".md")
        doc.save_as_markdown(markdown_path)
        log.info("Saved into: %s", markdown_path)

        # chunk
        log.info("Chunking ...")
        chunk_iter = self.chunker.chunk(dl_doc=doc)

        document = Document(
            id="1", title=str(pdf_path), language="en", source=str(pdf_path)
        )
        chunks = [
            Chunk(
                id=str(id),
                doc_id="1",
                page=id,
                text=self.chunker.contextualize(chunk),
            )
            for id, chunk in enumerate(chunk_iter)
        ]

        return document, chunks


# indexer = SmolDocling()
# doc, chunks = indexer.process_pdf(Path("docs/attention-is-all-you-need.pdf"))