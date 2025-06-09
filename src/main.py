import logging
import time

import gradio as gr

from core.pipeline import ask
from lib.schemas import RetrievedChunk
from lib.settings import settings
from lib.vectordb import VectorDB

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logs.log",
    filemode="a",
    encoding="utf-8",
)
log = logging.getLogger("app")


def main() -> None:
    log.info("Starting RAG in %s mode", settings.ENVIRONMENT)

    with gr.Blocks(fill_height=True) as ui:
        db = gr.State(VectorDB())
        local_storage = gr.BrowserState()
        chunks = gr.State([])

        with gr.Sidebar(position="left", open=False):
            gr.Markdown("# Model Options")
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.6,
                step=0.1,
                label="Temperature",
                info="Controls randomness: Lower = more deterministic, Higher = more creative.",
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.95,
                step=0.05,
                label="Top-p",
                info="Considers the smallest set of tokens whose cumulative probability exceeds p.",
            )
            max_tokens = gr.Slider(
                minimum=1,
                maximum=4000,
                value=1024,
                step=10,
                label="Max Tokens",
                info="Sets the maximum number of tokens in the response.",
            )
            frequency_penalty = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.1,
                label="Frequency Penalty",
                info="Reduces the likelihood of repeating common phrases.",
            )
            presence_penalty = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.1,
                label="Presence Penalty",
                info="Encourages new topics.",
            )

            gr.Markdown("# Indexing Options")
            advanced_indexing = gr.Checkbox(
                value=True,
                label="Extend Chunks with Contextual Information",
                info="Enabling this will improve chunks, but make indexing much more slow.",
            )
            chunk_size = gr.Number(
                value=1200,
                minimum=100,
                maximum=5000,
                step=10,
                label="Max size of each chunk",
            )

            gr.Markdown("# Retrieval Options")
            top_k = gr.Number(
                value=20,
                minimum=3,
                maximum=30,
                step=1,
                label="K Items to Retrieve",
            )
            top_r = gr.Number(
                value=10,
                minimum=1,
                maximum=20,
                step=1,
                label="R Items to Keep After the initial K items",
            )
            threshold = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.1,
                label="Retrieval Score Cutoff",
            )
            saved_message = gr.Markdown("", visible=False)

            gr.Markdown("# Advanced Options")
            use_query_expansion = gr.Checkbox(
                value=True,
                label="Expand the Query",
                info="Enabling this will improve retrieval, but slow the answer.",
            )
            use_query_decomposition = gr.Checkbox(
                value=True,
                label="Query Decomposition",
                info="Enabling this will improve retrieval for complex multi-hop queries, but slow the answer.",
            )

        with gr.Sidebar(position="right", open=False):
            gr.Markdown("# References")

            @gr.render(chunks)
            def render_chunks(chunks: list[RetrievedChunk]):
                for chunk in chunks:
                    with gr.Accordion(
                        label=f"id: {chunk.chunk.id}, doc_id: {chunk.chunk.doc_id}",
                        open=False,
                    ):
                        gr.Markdown(str(chunk))

        gr.ChatInterface(
            fn=ask,
            type="messages",
            multimodal=True,
            editable=True,
            flagging_mode="manual",
            flagging_options=[
                "Like",
                "Dislike",
                "Hallucination",
                "Inappropriate",
                "Harmful",
            ],
            flagging_dir=".",
            additional_inputs=[
                db,
                temperature,
                max_tokens,
                top_p,
                frequency_penalty,
                presence_penalty,
                advanced_indexing,
                chunk_size,
                top_k,
                top_r,
                threshold,
                use_query_expansion,
                use_query_decomposition,
            ],
            additional_outputs=[chunks],
            save_history=True,
        )

        @ui.load(
            inputs=[local_storage],
            outputs=[
                temperature,
                max_tokens,
                top_p,
                frequency_penalty,
                presence_penalty,
            ],
        )
        def load_from_local_storage(saved_values):
            defaults = {
                "temperature": 0.75,
                "max_tokens": 300,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
            }
            saved_values = saved_values or defaults
            return (
                saved_values["temperature"],
                saved_values["max_tokens"],
                saved_values["top_p"],
                saved_values["frequency_penalty"],
                saved_values["presence_penalty"],
            )

        @gr.on(
            [
                temperature.change,
                max_tokens.change,
                top_p.change,
                frequency_penalty.change,
                presence_penalty.change,
            ],
            inputs=[
                temperature,
                max_tokens,
                top_p,
                frequency_penalty,
                presence_penalty,
            ],
            outputs=[local_storage],
        )
        def save_to_local_storage(
            temp,
            max_tks,
            top_p,
            freq_pen,
            pres_pen,
        ):
            return {
                "temperature": temp,
                "max_tokens": max_tks,
                "top_p": top_p,
                "frequency_penalty": freq_pen,
                "presence_penalty": pres_pen,
            }

        @gr.on(local_storage.change, outputs=[saved_message])
        def show_saved_message():
            timestamp = time.strftime("%I:%M:%S %p")
            return gr.Markdown(
                f"✅ Saved to local storage at {timestamp}", visible=True
            )

    ui.queue(api_open=False).launch(
        max_file_size="300mb",
        share=settings.ENVIRONMENT == "prod",
    )


if __name__ == "__main__":
    main()
