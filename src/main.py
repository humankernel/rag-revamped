import logging
import time

import gradio as gr

from core.pipeline import ask
from lib.settings import settings
from lib.types import RetrievedChunk
from lib.vectordb import KnowledgeBase

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
        db = gr.State(KnowledgeBase("att", test=True))
        local_storage = gr.BrowserState()
        chunks = gr.State([])

        with gr.Sidebar(position="left", open=False):
            gr.Markdown("# Model Settings")
            temperature = gr.Slider(
                0.0,
                1.0,
                value=0.6,
                step=0.1,
                label="Temperature",
                info="Controls randomness: Lower = more deterministic, Higher = more creative.",
            )
            top_p = gr.Slider(
                0.0,
                1.0,
                value=0.95,
                step=0.05,
                label="Top-p",
                info="Considers the smallest set of tokens whose cumulative probability exceeds p.",
            )
            max_tokens = gr.Slider(
                1,
                4000,
                value=1024,
                step=10,
                label="Max Tokens",
                info="Sets the maximum number of tokens in the response.",
            )
            frequency_penalty = gr.Slider(
                0.0,
                2.0,
                value=0.5,
                step=0.1,
                label="Frequency Penalty",
                info="Reduces the likelihood of repeating common phrases.",
            )
            presence_penalty = gr.Slider(
                0.0,
                2.0,
                value=0.5,
                step=0.1,
                label="Presence Penalty",
                info="Encourages new topics.",
            )
            saved_message = gr.Markdown("", visible=False)

        with gr.Sidebar(position="right", open=False):
            gr.Markdown("# References")

            @gr.render(chunks)
            def render_chunks(chunks: list[RetrievedChunk]):
                for chunk in chunks:
                    with gr.Accordion(
                        label=f"id: {chunk.chunk.id}, doc_id: {chunk.chunk.doc_id}",
                        open=False,
                    ):
                        gr.Markdown(
                            f"scores: (dense:{chunk.scores['dense_score']:.2f}"
                            f", sparse:{chunk.scores['sparse_score']:.2f}"
                            f", hybrid:{chunk.scores['hybrid_score']:.2f}"
                            f", rerank:{chunk.scores['rerank_score']:.2f}) \n\n"
                            f"{chunk.chunk.text}",
                        )

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
