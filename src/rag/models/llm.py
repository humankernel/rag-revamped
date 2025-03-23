from functools import cache
from threading import Lock
from typing import Generator

from vllm import LLM, SamplingParams

from rag.settings import settings
from rag.utils.helpers import SingletonMeta


class LLMModel(metaclass=SingletonMeta):
    _model = None
    _lock: Lock = Lock()

    def __init__(self, model_name_or_path: str = settings.LLM_MODEL, **kwargs):
        # TODO: parse kwargs
        # TODO: explore bitsandbytes (https://docs.vllm.ai/en/stable/features/quantization/bnb.html)
        # TODO: bench .gguf vs .safetensors
        self._model = LLM(
            model=model_name_or_path,
            dtype=settings.DTYPE,
            gpu_memory_utilization=0.5,  # By increasing this utilization, you can provide more KV cache space.
            task="generate",
            seed=42,
            enforce_eager=False,  # Enable CUDA graphs for performance
            max_model_len=settings.CTX_WINDOW,
            max_num_seqs=2,  # Single sequence for simplicity, adjust if batching
            enable_chunked_prefill=True,  # allows to chunk large prefills into smaller chunks and batch them together with decode requests
        )

    @cache
    def count_tokens(self, text: str) -> int:
        assert self._model, "LLM model not initialized"
        tokens = self._model.get_tokenizer().encode(text)
        return len(tokens)

    def generate(self, prompt: str, model_params: dict = {}) -> str:
        assert prompt, "A prompt must be provided!"
        total_tokens = self.count_tokens(prompt)
        assert total_tokens < settings.CTX_WINDOW, "messages exceed ctx window"

        sampling_params = SamplingParams(
            max_tokens=model_params.get("max_tokens", 300),
            temperature=model_params.get("temperature", 0.6),
            top_p=model_params.get("top_p", 0.95),
            frequency_penalty=model_params.get("frequency_penalty", 0.5),
            presence_penalty=model_params.get("presence_penalty", 1.2),
            repetition_penalty=model_params.get("repetition_penalty", 1.2),
        )

        with self._lock:
            outputs = self._model.generate(prompt, sampling_params)
            return outputs[0].outputs[0].text

    def generate_stream(
        self, prompt: str, model_params: dict = {}
    ) -> Generator[str, None, None]:
        assert prompt, "A prompt must be provided!"
        total_tokens = self.count_tokens(prompt)
        assert total_tokens < settings.CTX_WINDOW, "messages exceed ctx window"

        sampling_params = SamplingParams(
            max_tokens=model_params.get("max_tokens", 300),
            temperature=model_params.get("temperature", 0.6),
            top_p=model_params.get("top_p", 0.95),
            frequency_penalty=model_params.get("frequency_penalty", 0.5),
            presence_penalty=model_params.get("presence_penalty", 1.2),
            repetition_penalty=model_params.get("repetition_penalty", 1.2),
        )

        with self._lock:
            outputs = self._model.generate(prompt, sampling_params)
            for output in outputs:
                yield output.outputs[0].text
