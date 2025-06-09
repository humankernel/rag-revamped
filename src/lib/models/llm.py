import logging
from typing import Generator, Type

import vllm
from openai import OpenAI
from pydantic import BaseModel
from vllm.sampling_params import GuidedDecodingParams

from lib.helpers import count_tokens
from lib.settings import settings
from lib.schemas import ChatMessage, GenerationParams

DEFAULT_PARAMS: GenerationParams = {
    "max_tokens": 500,
    "temperature": 0.25,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 1.2,
}

log = logging.getLogger("app")


class OpenAIClient:
    def __init__(self) -> None:
        log.info("llm: starting client in dev mode")
        self.model = OpenAI(
            api_key="API",
            base_url=settings.CLIENT_URL,
        )

    def generate(
        self,
        prompts: list[str],
        output_format: Type[BaseModel] | None = None,
        params: GenerationParams = {},
    ) -> list[str]:
        assert all(
            count_tokens(prompt) < settings.CTX_WINDOW for prompt in prompts
        )
        assert prompts

        params = DEFAULT_PARAMS | params
        all_outputs: list[str] = []

        for prompt in prompts:
            response = self.model.chat.completions.create(
                messages=[{"role": "assistant", "content": prompt}],
                model=settings.LLM_MODEL,
                extra_body={"guided_json": output_format.model_json_schema()}
                if output_format
                else None,
                **params,
            )
            response = response.choices[0].message.content
            if response:
                all_outputs.append(response)

        return all_outputs

    def generate_stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams = {},
    ) -> Generator[str, None, None]:
        assert len(messages) > 0

        params = DEFAULT_PARAMS | params
        response = self.model.chat.completions.create(
            messages=messages,  # type: ignore
            model=settings.LLM_MODEL,
            stream=True,
            **params,
        )
        for output in response:
            yield output.choices[0].delta.content or ""


class vLLMClient:
    def __init__(self) -> None:
        log.info(
            "llm: starting vllm with:\nmodel:%s - dtype:%s - max_model_len:%s",
            settings.LLM_MODEL,
            settings.DTYPE,
            settings.CTX_WINDOW,
        )
        self.model = vllm.LLM(
            model=settings.LLM_MODEL,
            dtype=settings.DTYPE,
            task="generate",
            max_model_len=settings.CTX_WINDOW,
            enable_chunked_prefill=True,
            reasoning_parser="deepseek_r1",
            guided_decoding_backend="xgrammar",
            enable_prefix_caching=True,
            max_num_seqs=8,
            max_num_batched_tokens=8192,  # batching
            gpu_memory_utilization=0.6,
        )

    def generate(
        self,
        prompts: list[str],
        output_format: Type[BaseModel] | None = None,
        params: GenerationParams = {},
    ) -> list[str]:
        assert all(
            count_tokens(prompt) < settings.CTX_WINDOW for prompt in prompts
        )
        assert prompts

        params = DEFAULT_PARAMS | params
        response = self.model.generate(
            prompts,
            sampling_params=vllm.SamplingParams(
                **params,
                guided_decoding=GuidedDecodingParams(
                    json=output_format.model_json_schema()
                )
                if output_format
                else None,
            ),
        )
        return [r.outputs[0].text for r in response]

    def generate_stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams = {},
    ) -> Generator[str, None, None]:
        assert len(messages) > 0

        params = DEFAULT_PARAMS | params
        response = self.model.chat(
            messages=messages, sampling_params=vllm.SamplingParams(**params)
        )
        for output in response:
            yield output.outputs[0].text
