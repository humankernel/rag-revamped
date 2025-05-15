from typing import Generator, NotRequired, Optional, Type, TypedDict

from pytest import param
import vllm
from openai import OpenAI
from pydantic import BaseModel
from vllm.sampling_params import GuidedDecodingParams

from lib.helpers import count_tokens
from settings import settings


class GenerationParams(TypedDict):
    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]


DEFAULT_PARAMS: GenerationParams = {
    "max_tokens": 100,
    "temperature": 0.25,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 1.2,
}


class OpenAIClient:
    def __init__(self) -> None:
        self.model = OpenAI(
            api_key="API",
            base_url=settings.CLIENT_URL,
        )

    def generate(
        self,
        prompt: str,
        output_format: Type[BaseModel] | None,
        params: GenerationParams | None,
    ) -> str | BaseModel:
        assert count_tokens(prompt) < settings.CTX_WINDOW, "Prompt too large!!"

        params = params if params else DEFAULT_PARAMS
        response = self.model.chat.completions.create(
            messages=[{"role": "assistant", "content": prompt}],
            model=settings.LLM_MODEL,
            extra_body={"guided_json": output_format.model_json_schema()}
            if output_format
            else None,
            **params,
        )
        response = response.choices[0].message.content or ""

        if output_format:
            return output_format.model_validate_json(response)

        return response

    def generate_stream(
        self,
        prompt: str,
        params: GenerationParams | None,
    ) -> Generator[str, None, None]:
        assert count_tokens(prompt) < settings.CTX_WINDOW, "Prompt too large!!"

        params = params if params else DEFAULT_PARAMS
        response = self.model.chat.completions.create(
            messages=[{"role": "assistant", "content": prompt}],
            model=settings.LLM_MODEL,
            stream=True,
            **params,
        )
        for output in response:
            yield output.choices[0].delta.content or ""


# TODO: explore bitsandbytes (https://docs.vllm.ai/en/stable/features/quantization/bnb.html)
class vLLMClient:
    def __init__(self) -> None:
        self.model = vllm.LLM(
            model=settings.LLM_MODEL,
            dtype=settings.DTYPE,
            task="generate",
            enforce_eager=False,
            max_model_len=settings.CTX_WINDOW,
            max_num_seqs=2,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.5,
        )

    def generate(
        self,
        prompt: str,
        output_format: Type[BaseModel] | None = None,
        params: GenerationParams = DEFAULT_PARAMS,
    ) -> str | BaseModel:
        assert count_tokens(prompt) < settings.CTX_WINDOW, "Prompt too large!!"

        params = {**DEFAULT_PARAMS, **params}
        response = self.model.generate(
            prompt,
            sampling_params=vllm.SamplingParams(
                **params,
                guided_decoding=GuidedDecodingParams(
                    json=output_format.model_json_schema()
                )
                if output_format
                else None,
            ),
        )
        response = response[0].outputs[0].text

        if output_format:
            return output_format.model_validate_json(response)

        return response

    def generate_stream(
        self,
        prompt: str,
        params: GenerationParams = DEFAULT_PARAMS,
    ) -> Generator[str, None, None]:
        assert count_tokens(prompt) < settings.CTX_WINDOW, "Prompt too large!!"

        params = {**DEFAULT_PARAMS, **params}
        response = self.model.generate(
            prompt, sampling_params=vllm.SamplingParams(**params)
        )
        for output in response:
            yield output.outputs[0].text
