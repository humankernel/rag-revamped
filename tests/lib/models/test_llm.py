from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
)
from pydantic import BaseModel

from lib.helpers import count_tokens
from lib.models.llm import DEFAULT_PARAMS, OpenAIClient, vLLMClient
from lib.settings import settings

# Fixtures --------------------------------------------------------------------


@pytest.fixture
def mock_tiktoken():
    with patch("tiktoken.get_encoding") as mock_get_encoding:
        mock_encoder = MagicMock()
        mock_encoder.encode.side_effect = lambda text: list(range(len(text) // 4))
        mock_get_encoding.return_value = mock_encoder
        yield {"get_encoding": mock_get_encoding, "encoder": mock_encoder}


@pytest.fixture
def mock_vllm():
    with patch("lib.models.llm.vllm.LLM") as mock_llm:
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance

        # Configure mock outputs
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Hello World")]
        mock_instance.generate.return_value = [mock_output]

        yield mock_instance


@pytest.fixture
def mock_openai():
    with patch("lib.models.llm.OpenAI") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Configure regular completion
        mock_completion = MagicMock(spec=ChatCompletion)
        mock_completion.choices = [MagicMock(message=MagicMock(content="Generated response"))]

        # Configure streaming completion chunks
        mock_chunk1 = MagicMock(spec=ChatCompletionChunk)
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Stream "), index=0, finish_reason=None)]
        mock_chunk2 = MagicMock(spec=ChatCompletionChunk)
        mock_chunk2.choices = [
            MagicMock(
                delta=MagicMock(content="response"),
                index=0,
                finish_reason="stop",
            )
        ]

        # Create a generator to simulate streaming
        def mock_stream_generator():
            yield mock_chunk1
            yield mock_chunk2

        # Configure the create method to return appropriate response type
        def create_mock(*args, **kwargs):
            if kwargs.get("stream"):
                return mock_stream_generator()
            return mock_completion

        mock_instance.chat.completions.create.side_effect = create_mock

        yield mock_instance


# Test Cases -------------------------------------------------------------------


def test_count_tokens(mock_tiktoken):
    result = count_tokens("12345678")
    assert result == 2
    mock_tiktoken["encoder"].encode.assert_called_once_with("12345678")


def test_vllm_generate(mock_vllm):
    client = vLLMClient()
    result = client.generate("test prompt")

    assert result == "Hello World"
    mock_vllm.generate.assert_called_once()

    # Verify sampling params
    sampling_params = mock_vllm.generate.call_args[1]["sampling_params"]
    assert sampling_params.max_tokens == DEFAULT_PARAMS["max_tokens"]
    assert sampling_params.temperature == DEFAULT_PARAMS["temperature"]


def test_vllm_generate_stream(mock_vllm):
    # Configure streaming response
    mock_output1 = MagicMock()
    mock_output1.outputs = [MagicMock(text="Hello")]
    mock_output2 = MagicMock()
    mock_output2.outputs = [MagicMock(text=" World")]
    mock_vllm.generate.return_value = [mock_output1, mock_output2]

    client = vLLMClient()
    stream = client.generate_stream("test prompt")
    results = list(stream)

    assert results == ["Hello", " World"]
    mock_vllm.generate.assert_called_once()


def test_openai_generate(mock_openai):
    client = OpenAIClient()
    result = client.generate("test prompt")

    assert result == "Generated response"
    mock_openai.chat.completions.create.assert_called_once_with(
        messages=[{"role": "assistant", "content": "test prompt"}],
        model=settings.LLM_MODEL,
        extra_body=None,
        **DEFAULT_PARAMS,
    )


def test_openai_generate_stream(mock_openai):
    client = OpenAIClient()
    stream = client.generate_stream("test prompt")
    results = list(stream)

    assert results == ["Stream ", "response"]
    mock_openai.chat.completions.create.assert_called_once_with(
        messages=[{"role": "assistant", "content": "test prompt"}],
        model=settings.LLM_MODEL,
        stream=True,
        **DEFAULT_PARAMS,
    )


def test_openai_format_guidance(mock_openai):
    class TestFormat(BaseModel):
        name: str

    client = OpenAIClient()
    client.generate("test prompt", output_format=TestFormat)

    call_args = mock_openai.chat.completions.create.call_args[1]
    assert "guided_json" in call_args["extra_body"]
    assert call_args["extra_body"]["guided_json"] == TestFormat.model_json_schema()


def test_token_limit_enforcement(mock_tiktoken, mock_vllm):
    mock_tiktoken["encoder"].encode.side_effect = lambda x: [0] * (settings.CTX_WINDOW + 1)

    client = vLLMClient()
    with pytest.raises(AssertionError, match="Prompt too large!!"):
        client.generate("any prompt")


def test_parameter_merging(mock_vllm):
    client = vLLMClient()
    custom_params = {"temperature": 0.9, "max_tokens": 500}
    client.generate("test", params=custom_params)

    expected_params = {**DEFAULT_PARAMS, **custom_params}
    sampling_params = mock_vllm.generate.call_args[1]["sampling_params"]

    assert sampling_params.temperature == expected_params["temperature"]
    assert sampling_params.max_tokens == expected_params["max_tokens"]
