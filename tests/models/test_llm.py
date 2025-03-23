from unittest.mock import MagicMock, patch

import pytest

from rag.models.llm import LLMModel


@pytest.fixture
def mock_llm():
    with patch("rag.models.llm.LLM") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x, **kwargs: list(
            range(len(x) // 4)
        )
        mock_instance.get_tokenizer.return_value = mock_tokenizer

        # Mock the output of generate fn
        mock_output = MagicMock()
        mock_output.outputs = [
            MagicMock(text="Hello"),
            MagicMock(text=" World"),
        ]
        mock_instance.generate.return_value = [mock_output]

        yield mock_instance


def test_singleton(mock_llm):
    """Ensure that LLMModel enforces the singleton pattern."""
    model1 = LLMModel()
    model2 = LLMModel()
    assert model1 is model2


def test_count_tokens(mock_llm):
    """Test that count_tokens correctly counts tokens and caches results."""
    model = LLMModel()
    mock_tokenizer = mock_llm.get_tokenizer.return_value

    # "hello world" has length 11, so 11 // 4 = 2 tokens
    assert model.count_tokens("hello world") == 2
    assert model.count_tokens("hello world") == 2, (
        "Cached token count incorrect"
    )
    assert mock_tokenizer.encode.call_count == 1, (
        "Tokenizer encode called more than once despite cache"
    )
    assert mock_tokenizer.encode.call_count == 1, (
        "Tokenizer encode call count incorrect after new input"
    )
    # "another text" has length 12, so 12 // 4 = 3 tokens
    assert model.count_tokens("another text") == 3, (
        "Token count incorrect for 'another text'"
    )
    assert mock_tokenizer.encode.call_count == 2, (
        "Tokenizer encode call count incorrect after new input"
    )


@patch("rag.models.llm.SamplingParams")
def test_generate_non_streaming(mock_sampling_params, mock_llm):
    # Mock SamplingParams instance
    mock_sampling_params_instance = MagicMock()
    mock_sampling_params.return_value = mock_sampling_params_instance

    model = LLMModel()
    result = model.generate("prompt")

    assert result == "Hello World"
    mock_llm.generate.assert_called_once_with(
        "prompt", sampling_params=mock_sampling_params_instance
    )
    mock_sampling_params.assert_called_once_with(
        max_tokens=300,
        temperature=0.6,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=1.2,
        repetition_penalty=1.2,
    )


@patch("rag.models.llm.SamplingParams")
def test_generate_streaming(mock_llm):
    model = LLMModel()
    stream = model.generate_stream("prompt")
    results = list(stream)

    assert results == ["Hello", " World"]
    mock_llm.generate.assert_called_once_with("prompt")


def test_generate_prompt_too_long(mock_llm):
    """Test that generate raises AssertionError when prompt tokens exceed CTX_WINDOW."""
    # Patch CTX_WINDOW to a small value
    with patch("rag.models.llm.settings.CTX_WINDOW", 10):
        # Set up mock tokenizer to return 11 tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(11))
        mock_llm.get_tokenizer.return_value = mock_tokenizer

        model = LLMModel()
        with pytest.raises(AssertionError, match="messages exceed ctx window"):
            model.generate("prompt")
