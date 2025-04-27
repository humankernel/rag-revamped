import pytest

from lib.models.llm import OpenAIClient, vLLMClient

# Fixtures ---------------------------------------------------------------------


@pytest.fixture
def openai_client():
    return OpenAIClient()


@pytest.fixture
def vllm_client():
    return vLLMClient()


# Test Cases -------------------------------------------------------------------

# --- Real LLM -----------------------------------------------------------------




# --- Mocked LLM ---------------------------------------------------------------
