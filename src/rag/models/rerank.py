from threading import Lock

import numpy as np
from FlagEmbedding import FlagReranker
from vllm import LLM

from rag.settings import settings
from rag.utils.helpers import SingletonMeta


class RerankerModel(metaclass=SingletonMeta):
    _model = None
    _lock: Lock = Lock()

    def __init__(self, model_name: str = settings.RERANKER_MODEL, **kwargs):
        self._model = FlagReranker(model_name, **kwargs)

    def compute_score(self, pairs: list[tuple[str, str]], **kwargs) -> np.array:
        assert self._model, "LLM model not initialized"
        with self._lock:
            return self._model.compute_score(pairs, normalize=True, **kwargs)


class RerankerVLLMModel(metaclass=SingletonMeta):
    _model = None
    _lock: Lock = Lock()

    def __init__(self, model_name: str = settings.RERANKER_MODEL, **kwargs):
        self._model = LLM(model_name, task="score", **kwargs)

    def compute_score(self, pairs: list[tuple[str, str]], **kwargs) -> np.array:
        assert self._model, "LLM model not initialized"
        assert all(
            isinstance(pair, tuple) and len(pair) == 2 for pair in pairs
        ), f"Pairs should be list of tuples of size 2: {pairs}"

        with self._lock:
            scores = [
                self._model.score(text1, text2)[0].outputs.score
                for text1, text2 in pairs
            ]
            # TODO: make sure they are normalized
            return scores
