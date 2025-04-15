from threading import Lock

import numpy as np
from FlagEmbedding import FlagReranker
from vllm import LLM

from settings import settings
from utils.helpers import SingletonMeta


class RerankerModel(metaclass=SingletonMeta):
    _model = None
    _lock: Lock = Lock()

    def __init__(self, model_name: str = settings.RERANKER_MODEL, **kwargs):
        self._model = FlagReranker(model_name, **kwargs)

    def compute_score(
        self, pairs: list[tuple[str, str]], **kwargs
    ) -> np.ndarray:
        assert self._model, "LLM model not initialized"
        with self._lock:
            scores = self._model.compute_score(pairs, normalize=True, **kwargs)
            return scores


class RerankerVLLMModel(metaclass=SingletonMeta):
    _model = None
    _lock: Lock = Lock()

    def __init__(self, model_name: str = settings.RERANKER_MODEL, **kwargs):
        self._model = LLM(model_name, task="score", **kwargs)

    def compute_score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
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
            return np.array(scores)
