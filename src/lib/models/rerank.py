import torch
import vllm

from lib.settings import settings


class RerankerModel:
    def __init__(self):
        self.model = vllm.LLM(
            model=settings.RERANKER_MODEL,
            task="score",
            enforce_eager=True,
        )

    def compute_score(
        self, text1: str | list[str], texts2: str | list[str]
    ) -> torch.Tensor:
        outputs = self.model.score(text1, texts2)
        scores = [o.outputs.score for o in outputs]

        assert all(0 <= score <= 1 for score in scores)
        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        return scores_tensor
        norm = torch.softmax(scores_tensor, dim=0)
        return norm


# model = RerankerModel()
# results = model.compute_score(
#     "What is the capital of France?",
#     [
#         "The capital of Brazil is Brasilia.",
#         "The capital of France is Paris.",
#     ],
# )
# print(results)
