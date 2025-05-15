
- [x] eval retrieval
- [ ] eval generation (individual)
- [ ] eval generation (end-to-end)

- [ ] impl citations
- [ ] input sanitization (fix: stop nonsensical queries)

- [ ] impl iterative retrieval (retrieve -> rerank -> retrieve) on identified knowledge gaps
    (fix for multi-hop QA)
    - [x] query decomposition (explain: hypothesis / search plan similar to deep research)
    this step maybe involves structured outputs and parallel / batched retrieval
- [x] impl benchmarks (latency, memory)
- [ ] if good results -> impl graphrag and retrieval router based on query type (global vs local search)


## NLP Metrics
BLEU and ROUGE: Measure n-gram overlap between generated and reference answers but ignore factual consistency14
BERTScore: Computes semantic similarity using BERT embeddings but remains vulnerable to adversarial attacks1.

## Generation Metrics (RAGAS)
- [ ] Faithfulness
  Ratio of answer statements supported by retrieved context.
  For example, a score of 0.91 indicates 91% of claims align with context

- [ ] Answer Relevance
  Semantic similarity between the question and generated answer, measured via cosine similarity of embeddings

- [ ] Context Relevance
  Proportion of context sentences directly relevant to the query

- [ ] Factual Correctness
  Manual or LLM-based verification of factual accuracy against ground truth


low faithfulness score flags retrieval issues, while low factual correctness indicates generator limitations


(retrieval metrics)
- Recall@k
  Measures the percentage of ground-truth relevant documents retrieved in the top k results.
  Research shows recall@10 strongly correlates with downstream QA accuracy, as missing critical documents directly limits the LLM’s ability to answer correctly.
- Precision@k
  Evaluates how many of the top k retrieved documents are truly relevant.
  Precision tradeoffs become critical when balancing retrieval speed and answer quality.
- Context Relevance
  Assesses whether retrieved documents contain sufficient information to answer the query, not just topical relevance.
  This can be measured by prompting an LLM to judge if the context supports answering the question.



ref:
(Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing)[https://arxiv.org/abs/2502.12962]
(LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs)[https://arxiv.org/abs/2502.06139v1]
(DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens)[https://arxiv.org/abs/2502.11493]
