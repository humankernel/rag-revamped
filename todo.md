

- [ ] initial chat message similar to notebooklm
- [ ] impl citations
- [ ] make demos to improve each components
    - [ ] wikiQA
    - [ ] chunking
    - [ ] retriever
- [ ] fix: initialize the llm only once

- [ ] preprocessing
    - [ ] fix RecursiveTextSplitter
    - [ ] preprocess pdf to extract (tables, formulas, images)

- [x] impl generation
- [ ] input sanitization (fix: stop nonsensical queries)
- [ ] rag as a tool use

- [ ] impl iterative retrieval
    - [x] impl gaps discovery
    - [x] impl hybrid search
    - [x] impl reranker
    - [ ] impl contextual compression + filters

- [ ] polish wikiQA ui
- [ ] eval retrieval
- [ ] eval generation (individual)
- [ ] eval generation (end-to-end)

- [ ] impl iterative retrieval (retrieve -> rerank -> retrieve) on identified knowledge gaps
    (fix for multi-hop QA)
    - [ ] query decomposition (explain: hypothesis / search plan similar to deep research)
    this step maybe involves structured outputs and parallel / batched retrieval
- [ ] recursive retrieval (hypothesis-driven query expansion)
- [ ] dynamic hybrid weighting
    e.g. for factoid questions, sparse methods uses 60-70% retrieval score
    open-ended queries shift weighting toward dense by x1.8
    - use lightweight meta-models that classify query intent
- [ ] impl benchmarks (latency, memory)
- [ ] if good results -> impl graphrag and retrieval router based on query type (global vs local search)

(datasets)
- Natural Questions (NQ): A large-scale QA dataset where answers are derived from Wikipedia passages
- TriviaQA (TQA): Focused on trivia questions requiring diverse factual knowledge
- WebQuestions (WQ) and CuratedTrec (CT): Smaller datasets used for transfer learning in RAG models
These datasets measure exact match (EM) and F1 scores for extractive QA but lack mechanisms to evaluate generative quality or retrieval faithfulness4
- RagChecker Benchmark: A cross-domain dataset spanning 10 domains (e.g., healthcare, law) with annotated claims for fine-grained evaluation of retriever and generator errors3

(nlp metrics)
BLEU and ROUGE: Measure n-gram overlap between generated and reference answers but ignore factual consistency14
BERTScore: Computes semantic similarity using BERT embeddings but remains vulnerable to adversarial attacks1.

(rag metrics)
- Faithfulness: ratio of answer statements supported by retrieved context.
    For example, a score of 0.91 indicates 91% of claims align with context
- Answer Relevance: Semantic similarity between the question and generated answer,
    measured via cosine similarity of embeddings
- Context Relevance: Proportion of context sentences directly relevant to the query
- Factual Correctness: Manual or LLM-based verification of factual accuracy against ground truth

low faithfulness score flags retrieval issues, while low factual correctness indicates generator limitations13

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
