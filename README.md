# Advanced RAG Chatbot

This project is a simple prototype developed for a [research paper](https://github.com/humankernel/rag-paper/blob/main/rcci_template.pdf) focused on democratizing AI tools for managing PDF documents in resource-limited contexts. It serves as an advanced Retrieval-Augmented Generation (RAG) pipeline example, showcasing how AI can facilitate document management.

## Features

-

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/humankernel/rag.git
cd rag
```

2. Install the dependencies: `uv sync`

(Optional) Setup PyPi cuban repos
```shell
# uv will use the cuban national pypi repos
# if you don't want this open the `pyproject.toml`
# and remove the following:
# [[tool.uv.index]]
# url = "http://nexus.prod.uci.cu/repository/pypi-all/simple"
# default = true
```

3. Setup `.env` variables
4. Execute `uv run main`, this will start the Gradio UI

(Optional) In a local environment (e.g dev mode) you can start a separated `vLLM` instance.
```shell
# https://docs.vllm.ai/en/stable/serving/engine_args.html
> vllm serve ./Qwen3-4B-AWQ --device cuda --gpu-memory-utilization 0.6 --max-model-len 4096
```

## Tests

```sh
pytest .
```

## License
This project is licensed under the MIT License - see the LICENSE file for details
