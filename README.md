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

2. Install the dependencies

```sh
uv sync

# uv will use the cuban national pypi repos 
# if you don't want this open the `pyproject.toml`
# and remove the following:
# [[tool.uv.index]]
# url = "http://nexus.prod.uci.cu/repository/pypi-all/simple"
# default = true
```

3. Execute `uv run main`, this will start the Gradio UI

## (Optional) Setup PyPi cuban repos

If your using uv you don't need to do this:

```ini
# edit
# linux: ~/.config/pip/pip.conf
# windows: ~\AppData\Roaming\pip\pip.ini

[global]
timeout = 120
index = http://nexus.prod.uci.cu/repository/pypi-all/pypi
index-url = http://nexus.prod.uci.cu/repository/pypi-all/simple
[install]
trusted-host = nexus.prod.uci.cu
```


## Run Tests

```sh
pytest .
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Make your changes and commit them (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details
