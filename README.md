# LLM Agent
A MemGPT-like conversational agent

## Installation
### Manual installation
1) Install Python dependencies
```sh
pip install -r requirements.txt
pip install -r client/requirements.txt
```

2) Install the required Ollama models
```sh
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

3) Install at least one other supported Ollama model
```sh
ollama pull openhermes
ollama pull deepseek-v2:16b-lite-chat-q4_0
ollama pull gemma2:2b-instruct-q5_0
```

4) Install Docker

5) Build the Docker images from the dockerfiles
```
docker build -f llm_os/dockerfiles/python_runner/Dockerfile -t python_runner .
```

### Automatic installation (Arch Linux only, may not work)
1) Run the script
```
bash install.sh
```

2) Install at least one supported Ollama model (not done automatically so we don't fill up your root partition without your consent)
```sh
ollama pull openhermes
ollama pull deepseek-v2:16b-lite-chat-q4_0
ollama pull gemma2:2b-instruct-q5_0

```

## Usage (CLI)
1) Configure LLM_Agent if you have not already done so
```sh
python3 config.py
```

2) Run LLM_Agent
```sh
python3 server.py
```

```sh
python3 client/cli-client.py
```

## Troubleshooting
If you are on Linux (non-arch) and playsound freezes up and doesn't play anything, run the following commands:
```sh
sudo apt-get install libx264-dev libjpeg-dev
sudo apt-get install libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-bad1.0-dev \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-tools \
     gstreamer1.0-gl \
     gstreamer1.0-gtk3
sudo apt-get install gstreamer1.0-qt5
sudo apt-get install gstreamer1.0-pulseaudio
```

## Improvements
- Use of "function paging" to reduce context usage of function schemas (TODO: list available function sets in system prompt)
- Splitting of "thoughts" field (string) into **"emotions"** array (to encourage more humanlike behaviour) field and "thoughts" **array** (to encourage chain of thought reasoninng)

## Note
- The code in llm_os is based on the MemGPT codebase [here](https://github.com/cpacker/MemGPT) (some parts are from the codebase, some parts are from the internet, some parts were from me)
- The system instructions have been modified from [here](https://github.com/cpacker/MemGPT/tree/c6325feef6d9d2154c0445e317bcc06a7eb27665/memgpt/prompts) with few edits
- The schema generator has been lifted from [here](https://github.com/cpacker/MemGPT/tree/c6325feef6d9d2154c0445e317bcc06a7eb27665/memgpt/functions/schema_generator.py) with few edits
- The base function set has been lifted from [here](https://github.com/cpacker/MemGPT/tree/c6325feef6d9d2154c0445e317bcc06a7eb27665/memgpt/functions/function_sets/base.py) with few edits
- The writing of some code in this repository has been assisted by AI
- AI assistance was used during the compression of the default system instructions

## References
- Packer, Charles, et al. ‘MemGPT: Towards LLMs as Operating Systems’. arXiv [Cs.AI], 2024, http://arxiv.org/abs/2310.08560. arXiv.
