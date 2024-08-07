# LLM Agent
A MemGPT-based conversational agent

## Installation
1) Install Python dependencies
```sh
pip install -r requirements.txt
```

2) Install ffmpeg

3) Install the required Ollama models
```sh
ollama pull llama3
ollama pull mistral 
ollama pull openhermes
ollama pull deepseek-v2:16b-lite-chat-q4_0
ollama pull nomic-embed-text
```

## Usage (CLI)
1) Configure LLM_Agent if you have not already done so
```sh
python3 config.py
```

2) Run LLM_Agent
```sh
python3 main.py
```

## Troubleshooting
If you are on Linux and playsound freezes up and doesn't play anything, run the following commands:
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

## Note
- The code in llm_os is based on the MemGPT codebase [here](https://github.com/cpacker/MemGPT) (some parts are from the codebase, some parts are from the internet)
- The system instructions have been lifted from [here](https://github.com/cpacker/MemGPT/tree/c6325feef6d9d2154c0445e317bcc06a7eb27665/memgpt/prompts) with few edits
- The schema generator has been lifted from [here](https://github.com/cpacker/MemGPT/tree/c6325feef6d9d2154c0445e317bcc06a7eb27665/memgpt/functions/schema_generator.py) with few edits
- The base function set has been lifted from [here](https://github.com/cpacker/MemGPT/tree/c6325feef6d9d2154c0445e317bcc06a7eb27665/memgpt/functions/function_sets/base.py) with few edits

## TODO
- Upgrade LLM_Agent's archival memory (add KG)
- Allow it to use function-calling to interact with its greater environment, search the web, and perform other actions
- Allow LLM_Agent to use end-to-end speech-to-speech (we need faster SLMS!)
- Allow LLM_Agent to speak to multiple users (group chat/conversation)

## References
- Packer, Charles, et al. ‘MemGPT: Towards LLMs as Operating Systems’. arXiv [Cs.AI], 2024, http://arxiv.org/abs/2310.08560. arXiv.
- Nussbaum, Zach, et al. ‘Nomic Embed: Training a Reproducible Long Context Text Embedder’. arXiv [Cs.CL], 2024, http://arxiv.org/abs/2402.01613. arXiv.
