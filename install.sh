#!/bin/bash

# Update system
sudo pacman -Syu

# Install required packages
## Pacman
sudo pacman -S python python-pip docker wget tar
## Shell scripts
curl -fsSL https://ollama.com/install.sh | sh
## Manual
### Piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz -P ~/
tar -xzf ~/piper_amd64.tar.gz -C ~/
chmod +x ~/piper/piper
rm ~/piper_amd64.tar.gz
### Piper voice
mkdir piper-voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx -P piper-voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json -P piper-voice

# Prepare pip
sudo rm /usr/lib/python3.11/EXTERNALLY-MANAGED

# Build docker images
docker build -f llm_os/dockerfiles/python_runner/Dockerfile -t python_runner .

# Install python packages
pip install -r requirements.txt
pip install -r client/requirements.txt
