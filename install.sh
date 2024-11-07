#!/bin/bash

# Update system
sudo pacman -Syu

# Install required packages
## Pacman
sudo pacman -S python python-pip docker wget tar git
## Shell scripts
curl -fsSL https://ollama.com/install.sh | sh

# Prepare pip
sudo rm /usr/lib/python3.12/EXTERNALLY-MANAGED

# Build Docker images
docker build -f llm_os/dockerfiles/python_runner/Dockerfile -t python_runner .

# Install Python packages
pip install -r requirements.txt
pip install -r client/requirements.txt

# Install required Ollama models
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
