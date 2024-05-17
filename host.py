from os import path

import ollama

from config import CONFIG

HOST_URL = CONFIG["server_url"]
HOST = ollama.Client(HOST_URL)
