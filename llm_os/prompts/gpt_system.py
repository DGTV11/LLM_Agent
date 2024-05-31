import os

def get_system_text(key):
    filename = f"{key}.txt"
    file_path = os.path.join(os.path.dirname(__file__), "llm_os", "prompts", "system", filename)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    else:
        raise FileNotFoundError(f"No file found for key {key}, path={file_path}")
