import configparser
from os import path


def get_config():
    config = configparser.ConfigParser()
    config_path = path.join(path.dirname(__file__), "config.ini")

    if not path.exists(config_path):
        return None

    config.read(config_path)

    server_url = config.get("Server", "server_url")

    model_name = config.get("Models", "model_name")

    google_api_key = config.get("Keys_and_IDs", "google_api_key")
    google_prog_search_engine_id = config.get(
        "Keys_and_IDs", "google_prog_search_engine_id"
    )
    huggingface_user_access_token = config.get(
        "Keys_and_IDs", "huggingface_user_access_token"
    )

    return {
        "server_url": server_url,
        "model_name": model_name,
        "google_api_key": google_api_key,
        "google_prog_search_engine_id": google_prog_search_engine_id,
        "huggingface_user_access_token": huggingface_user_access_token,
    }


CONFIG = get_config()

if __name__ == "__main__":
    config = configparser.ConfigParser()

    server_url = (
        input(
            "Please input Ollama server url (Default http://127.0.0.1:11434): "
        ).strip()
        or "http://127.0.0.1:11434"
    )
    if server_url[-1] == "/":
        server_url = server_url[:-1]
    config["Server"] = {"server_url": server_url}

    model_number = int(
        input(
            """
    Choose the model you want to use: 
    1) DeepSeek-V2 16B (Default, Q4_0 quant)
    2) OpenHermes (Mistral 7B finetune, default quant)
    3) Gemma 2 2B (Q5_0 quant)
    > """
        )
        or 1
    )

    match model_number:
        case 1:
            model_name = "deepseek-v2:16b-lite-chat-q4_0"
        case 2:
            model_name = "openhermes"
        case 3:
            model_name = "gemma2:2b-instruct-q5_0"
        case _:
            model_name = "deepseek-v2:16b-lite-chat-q4_0"

    config["Models"] = {
        "model_name": model_name,
    }

    google_api_key = input("Please input Google API Key: ").strip()
    google_prog_search_engine_id = input(
        "Please input Google Programmable Search Engine ID: "
    ).strip()
    huggingface_user_access_token = input(
        "Please input Hugging Face User Access Token: "
    ).strip()

    config["Keys_and_IDs"] = {
        "google_api_key": google_api_key,
        "google_prog_search_engine_id": google_prog_search_engine_id,
        "huggingface_user_access_token": huggingface_user_access_token,
    }

    with open(path.join(path.dirname(__file__), "config.ini"), "w") as configfile:
        config.write(configfile)
