from os import path
import configparser


def get_config():
    config = configparser.ConfigParser()
    config_path = path.join(path.dirname(__file__), "config.ini")

    if not path.exists(config_path):
        return None

    config.read(config_path)

    server_url = config.get("Server", "server_url")

    model_name = config.get("Models", "model_name")
    whisper_model_name = config.get("Models", "whisper_model_name")

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
        "whisper_model_name": whisper_model_name,
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

    whisper_model_number = int(
        input(
            """
    Choose the Whisper model you want to use: 
    1) tiny             -> 39M parameters, ~1GB VRAM required, ~32x relative speed
    2) tiny.en          -> 39M parameters, ~1GB VRAM required, ~32x relative speed, English-only
    3) base (default)   -> 74M parameters, ~1GB VRAM required, ~16x relative speed
    4) base.en          -> 74M parameters, ~1GB VRAM required, ~16x relative speed, English-only
    5) small            -> 244M parameters, ~2GB VRAM required, ~6x relative speed
    6) small.en         -> 244M parameters, ~2GB VRAM required, ~6x relative speed, English-only
    7) medium           -> 769M parameters, ~5GB VRAM required, ~2x relative speed
    8) medium.en        -> 769M parameters, ~5GB VRAM required, ~2x relative speed, English-only
    9) large            -> 1550M parameters, ~10GB VRAM required, ~1x relative speed
    > """
        )
        or 3
    )

    match whisper_model_number:
        case 1:
            whisper_model_name = "tiny"
        case 2:
            whisper_model_name = "tiny.en"
        case 3:
            whisper_model_name = "base"
        case 4:
            whisper_model_name = "base.en"
        case 5:
            whisper_model_name = "small"
        case 6:
            whisper_model_name = "small.en"
        case 7:
            whisper_model_name = "medium"
        case 8:
            whisper_model_name = "medium.en"
        case 9:
            whisper_model_name = "large"
        case _:
            whisper_model_name = "base"
    config["Models"] = {
        "model_name": model_name,
        "whisper_model_name": whisper_model_name,
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
