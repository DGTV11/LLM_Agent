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

    return {
        "server_url": server_url,
        "model_name": model_name,
        "whisper_model_name": whisper_model_name,
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
    config["Server"] = {"server_url": server_url}

    model_number = int(
        input(
            """
    Choose the model you want to use: 
    1) Llama-3 (default)
    2) OpenChat
    3) Mistral 7B 
    4) Phi-3
    > """
        )
        or 1
    )

    match model_number:
        case 1:
            model_name = "assistant_llama3"
        case 2:
            model_name = "assistant_openchat"
        case 3:
            model_name = "assistant_mistral"
        case 4:
            model_name = "assistant_phi_3"
        case _:
            model_name = "assistant_llama3"

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

    with open(path.join(path.dirname(__file__), "config.ini"), "w") as configfile:
        config.write(configfile)
else:
    if not CONFIG:
        raise ValueError("Config not found, please run config.py")
