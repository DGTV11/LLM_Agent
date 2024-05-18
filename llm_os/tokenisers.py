from tokenizers import Tokenizer

from config import CONFIG

def get_tokeniser_and_context_window(mode_lname):
    match model_name:
        case "llama3"
            tokenizer = Tokenizer.from_pretrained(
                "AdithyaSK/LLama3Tokenizer",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 8192
        case "mistral":
            tokenizer = Tokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 8192
        case "openchat":
            tokenizer = Tokenizer.from_pretrained(
                "openchat/openchat_3.5",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 8192
        case "phi3":
            tokenizer = Tokenizer.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 4096
        case _:
            raise ValueError(f"{modelname} is not a supported model.")

    return tokenizer, ctx_window

NOMIC_EMBED_TEXT_TOKENIZER = Tokenizer.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5",
    auth_token=CONFIG["huggingface_user_access_token"],
)
