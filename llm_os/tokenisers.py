from transformers import AutoTokenizer

from config import CONFIG

def get_tokeniser_and_context_window(model_name):
    match model_name:
        case "llama3"
            tokenizer = AutoTokenizer.from_pretrained(
                "AdithyaSK/LLama3Tokenizer",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 8192
            num_token_func = lambda text: len(tokenizer.encode(text))
            ct_num_token_func = lambda conv: len(tokenizer.apply_chat_template([{'role': msg['role'] if msg['role'] != 'system' else 'user', 'content': msg['content']} for msg in conv]))
        case "mistral":
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 8192
            num_token_func = lambda text: len(tokenizer.encode(text))
            ct_num_token_func = lambda conv: len(tokenizer.apply_chat_template(conv))
        case "openchat":
            tokenizer = AutoTokenizer.from_pretrained(
                "openchat/openchat_3.5",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 8192
            num_token_func = lambda text: len(tokenizer.encode(text))
            ct_num_token_func = lambda conv: len(tokenizer.apply_chat_template(conv))
        case "phi3":
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                auth_token=CONFIG["huggingface_user_access_token"],
            )
            ctx_window = 4096
            num_token_func = lambda text: len(tokenizer.encode(text))
            ct_num_token_func = lambda conv: len(tokenizer.apply_chat_template(conv))
        case _:
            raise ValueError(f"{modelname} is not a supported model.")

    return tokenizer, ctx_window, num_token_func, ct_num_token_func

NOMIC_EMBED_TEXT_TOKENIZER = Tokenizer.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5",
    auth_token=CONFIG["huggingface_user_access_token"],
)
