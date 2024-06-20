from os import path
from host import HOST

with open(path.join(path.dirname(__file__), "system.md")) as f:
    COMPRESSION_SYSTEM_PROMPT = f.read().strip()

with open(path.join(path.dirname(__file__), "unpack.md")) as f:
    DECOMPRESSION_SYSTEM_PROMPT = f.read().strip()


def spr_compress(model_name, ctx_window, text):
    result = HOST.chat(
        model=model_name,
        messages=[
            {
                'role': 'system',
                'content': COMPRESSION_SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': text
            }
        ],
        options={"num_ctx": ctx_window}
    )
    return result['message']['content']

def spr_decompress(model_name, ctx_window, text):
    result = HOST.chat(
        model=model_name,
        messages=[
            {
                'role': 'system',
                'content': DECOMPRESSION_SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': text
            }
        ],
        options={"num_ctx": ctx_window}
    )
    return result['message']['content']
