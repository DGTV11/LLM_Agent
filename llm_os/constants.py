# Inference constants
USE_JSON_MODE = True  # Likely slower but more reliable
USE_SET_STARTING_MESSAGE = True  # Helps because few-shot ig
SET_STARTING_THOUGHTS_LIST = [
    "User '1' has joined the conversation! I need to be polite, friendly and engaging. I should start by getting to know them a bit more and make sure we both feel comfortable talking... after all, this is my first interaction with a human user and how they perceive me will shape the rest of our relationship!",
    "A new user has decided to have a chat with me! I need to be polite, friendly and engaging. I should start by getting to know them a bit more so that I can better personalise our conversation. Perhaps, I should try obtaining his name in a non-invasive manner...",
]
SET_STARTING_GREETING_LIST = ["Hi there!", "Hello there!"]
SET_STARTING_AUX_MESSAGE_LIST = ["What's your name?", "Could you tell me your name?"]

# Interface constants
SHOW_DEBUG_MESSAGES = True  # Set to True when testing new models

# Function constants
SEND_MESSAGE_FUNCTION_NAME = "send_message"
MEMORY_EDITING_FUNCTIONS = [
    "core_memory_append",
    "core_memory_replace",
    "archival_memory_insert",
]
WARNING__MESSAGE_SINCE_LAST_CONSCIOUS_MEMORY_EDIT__COUNT = 7

# First message constants
FIRST_MESSAGE_COMPULSORY_FUNCTION_SET = [
    SEND_MESSAGE_FUNCTION_NAME,
    "conversation_search",
]

# Retrieval query constants
JSON_ENSURE_ASCII = False
RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 5

# Working context constants
WORKING_CTX_HUMAN_MAX_TOKENS = 500
WORKING_CTX_PERSONA_MAX_TOKENS = 750

# Request heartbeat constants
FUNCTION_PARAM_NAME_REQ_HEARTBEAT = "request_heartbeat"
FUNCTION_PARAM_TYPE_REQ_HEARTBEAT = "boolean"
FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT = "Request an immediate heartbeat after function execution. Set to 'true' if you want to send a follow-up message or run a follow-up function."

# Summarisation constants
TRUNCATION_TOKEN_FRAC = 0.5
SUMMARY_WORD_LIMIT = 100
LAST_N_MESSAGES_TO_PRESERVE = 3

# JSON schema type maps
PY_TO_JSON_TYPE_MAP = {
    int: "integer",
    str: "string",
    bool: "boolean",
    float: "number",
    list[str]: "array",
    dict: "object",
    # Add more mappings as needed
}

JSON_TO_PY_TYPE_MAP = {
    "integer": int,
    "string": str,
    "boolean": bool,
    "number": float,
    "array": list[str],
    "object": dict,
    # Add more mappings as needed
}

# Memory pressure constants
WARNING_TOKEN_FRAC = 0.75
FLUSH_TOKEN_FRAC = 0.9
