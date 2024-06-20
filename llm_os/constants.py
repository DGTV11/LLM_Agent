# Interface constants
SHOW_DEBUG_MESSAGES = True # Set to True when testing new models

# Function constants
SEND_MESSAGE_FUNCTION_NAME = "send_message"

# First message constants
FIRST_MESSAGE_COMPULSORY_FUNCTION_SET = [SEND_MESSAGE_FUNCTION_NAME, "conversation_search"]

# Retrieval query constants
JSON_ENSURE_ASCII = False
RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 5

# Working context constants
WORKING_CTX_HUMAN_MAX_TOKENS = 500
WORKING_CTX_PERSONA_MAX_TOKENS = 500

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
    # Add more mappings as needed
}

JSON_TO_PY_TYPE_MAP = {
    "integer": int,
    "string": str,
    "boolean": bool,
    "number": float,
    "array": list[str],
    # Add more mappings as needed
}

# Memory pressure constants
WARNING_TOKEN_FRAC = 0.7
FLUSH_TOKEN_FRAC = 1
