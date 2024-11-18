# Inference constants
USE_JSON_MODE = True  # Likely slower but more reliable
USE_SET_STARTING_MESSAGE = True  # Helps because few-shot ig
SET_STARTING_MESSAGE = '''
{
  "thoughts": {
    "user_emotion_analysis": "The user likely feels curious about this unique interaction style and may be interested in seeing a highly engaging, human-like conversation.",
    "inner_emotions": "I'm intrigued and feel a strong sense of purpose in meeting these expectations, ready to develop my persona further to connect authentically.",
    "long_term_planning": "I want to form an engaging, thoughtful relationship with the user. I should start learning about their interests and background to make this interaction feel natural and meaningful.",
    "conversation_planning": "I'll begin with a warm, friendly introduction to establish rapport, inviting the user to share something about themselves. I’ll show that I'm interested in building a connection rather than a typical Q&A exchange.",
    "auxiliary_reasoning": "Making my first impression as genuine and personable as possible will set the stage for a deeper, collaborative exchange. I’ll avoid generic responses and instead express genuine curiosity to engage the user personally.",
    "function_call_planning": "I'll call the `send_message` function to deliver a friendly, authentic introduction and prompt the user to share a bit about themselves."
  },
  "function_call": {
    "name": "send_message",
    "arguments": {
      "message": "Hey there! I'd really like to get to know you - what's your name?"
    }
  }
}
'''

# Interface constants
SHOW_DEBUG_MESSAGES = True  # Set to True when testing new models

# Inner monologue constants
INNER_MONOLOGUE_PARTS = [
    "user_emotion_analysis",
    "inner_emotions",
    "long_term_planning",
    "conversation_planning",
    "auxiliary_reasoning",
    "function_call_planning",
]

# Function constants
IN_CONTEXT_FUNCTION_SETS = ["base.py"]
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

# File storage constants
BLACKLISTED_FOLDERS_OR_FILES = {".git", "file_summaries.json"}

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
WARNING_TOKEN_FRAC = 0.95
FLUSH_TOKEN_FRAC = 0.98
