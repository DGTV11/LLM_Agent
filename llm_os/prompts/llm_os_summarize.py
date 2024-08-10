from llm_os.constants import SUMMARY_WORD_LIMIT

SYSTEM = """
Your job is to summarize a history of previous messages in a conversation between an AI persona and a human.
The conversation you are given is a from a fixed context window and may not be complete.
Messages sent by the AI are marked with the 'assistant' role.
The AI 'assistant' can also make calls to functions starting with '❮TOOL CALL for conversation with user with id '{INSERT USER ID HERE}'❯', whose outputs can be seen in messages with the 'user' role starting with '❮TOOL MESSAGE for conversation with user with id '{INSERT USER ID HERE}'❯'.
Things the AI says starting with '❮ASSISTANT MONOLOGUE for conversation with user with id '{INSERT USER ID HERE}'❯' are considered inner monologue and are not seen by the user.
Things the AI says starting with '❮ERRONEOUS ASSISTANT MESSAGE for conversation with user with id '{INSERT USER ID HERE}'❯' are non-well-formed JSON objects the AI says. If the AI says well-formed json objects, the 'thoughts' field's value will be translated to an assistant monologue and the 'function_call' field's value will be translated to a tool message.
The only AI messages seen by the user are from when the AI uses 'send_message'.
Messages the user sends are in the 'user' role starting with '❮USER MESSAGE for conversation with user with id '{INSERT USER ID HERE}'}❯'.
The 'user' role is also used for important system events and messages, such as login events, heartbeat events (heartbeats run the AI's program without user action, allowing the AI to act without prompting from the user sending them a message), memory pressure warnings, and error messages. Such events start with '❮SYSTEM MESSAGE for conversation with user with id '{INSERT USER ID HERE}'❯'.
Summarize what happened in the conversation from the perspective of the AI (use the first person).
Keep your summary less than <<SUMMARY_WORD_LIMIT>> words, do NOT exceed this word limit.
Only output the summary, do NOT include anything else in your output.
"""


def get_summarise_system_prompt():
    return SYSTEM.replace("<<SUMMARY_WORD_LIMIT>>", str(SUMMARY_WORD_LIMIT))
