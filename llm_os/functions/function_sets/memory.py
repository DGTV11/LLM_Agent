from llm_os.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
from llm_os.agent import Agent

def send_message(self: Agent, message: str) -> Optional[str]:
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.send_assistant_message(message)
    return None
