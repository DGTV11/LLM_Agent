# MODIFIED FROM MEMGPT REPO

import datetime
import json
import math
from typing import Optional

from llm_os.agent import Agent
from llm_os.constants import (
    JSON_ENSURE_ASCII,
    # MAX_PAUSE_HEARTBEATS,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)


def send_message(self: Agent, message: str) -> Optional[str]:
    """
    Sends a message to the human user. If you need to use other functions to respond to the user's query, use them before using this function.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.assistant_message(message)
    return None


'''
# Construct the docstring dynamically (since it should use the external constants)
pause_heartbeats_docstring = f"""
Temporarily ignore timed heartbeats. You may still receive messages from manual heartbeats and other events.

Args:
    minutes (int): Number of minutes to ignore heartbeats for. Max value of {MAX_PAUSE_HEARTBEATS} minutes ({MAX_PAUSE_HEARTBEATS // 60} hours).

Returns:
    str: Function status response
"""


def pause_heartbeats(self: Agent, minutes: int) -> Optional[str]:
    minutes = min(MAX_PAUSE_HEARTBEATS, minutes)

    # Record the current time
    self.pause_heartbeats_start = datetime.datetime.now(datetime.timezone.utc)
    # And record how long the pause should go for
    self.pause_heartbeats_minutes = int(minutes)

    return f"Pausing timed heartbeats for {minutes} min"

pause_heartbeats.__doc__ = pause_heartbeats_docstring
'''
