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


def search_ooc_function_schemas(
    self: Agent, query: str, page: Optional[int] = 0
) -> Optional[str]:
    """
    Search out-of-context function schemas (based on their descriptions) using semantic (embedding-based) search. Function schemas contain function names, descriptions, return types and descriptions, and parameters with types and descriptions. To be used when you want to find out how to do certain actions or access persistent memory units using functions not included in your context window by default.

    Args:
        query (str): String to search for.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.memory.search_function_description_embeddings(
        query,
        count=count,
        start=page * count,
    )
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = (
            f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        )
        results_str = (
            f"{results_pref} {json.dumps(results, ensure_ascii=JSON_ENSURE_ASCII)}"
        )
    return results_str
