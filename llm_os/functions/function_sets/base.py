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

### Functions / tools the agent can use
# All functions should return a response string (or None)
# If the function fails, throw an exception


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


def core_memory_append(self: Agent, section_name: str, content: str) -> Optional[str]:
    """
    Append to the contents of core memory.

    Args:
        section_name (str): Section of the memory to be edited ('persona' to edit your persona or 'human' to edit persona of human who last sent you a message).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.working_context.edit_append(section_name, content)
    return None


def core_memory_replace(
    self: Agent, section_name: str, old_content: str, new_content: str
) -> Optional[str]:
    """
    Replace the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        section_name (str): Section of the memory to be edited ('persona' to edit your persona or 'human' to edit persona of human who last sent you a message).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.working_context.edit_replace(section_name, old_content, new_content)
    return None


def conversation_search(
    self: Agent, query: str, page: Optional[int] = 0
) -> Optional[str]:
    """
    Search prior conversation history with the user you last conversed with using case-insensitive string matching.

    Args:
        query (str): String to search for.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

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
    results, total = self.memory.recall_storage.text_search(
        query, self.memory.working_context.last_2_human_ids[-1], count=count, start=page * count
    )
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = (
            f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        )
        results_formatted = [
            f"timestamp: '{d['timestamp']}', role: '{d['message']['role']}' - {d['message']['content']}"
            for d in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def conversation_search_date(
    self: Agent, start_date: str, end_date: str, page: Optional[int] = 0
) -> Optional[str]:
    """
    Search prior conversation history with the user you last conversed with using a date range.

    Args:
        start_date (str): The start of the date range to search, in the format 'YYYY-MM-DD'.
        end_date (str): The end of the date range to search, in the format 'YYYY-MM-DD'.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

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
    results, total = self.memory.recall_storage.date_search(
        start_date, end_date, self.memory.working_context.last_2_human_ids[-1], count=count, start=page * count
    )
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = (
            f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        )
        results_formatted = [
            f"timestamp: '{d['timestamp']}', role: '{d['message']['role']}' message: {d['message']['content']}"
            for d in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def archival_memory_insert(self: Agent, content: str) -> Optional[str]:
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

    Args:
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.archival_storage.insert(
        self.memory.working_context.last_2_human_ids[-1], content
    )
    return None


def archival_memory_search(
    self: Agent, query: str, page: Optional[int] = 0
) -> Optional[str]:
    """
    Search archival memory using semantic (embedding-based) search.

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
    results, total = self.memory.archival_storage.search(
        query, count=count, start=page * count
    )
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = (
            f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        )
        results_formatted = [
            f"timestamp: '{d['timestamp']}', memory: '{d['content']}'" for d in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str
