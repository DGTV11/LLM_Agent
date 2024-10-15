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
        query,
        self.memory.working_context.last_2_human_ids[-1],
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
        start_date,
        end_date,
        self.memory.working_context.last_2_human_ids[-1],
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
        results_formatted = [
            f"timestamp: '{d['timestamp']}', role: '{d['message']['role']}' message: {d['message']['content']}"
            for d in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def archival_memory_insert(self: Agent, content: str) -> Optional[str]:
    """
    Add to archival memory for your chat with the user you last conversed with. Make sure to phrase the memory contents such that it can be easily queried later.

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
    Search archival memory for your chat with the user you last conversed with using semantic (embedding-based) search.

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
            f"timestamp: '{d['timestamp']}', memory: '{d['content']}'" for d in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def file_memory_make_file(
    self: Agent, file_rel_path_parts: list[str] = 0
) -> Optional[str]:
    """
    Creates a new file in the folder assigned to your chat with the user you last conversed with.

    Args:
        file_rel_path_parts (list[str]): Relative path parts of the new file with the root directory being the assigned folder.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.make_file(
        self.memory.working_context.last_2_human_ids[-1], file_rel_path_parts
    )

def file_memory_make_folder(
    self: Agent, folder_rel_path_parts: list[str]
) -> Optional[str]:
    """
    Creates a new folder in the folder assigned to your chat with the user you last conversed with.

    Args:
        folder_rel_path_parts (list[str]): Relative path parts of the new folder with the root directory being the assigned folder.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.make_folder(
        self.memory.working_context.last_2_human_ids[-1], folder_rel_path_parts
    )

def file_memory_remove_file(
    self: Agent, file_rel_path_parts: list[str]
) -> Optional[str]:
    """
    Removes a file in the folder assigned to your chat with the user you last conversed with.

    Args:
        file_rel_path_parts (list[str]): Relative path parts of the file with the root directory being the assigned folder.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.remove_file(
        self.memory.working_context.last_2_human_ids[-1], file_rel_path_parts
    )

def file_memory_remove_folder(
    self: Agent, folder_rel_path_parts: list[str]
) -> Optional[str]:
    """
    Removes a folder in the folder assigned to your chat with the user you last conversed with.

    Args:
        folder_rel_path_parts (list[str]): Relative path parts of the folder with the root directory being the assigned folder.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.remove_folder(
        self.memory.working_context.last_2_human_ids[-1], folder_rel_path_parts
    )

def file_memory_append_to_file(
    self: Agent, file_rel_path_parts: list[str], text: str
) -> Optional[str]:
    """
    Appends text to a file in the folder assigned to your chat with the user you last conversed with.

    Args:
        file_rel_path_parts (list[str]): Relative path parts of the file with the root directory being the assigned folder.
        text (str): The string to be appended to the end of the file.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.append_to_file(
        self.memory.working_context.last_2_human_ids[-1], file_rel_path_parts, text
    )

def file_memory_replace_first_in_file(
    self: Agent, file_rel_path_parts: list[str], old_text: str, new_text: str
) -> Optional[str]:
    """
    Replace first occurence of text in a file in the folder assigned to your chat with the user you last conversed with.

    Args:
        file_rel_path_parts (list[str]): Relative path parts of the file with the root directory being the assigned folder.
        old_text (str): String to replace. Must be an exact match.
        new_text (str): Text to write to the file. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.replace_first_in_file(
        self.memory.working_context.last_2_human_ids[-1], file_rel_path_parts, old_text, new_text
    )

def file_memory_replace_all_in_file(
    self: Agent, file_rel_path_parts: list[str], old_text: str, new_text: str
) -> Optional[str]:
    """
    Replace all occurences of text in a file in the folder assigned to your chat with the user you last conversed with.

    Args:
        file_rel_path_parts (list[str]): Relative path parts of the file with the root directory being the assigned folder.
        old_text (str): String to replace. Must be an exact match.
        new_text (str): Text to write to the file. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.replace_all_in_file(
        self.memory.working_context.last_2_human_ids[-1], file_rel_path_parts, old_text, new_text
    )

def file_memory_browse_files(
    self: Agent, page: Optional[int] = 0
) -> Optional[str]:
    """
    Browse through (file path parts) + (file summary) pairs in the folder assigned to your chat with the user you last conversed with.

    Args:
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
    results, total = self.memory.file_storage.browse_files(
        self.memory.working_context.last_2_human_ids[-1], count=count, start=page * count
    )
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = (
            f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        )
        results_formatted = [
            f"file_path_parts: {res[0]}, file_summary: '{res[1]}'" for res in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str

def file_memory_read_file(
    self: Agent, file_rel_path_parts: list[str], page: Optional[int] = 0
) -> Optional[str]:
    """
    Read pages in a file in the folder assigned to your chat with the user you last conversed with.

    Args:
        file_rel_path_parts (list[str]): Relative path parts of the file with the root directory being the assigned folder.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Text in page
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    results, total = self.memory.file_storage.read_file(
        self.memory.working_context.last_2_human_ids[-1], file_rel_path_parts, count=count, start=page * count
    )
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(results) == 0:
        results_str = f"No results found."
    else:
        results_pref = (
            f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        )
        results_str = f"{results_pref} {json.dumps(results, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str

def file_memory_revert_to_last_commit(
    self: Agent
) -> Optional[str]:
    """
    Undos last edit in the folder assigned to your chat with the user you last conversed with.

    Args:
        None

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.revert_to_last_commit(
        self.memory.working_context.last_2_human_ids[-1]
    )

def file_memory_get_diff(
    self: Agent
) -> Optional[str]:
    """
    Gets Git diff between HEAD and HEAD~1 in the folder assigned to your chat with the user you last conversed with.

    Args:
        None

    Returns:
        str: Retrieved diff
    """
    return self.memory.file_memory.get_diff(
        self.memory.working_context.last_2_human_ids[-1]
    )
