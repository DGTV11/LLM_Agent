# MODIFIED FROM MEMGPT REPO (extracted from base.py)

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
        self.memory.working_context.last_2_human_ids[-1],
        file_rel_path_parts,
        old_text,
        new_text,
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
        self.memory.working_context.last_2_human_ids[-1],
        file_rel_path_parts,
        old_text,
        new_text,
    )


def file_memory_browse_files(self: Agent, page: Optional[int] = 0) -> Optional[str]:
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
        self.memory.working_context.last_2_human_ids[-1],
        file_rel_path_parts,
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


def file_memory_revert_n_commits(self: Agent, n: Optional[int] = 1) -> Optional[str]:
    """
    Undos n edits (commits) in the folder assigned to your chat with the user you last conversed with by creating n new edits that reverse the last n edits.

    Args:
        n (Optional[int]): How many commits to revert. Defaults to 1.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.revert_n_commits(
        self.memory.working_context.last_2_human_ids[-1], n
    )

def file_memory_reset_n_commits(self: Agent, n: Optional[int] = 1) -> Optional[str]:
    """
    Undos n edits (commits) in the folder assigned to your chat with the user you last conversed with by deleting the last n edits.

    Args:
        n (Optional[int]): How many commits to reset. Defaults to 1.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.file_memory.reset_n_commits(
        self.memory.working_context.last_2_human_ids[-1], n
    )


def file_memory_get_diff(self: Agent, n: Optional[int] = 1) -> Optional[str]:
    """
    Gets Git diff between HEAD and HEAD~n in the folder assigned to your chat with the user you last conversed with.

    Args:
        n (Optional[int]): How many commits previously from HEAD the target commit is. Defaults to 1.

    Returns:
        str: Retrieved diff
    """
    return self.memory.file_memory.get_diff(
        self.memory.working_context.last_2_human_ids[-1], n
    )

def file_memory_view_commit_history(self: Agent, page: Optional[int] = 0) -> Optional[str]:
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
            f"file_path_parts: {res[0]}, file_summary: '{res[1]}'" for res in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str

