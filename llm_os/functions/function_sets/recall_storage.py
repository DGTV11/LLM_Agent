# MODIFIED FROM MEMGPT REPO (extracted from base.py)

import math
from typing import Optional

from llm_os.agent import Agent
from llm_os.constants import (
    JSON_ENSURE_ASCII,
    # MAX_PAUSE_HEARTBEATS,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)


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
