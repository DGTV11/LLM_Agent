# MODIFIED FROM MEMGPT REPO (extracted from base.py)

import math
from typing import Optional

from llm_os.agent import Agent
from llm_os.constants import (
    JSON_ENSURE_ASCII,
    # MAX_PAUSE_HEARTBEATS,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)


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
            f"timestamp: '{d['timestamp']}', memory: '{d['content']}'" for d in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str
