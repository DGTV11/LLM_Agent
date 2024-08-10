import json
import math
from typing import Optional

from llm_os.agent import Agent
from config import CONFIG
from llm_os.constants import (
    JSON_ENSURE_ASCII,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)


def google_search(self: Agent, query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Retrieves possible website urls from search queries from Google's Custom Search JSON API (don't query this too many times - just query this as little as required to get factually correct information). You need to use the 'load_webpage_from_url' function to load specific webpages from the given search results after using this function.

    Args:
        query (str): Search query.
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

    results, total = self.web_interface.get_urls_from_query(
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
            f"url: '{res[0]}', title: '{res[1]}', snippet: '{res[2]}'"
            for res in results
        ]
        results_str = f"{results_pref} {json.dumps(results_formatted, ensure_ascii=JSON_ENSURE_ASCII)}"
    return results_str


def load_webpage_from_url(self: Agent, url: str) -> Optional[str]:
    """
    Retrieves the content of a webpage at a specified url. You should first retrieve possible urls with the 'google_search' function.

    Args:
        url (str): Url to retrieve the webpage content of.

    Returns:
        str: Query result string
    """

    return self.web_interface.load_webpage_from_url(url)
