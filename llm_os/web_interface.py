import requests
from bs4 import BeautifulSoup

from config import CONFIG
from llm_os.constants import (
    JSON_ENSURE_ASCII,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)


class WebInterface:
    def __init__(self):
        self.google_search_cache = {}

    def get_urls_from_query(self, query, count, start):
        if not self.google_search_cache.get(query, None):
            search_res = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": CONFIG["google_api_key"],
                    "cx": CONFIG["google_prog_search_engine_id"],
                    "q": query,
                },
            ).json()

            self.google_search_cache[query] = [
                (item["link"], item["title"], item["snippet"])
                for item in search_res["items"]
            ]

        results = self.google_search_cache[query]

        start = int(start if start else 0)
        count = int(count if count else len(results))
        end = min(count + start, len(results))

        return results[start:end], len(results)

    def load_webpage_from_url(self, url):
        r = requests.get(url)

        soup = BeautifulSoup(r.content, "html.parser")

        return str(r) + "\n" + soup.prettify()
