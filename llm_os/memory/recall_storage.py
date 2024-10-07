from os import path
from datetime import datetime
import json


class RecallStorage:
    def __init__(self, conv_name):
        self.rc_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "persistent_storage",
            conv_name,
            "recall_storage.json",
        )
        if path.exists(self.rc_path):
            self.__save_rc_path_dat_to_rs_cache()
        else:
            self.rs_cache = []
            self.__write_rs_cache_to_rc_path()

    def __len__(self):
        return len(self.rs_cache)

    def __write_rs_cache_to_rc_path(self):
        if not path.exists(self.rc_path):
            f = open(self.rc_path, "x")
            f.close()
        with open(self.rc_path, "w") as f:
            f.write(json.dumps(self.rs_cache))

    def __save_messaged(self, messaged):
        self.rs_cache.append(messaged)
        self.__write_rs_cache_to_rc_path()

    def __save_rc_path_dat_to_rs_cache(self):
        with open(self.rc_path, "r") as f:
            self.rs_cache = json.loads(f.read())

    @property
    def conv_messageds(self):
        return [
            messaged
            for messaged in self.rs_cache
            if messaged["type"] not in ["system", "tool"]
        ]

    def insert(self, messaged):
        # note: messaged must be in the form {'type': type, 'user_id': user_id, 'message': {'role': role, 'content': content}}
        recall_messaged = {
            "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d"),
            "user_id": messaged["user_id"],
            "type": messaged["type"],
            "message": messaged["message"],
        }
        self.__save_messaged(recall_messaged)

    def text_search(self, query_string, for_user_id, count=None, start=None):
        results = [
            messaged
            for messaged in self.conv_messageds
            if messaged["message"]["content"] is not None
            and query_string.lower() in messaged["message"]["content"].lower()
            and messaged["user_id"] == for_user_id
        ]

        start = int(start if start else 0)
        count = int(count if count else len(results))
        end = min(count + start, len(results))

        return results[start:end], len(results)

    def date_search(self, start_date, end_date, for_user_id, count=None, start=None):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        results = [
            messaged
            for messaged in self.conv_messageds
            if start_dt
            <= datetime.strptime(messaged["timestamp"], "%Y-%m-%d")
            <= end_dt
            and messaged["user_id"] == for_user_id
        ]

        start = int(start if start else 0)
        count = int(count if count else len(results))
        end = min(count + start, len(results))

        return results[start:end], len(results)
