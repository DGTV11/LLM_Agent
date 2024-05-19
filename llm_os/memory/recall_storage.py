from os import path
from datetime import now
import json

class RecallStorage:
    def __init__(self):
        self.rc_path = path.join(path.dirname(__file__), "persistent_storage", "recall_storage.json"))
        if path.exists(self.rc_path):
            with open(self.rc_path, "r") as f:
                self.rs_cache = json.loads(f.read())
        else:
            with open(self.rc_path, "w") as f:
                self.rs_cache = []
                f.write(json.dump(self.rs_cache))

    def append(self, message):
        #note: messages are in the form {'role': role, 'content': content}
        message["timestamp"] = now().astimezone().strftime("%Y-%m-%d")
        self.rs_cache.append(message)
        with open(self.rc_path, "w") as f:
            f.write(json.dumps(self.rs_cache))
