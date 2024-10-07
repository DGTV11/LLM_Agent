from os import path, mkdir
from datetime import datetime
import json
import pathlib

from git import Repo

from llm_os.constants import BLACKLISTED_FOLDERS_OR_FILES


class FileStorage:
    def __init__(self, conv_name):
        self.folder_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "persistent_storage",
            conv_name,
            "files",
        )
        if not path.exists(self.folder_path):
            mkdir(self.folder_path)

    def __len__(self):
        return len(self.rs_cache)

    def __write_file_hashes(self):

        with open(self.rc_path, "w") as f:
            f.write(json.dumps(self.rs_cache))

    def __read_file_hashes(self, messaged):
        self.rs_cache.append(messaged)
        self.__write_rs_cache_to_rc_path()

    def __save_rc_path_dat_to_rs_cache(self):
        with open(self.rc_path, "r") as f:
            self.rs_cache = json.loads(f.read())

    def __get_repo_path_from_user_id(self, user_id):
        return path.join(self.folder_path, user_id)
    
    def __load_repo(self, repo_path):
        if not path.exists(repo_path):
            mkdir(repo_path)
            repo = Repo.init(repo_path)

            hashes_path = path.join(repo_path, "file_summaries.json")
            f = open(hashes_path, "x")
            f.close()
            with open(hashes_path, "w") as f:
                f.write('{}')

            repo.index.add([hashes_path])
            repo.index.commit("Initial commit")
            
            return repo
        else:
            return Repo(repo_path)

    def get_file_paths(self, user_id):
        repo_path = __get_repo_path_from_user_id(user_id)
        repo_pathlib_dir = pathlib.Path(repo_path)
        return [str(item.relative_to(repo_pathlib_dir)) for item in repo_pathlib_dir.rglob("*") if item.is_file() and item.parts.isdisjoint(BLACKLISTED_FOLDERS_OR_FILES)]

    def make_file(self, user_id, file_rel_path_parts):
        repo_path = __get_repo_path_from_user_id(user_id)
        repo = __load_repo(repo_path)

        new_file_path = path.join(repo_path, *file_rel_path_parts)
        f = open(new_file_path, "x")
        f.close()

    def get_file_summary(self, user_id, file_rel_path_parts):
        repo_path = __get_repo_path_from_user_id(user_id)
        repo = __load_repo(repo_path)

        hashes_path = path.join(repo_path, "file_summaries.json")

        with open(hashes_path, "r+") as f:
            pass
