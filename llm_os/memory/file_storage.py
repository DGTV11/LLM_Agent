from os import path, mkdir
from datetime import datetime
import json
import pathlib
import hashlib

from semantic_text_splitter import MarkdownSplitter
from git import Repo

from llm_os.constants import BLACKLISTED_FOLDERS_OR_FILES, QWEN_2_5_TOKENIZER
from llm_os.prompts.spr.spr import spr_compress

from host import HOST

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

    def __len__(self): #TODO
        #return len(self.get_file_paths())
        pass

    def __write_file_summaries(self, user_id, hashes):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        summaries_path = path.join(repo_path, "file_summaries.json")

        with open(summaries_path, "w") as f:
            f.write(json.dumps(hashes))

        repo.index.add([summaries_path])
        repo.index.commit(f"Updated file_summaries.json for user {user_id}")

    def __read_file_summaries(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)

        summaries_path = path.join(repo_path, "file_summaries.json")

        with open(summaries_path, "r") as f:
            return json.loads(f)

    def __get_repo_path_from_user_id(self, user_id):
        return path.join(self.folder_path, user_id)
    
    def __load_repo(self, repo_path):
        if not path.exists(repo_path):
            mkdir(repo_path)
            repo = Repo.init(repo_path)

            summaries_path = path.join(repo_path, "file_summaries.json")
            f = open(summaries_path, "x")
            f.close()
            with open(summaries_path, "w") as f:
                f.write('{}')

            repo.index.add([summaries_path])
            repo.index.commit("Initial commit")
            
            return repo
        else:
            return Repo(repo_path)

    def __compute_file_hash(self, file_path, algorithm='sha256'):
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as file:
            # Read the file in chunks of 8192 bytes
            while chunk := file.read(8192):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()

    def get_file_paths(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo_pathlib_dir = pathlib.Path(repo_path)
        return [str(item.relative_to(repo_pathlib_dir)) for item in repo_pathlib_dir.rglob("*") if item.is_file() and item.parts.isdisjoint(BLACKLISTED_FOLDERS_OR_FILES)]

    def make_file(self, user_id, file_rel_path_parts):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        new_file_path = path.join(repo_path, *file_rel_path_parts)
        f = open(new_file_path, "x")
        f.close()

        #TODO: create placeholder summary here

    def get_file_summary(self, user_id, file_rel_path_parts):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        summaries = self.__read_file_summaries(user_id)

        file_rel_path_parts_tuple = tuple(file_rel_path_parts)
        file_path = path.join(repo_path, *file_rel_path_parts)
        file_hash = self.__compute_file_hash(file_path)

        if not (rel_path_parts_tuple in summaries and summaries[rel_path_parts_tuple]['file_hash'] == file_hash):
            summaries[rel_path_parts_tuple] = {'file_hash': file_hash}
            splitter = MarkdownSplitter.from_huggingface_tokenizer(QWEN_2_5_TOKENIZER, 8192)
            summary = ""

            with open(file_path, "r") as f:
                for chunk in splitter.chunks(f.read()):
                    summary += spr_compress('qwen2.5:0.5b', 8192, chunk) + "\n"

            summaries[rel_path_parts_tuple]['summary'] = summary

            self.__write_file_summaries(user_id, summaries)

        return summaries[rel_path_parts_tuple]['summary']
