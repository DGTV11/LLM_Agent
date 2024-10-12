from os import path, mkdir, remove
from datetime import datetime
import json
import pathlib
import hashlib

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from semantic_text_splitter import MarkdownSplitter
from git import Repo

from llm_os.constants import (
    BLACKLISTED_FOLDERS_OR_FILES,
    QWEN_2_5_TOKENIZER,
    NOMIC_EMBED_TOKENIZER,
)
from llm_os.prompts.spr.spr import spr_compress

from host import HOST_URL, HOST


class FileStorage:
    def __init__(self, conv_name, top_k=100):
        self.top_k = top_k

        self.folder_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "persistent_storage",
            conv_name,
            "files",
        )
        if not path.exists(self.folder_path):
            mkdir(self.folder_path)

        self.client = chromadb.PersistentClient(
            path=path.join(
                path.dirname(path.dirname(path.dirname(__file__))),
                "persistent_storage",
                conv_name + "-file_storage_embeddings",
            )
        )
        self.ef = OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url=f"{HOST_URL}/api/embed",
        )
        self.collection = self.client.get_or_create_collection(
            name="file_storage_embeddings", embedding_function=self.ef
        )

    def __len__(self):
        return sum(
            map(
                lambda id: len(self.get_file_rel_paths(id)),
                self.get_all_user_ids_with_folders,
            )
        )

    def get_all_user_ids_with_folders(self):
        return [f for f in os.listdir(self.folder_path) if os.path.isdir(f)]

    def __write_file_summaries(
        self, user_id, hashes, file_rel_path_parts=None, edit_mode=None
    ):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        summaries_path = path.join(repo_path, "file_summaries.json")

        with open(summaries_path, "w") as f:
            f.write(json.dumps(hashes))

        if not edit_mode:
            repo.index.add([summaries_path])
            repo.index.commit(f"Updated file_summaries.json for user {user_id}")
        else:
            repo.index.add([path.join(repo_path, *file_rel_path_parts), summaries_path])
            repo.index.commit(
                f"{edit_mode} {path.join(*file_rel_path_parts)} for user {user_id}"
            )

    def __read_file_summaries(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)

        summaries_path = path.join(repo_path, "file_summaries.json")

        with open(summaries_path, "r") as f:
            return json.loads(f)

    def get_file_summary(self, user_id, file_rel_path_parts, edit_mode=None):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        summaries = self.__read_file_summaries(user_id)

        file_rel_path_parts_tuple = tuple(file_rel_path_parts)
        file_path = path.join(repo_path, *file_rel_path_parts)
        file_hash = self.__compute_file_hash(file_path)

        if not (
            file_rel_path_parts_tuple in summaries
            and summaries[file_rel_path_parts_tuple]["file_hash"] == file_hash
        ):
            summaries[file_rel_path_parts_tuple] = {"file_hash": file_hash}
            splitter = MarkdownSplitter.from_huggingface_tokenizer(
                QWEN_2_5_TOKENIZER, 8192
            )
            summary = ""

            with open(file_path, "r") as f:
                for chunk in splitter.chunks(f.read()):
                    summary += spr_compress("qwen2.5:0.5b", 8192, chunk) + "\n"

            summaries[file_rel_path_parts_tuple]["summary"] = summary

            self.__write_file_summaries(
                user_id, summaries, file_rel_path_parts, edit_mode
            )

        return summaries[file_rel_path_parts_tuple]["summary"]

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
                f.write("{}")

            repo.index.add([summaries_path])
            repo.index.commit("Initial commit")

            return repo
        else:
            return Repo(repo_path)

    def __compute_file_hash(self, file_path, algorithm="sha256"):
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as file:
            # Read the file in chunks of 8192 bytes
            while chunk := file.read(8192):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def get_file_rel_paths(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo_pathlib_dir = pathlib.Path(repo_path)
        return [
            str(item.relative_to(repo_pathlib_dir))
            for item in repo_pathlib_dir.rglob("*")
            if item.is_file() and item.parts.isdisjoint(BLACKLISTED_FOLDERS_OR_FILES)
        ]

    def get_file_paths(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo_pathlib_dir = pathlib.Path(repo_path)
        return [
            str(item)
            for item in repo_pathlib_dir.rglob("*")
            if item.is_file() and item.parts.isdisjoint(BLACKLISTED_FOLDERS_OR_FILES)
        ]

    def get_file_rel_paths_parts(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo_pathlib_dir = pathlib.Path(repo_path)
        return [
            item.relative_to(repo_pathlib_dir).parts
            for item in repo_pathlib_dir.rglob("*")
            if item.is_file() and item.parts.isdisjoint(BLACKLISTED_FOLDERS_OR_FILES)
        ]

    # * File Memory embedding functions
    def initialise_embedding_collection(self):
        client = chromadb.EphemeralClient()
        ef = OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url=f"{HOST_URL}/api/embed",
        )
        collection = client.get_or_create_collection(
            name="file_memory_embeddings", embedding_function=ef
        )
        return collection

    def populate_embedding_collection(self, user_id, collection):
        splitter = MarkdownSplitter.from_huggingface_tokenizer(
            NOMIC_EMBED_TEXT_TOKENIZER, 8192
        )

        for file_path, file_rel_path_parts in zip(
            self.get_file_paths(user_id), self.get_file_rel_paths_parts(user_id)
        ):
            with open(file_path, "r") as f:
                chunk_list = splitter.chunks(f.read())

            hex_stringify = lambda chunk: hashlib.md5(chunk.encode("UTF-8")).hexdigest()
            ids = [hex_stringify(chunk) for chunk in chunk_list]
            metadatas = [
                {"file_rel_path_parts": file_rel_path_parts}
                for _ in range(len(chunk_list))
            ]
            collection.add(documents=chunk_list, metadatas=metadatas, ids=ids)

        return collection

    # * File Memory repository search functions
    def browse_files(self, user_id, count, start):
        results = self.get_file_rel_paths_parts(user_id)

        start = int(start if start else 0)
        count = int(count if count else len(results))
        end = min(count + start, len(results))

        return list(
            zip(
                results[start:end],
                map(
                    lambda item: self.get_file_summary(user_id, item),
                    results[start:end],
                ),
            )
        ), len(results)

    def embedding_search_files(self, user_id, query, count, start):
        collection = self.populate_embedding_collection(
            user_id, self.initialise_embedding_collection()
        )
        # TODO

    def string_search_files(self, user_id, string, count, start):
        pass

    # * File Memory single file search functions
    def read_file(self, user_id, file_rel_path_parts, count, start):
        results = self.get_file_rel_paths_parts(user_id)

        start = int(start if start else 0)
        count = int(count if count else len(results))
        end = min(count + start, len(results))

        return list(
            zip(
                results[start:end],
                map(
                    lambda item: self.get_file_summary(user_id, item),
                    results[start:end],
                ),
            )
        ), len(results)

    def embedding_search_file(self, user_id, query, count, start):
        collection = self.populate_embedding_collection(
            user_id, self.initialise_embedding_collection()
        )

    def string_search_file(self, user_id, string, count, start):
        pass

    # * File Memory edit functions
    def make_file(self, user_id, file_rel_path_parts):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        new_file_rel_path_parts_tuple = tuple(file_rel_path_parts)
        new_file_path = path.join(repo_path, *file_rel_path_parts)
        f = open(new_file_path, "x")
        f.close()

        with open(new_file_path, "w") as f:
            f.write("Nothing here (yet)")

        file_hash = self.__compute_file_hash(new_file_path)
        summaries[new_file_rel_path_parts_tuple] = {"file_hash": file_hash}

        summaries[new_file_rel_path_parts_tuple]["summary"] = "Empty file"

        self.__write_file_summaries(
            user_id, summaries, new_file_rel_path_parts, "Created"
        )

    def make_folder(self, user_id, folder_rel_path_parts):
        pass

    def remove_file(self, user_id, file_rel_path_parts):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        file_rel_path_parts_tuple = tuple(file_rel_path_parts)
        file_path = path.join(repo_path, *file_rel_path_parts)

        remove(file_path)

        del summaries[file_rel_path_parts_tuple]

        self.__write_file_summaries(user_id, summaries, file_rel_path_parts, "Removed")

    def append_to_file(self, user_id, file_rel_path_parts, text):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        file_path = path.join(repo_path, *file_rel_path_parts)

        with open(file_path, "a") as f:
            f.write(text)

        self.get_file_summary(user_id, file_rel_path_parts, "Modified")

    def replace_first_in_file(self, user_id, file_rel_path_parts, old_text, new_text):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        file_path = path.join(repo_path, *file_rel_path_parts)

        with open(file_path, "r+") as f:
            file_contents = f.read().replace(old_text, new_text, 1)
            f.write(file_contents)

        self.get_file_summary(user_id, file_rel_path_parts, "Modified")

    def replace_all_in_file(self):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        file_path = path.join(repo_path, *file_rel_path_parts)

        with open(file_path, "r+") as f:
            file_contents = f.read().replace(old_text, new_text)
            f.write(file_contents)

        self.get_file_summary(user_id, file_rel_path_parts, "Modified")

    # * File Memory version control functions
    def revert_to_last_commit(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        heads = repo.heads
        master = heads.master
        log = master.log()
        repo.git.revert(log[-1].hexsha, no_edit=True)

    def get_diff(self, user_id):
        repo_path = self.__get_repo_path_from_user_id(user_id)
        repo = self.__load_repo(repo_path)

        t = repo.head.commit.tree
        return repo.git.diff(t)
