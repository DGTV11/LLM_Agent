from os import path
import json

from llm_os.tokenisers import get_tokeniser_and_context_window
from llm_os.constants import (
    WORKING_CTX_PERSONA_MAX_TOKENS,
    WORKING_CTX_HUMAN_MAX_TOKENS,
)


class WorkingContext:
    def __init__(self, model_name: str, conv_name: str, persona: str, initial_human_id: int, initial_human_persona: str):
        self.wc_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "persistent_storage",
            conv_name,
            "working_context.json",
        )
        _, _, self.no_token_func, _ = get_tokeniser_and_context_window(model_name)

        if path.exists(self.wc_path):
            with open(self.wc_path, "r") as f:
                wc_cache = json.loads(f.read())
                self.last_2_human_ids = wc_cache["last_2_human_ids"]
                self.persona = wc_cache["persona"]
                self.humans = {int(k): v for k, v in wc_cache["humans"].items()}
            self.__update_working_context_ps()
        else:
            self.last_2_human_ids = []
            self.persona = persona
            self.humans = {}
            self.add_new_human_persona(initial_human_id, initial_human_persona)


    def __repr__(self):
        return '\n'.join(
            [
                f"""<persona>
                {self.persona}
                </persona>"""
            ] +
            [
                f"""<human id="{id}">
                {human}
                </human>""" for id, human in zip(self.last_2_human_ids, map(lambda id: self.humans[id], self.humans))
            ]
        )

    def add_new_human_persona(self, human_id, human):
        no_tokens_new_persona = self.no_token_func(human)
        if no_tokens_new_persona > WORKING_CTX_PERSONA_MAX_TOKENS:
            raise ValueError(
                f"Addition failed: Exceeds {WORKING_CTX_PERSONA_MAX_TOKENS} token limit (requested {no_tokens_new_persona})."
            )
        if human_id in self.humans:
            raise ValueError(
                f"Addition failed: Human persona with ID '{human_id}' already exists!"
            )

        self.humans[int(human_id)] = human
        self.__update_working_context_ps()

    def submit_used_human_id(self, human_id):
        if human_id in self.last_2_human_ids:
            self.last_2_human_ids.remove(human_id)
        self.last_2_human_ids.append(human_id)
        if len(self.last_2_human_ids) > 2:
            self.last_2_human_ids = self.last_2_human_ids[-2:]
        self.__update_working_context_ps()

    def __update_working_context_ps(self):
        if not path.exists(self.wc_path):
            f = open(self.wc_path, "x")
            f.close()
        with open(self.wc_path, "w") as f:
            f.write(json.dumps({"last_2_human_ids": self.last_2_human_ids, "persona": self.persona, "humans": self.humans}))

    def __str__(self):
        return self.__repr__()

    def edit_persona(self, new_persona):
        no_tokens_new_persona = self.no_token_func(new_persona)
        if no_tokens_new_persona > WORKING_CTX_PERSONA_MAX_TOKENS:
            raise ValueError(
                f"Edit failed: Exceeds {WORKING_CTX_PERSONA_MAX_TOKENS} token limit (requested {no_tokens_new_persona}). Consider summarizing existing core memories in 'persona' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            )

        self.persona = new_persona
        self.__update_working_context_ps()
        return self.no_token_func(self.persona)

    def edit_human(self, human_id, new_human):
        no_tokens_new_human = self.no_token_func(new_human)
        if no_tokens_new_human > WORKING_CTX_PERSONA_MAX_TOKENS:
            raise ValueError(
                f"Edit failed: Exceeds {WORKING_CTX_HUMAN_MAX_TOKENS} token limit (requested {no_tokens_new_human}). Consider summarizing existing core memories in '{human_id}' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            )

        self.humans[human_id] = new_human
        self.__update_working_context_ps()
        return self.no_token_func(self.humans[human_id])

    def edit(self, section, content):
        match section:
            case "persona":
                return self.edit_persona(content)
            case _:
                if self.humans.get(section):
                    return self.edit_human(section, content)
                raise KeyError(
                    f"Edit failed: No memory section named {section} (must be either 'persona' or a human's id like '{self.last_2_human_ids[-1]}')"
                )

    def edit_append(self, section, content, sep="\n"):
        match section:
            case "persona":
                new_content = self.persona + sep + content
                return self.edit_persona(new_content)
            case _:
                if self.humans.get(section):
                    new_content = self.humans[section] + sep + content
                    return self.edit_human(section, new_content)
                raise KeyError(
                    f"Edit failed: No memory section named {section} (must be either 'persona' or a human's id like '{self.last_2_human_ids[-1]}'))"
                )

    def edit_replace(self, section, old_content, new_content):
        if not old_content:
            raise ValueError(
                "Edit failed: Old content cannot be an empty string (must specify old_content to replace)"
            )

        match section:
            case "persona":
                if old_content not in self.persona:
                    raise ValueError(
                        f"Edit failed: Old content not found in 'persona' (make sure to use exact string)"
                    )
                new_persona = self.persona.replace(old_content, new_content)
                return self.edit_persona(new_persona)
            case _:
                if self.humans.get(section):
                    if old_content not in self.humans[section]:
                        raise ValueError(
                            f"Edit failed: Old content not found in 'human' (make sure to use exact string)"
                        )
                    new_human = self.humans[section].replace(old_content, new_content)
                    return self.edit_human(section, new_human)
                raise KeyError(
                    f"Edit failed: No memory section named {section} (must be either 'persona' or a human's id like '{self.last_2_human_ids[-1]}')"
                )
