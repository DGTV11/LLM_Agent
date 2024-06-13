from os import path
import json

from llm_os.tokenisers import get_tokeniser_and_context_window
from llm_os.constants import (
    WORKING_CTX_PERSONA_MAX_TOKENS,
    WORKING_CTX_HUMAN_MAX_TOKENS,
)


class WorkingContext:
    def __init__(self, model_name: str, conv_name: str, persona: str, human: str):
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
                self.persona = wc_cache["persona"]
                self.human = wc_cache["human"]
        else:
            self.persona = persona
            self.human = human

        self.__update_working_context_ps()

    def __repr__(self):
        return f"""<persona>
        {self.persona}
        </persona>
        <human>
        {self.human}
        </human>"""

    def __update_working_context_ps(self):
        if not path.exists(self.wc_path):
            f = open(self.wc_path, "x")
            f.close()
        with open(self.wc_path, "w") as f:
            f.write(json.dumps({"persona": self.persona, "human": self.human}))

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

    def edit_human(self, new_human):
        no_tokens_new_human = self.no_token_func(new_human)
        if no_tokens_new_human > WORKING_CTX_PERSONA_MAX_TOKENS:
            raise ValueError(
                f"Edit failed: Exceeds {WORKING_CTX_HUMAN_MAX_TOKENS} token limit (requested {no_tokens_new_human}). Consider summarizing existing core memories in 'human' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            )

        self.human = new_human
        self.__update_working_context_ps()
        return self.no_token_func(self.human)

    def edit(self, section, content):
        match section:
            case "persona":
                return self.edit_persona(content)
            case "human":
                return self.edit_human(content)
            case _:
                raise KeyError(
                    f"Edit failed: No memory section named {section} (must be either 'persona' or 'human')"
                )

    def edit_append(self, section, content, sep="\n"):
        match section:
            case "persona":
                new_content = self.persona + sep + content
                return self.edit_persona(new_content)
            case "human":
                new_content = self.human + sep + content
                return self.edit_human(new_content)
            case _:
                raise KeyError(
                    f"Edit failed: No memory section named {section} (must be either 'persona' or 'human')"
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
            case "human":
                if old_content not in self.human:
                    raise ValueError(
                        f"Edit failed: Old content not found in 'human' (make sure to use exact string)"
                    )
                new_human = self.human.replace(old_content, new_content)
                return self.edit_human(new_human)
            case _:
                raise KeyError(
                    f"Edit failed: No memory section named {section} (must be either 'persona' or 'human')"
                )
