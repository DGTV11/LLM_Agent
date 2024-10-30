# MODIFIED FROM MEMGPT REPO (extracted from base.py)

import datetime
import json
import math
from typing import Optional

from llm_os.agent import Agent
from llm_os.constants import (
    JSON_ENSURE_ASCII,
    # MAX_PAUSE_HEARTBEATS,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)


def core_memory_append(self: Agent, section_name: str, content: str) -> Optional[str]:
    """
    Append to the contents of core memory.

    Args:
        section_name (str): Section of the memory to be edited ('persona' to edit your persona or 'human' to edit persona of human who last sent you a message).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.working_context.edit_append(section_name, content)
    return None


def core_memory_replace(
    self: Agent, section_name: str, old_content: str, new_content: str
) -> Optional[str]:
    """
    Replace the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        section_name (str): Section of the memory to be edited ('persona' to edit your persona or 'human' to edit persona of human who last sent you a message).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.memory.working_context.edit_replace(section_name, old_content, new_content)
    return None
