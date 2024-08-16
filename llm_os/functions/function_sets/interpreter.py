import math
from typing import Optional

import docker
from agentrun import AgentRun

from llm_os.agent import Agent
from config import CONFIG
from llm_os.constants import (
    JSON_ENSURE_ASCII,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)


def execute_python_code(self: Agent, code: str) -> Optional[str]: # Adapted from: https://github.com/Jonathan-Adly/AgentRun
    """
    Sends a Python code snippet to the code execution environment and returns the output. The code execution environment can automatically import any library or package by importing. Your Python environment only allows for 512MB of RAM and 1GB of swap. Your Python code snippet has 3 minutes to run.

    Args:
        code (str): The code snippet to execute. Must be a valid python code. Must use print() to output the result.

    Returns:
        str: Python code execution result string
    """

    client = docker.from_env()
    for container in client.containers.list():
        if "python_runner" in container.attrs['Config']['Image']:
            runner_container = container
            break
    else:
        runner_container = client.containers.run("python_runner")

    runner = AgentRun(
        container_name = runner_container.id,
        dependencies_whitelist = [],
        cached_dependencies = [],
        default_timeout = 3 * 60,
        memory_limit = "512mb",
        memswap_limit = "1gb"
    )
    result = runner.execute_code_in_container(code)

    runner_container.stop()
    
    return result 
