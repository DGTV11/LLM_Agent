import json
from asyncio import Semaphore
from datetime import timedelta
from os import listdir, mkdir, path, rmdir
from time import time
from uuid import uuid4

import uvicorn
from config import CONFIG
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_os.agent import Agent
from llm_os.functions.load_functions import (
    get_function_dats_from_function_sets,
    load_all_function_sets,
)
from llm_os.interface import CLIInterface, ServerInterface
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.file_storage import FileStorage
from llm_os.memory.memory import Memory
from llm_os.memory.recall_storage import RecallStorage
from llm_os.memory.working_context import WorkingContext
from llm_os.prompts.gpt_system import get_system_text
from llm_os.tokenisers import get_tokeniser_and_context_window
from llm_os.web_interface import WebInterface

app = FastAPI()

sem = Semaphore(1)

loaded_agents = {}


class InitAgentRequestParams(BaseModel):
    agent_persona_name: str
    human_persona_name: str


class ConvNameGenericRequestParams(BaseModel):
    conv_name: str


class CreateHumanRequestParams(BaseModel):
    conv_name: str
    human_persona_name: str


class SendMessageParams(BaseModel):
    conv_name: str
    user_id: int
    message: str


def get_agent(conv_name):
    global loaded_agents

    if conv_name in loaded_agents:
        return loaded_agents[conv_name]

    # Load system instructions
    system_instructions = get_system_text("llm_agent_chat")

    # Load functions
    (
        in_context_function_dats,
        out_of_context_function_dats,
        out_of_context_function_sets,
    ) = get_function_dats_from_function_sets(load_all_function_sets())

    # Load interfaces
    interface = ServerInterface()
    web_interface = WebInterface()

    # Load agent
    working_context = WorkingContext(CONFIG["model_name"], conv_name, None, None, None)
    recall_storage = RecallStorage(conv_name)
    archival_storage = ArchivalStorage(conv_name)
    file_storage = FileStorage(
        conv_name, get_tokeniser_and_context_window(CONFIG["model_name"])[0]
    )

    agent = Agent(
        interface,
        web_interface,
        conv_name,
        CONFIG["model_name"],
        in_context_function_dats,
        out_of_context_function_dats,
        out_of_context_function_sets,
        system_instructions,
        working_context,
        archival_storage,
        recall_storage,
        file_storage,
    )

    loaded_agents[conv_name] = agent

    return agent


def init_agent(agent_persona_name, human_persona_name):
    global loaded_agents

    agent_persona_fp = path.join(
        path.dirname(__file__),
        "llm_os",
        "personas",
        "agents",
        agent_persona_name,
    )
    human_persona_fp = path.join(
        path.dirname(__file__),
        "llm_os",
        "personas",
        "humans",
        human_persona_name,
    )

    with open(agent_persona_fp, "r") as f:
        agent_persona_str = f.read()

    with open(human_persona_fp, "r") as f:
        human_persona_str = f.read()

    # Load system instructions
    system_instructions = get_system_text("llm_agent_chat")

    # Load functions
    (
        in_context_function_dats,
        out_of_context_function_dats,
        out_of_context_function_sets,
    ) = get_function_dats_from_function_sets(load_all_function_sets())

    # Load interfaces
    interface = ServerInterface()
    web_interface = WebInterface()

    # Create persistent storage directory
    ps_folders = list(
        filter(
            lambda s: s[0] != ".",
            listdir(path.join(path.dirname(__file__), "persistent_storage")),
        )
    )

    conv_name = f"{agent_persona_name.split('.')[0]}--{human_persona_name.split('.')[0]}@{uuid4().hex}-{uuid4().hex}"

    while conv_name in ps_folders:
        conv_name = f"{agent_persona_name.split('.')[0]}--{human_persona_name.split('.')[0]}@{uuid4().hex}-{uuid4().hex}"

    mkdir(path.join(path.dirname(__file__), "persistent_storage", conv_name))

    # Create agent
    working_context = WorkingContext(
        CONFIG["model_name"], conv_name, agent_persona_str, 1, human_persona_str
    )
    recall_storage = RecallStorage(conv_name)
    archival_storage = ArchivalStorage(conv_name)
    file_storage = FileStorage(
        conv_name, get_tokeniser_and_context_window(CONFIG["model_name"])[0]
    )

    agent = Agent(
        interface,
        web_interface,
        conv_name,
        CONFIG["model_name"],
        in_context_function_dats,
        out_of_context_function_dats,
        out_of_context_function_sets,
        system_instructions,
        working_context,
        archival_storage,
        recall_storage,
        file_storage,
    )

    loaded_agents[conv_name] = agent

    return conv_name


@app.get("/conversation-ids")
async def get_existing_conversation_ids():
    return {
        "conv_ids": list(
            filter(
                lambda s: s[0] != ".",
                listdir(path.join(path.dirname(__file__), "persistent_storage")),
            )
        )
    }


@app.get("/personas/agents")
async def get_agent_personas():
    return {
        "persona_names": listdir(
            path.join(path.dirname(__file__), "llm_os", "personas", "agents")
        )
    }


@app.get("/personas/humans")
async def get_human_personas():
    return {
        "persona_names": listdir(
            path.join(path.dirname(__file__), "llm_os", "personas", "humans")
        )
    }


@app.post("/agent")
async def init_agent(data: InitAgentRequestParams):
    agent_persona_name = data.agent_persona_name
    human_persona_name = data.human_persona_name

    conv_name = init_agent(agent_persona_name, human_persona_name)

    return {"conv_name": conv_name}


@app.delete("/agent")
async def delete_agent(data: ConvNameGenericRequestParams):
    conv_name = data.conv_name

    # Remove conversation
    try:
        rmdir(path.join(path.dirname(__file__), "persistent_storage", conv_name))
        return {"success": True}
    except OSError as error:
        return {"success": False}


@app.get("/agent/humans")
async def get_all_human_ids(data: ConvNameGenericRequestParams):
    conv_name = data.conv_name

    # Load working context
    working_context = WorkingContext(CONFIG["model_name"], conv_name, None, None, None)

    # Get all registered human ids
    human_ids = list(working_context.humans.keys())

    return {"human_ids": human_ids}


@app.post("/agent/humans")
async def create_human(data: CreateHumanRequestParams):
    conv_name = data.conv_name
    human_persona_name = data.human_persona_name

    # Load working context
    working_context = WorkingContext(CONFIG["model_name"], conv_name, None, None, None)

    # Get all registered human ids
    human_ids = list(working_context.humans.keys())

    # Load human persona
    human_persona_fp = path.join(
        path.dirname(__file__),
        "llm_os",
        "personas",
        "humans",
        human_persona_name,
    )

    with open(human_persona_fp, "r") as f:
        human_persona_str = f.read()

    # Add new user
    new_human_id = max(human_ids) + 1
    working_context.add_new_human_persona(new_human_id, human_persona_str)

    return {"new_human_id": new_human_id}


@app.post("/messages/send")
async def send_message(data: SendMessageParams):
    async with sem:
        # Load data
        conv_name = data.conv_name
        user_id = data.user_id
        message = data.message

        # Load agent
        agent = get_agent(conv_name)

        # Send message
        agent.interface.user_message(message)
        agent.memory.append_messaged_to_fq_and_rs(
            {
                "type": "user",
                "user_id": user_id,
                "message": {"role": "user", "content": message},
            }
        )

        # Generate agent responses
        def generate_agent_responses(agent_obj):
            total_start_time = time()

            heartbeat_request = True
            while heartbeat_request:
                start_time = time()
                _, heartbeat_request, _ = agent_obj.step(user_id)
                end_time = time()

                server_message_stack = agent_obj.interface.server_message_stack.copy()
                agent_obj.interface.server_message_stack = []

                yield json.dumps(
                    {
                        "server_message_stack": server_message_stack,
                        "ctx_info": {
                            "current_ctx_token_count": agent_obj.memory.main_ctx_message_seq_no_tokens,
                            "ctx_window": agent_obj.memory.ctx_window,
                        },
                        "duration": str(
                            timedelta(seconds=round(end_time - start_time, 2))
                        ),
                    }
                ) + "\n"

            total_end_time = time()

            yield json.dumps(
                {
                    "total_duration": str(
                        timedelta(seconds=round(total_end_time - total_start_time, 2))
                    )
                }
            ) + "\n"

        return StreamingResponse(generate_agent_responses(agent))


@app.post("/messages/send/first-message")
async def send_first_message(data: SendMessageParams):
    async with sem:
        # Load data
        conv_name = data.conv_name
        user_id = data.user_id
        message = data.message

        # Load agent
        agent = get_agent(conv_name)

        # Send message
        agent.interface.system_message(message)
        agent.memory.append_messaged_to_fq_and_rs(
            {
                "type": "system",
                "user_id": "user_id",
                "message": {"role": "user", "content": message},
            }
        )

        # Generate agent responses
        def generate_agent_responses(agent_obj):
            total_start_time = time()

            heartbeat_request = True
            while heartbeat_request:
                start_time = time()
                _, heartbeat_request, _ = agent_obj.step(user_id, is_first_message=True)
                end_time = time()

                server_message_stack = agent_obj.interface.server_message_stack.copy()
                agent_obj.interface.server_message_stack = []

                yield json.dumps(
                    {
                        "server_message_stack": server_message_stack,
                        "ctx_info": {
                            "current_ctx_token_count": agent_obj.memory.main_ctx_message_seq_no_tokens,
                            "ctx_window": agent_obj.memory.ctx_window,
                        },
                        "duration": str(
                            timedelta(seconds=round(end_time - start_time, 2))
                        ),
                    }
                ) + "\n"

            total_end_time = time()

            yield json.dumps(
                {
                    "total_duration": str(
                        timedelta(seconds=round(total_end_time - total_start_time, 2))
                    )
                }
            ) + "\n"

        return StreamingResponse(generate_agent_responses(agent))


@app.post("/messages/send/no-heartbeat")
async def send_message_without_heartbeat(data: SendMessageParams):
    async with sem:
        # Load data
        conv_name = data.conv_name
        user_id = data.user_id
        message = data.message

        # Load agent
        agent = get_agent(conv_name)

        # Send message
        agent.interface.system_message(message)
        agent.memory.append_messaged_to_fq_and_rs(
            {
                "type": "system",
                "user_id": user_id,
                "message": {"role": "user", "content": message},
            }
        )

        agent.memory.working_context.submit_used_human_id(user_id)

        server_message_stack = agent.interface.server_message_stack.copy()
        agent.interface.server_message_stack = []

        return {
            "server_message_stack": server_message_stack,
            "ctx_info": {
                "current_ctx_token_count": agent.memory.main_ctx_message_seq_no_tokens,
                "ctx_window": agent.memory.ctx_window,
            },
        }


if __name__ == "__main__":
    if not CONFIG:
        raise ValueError("Config not found, please run config.py")
    # app.run(host="0.0.0.0", port=4999)
    uvicorn.run(app, port=4999, host="0.0.0.0")
