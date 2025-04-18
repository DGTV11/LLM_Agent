from os import path, listdir, mkdir, rmdir
import threading
from time import time
from datetime import timedelta
from uuid import uuid4
import json

from flask import Flask, Response, request, jsonify, stream_with_context

from config import CONFIG

from llm_os.agent import Agent
from llm_os.interface import CLIInterface, ServerInterface
from llm_os.web_interface import WebInterface
from llm_os.tokenisers import get_tokeniser_and_context_window

from llm_os.memory.memory import Memory
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.recall_storage import RecallStorage
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.file_storage import FileStorage

from llm_os.prompts.gpt_system import get_system_text

from llm_os.functions.load_functions import (
    load_all_function_sets,
    get_function_dats_from_function_sets,
)

app = Flask(__name__)

sem = threading.Semaphore()

loaded_agents = {}


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
    in_context_function_dats, out_of_context_function_dats = (
        get_function_dats_from_function_sets(load_all_function_sets())
    )

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
        system_instructions,
        working_context,
        archival_storage,
        recall_storage,
        file_storage,
    )

    loaded_agents[conv_name] = agent

    return conv_name


@app.route("/conversation-ids", methods=["GET"])
def get_existing_conversation_ids():
    print('FINISHED RUNNING GET "/conversation-ids"')
    return jsonify(
        {
            "conv_ids": list(
                filter(
                    lambda s: s[0] != ".",
                    listdir(path.join(path.dirname(__file__), "persistent_storage")),
                )
            )
        }
    )


@app.route("/personas/agents", methods=["GET"])
def get_agent_personas():
    print('FINISHED RUNNING GET "/personas/agents"')
    return jsonify(
        {
            "persona_names": listdir(
                path.join(path.dirname(__file__), "llm_os", "personas", "agents")
            )
        }
    )


@app.route("/personas/humans", methods=["GET"])
def get_human_personas():
    print('FINISHED RUNNING GET "/personas/humans"')
    return jsonify(
        {
            "persona_names": listdir(
                path.join(path.dirname(__file__), "llm_os", "personas", "humans")
            )
        }
    )


@app.route("/agent", methods=["POST", "DELETE"])
def agent_methods():
    match request.method:
        case "POST":
            # Load personas
            data = request.get_json()
            agent_persona_name = data.get("agent_persona_name")
            human_persona_name = data.get("human_persona_name")

            conv_name = init_agent(agent_persona_name, human_persona_name)

            print('FINISHED RUNNING POST "/agent"')
            return jsonify({"conv_name": conv_name})
        case "DELETE":
            # Load conversation name
            data = request.get_json()
            conv_name = data.get("conv_name")

            # Remove conversation
            try:
                rmdir(
                    path.join(path.dirname(__file__), "persistent_storage", conv_name)
                )
                print('FINISHED RUNNING DELETE "/agent"')
                return jsonify({"success": True})
            except OSError as error:
                print('FINISHED RUNNING DELETE "/agent"')
                return jsonify({"success": False})


@app.route("/agent/humans", methods=["GET", "POST"])
def agent_human_methods():
    # Load data
    data = request.get_json()
    conv_name = data.get("conv_name")

    # Load working context
    working_context = WorkingContext(CONFIG["model_name"], conv_name, None, None, None)

    # Get all registered human ids
    human_ids = list(working_context.humans.keys())

    match request.method:
        case "GET":
            print('FINISHED RUNNING GET "/agent/humans"')
            return jsonify({"human_ids": human_ids})
        case "POST":
            # Load human persona
            human_persona_name = data.get("human_persona_name")

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

            print('FINISHED RUNNING POST "/agent/humans"')
            return jsonify({"new_human_id": new_human_id})


@app.route("/messages/send", methods=["POST"])
def send_message():
    sem.acquire()

    try:
        # Load data
        data = request.get_json()
        conv_name = data.get("conv_name")
        user_id = data.get("user_id")
        message = data.get("message")

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
        @stream_with_context
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

        print('FINISHED RUNNING POST "/messages/send"')
        return Response(generate_agent_responses(agent), mimetype="application/json")
    finally:
        sem.release()


@app.route("/messages/send/first-message", methods=["POST"])
def send_first_message():
    sem.acquire()

    try:
        # Load data
        data = request.get_json()
        conv_name = data.get("conv_name")
        user_id = data.get("user_id")
        message = data.get("message")

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
        @stream_with_context
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

        print('FINISHED RUNNING POST "/messages/send/first-message"')
        return Response(generate_agent_responses(agent), mimetype="application/json")
    finally:
        sem.release()


@app.route("/messages/send/no-heartbeat", methods=["POST"])
def send_message_without_heartbeat():
    sem.acquire()

    try:
        # Load data
        data = request.get_json()
        conv_name = data.get("conv_name")
        user_id = data.get("user_id")
        message = data.get("message")

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

        print('FINISHED RUNNING POST "/messages/send/no-heartbeat"')
        return jsonify(
            {
                "server_message_stack": server_message_stack,
                "ctx_info": {
                    "current_ctx_token_count": agent.memory.main_ctx_message_seq_no_tokens,
                    "ctx_window": agent.memory.ctx_window,
                },
            }
        )
    finally:
        sem.release()


if __name__ == "__main__":
    if not CONFIG:
        raise ValueError("Config not found, please run config.py")
    app.run(host="0.0.0.0", port=4999)
