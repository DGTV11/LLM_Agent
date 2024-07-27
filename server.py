from os import path, listdir, mkdir
from time import time
from datetime import timedelta
from uuid import uuid4
import json

from flask import Flask, Response, request, jsonify, stream_with_context

from config import CONFIG

from llm_os.agent import Agent
from llm_os.interface import CLIInterface, ServerInterface
from llm_os.web_interface import WebInterface

from llm_os.memory.memory import Memory
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.recall_storage import RecallStorage
from llm_os.memory.archival_storage import ArchivalStorage

from llm_os.prompts.gpt_system import get_system_text

from llm_os.functions.load_functions import (
    load_all_function_sets,
    get_function_dats_from_function_sets,
)

app = Flask(__name__)

@app.route('/conversation-ids', methods=['GET'])
def get_existing_conversation_ids():
    return jsonify({
        'conv_ids': list(
            filter(
                lambda s: s[0] != ".",
                listdir(path.join(path.dirname(__file__), "persistent_storage")),
            )
        )
    })

@app.route('/personas/agents', methods=['GET'])
def get_agent_personas():
    return jsonify({
        'persona_names': listdir(
            path.join(
                path.dirname(__file__), "llm_os", "personas", "agents"
            )
        )
    })

@app.route('/personas/humans', methods=['GET'])
def get_human_personas():
    return jsonify({
        'persona_names': listdir(
            path.join(
                path.dirname(__file__), "llm_os", "personas", "humans"
            )
        )
    })

@app.route('/agent/create-agent', methods=['POST'])
def create_agent_from_personas():
    # Load personas
    data = request.get_json()
    agent_persona_name = data.get('agent_persona_name')
    human_persona_name = human.get('human_persona_name')

    agent_persona_fp = path.join(
        path.dirname(__file__), "llm_os", "personas", "agents", agent_persona_name
    )
    human_persona_fp = path.join(
        path.dirname(__file__), "llm_os", "personas", "humans", human_persona_name
    )

    with open(chosen_agent_persona_fp, "r") as f:
        agent_persona_str = f.read()

    with open(chosen_human_persona_fp, "r") as f:
        human_persona_str = f.read()

    # Load system instructions
    system_instructions = get_system_text("llm_agent_chat")

    # Load functions
    function_dats = get_function_dats_from_function_sets(load_all_function_sets())

    # Load interfaces
    interface = CLIInterface()
    web_interface = WebInterface()

    # Create persistent storage directory
    conv_name = f"{chosen_agent_persona.split('.')[0]}--{chosen_human_persona.split('.')[0]}@{uuid4().hex}-{uuid4().hex}"

    while conv_name in ps_folders:
        conv_name = f"{chosen_agent_persona}--{chosen_human_persona}@{uuid4().hex}-{uuid4().hex}"

    mkdir(path.join(path.dirname(__file__), "persistent_storage", conv_name))

    # Create agent
    working_context = WorkingContext(
        CONFIG["model_name"], conv_name, agent_persona_str, human_persona_str
    )
    recall_storage = RecallStorage(conv_name)
    archival_storage = ArchivalStorage(conv_name)
    agent = Agent(
        interface,
        web_interface,
        conv_name,
        CONFIG["model_name"],
        function_dats,
        system_instructions,
        working_context,
        archival_storage,
        recall_storage,
    )

    return jsonify({
        'conv_name': conv_name
    })

@app.route('/messages/send', methods=['POST'])
def send_message():
    # Load data
    data = request.get_json()
    conv_name = data.get('conv_name')
    message = data.get('message')
    message_type = data.get('message_type')

    # Load system instructions
    system_instructions = get_system_text("llm_agent_chat")

    # Load functions
    function_dats = get_function_dats_from_function_sets(load_all_function_sets())

    # Load interfaces
    interface = ServerInterface()
    web_interface = WebInterface()

    # Load agent
    working_context = WorkingContext(
        CONFIG["model_name"], conv_name, None, None
    )
    recall_storage = RecallStorage(conv_name)
    archival_storage = ArchivalStorage(conv_name)
    agent = Agent(
        interface,
        web_interface,
        conv_name,
        CONFIG["model_name"],
        function_dats,
        system_instructions,
        working_context,
        archival_storage,
        recall_storage,
    )

    # Send message
    agent.interface.user_message(input_message)
    agent.memory.append_messaged_to_fq_and_rs(
        {
            "type": message_type,
            "message": {"role": "user", "content": input_message},
        }
    )

    # Generate agent responses
    @stream_with_context
    def generate_agent_responses(agent_obj):
        total_start_time = time()

        heartbeat_request = True
        while heartbeat_request:
            start_time = time()
            _, heartbeat_request, _ = agent_obj.step()
            end_time = time()

            server_message_stack = agent_obj.interface.server_message_stack.copy()
            agent_obj.interface.server_message_stack = []

            yield json.dumps({
                'server_message_stack': server_message_stack,
                'duration': timedelta(seconds=round(end_time - start_time, 2))
            }) + '\n'

        total_end_time = time()

        yield json.dumps({
            'total_duration': timedelta(seconds=round(total_end_time - total_start_time, 2))
        }) + '\n'   

    return Response(generate_agent_responses(agent), mimetype='application/json')

@app.route('/messages/send/no-heartbeat', methods=['POST'])
def send_message_without_heartbeat():
    # Load data
    data = request.get_json()
    conv_name = data.get('conv_name')
    message = data.get('message')
    message_type = data.get('message_type')

    # Load system instructions
    system_instructions = get_system_text("llm_agent_chat")

    # Load functions
    function_dats = get_function_dats_from_function_sets(load_all_function_sets())

    # Load interfaces
    interface = ServerInterface()
    web_interface = WebInterface()

    # Load agent
    working_context = WorkingContext(
        CONFIG["model_name"], conv_name, None, None
    )
    recall_storage = RecallStorage(conv_name)
    archival_storage = ArchivalStorage(conv_name)
    agent = Agent(
        interface,
        web_interface,
        conv_name,
        CONFIG["model_name"],
        function_dats,
        system_instructions,
        working_context,
        archival_storage,
        recall_storage,
    )

    # Send message
    agent.interface.user_message(input_message)
    agent.memory.append_messaged_to_fq_and_rs(
        {
            "type": message_type,
            "message": {"role": "user", "content": input_message},
        }
    )

    return jsonify({
        "type": message_type,
        "message": {"role": "user", "content": input_message},
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0')
