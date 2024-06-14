from os import path, listdir, mkdir
from uuid import uuid4

from config import CONFIG

from llm_os.agent import Agent
from llm_os.interface import CLIInterface

from llm_os.memory.memory import Memory
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.recall_storage import RecallStorage
from llm_os.memory.archival_storage import ArchivalStorage

from llm_os.prompts.gpt_system import get_system_text

from llm_os.functions.load_functions import (
    load_all_function_sets,
    get_function_dats_from_function_sets,
)

if __name__ == "__main__":
    if not CONFIG:
        raise ValueError("Config not found, please run config.py")

    conv_name = None

    function_dats = get_function_dats_from_function_sets(load_all_function_sets())
    system_instructions = get_system_text("llm_agent_chat")
    interface = CLIInterface()

    has_prev_conv = False
    if ps_folders := listdir(path.join(path.dirname(__file__), "persistent_storage")):
        use_existing_conv = (
            True if input("Use existing conv? (y/n) ").strip().lower() == "y" else False
        )
        if use_existing_conv:
            existing_conv_enum = list(enumerate(ps_folders, start=1))
            existing_conv_enum_dict = {idx: name for idx, name in existing_conv_enum}

            print(
                "Please choose the number corresponding to your chosen existing conversation"
            )
            for folder_idx, folder_name in existing_conv_enum:
                print(f"{folder_idx} -> {folder_name}")

            while True:
                try:
                    chosen_folder_num = int(input("> "))
                    if chosen_folder_num not in existing_conv_enum_dict:
                        print(
                            "Input is not in the given conversation numbers! Please try again!"
                        )
                        continue
                    break
                except ValueError:
                    print("Input is not an integer! Please try again!")

            conv_name = existing_conv_enum_dict[chosen_folder_num]

            working_context = WorkingContext(
                CONFIG["model_name"], conv_name, None, None
            )
            recall_storage = RecallStorage(conv_name)
            archival_storage = ArchivalStorage(conv_name)
            agent = Agent(
                interface,
                conv_name,
                CONFIG["model_name"],
                function_dats,
                system_instructions,
                working_context,
                archival_storage,
                recall_storage,
            )
            has_prev_conv = True

    if not conv_name:
        agent_persona_folder_path = path.join(
            path.dirname(__file__), "llm_os", "personas", "agents"
        )
        human_persona_folder_path = path.join(
            path.dirname(__file__), "llm_os", "personas", "humans"
        )
        agent_persona_folder = listdir(agent_persona_folder_path)
        human_persona_folder = listdir(human_persona_folder_path)

        # Choose agent persona
        agent_persona_folder_enum = list(enumerate(agent_persona_folder, start=1))
        agent_persona_folder_enum_dict = {
            idx: name for idx, name in agent_persona_folder_enum
        }

        print("Please choose the number corresponding to your chosen agent persona")
        for persona_idx, persona_name in agent_persona_folder_enum:
            print(f"{persona_idx} -> {persona_name.split('.')[0]}")

        while True:
            try:
                chosen_agent_persona_num = int(input("> "))
                if chosen_agent_persona_num not in agent_persona_folder_enum_dict:
                    print(
                        "Input is not in the given persona numbers! Please try again!"
                    )
                    continue
                break
            except ValueError:
                print("Input is not an integer! Please try again!")

        chosen_agent_persona = agent_persona_folder_enum_dict[chosen_agent_persona_num]
        chosen_agent_persona_fp = path.join(
            agent_persona_folder_path, chosen_agent_persona
        )

        # Choose human persona
        human_persona_folder_enum = list(enumerate(human_persona_folder, start=1))
        human_persona_folder_enum_dict = {
            idx: name for idx, name in human_persona_folder_enum
        }

        print("Please choose the number corresponding to your chosen human persona")
        for persona_idx, persona_name in human_persona_folder_enum:
            print(f"{persona_idx} -> {persona_name.split('.')[0]}")

        while True:
            try:
                chosen_human_persona_num = int(input("> "))
                if chosen_human_persona_num not in agent_persona_folder_enum_dict:
                    print(
                        "Input is not in the given persona numbers! Please try again!"
                    )
                    continue
                break
            except ValueError:
                print("Input is not an integer! Please try again!")

        chosen_human_persona = human_persona_folder_enum_dict[chosen_agent_persona_num]
        chosen_human_persona_fp = path.join(
            human_persona_folder_path, chosen_human_persona
        )

        # Load personas
        with open(chosen_agent_persona_fp, "r") as f:
            agent_persona_str = f.read()

        with open(chosen_human_persona_fp, "r") as f:
            human_persona_str = f.read()

        # Create agent
        conv_name = f"{chosen_agent_persona.split('.')[0]}--{chosen_human_persona.split('.')[0]}@{uuid4().hex}-{uuid4().hex}"

        while conv_name in ps_folders:
            conv_name = f"{chosen_agent_persona}--{chosen_human_persona}@{uuid4().hex}-{uuid4().hex}"

        mkdir(path.join(path.dirname(__file__), "persistent_storage", conv_name))

        working_context = WorkingContext(
            CONFIG["model_name"], conv_name, agent_persona_str, human_persona_str
        )
        recall_storage = RecallStorage(conv_name)
        archival_storage = ArchivalStorage(conv_name)
        agent = Agent(
            interface,
            conv_name,
            CONFIG["model_name"],
            function_dats,
            system_instructions,
            working_context,
            archival_storage,
            recall_storage,
        )

    # Main conversation loop
    no_tokens_in_ctx = agent.memory.main_ctx_message_seq_no_tokens
    ctx_window = agent.memory.ctx_window
    print(f"Context info: {no_tokens_in_ctx}/{ctx_window} tokens ({round((no_tokens_in_ctx/ctx_window)*100, 2)}%)")

    interface_message = f'User \'{conv_name.split("@")[0].split("--")[1]}\' entered the conversation. You should greet the user{" based on your previous conversation" if has_prev_conv else ""} using the \'send_message\' function.'
    try:
        agent.interface.system_message(interface_message)
        agent.memory.append_messaged_to_fq_and_rs(
            {
                "type": "system",
                "message": {
                    "role": "user",
                    "content": interface_message,
                },
            }
        )
        heartbeat_request = True
        while heartbeat_request:
            _, heartbeat_request, _ = agent.step()

        print("/help for commands, /exit to exit")
        while True:
            no_tokens_in_ctx = agent.memory.main_ctx_message_seq_no_tokens
            ctx_window = agent.memory.ctx_window
            input_message = input(f"{no_tokens_in_ctx}/{ctx_window} tokens ({round((no_tokens_in_ctx/ctx_window)*100, 2)}%) > ")

            match input_message.strip():
                case "/help":
                    print("HELP")
                    print("/reload_functions -> reloads function sets")
                    print("/exit -> exit conversation")
                case "/reload_functions":
                    function_dats = get_function_dats_from_function_sets(load_all_function_sets())
                    agent.memory.function_dats = function_dats
                case "/exit":
                    break
                case _:
                    agent.interface.user_message(input_message)
                    agent.memory.append_messaged_to_fq_and_rs(
                        {
                            "type": "system",
                            "message": {"role": "user", "content": input_message},
                        }
                    )
                    heartbeat_request = True
                    while heartbeat_request:
                        _, heartbeat_request, _ = agent.step()
    except KeyboardInterrupt:
        print('Received keyboard interrupt. Exiting...')
    finally:    
        interface_message = f'User \'{conv_name.split("@")[0].split("--")[1]}\' exited the conversation'
        agent.interface.system_message(interface_message)
        agent.memory.append_messaged_to_fq_and_rs(
            {
                "type": "system",
                "message": {
                    "role": "user",
                    "content": interface_message,
                },
            }
        )

        no_tokens_in_ctx = agent.memory.main_ctx_message_seq_no_tokens
        ctx_window = agent.memory.ctx_window
        print(f"Context info: {no_tokens_in_ctx}/{ctx_window} tokens ({round((no_tokens_in_ctx/ctx_window)*100, 2)}%)")
