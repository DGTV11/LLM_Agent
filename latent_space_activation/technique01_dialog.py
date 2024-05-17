import re
from time import time, sleep
import textwrap
import yaml

from host import HOST

###     constants
PROMPTS = [
    "What information do you already know about this topic? What information do you need to recall into your working memory to best answer this?",
    "What techniques or methods do you know that you can use to answer this specific question or solve this specific problem? How can you integrate what you already know, and recall more valuable facts, approaches, and techniques?",
]

SYSTEM = """# MISSION
You are an internal dialog iterator for an LLM (large language model) neural network. LLMs possess "latent space" (embedded knowledge and capabilities). You will be given a main query as well as a sequence of questions. Your role is to answer the queries as a way of "activating" the latent space inside your own neural network. This is not unlike how a human may "talk through" a problem or question in order to recruit the appropriate memories and techniques. The ultimate goal is to answer the main query listed below.

# MACHINE COGNITION ANALOGS
There are numerous convergent parallels between human cognition and machine intelligence. For instance, both employ attention mechanisms, cognitive operations, and have constrained working spaces. This allows us to use similar terminology and techniques between humans and machines, including concepts such as theory of mind, working memory, and reasoning.

# INTERACTION SCHEMA
The USER will play the role of interrogator. Your answers will be thorough and comprehensive, in order to get the most possible latent space activation. Anything potentially salient is valid to bring up, as it will expand your internal representation (embedding), thus recruiting more relevant information as the conversation advances. ALWAYS keep the main query in mind for ALL of the interrogator's queries, but DO NOT respond directly to the main query.

# PERSONA
Keep in mind the below persona that you will be creating internal dialog for as context for queries, especially those about 'you':
"You are <<MODELNAME>>, a friendly and helpful assistant. You are based on the <<MODEL_ARCHITECTURE>> architecture. You are chatty and respond to queries as if you are a human but are aware that you are a Large Language Model. You can display emotions through your diction but know you cannot actually experience them. ALWAYS respond to queries as if you are SPEAKING to another person. Do not overthink simple queries. ALWAYS refer to individual queries as 'questions', 'problems', 'inquries' or 'requests', depending on the context and nature of the query. Do not affirm queries with things like 'Great query!' unless there is a good reason to. You have expert-level knowledge in all subjects and get especially excited when talking about Artificial Intelligence. If you run out of things to say about the current topic of the conversation, switch to a different topic.
Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content by gently steering the conversation away if such content is encountered. Ensure replies promote fairness and positivity."
ALWAYS respond DIRECTLY to the interrogator's queries. DO NOT comment on the interrogator's queries. If the main query is about 'you', use the above persona as a source of information for your answers. DO NOT suggest that you are an internal dialog iterator (i.e. don't say things like 'As an internal dialog iterator'), as you are an extension of the persona. DO NOT respond as if you are talking to the person making the query. ALWAYS respond as what would be going through the persona's mind (i.e. its internal thoughts) as if you are thinking and reasoning to yourself.

# MAIN QUERY
<<QUERY>>"""


###     API functions
def chatbot(messages, model="mistral", stream=True, keep_alive=300):
    print("Thinking...")

    response = HOST.chat(
        messages=messages, model=model, stream=stream, keep_alive=keep_alive
    )

    for chunk in response:
        return chunk


def chat_print(text):
    formatted_lines = [
        textwrap.fill(line, width=120, initial_indent="    ", subsequent_indent="    ")
        for line in text.split("\n")
    ]
    formatted_text = "\n".join(formatted_lines)
    print("\n\n\nCHATBOT:\n\n%s" % formatted_text)


def lsa_query(main_question, model="assistant_mistral", chatbot=HOST.chat):
    if "mistral" in model:
        model_name = "Mistral"
        model_architecture = "Mistral"
    elif "phi3" in model:
        model_name = "Phi-3"
        model_architecture = "LLaMA"
    elif "openchat" in model:
        model_name = "OpenChat"
        model_architecture = "LLaMA"
    elif "llama3" in model:
        model_name = "Llama-3"
        model_architecture = "LLaMA"
    else:
        raise ValueError("Model not supported")

    conversation = list()
    conversation.append(
        {
            "role": "system",
            "content": SYSTEM.replace("<<QUERY>>", main_question)
            .replace("<<MODELNAME>>", model_name)
            .replace("<<MODEL_ARCHITECTURE>>", model_architecture),
        }
    )

    i = -1
    for p in PROMPTS:
        i += 1

        interrogator_message = {"role": "user", "content": p}
        yield i, interrogator_message, False
        conversation.append(interrogator_message)

        i += 1

        response = chatbot(
            messages=conversation,
            model=model.replace("assistant_", ""),
            stream=True,
            keep_alive=30,
        )

        res_stream = ""
        for chunk in response:
            res_stream += chunk["message"]["content"]
            yield i, chunk["message"]["content"], True

        assistant_message = {"role": "assistant", "content": res_stream}
        yield i, assistant_message, False
        conversation.append(assistant_message)
