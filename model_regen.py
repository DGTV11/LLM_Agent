import os

from host import HOST

def gen_model_from_modelfile(
    name,
    base_path,
    start_function: callable = lambda: None,
    end_function: callable = lambda: None,
):
    start_function()

    with open(os.path.join(os.path.dirname(base_path), "modelfiles", name)) as file:
        HOST.create(model=name, modelfile=file.read())

    end_function()

def regen_models(base_path):
    for file in os.listdir(os.path.join(os.path.dirname(base_path), "modelfiles")):
        if file[0] in [".", "_"]:
            continue
        gen_model_from_modelfile(
            f"{file}",
            base_path,
            lambda: print(f"Creating the '{file}' model..."),
            lambda: print(f"'{file}' model created!"),
        )
