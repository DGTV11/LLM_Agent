import importlib, inspect, os

from llm_os.functions.schema_generator import generate_schema


class FunctionSet:
    def __init__(self, path):
        self.path = path
        self.func_set_name = os.path.basename(self.path).split(".")[0]

        spec = importlib.util.spec_from_file_location(self.func_set_name, path)
        self.module = importlib.util.module_from_spec(spec)

    def is_mod_function(self, func):
        return inspect.isfunction(func) and inspect.getmodule(func) == self.mod

    @property
    def function_dict(self):
        func_dict = {}
        for attr_name in dir(self.module):
            attr = getattr(self.module, attr_name)
            if self.is_mod_function(attr):
                if attr_name in func_dict:
                    raise Exception(f"Duplicate function: {attr_name}")
                func_dict[attr_name] = attr
        if len(func_dict.items()) == 0:
            raise Exception(f"No functions found in {self.path}")
        return func_dict

    @property
    def function_schemas_and_functions(self):
        return {
            func_name: {
                "python_function": generate_schema(func),
                "json_schema": generate_schema(func),
            }
            for func_name, func in self.function_dict.items()
        }


def load_all_function_sets():
    # functions_path = os.path.join(
    #    os.path.dirname(__file__), "llm_os", "functions", "function_sets"
    # )

    functions_path = os.path.join(os.path.dirname(__file__), "function_sets")
    user_functions_path = os.path.join(functions_path, "user_functions")

    function_sets = []

    for path in [functions_path, user_functions_path]:
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if (
                os.path.isfile(filepath)
                and filename.endswith(".py")
                and not (filename.startswith("_") or filename.startswith("."))
            ):
                try:
                    function_sets.append(FunctionSet(filepath))
                    print(f"Loaded function set {filename}")
                except SyntaxError as e:
                    print(
                        f"Skipped loading function set {filename} due to a syntax error: {e}"
                    )
                except Exception as e:
                    print(
                        f"Skipped loading function set {filename} due to an error: {e}"
                    )

    return function_sets


def get_function_dats_from_function_sets(function_sets):
    func_dict = {}

    for function_set in function_sets:
        for func_name, func_dat in function_set.function_schemas_and_functions:
            if func_name in func_dict:
                raise Exception(f"Duplicate function: {func_name}")
            func_dict[func_name] = func_dat
    return func_dict
