import importlib, inspect, os, sys

from llm_os.functions.schema_generator import generate_schema
from llm_os.constants import IN_CONTEXT_FUNCTION_SETS


class FunctionSet:
    def __init__(self, path):
        self.path = path
        self.module_name = os.path.splitext(os.path.basename(self.path))[0]

        spec = importlib.util.spec_from_file_location(self.module_name, path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = self.module
        spec.loader.exec_module(self.module)

    @property
    def function_dict(self):
        func_dict = {}
        for func_name, func in inspect.getmembers(self.module, inspect.isfunction):
            if func_name in func_dict:
                raise Exception(f"Duplicate function: {func_name}")
            func_dict[func_name] = func
        if len(func_dict.items()) == 0:
            raise Exception(f"No functions found in {self.path}")
        return func_dict

    @property
    def function_schemas_and_functions(self):
        return {
            func_name: {
                "python_function": func,
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

    function_set_dict = {}

    for path in [functions_path, user_functions_path]:
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if (
                os.path.isfile(filepath)
                and filename.endswith(".py")
                and not (filename.startswith("_") or filename.startswith("."))
            ):
                try:
                    function_set_dict[filename] = FunctionSet(filepath)
                    print(f"Loaded function set {filename}")
                except SyntaxError as e:
                    print(
                        f"Skipped loading function set {filename} due to a syntax error: {e}"
                    )
                except Exception as e:
                    print(
                        f"Skipped loading function set {filename} due to an error: {e}"
                    )

    return function_set_dict


def get_function_dats_from_function_sets(function_set_dict):
    in_context_func_dict = {}
    out_of_context_func_dict = {}

    for function_set_name, function_set in function_set_dict.items():
        for func_name, func_dat in function_set.function_schemas_and_functions.items():
            if func_name in func_dict:
                raise Exception(f"Duplicate function: {func_name}")
            if function_set_name in IN_CONTEXT_FUNCTION_SETS:
                in_context_func_dict[func_name] = func_dat
            else:
                out_of_context_func_dict[func_name] = func_dat
    return in_context_func_dict, out_of_context_func_dict
