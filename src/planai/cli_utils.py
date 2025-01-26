import importlib.util
import os
import sys
from typing import Any, Optional

from llm_interface import LLMInterface

from planai import LLMTaskWorker


def load_module_from_file(
    python_file: str, search_path: Optional[str] = None
) -> Optional[Any]:
    """
    Dynamically load a module from a given Python file with an optional search path.

    :param python_file: The path to the Python file.
    :param search_path: An optional path to include in the module search path.
    :return: The loaded module if successful, else None.
    """
    python_file = os.path.abspath(python_file)
    if search_path:
        search_path = os.path.abspath(search_path)
        if search_path not in sys.path:
            sys.path.insert(0, search_path)

    try:
        # Derive the module's name and package structure
        relative_path = os.path.relpath(python_file, start=search_path)
        module_parts = os.path.splitext(relative_path.replace(os.path.sep, "."))[
            0
        ].split(".")
        package = ".".join(module_parts[:-1])
        module_name = module_parts[-1]

        # Load the target module
        spec = importlib.util.spec_from_file_location(
            f"{package}.{module_name}", python_file
        )
        if spec is None:
            print(f"Could not load spec for file {python_file}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package}.{module_name}"] = module
        spec.loader.exec_module(module)

        return module

    except Exception as e:
        print(f"Error loading module from '{python_file}': {e}")
        return None

    finally:
        # Clean up the path entry
        if search_path and search_path in sys.path:
            sys.path.remove(search_path)


def get_class_from_module(module: Any, class_name: str) -> Optional[type]:
    """
    Load a class from a given module.

    :param module: The module object containing the class.
    :param class_name: The name of the class to be loaded.
    :return: The class object if found, else None.
    """
    try:
        cls = getattr(module, class_name, None)
        if cls is None:
            print(f"Class '{class_name}' not found in module '{module.__name__}'")
        return cls
    except Exception as e:
        print(
            f"Error loading class '{class_name}' from module '{module.__name__}': {e}"
        )
        return None


def instantiate_llm_class_from_module(
    module: Any, class_name: str, llm: LLMInterface
) -> Optional[LLMTaskWorker]:
    """
    Load a class from a given module and instantiate it.

    :param module: The module object containing the class.
    :param class_name: The name of the class to instantiate.
    :return: An instance of the specified class or None if instantiation fails.
    """
    try:
        cls = getattr(module, class_name, None)
        if cls is None:
            print(f"Class '{class_name}' not found in module '{module.__name__}'.")
            return None

        # Create an instance of the class, passing the required 'llm' parameter
        instance = cls(llm=llm)
        if not isinstance(instance, LLMTaskWorker):
            print(
                f"Class '{class_name}' in module '{module.__name__}' does not inherit from 'LLMTaskWorker'."
            )
            return None
        return instance

    except Exception as e:
        print(
            f"Error instantiating class '{class_name}' from module '{module.__name__}': {e}"
        )
        return None
