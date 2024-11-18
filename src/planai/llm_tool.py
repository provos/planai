from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, get_type_hints
import inspect
import json


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any] = field(repr=False)

    def execute(self, **kwargs) -> Any:
        return self.func(**kwargs)


def _type_to_json_schema(type_hint: Type) -> Dict[str, Any]:
    """Convert Python type hints to JSON Schema types."""
    if type_hint == str:
        return {"type": "string"}
    elif type_hint == int:
        return {"type": "integer"}
    elif type_hint == float:
        return {"type": "number"}
    elif type_hint == bool:
        return {"type": "boolean"}
    elif type_hint == list or getattr(type_hint, "__origin__", None) == list:
        item_type = Any
        if hasattr(type_hint, "__args__"):
            item_type = type_hint.__args__[0]
        return {"type": "array", "items": _type_to_json_schema(item_type)}
    elif type_hint == dict or getattr(type_hint, "__origin__", None) == dict:
        return {"type": "object"}
    elif hasattr(type_hint, "__origin__") and type_hint.__origin__ == Optional:
        return _type_to_json_schema(type_hint.__args__[0])
    else:
        return {"type": "string"}


def create_tool(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    """
    Create a Tool instance from a Python function using its type hints and docstring.

    Args:
        func (Callable): The function to convert into a tool
        name (Optional[str]): Optional custom name for the tool. If not provided, uses the function name
        description (Optional[str]): Optional custom description. If not provided, uses the function's docstring

    Returns:
        Tool: A Tool instance representing the function

    Example:
        @create_tool
        def get_weather(location: str, units: str = "celsius") -> str:
            '''Get the weather for a specific location.

            Args:
                location: The city or location to get weather for
                units: Temperature units (celsius or fahrenheit)
            '''
            # Function implementation
            pass
    """
    # Get function metadata
    func_name = name or func.__name__
    func_doc = inspect.getdoc(func) or ""
    func_desc = description or func_doc.split("\n\n")[0] if func_doc else func_name

    # Get type hints
    type_hints = get_type_hints(func)

    # Parse docstring for parameter descriptions
    param_docs = {}
    if func_doc:
        current_param = None
        for line in func_doc.split("\n"):
            line = line.strip()
            if line.startswith("Args:"):
                continue
            if ":" in line and not line.endswith(":"):
                param_name, param_desc = line.split(":", 1)
                param_name = param_name.strip()
                param_docs[param_name] = param_desc.strip()
            elif current_param and line:
                param_docs[current_param] += " " + line

    # Get default values
    signature = inspect.signature(func)

    # Build parameters schema
    parameters = {"type": "object", "properties": {}, "required": []}

    for param_name, param in signature.parameters.items():
        if param_name == "self":  # Skip self parameter for methods
            continue

        param_schema = _type_to_json_schema(type_hints.get(param_name, Any))

        # Add description if available
        if param_name in param_docs:
            param_schema["description"] = param_docs[param_name]

        # Handle default values
        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            parameters["required"].append(param_name)

        parameters["properties"][param_name] = param_schema

    return Tool(name=func_name, description=func_desc, parameters=parameters, func=func)


def tool(name: Optional[str] = None, description: Optional[str] = None) -> Callable:
    """
    Decorator to create a Tool from a function.

    Args:
        name (Optional[str]): Optional custom name for the tool
        description (Optional[str]): Optional custom description

    Returns:
        Callable: Decorator function that creates a Tool

    Example:
        @tool(name="weather", description="Get weather information")
        def get_weather(location: str, units: str = "celsius") -> str:
            '''Get the weather for a specific location.'''
            # Function implementation
            pass
    """

    def decorator(func: Callable) -> Tool:
        return create_tool(func, name=name, description=description)

    return decorator
