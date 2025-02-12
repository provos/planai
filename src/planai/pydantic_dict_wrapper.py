from typing import Any, ClassVar, Optional, Type

from pydantic import create_model

from .task import Task


class PydanticDictWrapper:
    """This class creates a pydantic model from a dict object that can be used in the pre_process method of LLMTaskWorker."""

    def __init__(self, data: dict, name: Optional[str] = None):
        self._model = self._create_model(data, name)
        self._instance = self._model(**data)

    def _get_type_and_default(self, value: Any) -> tuple:
        """Determine the type and default value for a field."""
        if value is None:
            return (Optional[Any], None)
        if isinstance(value, dict):
            nested_model = self._create_model(value)
            return (nested_model, ...)
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                nested_model = self._create_model(value[0])
                return (list[nested_model], ...)
            return (list[type(value[0])] if value else list, ...)
        return (type(value), ...)

    def _create_model(self, data: dict, name: Optional[str] = None) -> Type[Task]:
        """Create a Pydantic model from a dictionary using Task as base."""
        fields = {key: self._get_type_and_default(value) for key, value in data.items()}

        # Create an intermediate base class with proper type annotations
        class AnnotatedTask(Task):
            type_: ClassVar[str] = "base_task"
            # Add other Task attributes here with proper annotations
            model_config = {"arbitrary_types_allowed": True}

        return create_model(
            name if name else "DynamicTaskModel", __base__=AnnotatedTask, **fields
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the Pydantic instance."""
        return getattr(self._instance, name)

    def dict(self) -> dict:
        """Return the model as a dictionary."""
        return self._instance.model_dump()

    def json(self) -> str:
        """Return the model as a JSON string."""
        return self._instance.model_dump_json()

    @property
    def model(self) -> Type[Task]:
        """Return the underlying Pydantic model class."""
        return self._model

    @property
    def instance(self) -> Task:
        """Return the Pydantic model instance."""
        return self._instance
