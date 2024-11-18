from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any] = field(repr=False)

    def execute(self, **kwargs) -> Any:
        return self.func(**kwargs)
