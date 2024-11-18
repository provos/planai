from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
