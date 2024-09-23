from queue import Queue
from typing import List

from pydantic import BaseModel, Field, PrivateAttr


class UserInputRequest(BaseModel):
    task_id: str
    instruction: str
    accepted_mime_types: List[str] = Field(
        default_factory=lambda: ["text/html", "application/pdf"]
    )
    _response_queue: Queue = PrivateAttr(default_factory=Queue)
