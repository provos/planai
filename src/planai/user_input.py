from queue import Queue
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field, PrivateAttr

ProvenanceChain = Tuple[Tuple[str, int], ...]


class UserInputRequest(BaseModel):
    """
    Allows a PlanAI graph to ask for out-of-band user input.
    A consumer of a UserInputRequest needs to return a response
    via the respond() method.
    The request is a blocking call until the response is received.
    """

    task_id: str
    instruction: str
    provenance: Optional[ProvenanceChain] = None
    accepted_mime_types: List[str] = Field(
        default_factory=lambda: ["text/html", "application/pdf"]
    )
    _response_queue: Queue[Tuple[Any, Optional[str]]] = PrivateAttr(
        default_factory=Queue
    )

    def respond(self, response: Any, mime_type: Optional[str] = None) -> None:
        """
        Respond to the request with a response and an optional mime type.
        """
        if mime_type is not None and mime_type not in self.accepted_mime_types:
            raise ValueError(f"Mime type {mime_type} not accepted")

        self._response_queue.put((response, mime_type))
