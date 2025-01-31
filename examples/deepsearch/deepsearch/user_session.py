from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from pydantic import BaseModel

from planai import ChatMessage


class Session(BaseModel):
    id: str
    messages: List[ChatMessage]


class UserSessionManager:

    def __init__(self, directory: str):
        self.directory = directory
        self.sessions: Dict[str, Session] = {}
        self.lock = Lock()

        if not Path(directory).exists():
            Path(directory).mkdir(parents=True)

        self.load_sessions()

    def load_sessions(self):
        for path in Path(self.directory).rglob("*.json"):
            data = path.read_text()
            session = Session.model_validate_json(data)
            with self.lock:
                self.sessions[session.id] = session

    def add_message(self, session_id: str, message: ChatMessage):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(id=session_id, messages=[])
            self.sessions[session_id].messages.append(message)

    def save_session(self, session_id: str):
        with self.lock:
            session = self.sessions[session_id]
        path = Path(self.directory) / f"{session_id}.json"
        path.write_text(session.model_dump_json(indent=2))

    def list_sessions(self) -> List[Dict]:
        with self.lock:
            response = []
            for id, session in self.sessions.items():
                response.append(
                    {"id": id, "first_message": session.messages[0].content}
                )
            return response

    def get_session(self, session_id: str) -> Optional[Session]:
        with self.lock:
            return self.sessions.get(session_id)
