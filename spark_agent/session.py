from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spark_agent.core.prompt_engine import ChatMessage

type JsonObject = dict[str, Any]

SESSION_DIR = ".spark-agent/sessions"
LATEST_FILE = ".spark-agent/latest-session"


@dataclass(frozen=True, slots=True)
class AgentSession:
    session_id: str
    path: Path
    events: tuple[ChatMessage, ...]


class SessionStore:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.sessions_dir = self.repo_root / SESSION_DIR
        self.latest_path = self.repo_root / LATEST_FILE

    def create(self, *, title: str | None = None) -> AgentSession:
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        path = self._session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._append_record(
            path,
            {
                "type": "meta",
                "session_id": session_id,
                "created_at": int(time.time()),
                "title": title or "",
            },
        )
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_path.write_text(session_id + "\n", encoding="utf-8")
        return AgentSession(session_id=session_id, path=path, events=())

    def load(self, session_id: str) -> AgentSession:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        return AgentSession(session_id=session_id, path=path, events=tuple(self._read_events(path)))

    def load_latest(self) -> AgentSession:
        if not self.latest_path.exists():
            raise FileNotFoundError("no latest session found")
        session_id = self.latest_path.read_text(encoding="utf-8").strip()
        if not session_id:
            raise FileNotFoundError("latest session marker is empty")
        return self.load(session_id)

    def append_events(self, session: AgentSession, events: list[ChatMessage]) -> AgentSession:
        for event in events:
            self._append_record(
                session.path,
                {
                    "type": "event",
                    "timestamp": int(time.time()),
                    "message": dict(event),
                },
            )
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_path.write_text(session.session_id + "\n", encoding="utf-8")
        return AgentSession(
            session_id=session.session_id,
            path=session.path,
            events=(*session.events, *(dict(event) for event in events)),
        )

    def _session_path(self, session_id: str) -> Path:
        if "/" in session_id or "\\" in session_id or session_id in {"", ".", ".."}:
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.sessions_dir / f"{session_id}.jsonl"

    @staticmethod
    def _append_record(path: Path, record: JsonObject) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _read_events(path: Path) -> list[ChatMessage]:
        events: list[ChatMessage] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") != "event":
                    continue
                message = record.get("message")
                if isinstance(message, dict):
                    events.append(dict(message))
        return events
