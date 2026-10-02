"""
In-memory session store.

Keeps chat_history per session_id.
Memory is process-local — it resets automatically when the server restarts,
satisfying the requirement that sessions reset between server runs.
No external database or cache required.
"""
from __future__ import annotations

import threading
from typing import Optional

# Thread-safe in-process store: {session_id: [{"role": ..., "content": ...}]}
_store: dict[str, list[dict]] = {}
_lock = threading.Lock()


def get_history(session_id: str) -> list[dict]:
    """Return the conversation history for a session (empty list if new)."""
    with _lock:
        return list(_store.get(session_id, []))


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the session history."""
    with _lock:
        if session_id not in _store:
            _store[session_id] = []
        _store[session_id].append({"role": role, "content": content})


def reset_session(session_id: str) -> None:
    """Clear the history for the given session."""
    with _lock:
        _store.pop(session_id, None)


def list_sessions() -> list[str]:
    """Return all active session IDs (for debugging)."""
    with _lock:
        return list(_store.keys())
