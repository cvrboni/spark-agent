from __future__ import annotations

from spark_agent.session import SessionStore


def test_session_store_persists_and_loads_events(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(title="initial task")

    session = store.append_events(session, [{"role": "user", "content": "hello"}])
    loaded = store.load(session.session_id)
    latest = store.load_latest()

    assert loaded.events == ({"role": "user", "content": "hello"},)
    assert latest.session_id == session.session_id
    assert latest.events == loaded.events
