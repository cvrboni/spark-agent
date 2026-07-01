from __future__ import annotations

from io import StringIO

from spark_agent.ui import TerminalUI


def test_terminal_ui_plain_info_output() -> None:
    stream = StringIO()
    ui = TerminalUI(stream=stream, rich=False)

    ui.info("collecting local repository snapshot")

    assert "[spark-agent] collecting local repository snapshot" in stream.getvalue()


def test_terminal_ui_plain_session_header() -> None:
    stream = StringIO()
    ui = TerminalUI(stream=stream, rich=False)

    ui.session_header(
        session_id="session-1",
        action="created",
        model="deepseek-v4-flash",
        sandbox_backend="local",
        repo="spark-agent",
    )

    output = stream.getvalue()
    assert "SparkAgent created session session-1" in output
    assert "Type /help" in output


def test_terminal_ui_plain_approval_request() -> None:
    stream = StringIO()
    ui = TerminalUI(stream=stream, rich=False)

    ui.approval_request("run_command", "run_command: pytest")

    assert "approval required" in stream.getvalue()
    assert "run_command: pytest" in stream.getvalue()
