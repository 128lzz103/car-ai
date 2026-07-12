"""通过本地假 Provider 黑盒验证 CLI → Agent → Tool 完整链路。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest


def _reply_for(request_number: int) -> dict[str, Any]:
    if request_number == 1:
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_e2e",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": "ci-output.txt", "content": "created by e2e"}
                        ),
                    },
                }
            ],
        }
    else:
        message = {"content": "E2E complete"}
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.mark.e2e
def test_headless_cli_executes_provider_tool_loop(tmp_path: Path):
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(_reply_for(len(requests))).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    environment = os.environ.copy()
    environment.update(
        {
            "MINICODER_PROVIDER": "openai",
            "MINICODER_MODEL": "fake-e2e-model",
            "MINICODER_MAX_RETRIES": "0",
            "MINICODER_AUTOSAVE": "false",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "NO_PROXY": "127.0.0.1,localhost",
            "PYTHONUTF8": "1",
        }
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "minicoder.cli",
                "-p",
                "create the requested file",
                "--permission-mode",
                "allow",
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "E2E complete"
    assert (tmp_path / "ci-output.txt").read_text(encoding="utf-8") == "created by e2e"
    assert len(requests) == 2
    assert requests[0]["model"] == "fake-e2e-model"
    assert any(tool["function"]["name"] == "write_file" for tool in requests[0]["tools"])
    tool_results = [message for message in requests[1]["messages"] if message["role"] == "tool"]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_call_id"] == "call_e2e"
