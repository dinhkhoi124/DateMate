"""eval/llamacpp_backend.py against a stub llama-server on 127.0.0.1 (no model, no external network)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from datamate.config import LLMConfig
from eval.llamacpp_backend import LlamaCppBackend


class _Stub(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append(body)
        out = {"choices": [{"message": {"content": f"TYPE: number\n```sql\nSELECT {body['seed']}\n```"}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_samples_seeds_and_thinking_off(server):
    _Stub.seen.clear()
    cfg = LLMConfig(backend="replay", model_id="stub", seed=10)
    be = LlamaCppBackend(server, cfg, workers=2)
    outs = be.complete(["p1", "p2"], n=3, free=True, max_tokens=50)
    assert [len(o) for o in outs] == [3, 3]
    assert sorted(b["seed"] for b in _Stub.seen) == [10, 10, 11, 11, 12, 12]
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} and b["max_tokens"] == 50 for b in _Stub.seen)
    assert be.complete(["p"], n=1, free=False) == [[]]          # guided JSON is not served by this backend


def test_only_local_servers():
    with pytest.raises(ValueError):
        LlamaCppBackend("http://example.com:8080", LLMConfig(backend="replay"))
