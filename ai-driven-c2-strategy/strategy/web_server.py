from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .simulation import Simulation


class StrategyWebServer:
    def __init__(self, simulation: Simulation, host: str, port: int, web_dir: Path) -> None:
        self.simulation = simulation
        self.host = host
        self.port = port
        self.web_dir = web_dir

    def serve_forever(self) -> None:
        simulation = self.simulation
        web_dir = self.web_dir

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib API name.
                if self.path in {"/", "/index.html"}:
                    self._send_file(web_dir / "index.html", "text/html; charset=utf-8")
                elif self.path == "/app.js":
                    self._send_file(web_dir / "app.js", "text/javascript; charset=utf-8")
                elif self.path == "/style.css":
                    self._send_file(web_dir / "style.css", "text/css; charset=utf-8")
                elif self.path == "/state.json":
                    self._send_json(simulation.snapshot())
                else:
                    self.send_error(404)

            def log_message(self, format: str, *args: object) -> None:
                return

            def _send_file(self, path: Path, content_type: str) -> None:
                try:
                    data = path.read_bytes()
                except FileNotFoundError:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, payload: dict) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.serve_forever()
