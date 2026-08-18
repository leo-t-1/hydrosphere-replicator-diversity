"""Vercel serverless endpoint: create a simulator session (stateless).

POST /api/session          -> initial state + opaque state blob
DELETE /api/session/:id    -> 204 (rewritten here; nothing to delete server-side)
GET /api/session, /api/health -> health check
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _stateless import create_session_state


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self._send_json({"ok": True, "server": "VercelStatelessSimulator/1.0"})

    def do_DELETE(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object")
            state = create_session_state(data)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(state, status=201)
