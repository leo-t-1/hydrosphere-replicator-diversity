"""Vercel serverless endpoint: advance a simulator session (stateless).

POST /api/session/:id/step (rewritten to /api/step) with body
{"n_steps": N, "blob": "<opaque state blob from the previous response>"}
-> new state + new blob. The real simulation.py step() runs server-side.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _stateless import step_session_state


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object")
            blob = data.get("blob")
            if not blob:
                raise ValueError(
                    "Session not found (stateless backend: request is missing "
                    "the state blob — reset the session)")
            state = step_session_state(blob, data.get("n_steps", 1))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(state)
