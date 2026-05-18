#!/usr/bin/env python3
"""
Serve a Python-backed simulator frontend that runs the real Simulation class.

Usage:
    python3 backend_simulator_server.py
Then open:
    http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from simulation import SimConfig, Simulation


BASE = Path(__file__).resolve().parent
HTML_PATH = BASE / "simulator_backend.html"
SESSION_TTL_SECONDS = 60 * 60
MAX_STEP_BATCH = 200
VALID_PATTERNS = {"uniform", "gradient", "patches", "hotspots"}
VALID_ABIOGENESIS_MODES = {"uniform", "fixed_theta"}


@dataclass
class SessionRecord:
    sim: Simulation
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create(self, cfg: SimConfig) -> tuple[str, SessionRecord]:
        sim = Simulation(cfg)
        sim.ensure_metrics_recorded()
        session_id = uuid.uuid4().hex
        record = SessionRecord(sim=sim)
        with self._lock:
            self._cleanup_locked()
            self._sessions[session_id] = record
        return session_id, record

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            self._cleanup_locked()
            record = self._sessions.get(session_id)
            if record is not None:
                record.last_access = time.time()
            return record

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _cleanup_locked(self):
        cutoff = time.time() - SESSION_TTL_SECONDS
        expired = [sid for sid, rec in self._sessions.items() if rec.last_access < cutoff]
        for sid in expired:
            self._sessions.pop(sid, None)


STORE = SessionStore()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_config(data: dict[str, Any]) -> SimConfig:
    grid_size = int(data.get("grid_size", 40))
    heterogeneity = float(data.get("heterogeneity", 0.8))
    mixing_rate = float(data.get("mixing_rate", 0.02))
    niche_width = float(data.get("niche_width", 20.0))
    pattern = str(data.get("pattern", "patches"))
    seed = int(data.get("seed", 42))
    abiogenesis_mode = str(data.get("abiogenesis_mode", "uniform"))
    abiogenesis_theta = float(data.get("abiogenesis_theta", 0.5))

    if pattern not in VALID_PATTERNS:
        raise ValueError(f"Invalid pattern: {pattern}")
    if abiogenesis_mode not in VALID_ABIOGENESIS_MODES:
        raise ValueError(f"Invalid abiogenesis_mode: {abiogenesis_mode}")

    return SimConfig(
        grid_size=max(10, min(120, grid_size)),
        heterogeneity=clamp(heterogeneity, 0.0, 1.0),
        mixing_rate=clamp(mixing_rate, 0.0, 1.0),
        niche_width=clamp(niche_width, 1.0, 100.0),
        pattern=pattern,
        mixing_mode="global",
        seed=seed,
        abiogenesis_mode=abiogenesis_mode,
        abiogenesis_theta=clamp(abiogenesis_theta, 0.0, 1.0),
        n_ticks=10**9,
    )


def build_state(session_id: str, sim: Simulation) -> dict[str, Any]:
    metrics = sim.get_current_metrics()
    history = sim.get_history_series(max_points=300)
    snapshot = sim.get_grid_snapshot()
    return {
        "session_id": session_id,
        "config": sim.cfg.to_dict(),
        "metrics": metrics,
        "history": history,
        "snapshot": {
            "N": int(snapshot["N"]),
            "grid": snapshot["grid"].tolist(),
            "theta": snapshot["theta"].tolist(),
            "env_map": snapshot["env_map"].tolist(),
        },
    }


class BackendSimulatorHandler(BaseHTTPRequestHandler):
    server_version = "BackendSimulator/1.0"

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._write_common_headers("application/json")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/simulator_backend.html"}:
            self._serve_html()
            return

        if parsed.path == "/api/health":
            self._send_json({"ok": True, "server": self.server_version})
            return

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 4 and parts[:2] == ["api", "session"] and parts[3] == "state":
            session_id = parts[2]
            record = STORE.get(session_id)
            if record is None:
                self._send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)
                return
            with record.lock:
                state = build_state(session_id, record.sim)
            self._send_json(state)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]

        if parsed.path == "/api/session":
            try:
                data = self._read_json_body()
                cfg = parse_config(data)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            session_id, record = STORE.create(cfg)
            with record.lock:
                state = build_state(session_id, record.sim)
            self._send_json(state, status=HTTPStatus.CREATED)
            return

        if len(parts) == 4 and parts[:2] == ["api", "session"] and parts[3] == "step":
            session_id = parts[2]
            record = STORE.get(session_id)
            if record is None:
                self._send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                data = self._read_json_body()
                n_steps = int(data.get("n_steps", 1))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            n_steps = max(1, min(MAX_STEP_BATCH, n_steps))
            with record.lock:
                for _ in range(n_steps):
                    record.sim.step()
                state = build_state(session_id, record.sim)
            self._send_json(state)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 3 and parts[:2] == ["api", "session"]:
            deleted = STORE.delete(parts[2])
            status = HTTPStatus.NO_CONTENT if deleted else HTTPStatus.NOT_FOUND
            self.send_response(status)
            self._write_common_headers("application/json")
            self.end_headers()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args):
        return

    def _serve_html(self):
        if not HTML_PATH.exists():
            self._send_json({"error": f"Missing HTML file: {HTML_PATH.name}"}, status=HTTPStatus.NOT_FOUND)
            return
        content = HTML_PATH.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._write_common_headers("text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._write_common_headers("application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(data)

    def _write_common_headers(self, content_type: str):
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def main():
    parser = argparse.ArgumentParser(description="Serve the Python-backed simulator frontend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), BackendSimulatorHandler)
    print(f"Serving Python-backed simulator at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
