"""
Stateless adapter for running the real Simulation class on serverless hosts.

Vercel functions keep no memory between requests, so the full simulation
state (arrays, RNG state, tick counter, chart history) is serialized into an
opaque blob that the browser holds and sends back with every step request.
The scientific model in simulation.py is untouched: we rebuild a Simulation,
overwrite its state arrays and RNG state exactly, and call the same step().
"""

from __future__ import annotations

import base64
import gzip
import json
import uuid
import zlib
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from simulation import SimConfig, Simulation
from backend_simulator_server import MAX_STEP_BATCH, build_state, parse_config  # noqa: F401

BLOB_VERSION = 1
MAX_BLOB_BYTES = 8 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 96 * 1024 * 1024
HISTORY_POINTS = 300
HISTORY_KEYS = ("ticks", "theta_std", "shannon_eco", "population")


def _encode_array(arr: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def _decode_array(text: str, dtype, expected_len: int) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(text), dtype=dtype)
    if arr.shape[0] != expected_len:
        raise ValueError("State blob does not match grid size")
    return arr.copy()


def encode_blob(sim: Simulation, history_series: dict) -> str:
    payload = {
        "v": BLOB_VERSION,
        "cfg": sim.cfg.to_dict(),
        "tick": int(sim.tick),
        "next_id": int(sim.next_id),
        "rng": sim.rng.bit_generator.state,
        "grid": _encode_array(sim.grid, np.int32),
        "theta": _encode_array(sim.theta, np.float64),
        "rep": _encode_array(sim.rep_rate, np.float64),
        "res": _encode_array(sim.resources, np.float64),
        "env": _encode_array(sim.env_map, np.float64),
        "hist": {k: list(history_series.get(k, []))[-HISTORY_POINTS:] for k in HISTORY_KEYS},
    }
    raw = json.dumps(payload).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 6)).decode("ascii")


def decode_blob(blob: str) -> dict:
    compressed = base64.b64decode(blob.encode("ascii"), validate=True)
    if len(compressed) > MAX_BLOB_BYTES:
        raise ValueError("State blob too large")
    decomp = zlib.decompressobj(wbits=31)
    raw = decomp.decompress(compressed, MAX_DECOMPRESSED_BYTES)
    if decomp.unconsumed_tail:
        raise ValueError("State blob too large")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("v") != BLOB_VERSION:
        raise ValueError("Unsupported state blob version")
    return payload


def restore_simulation(payload: dict) -> Simulation:
    cfg = SimConfig.from_dict(payload["cfg"])
    cfg.grid_size = max(10, min(120, int(cfg.grid_size)))
    S = cfg.grid_size * cfg.grid_size

    # Cheap construction: skip the expensive env build and seeding, then
    # overwrite every piece of state exactly from the blob.
    sim = Simulation(replace(cfg, pattern="uniform", n_initial=0))
    sim.cfg = cfg
    sim.grid = _decode_array(payload["grid"], np.int32, S)
    sim.theta = _decode_array(payload["theta"], np.float64, S)
    sim.rep_rate = _decode_array(payload["rep"], np.float64, S)
    sim.resources = _decode_array(payload["res"], np.float64, S)
    sim.env_map = _decode_array(payload["env"], np.float64, S)
    sim.next_id = int(payload["next_id"])
    sim.tick = int(payload["tick"])
    rng_state = payload["rng"]
    if rng_state.get("bit_generator") != "PCG64":
        raise ValueError("Unsupported RNG state")
    sim.rng.bit_generator.state = rng_state
    sim.history = []
    return sim


def create_session_state(config_data: dict) -> dict:
    cfg = parse_config(config_data)
    sim = Simulation(cfg)
    sim.ensure_metrics_recorded()
    state = build_state(uuid.uuid4().hex, sim)
    state["blob"] = encode_blob(sim, state["history"])
    return state


def step_session_state(blob: str, n_steps: int) -> dict:
    n_steps = max(1, min(MAX_STEP_BATCH, int(n_steps)))
    payload = decode_blob(blob)
    sim = restore_simulation(payload)
    for _ in range(n_steps):
        sim.step()

    carried = payload.get("hist", {})
    merged = {}
    for key, hist_key in (("ticks", "tick"), ("theta_std", "theta_std"),
                          ("shannon_eco", "shannon_eco"), ("population", "population")):
        old = list(carried.get(key, []))
        new = [row[hist_key] for row in sim.history]
        merged[key] = (old + new)[-HISTORY_POINTS:]

    state = build_state(uuid.uuid4().hex, sim)
    state["history"] = merged
    state["blob"] = encode_blob(sim, merged)
    return state
