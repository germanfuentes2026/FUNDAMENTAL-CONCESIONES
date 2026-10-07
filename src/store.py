"""Persistencia simple de balances cargados (JSON local)."""
from __future__ import annotations

import json
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "data" / "balances.json"
_PRIORITY = {"own": 1, "comparative": 0, "manual": 2}


def load() -> list[dict]:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def save(records: list[dict]) -> None:
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # filesystem de solo lectura (p. ej. hosting): queda en sesión


def merge(records: list[dict], new: list[dict]) -> list[dict]:
    """Un registro por (empresa, FY). Prioridad: manual > balance propio > comparativo."""
    out = {(r["company"].strip().lower(), int(r["fy"])): r for r in records}
    for r in new:
        k = (r["company"].strip().lower(), int(r["fy"]))
        old = out.get(k)
        if old is None or _PRIORITY.get(r.get("origin"), 0) >= _PRIORITY.get(old.get("origin"), 0):
            out[k] = r
    return sorted(out.values(), key=lambda r: (r["company"].lower(), int(r["fy"])))
