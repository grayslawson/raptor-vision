"""Map store: a JSON body plus named files (``panorama.jpg``, ``labels.png``, ``overlay.png``) per camera and kind.

``put(camera, kind, body, files)``, ``latest(camera, kind)``, ``read_file(map, name)``. Retention is by count: only the
newest ``keep`` maps of one camera and kind stay (Osprey's ``keep_maps``, 4). Plain folders and files, stdlib only:
``<root>/<camera>/<kind>/<id>/map.json`` next to the files. Files are 0600, folders 0700.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

_BODY = "map.json"


def _safe(part: str, what: str) -> str:
    if not part or part in (".", "..") or "/" in part or "\\" in part or "\0" in part:
        raise ValueError(f"bad {what}: {part!r}")
    return quote(part, safe="")


def _write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


class MapStore:
    def __init__(self, root: str | Path, keep: int = 4, clock: Callable[[], float] = time.time) -> None:
        if keep < 1:
            raise ValueError("keep must be at least 1")
        self.root, self.keep, self._clock, self._lock = Path(root), keep, clock, threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _dir(self, camera: str, kind: str) -> Path:
        return self.root / _safe(camera, "camera") / _safe(kind, "kind")

    @staticmethod
    def _ids(base: Path) -> list[int]:
        return sorted((int(p.name) for p in base.iterdir() if p.name.isdigit() and p.is_dir()), reverse=True) if base.is_dir() else []

    def put(self, camera: str, kind: str, body: Mapping[str, Any], files: Mapping[str, bytes] | None = None) -> dict[str, Any]:
        """Save a map and return its record: ``{"id", "camera", "kind", "ts", **body}``. Older maps beyond ``keep`` are deleted."""
        names = dict(files or {})
        for name in names:
            if name == _BODY:
                raise ValueError(f"{_BODY} is reserved")
            _safe(name, "file name")
        base = self._dir(camera, kind)
        with self._lock:
            ids = self._ids(base)
            map_id = (ids[0] + 1) if ids else 1
            rec = {"id": map_id, "camera": camera, "kind": kind, "ts": self._clock(), **body}
            path = base / str(map_id)
            path.mkdir(parents=True, mode=0o700)
            for name, data in names.items():
                _write(path / name, data)
            _write(path / _BODY, json.dumps(rec).encode())
            for old in ids[self.keep - 1:]:
                shutil.rmtree(base / str(old), ignore_errors=True)
        return rec

    def _read(self, camera: str, kind: str, map_id: int) -> dict[str, Any] | None:
        try:
            rec = json.loads((self._dir(camera, kind) / str(map_id) / _BODY).read_bytes())
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) else None

    def latest(self, camera: str, kind: str) -> dict[str, Any] | None:
        for map_id in self._ids(self._dir(camera, kind)):
            if (rec := self._read(camera, kind, map_id)) is not None:
                return rec
        return None

    def read_file(self, map: Mapping[str, Any], name: str) -> bytes | None:
        """A file of a map (the record ``put`` or ``latest`` returned), or None when it is gone."""
        if name == _BODY:
            return None
        try:
            return (self._dir(str(map["camera"]), str(map["kind"])) / str(int(map["id"])) / _safe(name, "file name")).read_bytes()
        except (OSError, KeyError, ValueError, TypeError):
            return None
