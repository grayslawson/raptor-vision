"""Shared fakes: a synthetic scene and a stub model endpoint that records every request. No network beyond 127.0.0.1."""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from PIL import Image, ImageDraw

from raptor_vision.types import Frame

W, H = 640, 360
SKY_Y = 0.30
TREE = (0.70, 0.30, 1.0, 0.60)  # x0 y0 x1 y1
ROAD = (0.20, 0.65, 0.60, 1.0)


def scene_jpeg(night: bool = False) -> bytes:
    img = Image.new("RGB", (W, H), (70, 120, 60))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, SKY_Y * H), fill=(100, 140, 200))
    d.rectangle((TREE[0] * W, TREE[1] * H, TREE[2] * W, TREE[3] * H), fill=(20, 80, 20))
    d.rectangle((ROAD[0] * W, ROAD[1] * H, ROAD[2] * W, ROAD[3] * H), fill=(110, 110, 115))
    if night:
        img = img.convert("L").convert("RGB")
    out = io.BytesIO()
    img.save(out, "JPEG", quality=90)
    return out.getvalue()


def frame(night: bool = False) -> Frame:
    return Frame(scene_jpeg(night), W, H, "night" if night else "day", 0.0)


def rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


CHAT_REGIONS = {"regions": [
    {"label": "sky", "polygon": rect(0, 0, 1, 0.36), "confidence": 0.9},
    {"label": "driveway", "polygon": rect(0.15, 0.6, 0.65, 1.0), "confidence": 0.8},
    {"label": "foliage", "polygon": rect(0.66, 0.26, 1.0, 0.64), "confidence": 0.7},
    {"label": "timestamp", "polygon": rect(0.02, 0.02, 0.30, 0.08), "confidence": 0.9},
    {"label": "neighbour", "polygon": rect(0.0, 0.31, 0.1, 0.5), "confidence": 0.6},
]}


class Stub:
    """A model endpoint. ``reply(request_json) -> (status, body)``; every request is kept in ``calls``."""

    def __init__(self, reply: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]) -> None:
        self.calls: list[dict[str, Any]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                outer.calls.append(req)
                code, payload = reply(req)
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub() -> Iterator[Callable[[Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]], Stub]]:
    made: list[Stub] = []

    def make(reply: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]) -> Stub:
        s = Stub(reply)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def choice(label: str, p: float = 0.9, entry: float | None = None) -> dict[str, Any]:
    """A decisions reply: the label, and optionally an entry answer."""
    ans: dict[str, Any] = {"kind": {"type": "choice", "choice": label, "probabilities": {label: p}}}
    if entry is not None:
        ans["entry"] = {"type": "noul", "noul": entry}
    return {"model": "d", "answers": ans}


def chat(content: dict[str, Any]) -> dict[str, Any]:
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    return {"model": "stub", "choices": [{"message": {"content": json.dumps(content)}}], "usage": usage}
