from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from raptor_vision.frigate import Change, FrigateClient, FrigateError, Writer, WriteRefused

SQUARE = ((0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5))


class Stub:
    def __init__(self, version: str = "0.18.3-abc") -> None:
        self.version = version
        self.cfg: dict[str, Any] = {
            "cameras": {
                "front": {
                    "zones": {"porch_locked": {"coordinates": "0,0,1,0,1,1", "enabled": True}, "mine": {"coordinates": "0,0,1,0,1,1"}},
                    "motion": {"mask": {}},
                    "objects": {"mask": {}},
                }
            }
        }
        self.requests: list[tuple[str, str, Any]] = []
        self.fail_set = False
        self.drop_after_set = False  # accept set but do not change anything (read-back mismatch)

    def raw(self) -> str:
        return json.dumps(self.cfg, sort_keys=True)


def _set_path(cfg: dict[str, Any], path: list[str], value: Any) -> None:
    for p in path[:-1]:
        cfg = cfg.setdefault(p, {})
    if value in (None, ""):
        cfg.pop(path[-1], None)
    else:
        cfg[path[-1]] = value


def _flat(d: dict[str, Any], pre: list[str]) -> Iterator[tuple[list[str], Any]]:
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _flat(v, [*pre, k])
        else:
            yield [*pre, k], v


@pytest.fixture
def stub() -> Iterator[tuple[Stub, FrigateClient]]:
    st = Stub()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self) -> None:
            st.requests.append(("GET", self.path, None))
            if self.path == "/api/version":
                self._send(200, st.version, "text/plain")
            elif self.path == "/api/config":
                self._send(200, json.dumps(st.cfg))
            elif self.path == "/api/config/raw":
                self._send(200, json.dumps(st.raw()), "text/plain")  # Frigate JSON-encodes the string
            else:
                self._send(404, "{}")

        def _body(self) -> str:
            return self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()

        def do_PUT(self) -> None:
            body = json.loads(self._body())
            st.requests.append(("PUT", self.path, body))
            if st.fail_set:
                return self._send(400, json.dumps({"success": False, "message": "invalid"}))
            if not st.drop_after_set:
                for path, v in _flat(body["config_data"], []):
                    _set_path(st.cfg, path, v)
            self._send(200, json.dumps({"success": True}))

        def do_POST(self) -> None:
            text = self._body()
            st.requests.append(("POST", self.path, text))
            st.cfg = json.loads(text)
            self._send(200, json.dumps({"success": True}))

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield st, FrigateClient(f"http://127.0.0.1:{srv.server_port}", timeout=3)
    srv.shutdown()


def zone(name: str = "rv_drive", **kw: Any) -> Change:
    return Change("zone", "front", name, SQUARE, objects=("car",), **kw)


def writes(st: Stub) -> list[tuple[str, str, Any]]:
    return [r for r in st.requests if r[0] != "GET"]


def test_client_has_no_public_write_and_default_is_readonly(stub: Any, tmp_path: Path) -> None:
    st, c = stub
    w = Writer(c)
    plan = w.plan([zone()])
    assert "+ zone front/rv_drive" in plan.diff and plan.calls[0].path == "/api/config/set"
    with pytest.raises(WriteRefused, match="read-only"):
        w.apply(plan, backup_dir=tmp_path)
    assert writes(st) == [] and not hasattr(c, "put") and not hasattr(c, "post")


def test_version_gate(stub: Any) -> None:
    st, c = stub
    st.version = "0.19.0"
    with pytest.raises(WriteRefused, match="not a supported version"):
        Writer(c).plan([zone()])
    st.version = "0.17.1"
    Writer(c).plan([zone()])  # previous stable ok for zones
    with pytest.raises(WriteRefused, match="masks are only written"):
        Writer(c).plan([Change("motion_mask", "front", "rv_m", SQUARE)])
    st.version = "0.19.0"
    Writer(c, supported=("0.19",)).plan([zone()])


def test_locked_foreign_and_bad_input(stub: Any) -> None:
    _, c = stub
    w = Writer(c)
    with pytest.raises(WriteRefused, match="marked locked"):
        w.plan([zone("porch_locked", allow_foreign=True)])
    with pytest.raises(WriteRefused, match="not created by us"):
        w.plan([zone("mine")])
    w.plan([zone("mine", allow_foreign=True)])
    with pytest.raises(WriteRefused, match="locked by the caller"):
        Writer(c, locked=("front",)).plan([zone()])
    with pytest.raises(WriteRefused, match="no such camera"):
        w.plan([Change("zone", "nope", "rv_a", SQUARE)])
    with pytest.raises(WriteRefused, match="fewer than 3"):
        w.plan([Change("zone", "front", "rv_a", SQUARE[:2])])
    with pytest.raises(WriteRefused, match="relative"):
        w.plan([Change("zone", "front", "rv_a", ((0, 0), (2, 0), (1, 1)))])


def test_locked_friendly_name_marker(stub: Any) -> None:
    st, c = stub
    st.cfg["cameras"]["front"]["zones"]["rv_x"] = {"coordinates": "0,0,1,0,1,1", "friendly_name": "Gate [LOCKED]"}
    with pytest.raises(WriteRefused, match="marked locked"):
        Writer(c).plan([zone("rv_x")])


def test_privacy_mask_needs_approval(stub: Any, tmp_path: Path) -> None:
    st, c = stub
    w = Writer(c, allow_write=True)
    plan = w.plan([Change("motion_mask", "front", "rv_neighbour", SQUARE, privacy=True)])
    assert plan.needs_approval
    with pytest.raises(WriteRefused, match="approval required"):
        w.apply(plan, backup_dir=tmp_path)
    assert writes(st) == []
    res = w.apply(plan, approved_by="owner", backup_dir=tmp_path)
    assert res.approved_by == "owner"
    assert "rv_neighbour" in st.cfg["cameras"]["front"]["motion"]["mask"]


def test_apply_backup_readback_and_rollback(stub: Any, tmp_path: Path) -> None:
    st, c = stub
    before = copy.deepcopy(st.cfg)
    w = Writer(c, allow_write=True)
    res = w.apply(w.plan([zone(), Change("object_mask", "front", "rv_sky", SQUARE)]), backup_dir=tmp_path)
    put = [r for r in writes(st) if r[0] == "PUT"]
    assert {r[2]["update_topic"] for r in put} == {"config/cameras/front/zones", "config/cameras/front/objects"}
    assert all(r[2]["requires_restart"] == 0 for r in put)
    assert st.cfg["cameras"]["front"]["zones"]["rv_drive"]["objects"] == ["car"]
    backup = Path(res.backup_path)
    assert json.loads(backup.read_text()) == before  # raw config as it was
    # handle round-trips through JSON (Kestrel stores it)
    from raptor_vision.frigate import RollbackHandle

    assert w.rollback(RollbackHandle.from_json(res.handle.to_json()))
    assert st.cfg == before
    # a second plan makes a second backup, the first is never overwritten
    res2 = w.apply(w.plan([zone()]), backup_dir=tmp_path)
    assert res2.backup_path != res.backup_path and backup.exists()


def test_modify_own_item_and_rollback_restores_prior(stub: Any, tmp_path: Path) -> None:
    st, c = stub
    st.cfg["cameras"]["front"]["zones"]["rv_drive"] = {"coordinates": "0,0,1,0,1,1", "enabled": True, "objects": ["person"]}
    w = Writer(c, allow_write=True)
    plan = w.plan([zone()])
    assert plan.diff.startswith("~ zone")
    res = w.apply(plan, backup_dir=tmp_path)
    assert st.cfg["cameras"]["front"]["zones"]["rv_drive"]["objects"] == ["car"]
    assert w.rollback(res.handle)
    assert st.cfg["cameras"]["front"]["zones"]["rv_drive"]["objects"] == ["person"]


def test_noop_plan_refused(stub: Any, tmp_path: Path) -> None:
    _, c = stub
    w = Writer(c, allow_write=True)
    w.apply(w.plan([zone()]), backup_dir=tmp_path)
    with pytest.raises(WriteRefused, match="nothing to change"):
        w.plan([zone()])


def test_stale_plan_refused(stub: Any, tmp_path: Path) -> None:
    st, c = stub
    w = Writer(c, allow_write=True)
    plan = w.plan([zone()])
    st.cfg["cameras"]["front"]["zones"]["hand_edit"] = {"coordinates": "0,0,1,0,1,1"}
    with pytest.raises(WriteRefused, match="stale"):
        w.apply(plan, backup_dir=tmp_path)
    assert writes(st) == [] and list(tmp_path.iterdir()) == []


def test_set_rejected_by_frigate_rolls_back(stub: Any, tmp_path: Path) -> None:
    st, c = stub
    st.fail_set = True
    w = Writer(c, allow_write=True)
    with pytest.raises(FrigateError, match="apply failed"):
        w.apply(w.plan([zone()]), backup_dir=tmp_path)


def test_readback_mismatch_detected(stub: Any, tmp_path: Path) -> None:
    st, c = stub
    st.drop_after_set = True
    w = Writer(c, allow_write=True)
    with pytest.raises(FrigateError, match="read-back mismatch"):
        w.apply(w.plan([zone()]), backup_dir=tmp_path)


def test_other_frigate_plan_refused(stub: Any, tmp_path: Path) -> None:
    _, c = stub
    plan = Writer(c).plan([zone()])
    other = Writer(FrigateClient("http://127.0.0.1:9"), allow_write=True)
    with pytest.raises(WriteRefused, match="different Frigate"):
        other.apply(plan, backup_dir=tmp_path)


def test_network_errors_and_token() -> None:
    w = Writer(FrigateClient("http://127.0.0.1:9", timeout=1), allow_write=True)
    with pytest.raises(FrigateError, match="network"):
        w.plan([zone()])
    with pytest.raises(ValueError):
        FrigateClient("ftp://x")
    with pytest.raises(FrigateError, match="token file"):
        FrigateClient("http://127.0.0.1:9", token_file="/nonexistent/t").version()
