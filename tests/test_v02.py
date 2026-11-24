"""v0.2: fallback chain, vote, entry pass. Stub endpoints only; no network beyond 127.0.0.1."""

from __future__ import annotations

import base64
import io
import random
from typing import Any

import pytest
from PIL import Image

from raptor_vision.entry import match_confirmed, parse_entries, propose_entries, protect
from raptor_vision.geometry import iou
from raptor_vision.label import Labeller, parse_regions, vote_regions
from raptor_vision.providers import ChatProvider, build_provider, parse_provider
from raptor_vision.tasks import parse_task
from raptor_vision.types import Frame, Region

from .conftest import Stub, chat, frame, rect


def provider(s: Stub) -> ChatProvider:
    p = build_provider("p", parse_provider("p", {"kind": "chat", "base_url": s.url, "path": "/chat/completions", "allow_frames": True}))
    assert isinstance(p, ChatProvider)
    return p


def reg(label: str, box: tuple[float, float, float, float], conf: float = 0.8) -> Region:
    return Region(label, [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])], conf, "t")


def by_model(replies: dict[str, Any]) -> Any:
    """A stub reply picking the body by the requested model; a missing model is a 500."""
    return lambda req: (200, chat(replies[req["model"]])) if req["model"] in replies else (500, {})


SKY = {"regions": [{"label": "sky", "polygon": rect(0, 0, 1, 0.4), "confidence": 0.9}]}
TASK = parse_task("t", {"mode": "model", "provider": "p", "model": "m1", "send_frames": True, "min_confidence": 0.3,
                        "fallback": [{"model": "m2"}, {"model": "m3"}]})


def test_fallback_moves_on_after_failure_and_after_empty(stub: Any) -> None:
    s: Stub = stub(by_model({"m2": {"regions": []}, "m3": SKY}))  # m1 fails (500), m2 answers with nothing, m3 draws
    out = Labeller(TASK, provider(s)).label(frame())
    assert [c["model"] for c in s.calls] == ["m1", "m2", "m3"] and [r.label for r in out] == ["sky"]
    s2: Stub = stub(by_model({"m1": SKY, "m2": SKY}))
    Labeller(TASK, provider(s2)).label(frame())
    assert [c["model"] for c in s2.calls] == ["m1"]  # the first usable answer stops the chain


def test_fallback_all_failed_is_empty_or_raises_when_strict(stub: Any) -> None:
    s: Stub = stub(by_model({}))
    assert Labeller(TASK, provider(s)).label(frame()) == [] and len(s.calls) == 3
    with pytest.raises(Exception, match=r"HTTP|status|500|error"):
        Labeller(TASK, provider(s), strict=True).label(frame())


def test_lenient_parse_rescales_and_defaults_confidence() -> None:
    out = parse_regions({"regions": [{"label": "sky", "polygon": [[0, 0], [1000, 0], [1000, 400], [0, 400]]},
                                     {"label": "yard", "polygon": [[0, 500], [640, 500], [640, 360], [0, 360]], "confidence": 85}]},
                        "x", 0.3, size=(640, 360))
    assert out[0].polygon[2] == (1.0, 0.4) and out[0].confidence == 0.5 and out[1].confidence == 0.85


def test_vote_keeps_the_majority_per_pixel() -> None:
    a = [reg("sky", (0, 0, 1, 0.5)), reg("yard", (0, 0.5, 1, 1))]
    b = [reg("sky", (0, 0, 1, 0.5)), reg("street", (0, 0.5, 1, 1))]
    c = [reg("sky", (0, 0, 1, 0.3)), reg("street", (0, 0.5, 1, 1))]
    out = {r.label: r for r in vote_regions([a, b, c], "vote:x")}
    assert set(out) == {"sky", "street"}  # yard had one vote of three
    assert iou(out["sky"].polygon, rect_poly(0, 0, 1, 0.5)) > 0.9 and iou(out["street"].polygon, rect_poly(0, 0.5, 1, 1)) > 0.9
    assert out["sky"].source == "vote:x"


def rect_poly(*b: float) -> list[tuple[float, float]]:
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def test_vote_mode_asks_every_model_and_survives_one_failure(stub: Any) -> None:
    s: Stub = stub(by_model({"a": SKY, "b": SKY}))
    out = Labeller(TASK, provider(s), vote=("a", "b", "c")).label(frame())
    assert sorted(c["model"] for c in s.calls) == ["a", "b", "c"] and [r.label for r in out] == ["sky"] and out[0].source == "vote:a+b+c"
    s1: Stub = stub(by_model({"a": SKY}))
    assert Labeller(TASK, provider(s1), vote=("a", "b")).label(frame())[0].source.startswith("chat:")  # one answer: used as is


def test_parse_entries_maps_tile_boxes_to_the_frame() -> None:
    out = parse_entries({"entries": [{"kind": "garage", "box": [100, 200, 300, 600], "confidence": 0.7},
                                     {"kind": "gate", "box": [0, 0, 1000, 1000]}, {"kind": "door", "box": [5, 5, 5, 9]},
                                     {"box": "bad"}, "junk"]}, 0.45, 1.0, "chat:x")
    assert [r.label for r in out] == ["door", "gate"] and out[0].extra["needs_confirmation"] is True and out[0].note == "confirm"
    assert out[0].polygon[0] == pytest.approx((0.45 + 0.1 * 0.55, 0.2)) and out[0].extra["kind"] == "garage"
    with pytest.raises(ValueError, match="no entries"):
        parse_entries({}, 0, 1, "x")


def test_propose_entries_uses_two_zoomed_tiles_and_dedupes(stub: Any) -> None:
    box = {"entries": [{"kind": "door", "box": [400, 300, 600, 700], "confidence": 0.8}]}
    s: Stub = stub(lambda req: (200, chat(box)))
    out = propose_entries(frame(), provider(s), "m")
    assert len(s.calls) == 2 and all(r.label == "door" and r.extra["needs_confirmation"] for r in out)
    sizes = [Image.open(io.BytesIO(base64.b64decode(c["messages"][1]["content"][1]["image_url"]["url"].split(",")[1]))).size
             for c in s.calls]
    assert sizes == [(704, 720), (704, 720)]  # 55% of 640 wide, 2x
    bad: Stub = stub(lambda req: (500, {}))
    assert propose_entries(frame(), provider(bad), "m") == []


def test_protect_grows_entries_and_keeps_the_porch() -> None:
    door = reg("door", (0.5, 0.4, 0.6, 0.7))
    porch = reg("porch", (0.4, 0.6, 0.7, 0.9))
    sky = reg("sky", (0, 0, 1, 0.3))
    out = protect([sky, porch], [door], 0.05)
    assert len(out) == 2
    assert [v for p in out[0] for v in p] == pytest.approx([v for p in rect_poly(0.45, 0.35, 0.65, 0.75) for v in p])
    assert out[1] == porch.polygon
    edge = protect([], [reg("door", (0.0, 0.9, 0.1, 1.0))], 0.05)[0]
    assert min(x for x, _ in edge) == 0.0 and max(y for _, y in edge) == 1.0  # clamped to the frame


def noisy_frame(patch: tuple[int, int], seed: int = 1, shift: int = 0) -> tuple[Frame, bytes]:
    rnd = random.Random(seed)
    img = Image.new("RGB", (640, 360), (120, 130, 125))
    tex = Image.new("L", (10, 20))  # blocky texture: 4 px cells, like real detail, so a 2x downscale keeps it
    tex.putdata([rnd.randrange(30, 230) for _ in range(10 * 20)])
    tex = tex.resize((40, 80), Image.Resampling.NEAREST)
    img.paste(tex.convert("RGB"), patch)
    out = io.BytesIO()
    img.save(out, "PNG")
    crop = io.BytesIO()
    img.crop((patch[0], patch[1], patch[0] + 40, patch[1] + 80)).save(crop, "PNG")
    jpeg = io.BytesIO()
    img.point(lambda v: min(255, v + shift)).save(jpeg, "JPEG", quality=92)
    return Frame(jpeg.getvalue(), 640, 360, "day", 0.0), crop.getvalue()


def test_template_refind_on_a_synthetic_picture() -> None:
    _, crop = noisy_frame((400, 150))
    later, _ = noisy_frame((400, 150), shift=25)  # same view, a little brighter, re-compressed
    hit = match_confirmed(later, [("door", crop)])
    assert len(hit) == 1 and hit[0].label == "door" and hit[0].source == "template"
    assert iou(hit[0].polygon, rect_poly(400 / 640, 150 / 360, 440 / 640, 230 / 360)) > 0.6
    moved, _ = noisy_frame((80, 40))
    assert match_confirmed(moved, [("door", crop)])[0].polygon[0][0] < 0.3  # follows the crop to where it is now
    blank = io.BytesIO()
    Image.new("RGB", (640, 360), (120, 130, 125)).save(blank, "JPEG")
    assert match_confirmed(Frame(blank.getvalue(), 640, 360, "day", 0.0), [("door", crop)]) == []
