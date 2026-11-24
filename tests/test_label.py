"""The labeller (Osprey's tile call, Kestrel's label_with_decider and label_with_chat, as one)."""

from __future__ import annotations

from typing import Any

import pytest

from raptor_vision.label import DEFAULT_LABELS, Labeller, parse_regions
from raptor_vision.providers import ProviderError, build_provider, parse_provider
from raptor_vision.tasks import TaskConfig
from raptor_vision.types import Region

from .conftest import CHAT_REGIONS, Stub, chat, choice, frame, rect

TASK = TaskConfig(mode="model", provider="p", model="m1", send_frames=True)


def provider(s: Stub, kind: str = "decisions", **raw: Any) -> Any:
    path = "/decisions" if kind == "decisions" else "/chat/completions"
    return build_provider("p", parse_provider("p", {"kind": kind, "base_url": s.url, "path": path, "allow_frames": True, **raw}))


def test_decider_labels_tiles_and_merges_regions(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, choice("sky", 0.9)))
    regions = Labeller(TASK, provider(s)).label(frame())
    assert len(regions) == 1 and regions[0].label == "sky" and regions[0].polygon[0] == (0.0, 0.0)
    assert regions[0].source == "decider:d" and regions[0].extra == {"entry": 0.0, "closer": 0.0}
    req = s.calls[0]
    assert len(s.calls) == 48 and req["questions"]["kind"]["type"] == "choice" and req["state"][1]["type"] == "image_url"
    assert set(req["questions"]) == {"kind", "entry", "closer"} and "wall_eaves" not in req["questions"]["kind"]["criteria"]


def test_labels_subset_and_context_reach_the_model(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, choice("sky", 0.9)))
    Labeller(TASK, provider(s), grid=(2, 2)).label(frame(), labels=("sky", "street"), context="Azimuth -80")
    assert set(s.calls[0]["questions"]["kind"]["criteria"]) == {"sky", "street"}
    assert "Azimuth -80" in s.calls[0]["state"][0] and "(day)" in s.calls[0]["state"][0]


def test_a_failing_model_gives_the_rules_floor_not_an_exception(stub: Any) -> None:
    s: Stub = stub(lambda req: (500, {"error": "boom"}))
    regions = Labeller(TASK, provider(s)).label(frame())
    assert regions and {r.source for r in regions} == {"rules"} and "sky" in {r.label for r in regions}
    assert Labeller(TASK, provider(s), rules_floor=False).label(frame()) == []


def test_strict_raises_when_most_tiles_fail(stub: Any) -> None:
    s: Stub = stub(lambda req: (500, {"error": "boom"}))
    with pytest.raises(ProviderError, match="most tile questions failed"):
        Labeller(TASK, provider(s), strict=True, rules_floor=False).label(frame())


def test_garbage_answers_are_dropped(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, choice("lava", 0.9)))
    assert all(r.source == "rules" for r in Labeller(TASK, provider(s)).label(frame()))
    s2: Stub = stub(lambda req: (200, {"answers": "nope"}))
    assert all(r.source == "rules" for r in Labeller(TASK, provider(s2)).label(frame()))


def test_the_model_is_not_called_unless_the_task_allows_it(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, choice("street")))
    p = provider(s)
    for task in (TaskConfig(mode="rules"), TaskConfig(mode="model", provider="p", model="m", send_frames=False)):
        assert all(r.source == "rules" for r in Labeller(task, p).label(frame()))
    nopic = provider(s, allow_frames=False)
    assert all(r.source == "rules" for r in Labeller(TASK, nopic).label(frame()))
    assert all(r.source == "rules" for r in Labeller(TASK, None).label(frame()))
    assert s.calls == []


def test_a_sure_enough_answer_only(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, choice("street", 0.5, entry=0.8)))
    lab = Labeller(TASK, provider(s), grid=(4, 3), merge=False)
    regions = lab.label(frame())
    street = [r for r in regions if r.label == "street"]
    assert not street  # 0.5 is under min_confidence 0.7
    unknown = [r for r in regions if r.label == "unknown"]
    assert unknown and unknown[0].extra["entry"] == 0.8 and unknown[0].confidence == 0.5
    lab2 = Labeller(TaskConfig(mode="model", provider="p", model="m1", send_frames=True, min_confidence=0.4), provider(s),
                    grid=(4, 3), merge=False)
    tiles = [r for r in lab2.label(frame()) if r.label == "street"]
    assert len(tiles) == 12 and tiles[0].polygon == [(0.0, 0.0), (0.25, 0.0), (0.25, 1 / 3), (0.0, 1 / 3)]


def test_calls_stop_at_the_providers_budget_and_the_rest_is_rules(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, choice("street", 0.9)))
    regions = Labeller(TASK, provider(s, max_calls_per_hour=5), grid=(4, 3), merge=False).label(frame())
    assert len(s.calls) == 5 and sum(r.source == "decider:d" for r in regions) == 5
    assert {r.source for r in regions} <= {"decider:d", "rules"}


def test_chat_whole_picture(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, chat(CHAT_REGIONS)))
    task = TaskConfig(mode="model", provider="p", model="m1", send_frames=True, min_confidence=0.3)
    regions = Labeller(task, provider(s, "chat")).label(frame(), labels=("sky", "driveway", "foliage"), context="front door camera")
    assert [r.label for r in regions] == ["sky", "driveway", "foliage"] and regions[0].source == "chat:stub"
    assert len(s.calls) == 1 and "front door camera" in s.calls[0]["messages"][1]["content"][0]["text"]


def test_chat_failure_returns_nothing_unless_strict(stub: Any) -> None:
    s: Stub = stub(lambda req: (500, {}))
    assert Labeller(TASK, provider(s, "chat")).label(frame()) == []
    with pytest.raises(ProviderError):
        Labeller(TASK, provider(s, "chat"), strict=True).label(frame())
    s2: Stub = stub(lambda req: (200, chat({"nothing": 1})))
    assert Labeller(TASK, provider(s2, "chat")).label(frame()) == []
    with pytest.raises(ValueError, match="no regions list"):
        Labeller(TASK, provider(s2, "chat"), strict=True).label(frame())


def test_chat_tile_by_tile(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, chat({"label": "sky", "confidence": 0.9, "entry": 0.1, "closer": 0.2})))
    regions = Labeller(TASK, provider(s, "chat"), grid=(2, 2)).label(frame())
    assert len(s.calls) == 4 and len(regions) == 1 and regions[0].extra == {"entry": 0.1, "closer": 0.2}
    assert regions[0].source == "chat:stub"


def test_every_model_call_is_logged(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, choice("sky", 0.9)))
    recs: list[dict[str, Any]] = []
    Labeller(TASK, provider(s), name="survey.label", grid=(2, 2), log=recs.append).label(frame())
    assert len(recs) == 4 and all(r["ok"] and r["task"] == "survey.label" and r["model"] == "m1" for r in recs)


def test_bad_label_subsets_are_refused() -> None:
    lab = Labeller(TaskConfig(), None)
    with pytest.raises(ValueError, match="one label"):
        lab.label(frame(), labels=("building", "wall_eaves"))
    with pytest.raises(ValueError, match="unknown label"):
        lab.label(frame(), labels=("lava",))
    assert "wall_eaves" not in DEFAULT_LABELS and "building" in DEFAULT_LABELS


def test_parse_regions_drops_bad_entries() -> None:
    bad = {"regions": [
        {"label": "lava", "polygon": rect(0, 0, 1, 1), "confidence": 0.9},
        {"label": "sky", "polygon": [[0, 0], [1, 1]], "confidence": 0.9},
        {"label": "sky", "polygon": rect(0, 0, 1, 0.3), "confidence": 0.1},
        {"label": "sky", "polygon": "nope", "confidence": 0.9},
        {"label": "sky", "polygon": [[0, 0], [2.5, 0], [1, 0.3]], "confidence": 0.9},  # clamped, still valid
        {"label": "yard", "polygon": rect(0, 0.5, 1, 1), "confidence": True},
    ]}
    out = parse_regions(bad, "chat:x")
    assert [r.label for r in out] == ["sky"] and max(x for x, _ in out[0].polygon) == 1.0
    with pytest.raises(ValueError):
        parse_regions({}, "chat:x")


def test_parse_regions_honours_label_subset() -> None:
    out = parse_regions(CHAT_REGIONS, "chat:x", labels=("sky", "driveway"))
    assert {r.label for r in out} == {"sky", "driveway"}
    assert isinstance(out[0], Region)
