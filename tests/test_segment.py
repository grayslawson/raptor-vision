"""Needs the [segment] extra (numpy). Ported from Kestrel test_scene."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from raptor_vision.refine import add_unclaimed, refine_region  # noqa: E402
from raptor_vision.segment import S  # noqa: E402
from raptor_vision.types import Region  # noqa: E402

from .conftest import ROAD, SKY_Y, TREE, rect  # noqa: E402


def truth_surface(w: int = 128, h: int = 72) -> np.ndarray[Any, Any]:  # noqa: F821
    m = np.full((h, w), S["grass"], dtype=np.uint8)
    m[: int(SKY_Y * h)] = S["sky"]
    m[int(TREE[1] * h): int(TREE[3] * h), int(TREE[0] * w):] = S["tree"]
    m[int(ROAD[1] * h):, int(ROAD[0] * w): int(ROAD[2] * w)] = S["road"]
    return m


def poly(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x, y) for x, y in rect(x0, y0, x1, y1)]  # type: ignore[misc]


def test_refinement_snaps_sky_edge_and_driveway() -> None:
    sky = refine_region(Region("sky", poly(0, 0, 1, 0.36), 0.9, "chat:x"), truth_surface())
    assert sky.refined and abs(max(y for _, y in sky.polygon) - SKY_Y) < 0.04
    drive = refine_region(Region("driveway", poly(0.15, 0.6, 0.65, 1.0), 0.8, "chat:x"), truth_surface())
    assert drive.refined and abs(min(x for x, _ in drive.polygon) - ROAD[0]) < 0.05


def test_refinement_keeps_original_when_segmenter_disagrees() -> None:
    r = refine_region(Region("sky", poly(0, 0.7, 1, 1.0), 0.9, "chat:x"), truth_surface())
    assert not r.refined and "disagrees" in r.note and r.confidence < 0.9


def test_timestamp_is_never_refined_and_wall_eaves_follows_building() -> None:
    r = refine_region(Region("timestamp", poly(0.02, 0.02, 0.3, 0.08), 0.9, "chat:x"), truth_surface())
    assert not r.refined and r.note == ""
    surface = np.full((72, 128), S["building"], dtype=np.uint8)
    w = refine_region(Region("wall_eaves", poly(0.1, 0.1, 0.5, 0.5), 0.9, "chat:x"), surface)
    assert w.refined


def test_add_unclaimed_finds_sky_the_model_missed() -> None:
    out = add_unclaimed([Region("driveway", poly(0.15, 0.6, 0.65, 1.0), 0.8, "chat:x")], truth_surface())
    assert {r.label for r in out} >= {"sky", "foliage"} and all(r.source == "segmenter" for r in out if r.label != "driveway")
