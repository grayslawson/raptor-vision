from __future__ import annotations

from pathlib import Path

import pytest

from raptor_vision.interest import PRIOR, interest
from raptor_vision.labels import ALIASES, COLORS, CRITERIA, LABELS, UNKNOWN, canonical, check_labels
from raptor_vision.mapstore import MapStore


def test_one_vocabulary_of_seventeen_labels() -> None:
    assert len(LABELS) == 17 and len(set(LABELS)) == 17
    assert set(CRITERIA) == set(LABELS) and set(COLORS) == {*LABELS, UNKNOWN}
    assert set(PRIOR) == {*(k for k in LABELS if k not in ALIASES), UNKNOWN}


def test_wall_eaves_is_an_alias_of_building() -> None:
    assert canonical("wall_eaves") == "building" and canonical("door") == "door"
    assert CRITERIA["wall_eaves"] == CRITERIA["building"]
    assert interest("wall_eaves", 0, None, False) == interest("building", 0, None, False)
    check_labels(["wall_eaves", "door"])
    with pytest.raises(ValueError):
        check_labels(["wall_eaves", "building"])


def test_interest_weights_and_absent_terms() -> None:
    assert interest("door", 1.0, 1.0, True) == 1.0
    assert interest("sky", 1.0, 1.0, False) == 0.0 + round(0.3 * 1.0 / 0.75, 4)  # prior 0, heat 1, no model: 0.3 / 0.75
    assert interest("driveway", 0, None, False) == 0.8
    assert interest("driveway", 0, 0.0, True) == round(0.45 * 0.8 / 1.0, 4)
    assert interest("lava", 0, None, False) == PRIOR["other"]


def test_map_store_put_latest_files_and_retention(tmp_path: Path) -> None:
    now = [100.0]
    st = MapStore(tmp_path / "maps", keep=2, clock=lambda: now[0])
    assert st.latest("front", "day") is None
    first = st.put("front", "day", {"regions": [1]}, {"panorama.jpg": b"P1", "labels.png": b"L1"})
    assert first["id"] == 1 and first["ts"] == 100.0 and first["regions"] == [1]
    now[0] = 200.0
    st.put("front", "day", {"regions": [2]}, {"panorama.jpg": b"P2"})
    st.put("front", "night", {"regions": ["n"]})
    third = st.put("front", "day", {"regions": [3]}, {"panorama.jpg": b"P3"})
    latest = st.latest("front", "day")
    assert latest == third and latest["id"] == 3
    assert st.read_file(latest, "panorama.jpg") == b"P3"
    assert st.read_file(first, "panorama.jpg") is None  # beyond keep=2
    assert st.latest("front", "night") is not None  # another kind has its own count
    assert st.read_file(latest, "missing.png") is None and st.read_file(latest, "map.json") is None


def test_map_store_refuses_path_tricks(tmp_path: Path) -> None:
    st = MapStore(tmp_path)
    for bad in ("../x", "a/b", "..", ""):
        with pytest.raises(ValueError):
            st.put("front", "day", {}, {bad: b"x"})
    with pytest.raises(ValueError):
        st.put("../evil", "day", {})
    rec = st.put("front", "day", {}, {"a.jpg": b"x"})
    assert st.read_file(rec, "../../etc/passwd") is None and not list(tmp_path.parent.glob("evil*"))
    with pytest.raises(ValueError):
        st.put("front", "day", {}, {"map.json": b"x"})
