"""Ported from Kestrel test_scene; masks are lists of rows, so no numpy is needed."""

from __future__ import annotations

from raptor_vision import geometry as g


def mask(h: int, w: int, *boxes: tuple[int, int, int, int], hole: tuple[int, int, int, int] | None = None) -> list[list[bool]]:
    m = [[False] * w for _ in range(h)]
    for y0, y1, x0, x1 in boxes:
        for y in range(y0, y1):
            for x in range(x0, x1):
                m[y][x] = True
    if hole:
        for y in range(hole[0], hole[1]):
            for x in range(hole[2], hole[3]):
                m[y][x] = False
    return m


def test_coords_roundtrip() -> None:
    poly = [(0.1, 0.2), (0.5, 0.2), (0.5, 0.9)]
    assert g.fmt_coords(poly) == "0.1,0.2,0.5,0.2,0.5,0.9"
    assert g.parse_coords(g.fmt_coords(poly)) == poly


def test_mask_to_polygon_rect_and_hole_ignored() -> None:
    (p,) = g.mask_to_polygons(mask(20, 40, (5, 15, 10, 30), hole=(8, 10, 14, 16)))
    assert g.area(p) == 0.25 and len(p) == 4


def test_mask_to_polygon_l_shape_and_small_blob_dropped() -> None:
    polys = g.mask_to_polygons(mask(40, 40, (10, 30, 10, 15), (25, 30, 10, 30), (0, 2, 0, 2)))
    assert len(polys) == 1 and len(polys[0]) == 6


def test_empty_mask() -> None:
    assert g.mask_to_polygons([]) == []


def test_validity_rejects_degenerate_and_out_of_frame() -> None:
    assert not g.is_valid([(0, 0), (1, 1)])
    assert not g.is_valid([(0, 0), (1, 0), (2, 1)])
    assert not g.is_valid([(0, 0), (0.5, 0.5), (1, 1)])  # no area
    assert g.is_valid([(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)])


def test_coverage_overlap_and_iou() -> None:
    a, b = [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], [(0.25, 0), (0.75, 0), (0.75, 1), (0.25, 1)]
    assert abs(g.coverage([a, b]) - 0.75) < 0.03
    assert 0.3 < g.iou(a, b) < 0.37
    assert 0.45 < g.overlap_fraction(a, b) < 0.55
    assert g.coverage([]) == 0.0
