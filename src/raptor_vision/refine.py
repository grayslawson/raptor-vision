"""Needs the ``[segment]`` extra. Cascade step two: snap the model's rough polygons to real surface edges with the CPU segmentation map.

The language model proposes a region; the segmenter says which pixels are the right kind of surface.
Where the two disagree, the original polygon is kept and flagged. Nothing here calls a network.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .geometry import area, mask_to_polygons, rasterize
from .labels import canonical
from .segment import S
from .types import Region

# Which surface groups can plausibly be each label. A label missing here is never refined.
COMPAT: dict[str, tuple[str, ...]] = {
    "driveway": ("road", "path", "ground", "floor"),
    "street": ("road",),
    "sidewalk": ("sidewalk", "path"),
    "yard": ("grass", "ground", "plant"),
    "sky": ("sky",),
    "foliage": ("tree", "plant"),
    "porch": ("floor", "ground", "building", "path"),
    "door": ("door", "building"),
    "building": ("building", "door"),
    "neighbour": ("building", "fence", "grass", "tree", "plant", "ground", "door"),
}
MIN_AGREEMENT = 0.25  # below this the segmenter and the model disagree: keep the model's polygon
MAX_SHRINK = 0.3  # a refined polygon smaller than this share of the original is an over-erosion: keep the original


def _morph(mask: np.ndarray, size: int, grow: bool) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(size) if grow else ImageFilter.MinFilter(size))
    return np.asarray(img) > 127


def refine_region(region: Region, surface: np.ndarray, reach: float = 0.06) -> Region:
    groups = COMPAT.get(canonical(region.label))
    if groups is None:
        return region
    h, w = surface.shape
    inside = np.asarray(rasterize(region.polygon, w, h), dtype=bool)
    if not inside.any():
        return region
    ok = np.isin(surface, [S[g] for g in groups])
    agree = float(np.count_nonzero(inside & ok)) / float(np.count_nonzero(inside))
    if agree < MIN_AGREEMENT:
        region.note = f"segmenter disagrees ({agree:.0%} of the area is the expected surface)"
        region.confidence = round(region.confidence * 0.6, 3)
        return region
    k = max(3, int(reach * w) | 1)
    cand = _morph(inside, k, grow=True) & ok
    cand = _morph(_morph(cand, 3, grow=False), 3, grow=True)  # drop specks
    polys = mask_to_polygons(cand.tolist(), max_n=1)
    if not polys or area(polys[0]) < MAX_SHRINK * area(region.polygon):
        region.note = "refinement discarded (too little left)"
        return region
    region.polygon, region.refined = polys[0], True
    region.note = f"snapped to segmentation ({agree:.0%} agreement)"
    return region


def refine_all(regions: list[Region], surface: np.ndarray) -> list[Region]:
    return [refine_region(r, surface) for r in regions]


def add_unclaimed(regions: list[Region], surface: np.ndarray, min_share: float = 0.04) -> list[Region]:
    """Large sky or foliage areas that the model missed come from the segmenter alone, at low confidence."""
    out = list(regions)
    for label, groups in (("sky", ("sky",)), ("foliage", ("tree",))):
        if any(r.label == label for r in regions):
            continue
        m = np.isin(surface, [S[g] for g in groups])
        if m.mean() < min_share:
            continue
        for poly in mask_to_polygons(m.tolist(), max_n=2):
            out.append(Region(label, poly, 0.5, "segmenter", refined=True, note="found by segmentation only"))
    return out
