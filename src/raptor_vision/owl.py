"""OWLv2 (Apache-2.0) for the entry pass. Needs the ``[entry]`` extra: ``pip install "raptor-vision[entry]"``. CPU, 2-6 s a call."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from PIL import Image

from .geometry import clamp

MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a door", "a gate", "a garage door", "an entrance"]
Box = tuple[float, float, float, float]


@lru_cache(maxsize=1)
def _load(name: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError as exc:
        raise ImportError('OWLv2 needs the extra: pip install "raptor-vision[entry]"') from exc
    return torch, Owlv2Processor.from_pretrained(name), Owlv2ForObjectDetection.from_pretrained(name).eval()


def _rel(boxes: Any, w: int, h: int) -> list[Box]:
    # OWLv2 pads to a square at the bottom/right, so pixel boxes divide by the real width and height
    return [(clamp(b[0] / w), clamp(b[1] / h), clamp(b[2] / w), clamp(b[3] / h)) for b in boxes.tolist()]


def zero_shot(img: Image.Image, tiles: Sequence[tuple[float, float]], *, top: int = 3, threshold: float = 0.03,
              model: str = MODEL) -> list[tuple[str, Box, float]]:
    """Top ``top`` door/gate boxes per tile (x range given as fractions of ``img``), as (label, box, score) in frame fractions."""
    torch, proc, net = _load(model)
    out: list[tuple[str, Box, float]] = []
    for x0, x1 in tiles:
        tile = img.crop((round(x0 * img.width), 0, round(x1 * img.width), img.height))
        s = max(tile.size)
        with torch.no_grad():
            res = proc.post_process_grounded_object_detection(
                net(**proc(text=[QUERIES], images=tile, return_tensors="pt")), threshold=threshold, target_sizes=torch.tensor([[s, s]]))[0]
        hits = sorted(zip(_rel(res["boxes"], tile.width, tile.height), res["scores"].tolist(), res["labels"].tolist(), strict=False),
                      key=lambda t: -t[1])[:top]
        out += [("gate" if "gate" in QUERIES[lab] else "door", (x0 + b[0] * (x1 - x0), b[1], x0 + b[2] * (x1 - x0), b[3]), sc)
                for b, sc, lab in hits]
    return out


def image_guided(img: Image.Image, crop: Image.Image, *, top: int = 3, model: str = MODEL) -> list[tuple[Box, float]]:
    """Boxes in ``img`` that look like ``crop``, best first, in frame fractions. Filter by size: the best box can be a bush."""
    torch, proc, net = _load(model)
    s = max(img.size)
    with torch.no_grad():
        res = proc.post_process_image_guided_detection(
            net.image_guided_detection(**proc(images=img, query_images=crop, return_tensors="pt")),
            threshold=0.0, nms_threshold=0.3, target_sizes=torch.tensor([[s, s]]))[0]
    boxes = _rel(res["boxes"], img.width, img.height)
    return sorted(zip(boxes, res["scores"].tolist(), strict=False), key=lambda t: -t[1])[:top]
