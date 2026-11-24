"""CPU semantic segmentation (ONNX) reduced to a handful of surface groups. Needs the ``[segment]`` extra.

The model is not bundled: point ``OnnxSegmenter`` at an ADE20K SegFormer export. Its licence is the user's to accept.
Without a model, callers simply skip refinement and regions stay unrefined.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

SURFACES = (
    "other", "road", "sidewalk", "path", "ground", "floor", "grass", "plant", "tree", "sky",
    "building", "door", "fence", "vehicle", "person", "water",
)
S = {n: i for i, n in enumerate(SURFACES)}

# ADE20K class name (first synonym) -> surface group. Anything not listed is "other".
ADE_TO_SURFACE = {
    "road": "road", "sidewalk": "sidewalk", "path": "path", "runway": "road", "earth": "ground", "sand": "ground",
    "dirt track": "ground", "floor": "floor", "rug": "floor", "grass": "grass", "field": "grass", "plant": "plant",
    "flower": "plant", "tree": "tree", "palm": "tree", "sky": "sky", "building": "building", "house": "building",
    "wall": "building", "skyscraper": "building", "hovel": "building", "door": "door", "fence": "fence",
    "railing": "fence", "car": "vehicle", "bus": "vehicle", "truck": "vehicle", "van": "vehicle", "person": "person",
    "water": "water", "sea": "water", "river": "water", "swimming pool": "water", "pool": "water",
}


class Segmenter(Protocol):
    def segment(self, jpeg: bytes) -> np.ndarray:
        """Return a (H, W) uint8 array of SURFACES indices, covering the whole picture."""
        ...


class OnnxSegmenter:
    def __init__(self, model: str | Path, labels: str | Path | None = None, size: int = 512) -> None:
        import onnxruntime as ort  # optional dependency, imported late

        model = Path(model)
        labels = Path(labels) if labels else model.with_name("config.json")
        id2label = json.loads(labels.read_text())["id2label"]
        self.lut = np.zeros(max(int(k) for k in id2label) + 1, dtype=np.uint8)
        for k, name in id2label.items():
            self.lut[int(k)] = S[ADE_TO_SURFACE.get(str(name).split(",")[0].strip().lower(), "other")]
        self.size = size
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def segment(self, jpeg: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        logits = self.sess.run(None, {self.input: x.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        return self.lut[np.argmax(logits[0], axis=0)]
