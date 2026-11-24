"""The core must import and label with numpy and onnxruntime absent (Osprey ships stdlib and Pillow only)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

CODE = textwrap.dedent(
    """
    import sys

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("numpy", "onnxruntime"):
                raise ImportError(f"{name} is blocked")

    sys.meta_path.insert(0, Block())
    import raptor_vision, raptor_vision.types, raptor_vision.providers, raptor_vision.tasks, raptor_vision.labels
    import raptor_vision.label, raptor_vision.entry, raptor_vision.geometry, raptor_vision.interest, raptor_vision.mapstore
    from PIL import Image
    import io
    from raptor_vision.label import Labeller
    from raptor_vision.tasks import TaskConfig
    from raptor_vision.types import Frame

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (100, 140, 200)).save(buf, "JPEG")
    regions = Labeller(TaskConfig(), None).label(Frame(buf.getvalue(), 64, 48, "day", 0.0))
    assert [r.label for r in regions] == ["sky"], regions
    assert not any(m.split(".")[0] in ("numpy", "onnxruntime") for m in sys.modules)
    try:
        import raptor_vision.refine
    except ImportError:
        print("ok")
    """
)


def test_core_imports_and_labels_without_numpy() -> None:
    done = subprocess.run([sys.executable, "-I", "-c", CODE], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
