# raptor-vision

The part of Kestrel and Osprey that looks at a security-camera picture and says what is where. Give it a frame and a model
and it gives back labelled regions (driveway, door, sky, street and so on) as polygons. It never moves a camera. The one
thing it can write to an NVR is Frigate zones and masks, through the guarded writer below, and only when you switch writing on.

Used by [Kestrel](https://github.com/grayslawson/kestrel) (proposes Frigate zones and masks) and
[Osprey](https://github.com/grayslawson/osprey) (Assist and Survey, for PTZ cameras). Each product still runs on its own.

## Install

```
pip install raptor-vision            # the core needs only Pillow
pip install "raptor-vision[segment]" # adds numpy and onnxruntime for segmentation and polygon refinement
pip install "raptor-vision[entry]"   # adds transformers and torch for OWLv2 in the entry pass
```

Python 3.11 or newer.

## The contract in short

- `Frame(jpeg, width, height, kind, ts, key="")`: one picture. `kind` is `day` or `night`. `key` is opaque to the library.
- `Region(label, polygon, confidence, source, refined, note, extra)`: a polygon in 0..1 of the frame.
- `Labeller(task, provider).label(frame, labels=..., context="") -> list[Region]`. It never raises on a model failure: it
  returns what the rules found (pass `strict=True` to raise `ProviderError` instead).
- For a chat provider, `label()` draws whole-frame polygons (the default). `TaskConfig.fallback` lists models to try in order
  when a call fails or returns no usable region; `Labeller(..., vote=("a", "b", "c"))` takes a per-pixel majority of 2 or 3
  models (a trial). A tile `grid` and decision providers stay available, for naming or voting on a region rather than drawing.
- `raptor_vision.entry`: `propose_entries(frame, provider, model)` proposes doors and gates (each `Region` has
  `extra["needs_confirmation"]` and note `confirm`) for a person to confirm once; `match_confirmed(frame, crops)` finds the
  confirmed ones again later; `protect(regions, entries, margin)` returns the area masks must never cover, porch included.
- One vocabulary of 17 labels in `raptor_vision.labels`; each product passes the subset it uses. `wall_eaves` is an alias of
  `building`.
- `ProviderConfig` and `TaskConfig` are plain config; `interest(...)` scores a region 0..1; `MapStore` keeps a JSON body plus
  named files per camera and kind, newest few only.
- With `[segment]`: `raptor_vision.segment.OnnxSegmenter` and `raptor_vision.refine` snap model polygons to real edges.

## Writing zones and masks to Frigate

`raptor_vision.frigate` is the one place that changes a Frigate config, so nobody ever has to copy and paste YAML. It is
stdlib only and read-only until you pass `allow_write=True`.

```python
from raptor_vision.frigate import Change, FrigateClient, Writer

w = Writer(FrigateClient("http://frigate:5000", token_file="/run/secrets/frigate"), allow_write=True)
plan = w.plan([Change("zone", "front", "rv_driveway", [(0.1, 0.5), (0.9, 0.5), (0.9, 0.95), (0.1, 0.95)], objects=("car",))])
print(plan.diff)                                   # show this to the person
result = w.apply(plan, approved_by="grays", backup_dir="/data/frigate-backups")
w.rollback(result.handle)                          # if they change their mind
```

It checks the Frigate version, refuses locked or hand-made items, backs up the config, refuses a stale plan, reads the result
back, and can roll back. Privacy masks always need `approved_by`. Only one writer per Frigate: Kestrel when present, else
Osprey. Details in [docs/frigate-writer.md](docs/frigate-writer.md).

## History

The contract was written in Osprey's [docs/survey.md, "Scene layer"](https://github.com/grayslawson/osprey/blob/main/docs/survey.md#scene-layer).

MIT licence. See NOTICE for where the code came from.

## Choosing models

No model name is built into the library: you name the model per task. The November 2026 bake-offs
([numbers](docs/bakeoff-2026-11.md)) found that one whole-frame polygon call from a cheap vision model beats every tile grid,
box detector and segmenter design, at about $0.0001 a frame, and that a second and third model are worth keeping as fallbacks.

Recommended task configs (model first, then `fallback` in order; the names are gateway aliases, use your own):

| use | model | fallback 1 | fallback 2 |
|---|---|---|---|
| fixed camera (Kestrel): best key-zone recall, least false area | glm-5.3-flash | hy4-flash | ling-3.0-flash-vl |
| PTZ survey (Osprey): best shapes on PTZ and night frames | hy4-flash | glm-5.3-flash | ling-3.0-flash-vl |

```python
task = parse_task("survey.label", {"mode": "model", "provider": "gw", "model": "hy4-flash", "send_frames": True,
                                   "min_confidence": 0.3,
                                   "fallback": [{"model": "glm-5.3-flash"}, {"model": "ling-3.0-flash-vl"}]})
regions = Labeller(task, provider).label(frame)
```

- Do not use tile grids to draw. Use a decision model only to name or vote on a region you already drew.
- Segmenter refine gave no gain on the strong models; leave it off.
- Vote (`vote=("hy4-flash", "glm-5.3-flash", "ling-3.0-flash-vl")`) scored best (0.603 mIoU against 0.577 for the best single
  model) but the gain is inside the noise of 24 frames and it triples the calls. Try it; do not rely on it.
- Doors are not drawn by the labeller. Run the entry pass once per camera view, have a person confirm, then `protect`.
