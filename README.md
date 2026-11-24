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
```

Python 3.11 or newer.

## The contract in short

- `Frame(jpeg, width, height, kind, ts, key="")`: one picture. `kind` is `day` or `night`. `key` is opaque to the library.
- `Region(label, polygon, confidence, source, refined, note, extra)`: a polygon in 0..1 of the frame.
- `Labeller(task, provider).label(frame, labels=..., context="") -> list[Region]`. It never raises on a model failure: it
  returns what the rules found (pass `strict=True` to raise `ProviderError` instead).
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
