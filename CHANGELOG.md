# Changelog

## 0.2.0 (unreleased)

The labeller after the 2026-11 bake-offs (docs/bakeoff-2026-11.md).

- `label`: a chat provider draws whole-frame polygons by default; `TaskConfig.fallback` models are tried in order when a call
  fails or returns no usable region (`strict` raises only when every model failed). Decision models and tile grids are for
  naming or voting on a region.
- `label`: `Labeller(vote=...)` and `vote_regions`: per-pixel majority of 2 or 3 models (trial, off by default).
- `label`: `parse_regions` accepts a missing confidence (0.5), 0..1000 or pixel coordinates, and percent confidences.
- `entry` (new): `propose_entries`, `match_confirmed` (Pillow template match), `protect`. Optional `[entry]` extra
  (transformers, torch) adds OWLv2 zero-shot and image-guided detection.
- Docs: "Choosing models" in the README with the recommended task configs.


- `frigate`: the shared, safe Frigate config writer (`FrigateClient`, `Writer`, `Change`). Read-only by default. See docs/frigate-writer.md.

## 0.1.0 (unreleased)

First release: the scene layer, merged from Osprey Assist and Kestrel. No consumer has switched to it yet.

- `types`: `Frame`, `Region`.
- `providers`: `ProviderConfig` (all keys of both products, including `private` and `extra_body`), chat and decisions
  providers, call budget, per-model cooldown.
- `tasks`: `TaskConfig` and `parse_task`.
- `labels`: one 17-label vocabulary, shared criteria sentences, `wall_eaves` as an alias of `building`.
- `label`: `Labeller` (tile/decider labeller and chat whole-picture labeller, rules floor, never raises on a model failure).
- `geometry`, `interest`, `mapstore`.
- Optional `[segment]` extra: `segment` and `refine` (numpy, onnxruntime).
