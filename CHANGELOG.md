# Changelog

## Unreleased

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
