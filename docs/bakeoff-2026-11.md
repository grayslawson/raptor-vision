# Bake-off results, November 2026

Two small studies chose the defaults in this release. Both used 24 outdoor frames from one home (8 from a fixed
wide-angle camera, 16 from a PTZ camera), cheap vision models through an OpenAI-compatible gateway, and ground truth drawn
by hand by the author. Treat differences under about 0.03 mIoU as noise. Frames are not published.

## Region drawing

Mean IoU per class over the classes present, averaged over frames (a missing class scores 0). Key-zone recall: street,
door and porch found at IoU >= 0.4. False area: share of the frame labelled with something the ground truth does not have
there. `chat_<model>` is one whole-frame polygon call; `+refine` adds the `[segment]` edge snapping; `dec*`/`chk*` are tile
grids of decision models; `seg*` is a segmenter that draws with a model naming; `box_*` is a box detector. Real $ is the
provider cost per frame.

| # | method | mIoU | key-zone recall | false area | fixed | PTZ | day | night | real $/frame | booked $/frame | s/frame | calls/frame | failed frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `chat_hy4-flash` | 0.577 | 0.43 | 0.33 | 0.287 | 0.723 | 0.615 | 0.515 | 0.00013 | 0.00013 | 9.4 | 1 | 0 |
| 2 | `chat_hy4-flash+refine` | 0.576 | 0.41 | 0.32 | 0.288 | 0.720 | 0.626 | 0.492 | 0.00013 | 0.00013 | 9.4 | 1 | 0 |
| 3 | `chat_glm-5.3-flash+refine` | 0.564 | 0.62 | 0.20 | 0.314 | 0.689 | 0.635 | 0.445 | 0.00009 | 0.00036 | 7.3 | 1 | 0 |
| 4 | `chat_glm-5.3-flash` | 0.561 | 0.68 | 0.22 | 0.304 | 0.690 | 0.629 | 0.448 | 0.00009 | 0.00036 | 7.3 | 1 | 0 |
| 5 | `chat_ling-3.0-flash-vl+refine` | 0.547 | 0.54 | 0.29 | 0.301 | 0.670 | 0.599 | 0.459 | 0.00013 | 0.00013 | 6.2 | 1 | 0 |
| 6 | `chat_deepseek-v4.1-flash+refine` | 0.526 | 0.38 | 0.29 | 0.252 | 0.663 | 0.583 | 0.430 | 0.00011 | 0.00011 | 4.3 | 1 | 0 |
| 7 | `chat_ling-3.0-flash-vl` | 0.524 | 0.54 | 0.32 | 0.289 | 0.642 | 0.564 | 0.457 | 0.00013 | 0.00013 | 6.2 | 1 | 0 |
| 8 | `chat_deepseek-v4.1-flash` | 0.500 | 0.27 | 0.33 | 0.241 | 0.630 | 0.546 | 0.425 | 0.00011 | 0.00011 | 4.3 | 1 | 0 |
| 9 | `chat_qwen3.7-flash+refine` | 0.487 | 0.46 | 0.16 | 0.229 | 0.615 | 0.533 | 0.411 | 0.00017 | 0.00017 | 11.1 | 1 | 0 |
| 10 | `chat_gemma-4-26b+refine` | 0.482 | 0.41 | 0.34 | 0.262 | 0.591 | 0.527 | 0.405 | 0.00011 | 0.00018 | 6.0 | 1 | 0 |
| 11 | `chat_gemma-4-26b` | 0.481 | 0.41 | 0.40 | 0.270 | 0.586 | 0.509 | 0.434 | 0.00011 | 0.00018 | 6.0 | 1 | 0 |
| 12 | `chat_qwen3.7-flash` | 0.466 | 0.41 | 0.15 | 0.216 | 0.592 | 0.502 | 0.407 | 0.00017 | 0.00017 | 11.1 | 1 | 0 |
| 13 | `chk_glm-5.3-flash-respan_span-02` | 0.464 | 0.38 | 0.17 | 0.266 | 0.562 | 0.507 | 0.391 | 0.00039 | 0.00395 | 13.7 | 7 | 0 |
| 14 | `chk_hy4-flash-respan_span-02` | 0.460 | 0.16 | 0.26 | 0.226 | 0.577 | 0.499 | 0.395 | 0.00017 | 0.00561 | 22.2 | 11 | 0 |
| 15 | `chat_mistral-small-3.2+refine` | 0.356 | 0.16 | 0.35 | 0.168 | 0.450 | 0.379 | 0.319 | 0.00019 | 0.00019 | 12.0 | 1 | 0 |
| 16 | `seg0_segformer-nolabelmodel` | 0.343 | 0.19 | 0.31 | 0.144 | 0.443 | 0.446 | 0.172 | 0.00000 | 0.00000 | 0.6 | 1 | 0 |
| 17 | `box_typesafe_jev-1.14` | 0.339 | 0.51 | 0.25 | 0.207 | 0.405 | 0.390 | 0.254 | 0.00008 | 0.00050 | 1.3 | 1 | 0 |
| 18 | `chat_mistral-small-3.2` | 0.337 | 0.16 | 0.41 | 0.156 | 0.427 | 0.352 | 0.311 | 0.00019 | 0.00019 | 12.0 | 1 | 0 |
| 19 | `seg0_evit-nolabelmodel` | 0.321 | 0.11 | 0.32 | 0.123 | 0.419 | 0.417 | 0.160 | 0.00000 | 0.00000 | 0.5 | 1 | 0 |
| 20 | `dec_typesafe_jev-1.13+refine` | 0.308 | 0.16 | 0.26 | 0.229 | 0.348 | 0.364 | 0.216 | 0.00118 | 0.01200 | 20.7 | 24 | 0 |
| 21 | `dec_perplexity_pplx-decider-v1.3-free+refine` | 0.291 | 0.19 | 0.14 | 0.241 | 0.316 | 0.358 | 0.179 | 0.00000 | 0.01200 | 21.6 | 24 | 0 |
| 22 | `dec2_typesafe_jev-1.13` | 0.290 | 0.19 | 0.34 | 0.232 | 0.318 | 0.356 | 0.179 | 0.00477 | 0.05008 | 68.4 | 100 | 0 |
| 23 | `dec_typesafe_jev-1.13` | 0.283 | 0.08 | 0.33 | 0.197 | 0.326 | 0.333 | 0.200 | 0.00118 | 0.01200 | 20.7 | 24 | 0 |
| 24 | `dec_respan_span-02+refine` | 0.278 | 0.32 | 0.26 | 0.219 | 0.308 | 0.337 | 0.180 | 0.00013 | 0.01200 | 26.5 | 24 | 0 |
| 25 | `dec_perplexity_pplx-decider-v1.3-free` | 0.267 | 0.00 | 0.21 | 0.203 | 0.299 | 0.330 | 0.162 | 0.00000 | 0.01200 | 21.6 | 24 | 0 |
| 26 | `dec_liquid_d1.2+refine` | 0.263 | 0.27 | 0.19 | 0.202 | 0.294 | 0.340 | 0.135 | 0.00024 | 0.01200 | 25.5 | 24 | 0 |
| 27 | `dec_respan_span-02` | 0.256 | 0.22 | 0.31 | 0.188 | 0.289 | 0.311 | 0.163 | 0.00013 | 0.01200 | 26.5 | 24 | 0 |
| 28 | `dec2_respan_span-02` | 0.243 | 0.05 | 0.29 | 0.198 | 0.265 | 0.293 | 0.159 | 0.00054 | 0.05092 | 105.4 | 102 | 0 |
| 29 | `dec_liquid_d1.2` | 0.241 | 0.11 | 0.24 | 0.170 | 0.277 | 0.313 | 0.122 | 0.00024 | 0.01200 | 25.5 | 24 | 0 |
| 30 | `seg_segformer-chat-hy4-flash` | 0.236 | 0.14 | 0.49 | 0.075 | 0.316 | 0.324 | 0.088 | 0.00009 | 0.00009 | 17.5 | 5 | 0 |
| 31 | `seg_evit-chat-hy4-flash` | 0.228 | 0.05 | 0.50 | 0.107 | 0.288 | 0.303 | 0.101 | 0.00010 | 0.00010 | 23.9 | 6 | 0 |
| 32 | `seg_segformer-dec-respan_span-02` | 0.183 | 0.08 | 0.59 | 0.101 | 0.224 | 0.239 | 0.088 | 0.00003 | 0.00244 | 6.6 | 5 | 0 |
| 33 | `seg_evit-dec-respan_span-02` | 0.159 | 0.03 | 0.62 | 0.097 | 0.190 | 0.202 | 0.087 | 0.00003 | 0.00284 | 8.0 | 6 | 0 |
| 34 | `chat_gpt-5-nano+refine` | 0.110 | 0.08 | 0.21 | 0.088 | 0.121 | 0.110 | 0.110 | 0.00019 | 0.00019 | 4.0 | 1 | 0 |
| 35 | `chat_gpt-5-nano` | 0.097 | 0.03 | 0.21 | 0.080 | 0.106 | 0.092 | 0.105 | 0.00019 | 0.00019 | 4.0 | 1 | 0 |

Per-class IoU for the top methods:

| method | street | yard | foliage | sky | building | sidewalk | porch | door | timestamp | neighbour | timestamp area ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `chat_hy4-flash` | 0.45 | 0.65 | 0.64 | 0.19 | 0.56 | 0.09 | 0.36 | 0.10 | 0.67 | 0.09 | 0.8 |
| `chat_hy4-flash+refine` | 0.46 | 0.61 | 0.64 | 0.19 | 0.57 | 0.09 | 0.37 | 0.13 | 0.67 | 0.08 | 0.8 |
| `chat_glm-5.3-flash+refine` | 0.45 | 0.67 | 0.61 | 0.14 | 0.44 | 0.04 | 0.49 | 0.04 | 0.77 | 0.01 | 1.2 |
| `chat_glm-5.3-flash` | 0.45 | 0.67 | 0.59 | 0.14 | 0.38 | 0.04 | 0.53 | 0.05 | 0.77 | 0.00 | 1.2 |
| `chat_ling-3.0-flash-vl+refine` | 0.42 | 0.68 | 0.54 | 0.09 | 0.46 | 0.09 | 0.55 | 0.08 | 0.68 | 0.05 | 0.8 |
| `chat_deepseek-v4.1-flash+refine` | 0.31 | 0.63 | 0.58 | 0.10 | 0.41 | 0.06 | 0.47 | 0.04 | 0.71 | 0.00 | 1.0 |
| `chat_ling-3.0-flash-vl` | 0.38 | 0.68 | 0.49 | 0.09 | 0.45 | 0.09 | 0.51 | 0.09 | 0.68 | 0.06 | 0.8 |
| `chat_deepseek-v4.1-flash` | 0.25 | 0.62 | 0.52 | 0.10 | 0.40 | 0.06 | 0.44 | 0.06 | 0.71 | 0.00 | 1.0 |
| `chat_qwen3.7-flash+refine` | 0.40 | 0.71 | 0.60 | 0.09 | 0.40 | 0.09 | 0.49 | 0.00 | 0.32 | 0.05 | 0.8 |
| `chat_gemma-4-26b+refine` | 0.20 | 0.55 | 0.59 | 0.10 | 0.50 | 0.00 | 0.39 | 0.01 | 0.73 | 0.00 | 0.9 |
| `chat_gemma-4-26b` | 0.21 | 0.58 | 0.58 | 0.10 | 0.49 | 0.00 | 0.37 | 0.00 | 0.73 | 0.00 | 0.9 |
| `chat_qwen3.7-flash` | 0.37 | 0.73 | 0.54 | 0.07 | 0.38 | 0.09 | 0.44 | 0.00 | 0.32 | 0.05 | 0.8 |

What it shows:

- Whole-frame chat polygons win (0.47 to 0.58 against 0.24 to 0.34 for every tile, box and segmenter design), at one call per
  frame and about $0.0001. Tile grids come out blocky and mislabel eaves and clock text; a finer grid at the edges did not help.
- The segmenter refine changes the strong models by about +/-0.01, so it is off by default; it helps weak models a little.
- A decider checking each polygon removed false area but also dropped correct polygons, so mIoU fell.
- The first model draws better shapes on PTZ and night frames, the second has the best key-zone recall and least false area,
  the third is a cheap fallback. gpt-5-nano (rectangles, 0.10) and mistral-small (0.34) were poor.

### Combining models (offline, from the saved drawings)

Per-class routing, chosen leave-one-frame-out, did not beat the best single model. A per-pixel majority vote did best, but
the gain over the best single model (+0.026) is inside the noise, and it triples the calls, so `vote` is a trial.

| combo | mIoU | key recall | false area | fixed | PTZ | day | night | real $/frame |
|---|---|---|---|---|---|---|---|---|
| single hy4-flash | 0.577 | 0.43 | 0.33 | 0.287 | 0.723 | 0.615 | 0.515 | 0.00013 |
| single glm-5.3-flash | 0.561 | 0.68 | 0.22 | 0.304 | 0.690 | 0.629 | 0.448 | 0.00009 |
| CV route-2 (hy4+glm) | 0.557 | 0.46 | 0.18 | 0.284 | 0.693 | 0.608 | 0.471 | 0.00021 |
| CV route-3 (hy4+glm+ling) | 0.555 | 0.43 | 0.18 | 0.285 | 0.690 | 0.617 | 0.452 | 0.00035 |
| CV route-3b (hy4+glm+qwen) | 0.574 | 0.46 | 0.16 | 0.308 | 0.707 | 0.623 | 0.492 | 0.00038 |
| CV route-all-7 | 0.573 | 0.46 | 0.16 | 0.304 | 0.707 | 0.624 | 0.487 | 0.00055 |
| vote 2-of-3 (hy4,glm,ling) | 0.603 | 0.68 | 0.14 | 0.336 | 0.737 | 0.658 | 0.513 | 0.00035 |
| vote 2-of-3 (hy4,glm,qwen) | 0.597 | 0.57 | 0.12 | 0.287 | 0.753 | 0.652 | 0.507 | 0.00038 |

## Doors and entry points

Ground truth: the 8 fixed-camera frames that show a door (the PTZ frames have none, so every box there is a false positive).
"Door found" means a box centre falls inside the door quad.

| method | recall IoU>=.3 | recall IoU>=.5 | door found (centre in GT) | mean best IoU | FP boxes/frame | s/frame | frames |
|---|---|---|---|---|---|---|---|
| VLM 2 tiles (L+R, 2x) hy4-flash | 3/8 | 0/8 | 3/8 | 0.18 | 0.21 | 8.9 | 24 |
| VLM 2 tiles (L+R, 2x) glm-5.3-flash | 2/8 | 0/8 | 6/8 | 0.23 | 0.33 | 4.7 | 24 |
| VLM full-frame glm-5.3-flash | 1/8 | 0/8 | 2/8 | 0.09 | 0.46 | 2.4 | 24 |
| OWLv2-base zero-shot (4 prompts) | 1/8 | 0/8 | 1/8 | 0.06 | 1.21 | 6.4 | 24 |
| VLM full-frame qwen3.7-flash | 1/8 | 0/8 | 1/8 | 0.06 | 0.21 | 2.5 | 24 |
| VLM 2 tiles (L+R, 2x) ling-3.0-flash-vl | 1/8 | 0/8 | 1/8 | 0.05 | 0.00 | 2.8 | 24 |
| OWLv2 one-shot (door crop from other frame) | 0/8 | 0/8 | 8/8 | 0.20 | 0.67 | 4.9 | 24 |
| OWLv2 right-crop zero-shot | 0/8 | 0/8 | 7/8 | 0.16 | 0.71 | 2.7 | 24 |
| VLM full-frame hy4-flash | 0/8 | 0/8 | 1/8 | 0.13 | 0.25 | 4.8 | 24 |
| Grounding DINO tiny (512px) | 0/8 | 0/8 | 0/8 | 0.07 | 3.00 | 7.6 | 24 |
| Florence-2-base-ft (door grounding+OD) | 0/8 | 0/8 | 0/8 | 0.02 | 2.96 | 51.0 | 24 |
| VLM full-frame gemma-4-26b | 0/8 | 0/8 | 0/8 | 0.00 | 0.33 | 2.6 | 24 |

- No method localises the door at IoU 0.5 (0 of 8 everywhere). It is a white panel on white siding, skewed by the lens.
- Tight zoomed tiles help every model, hence the two 2x tiles in `raptor_vision.entry`.
- One example crop of the door (what a person confirms once) lets OWLv2 image-guided detection put a box on the door in 8 of 8
  frames; boxes are loose and the best-scoring one can be a bush, so keep the top three and filter by size.
- SAM masks added nothing. YOLO-World and ultralytics models were skipped for their AGPL/GPL licences; OWLv2 is Apache-2.0.
- So doors are a separate pass that proposes boxes for a person to confirm, never an automatic mask.
