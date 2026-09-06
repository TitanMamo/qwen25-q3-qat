# Training runs (complete record)

Method: self-distillation — frozen teacher, full-vocabulary KL + 0.1 CE anchor, STE fake-quant, wikitext corpus. Internal eval: 100-row wikitext CE (`eval_ce`; stock 0.5B = 3.1334 fp32, earlier 10-row est 2.8317) + heldout KL on 20 blocks + post-snap `q3`/`dg` (quantized CE and transfer gap). Final verdict always by `llama-perplexity` on the exported GGUF (wikitext2, 512 chunks, n_ctx 512, GPU). Bar: Q3 ppl ≤ 16.5 (stock F16 14.74).

Common local setup: GTX 1660 Ti 5.6 GB, batch 1 × 512 tokens, CPU-offload AdamW (fp32 master), ~5 s/it. Cloud: Modal L4, bf16 student + fp16 teacher, 8-bit Adam, torch.compile + grad checkpointing.

## V1 — top-K targets (failed as designed): 1028.5

Distill targets = renormalized teacher top-32 from npz (no on-the-fly teacher). Renormalized top-K carries zero mass on ~20% of true tokens → tail erosion. Result ppl 923–1028. Lesson: full-vocab targets are non-negotiable (LLM-QAT never used top-K).

## RUN 2 — full-vocab KL, self-generated corpus: 131.6

On-the-fly frozen teacher, full-vocab KL, delayed STE (engage @40%), LR 2e-5 cosine, 2900 steps. Eval trajectory (stock 2.83 scale):

| step | KL | fp | q3 | dg |
|---|---|---|---|---|
| 400 | 0.28 | 3.27 | 5.74 | +2.47 |
| 800 | 0.85 | 4.26 | 5.67 | +1.41 |
| 1200 | 1.91 | 5.28 | 5.50 | +0.22 |
| 1600–2400 | ~1.5 | ~5.0–5.1 | ~5.0–5.1 | → +0.00 |

Student converged to the teacher *on generated text* while losing generality — the corpus was teacher-generated, not general text. Post-STE dg → 0 (first proof grid transfer is lossless), but fp alone ≈ e^5.03. FAIL by corpus design.

## RUN 3 — general corpus + teacher-context bugfix: 60.1

Two changes: corpus = wikitext-103-train blocks; bugfix — the old teacher scorer fed two context-free 256-token slices (targets for positions 256–511 corrupted), replaced by one full-context 512 pass. Final: fp 4.43 / q3 4.43 / dg +0.00 → GGUF 60.11 ± 0.57. QAT moved q3 6.3 → 4.43 (naive int3 snap of stock); remaining gap is fp drift (+1.59 CE over stock), not quantization. 256 min.

## RUN 4 — gentle LR + 3.4x data, stopped @5008/10000: FAIL class

LR 5e-6 (pre-registered "gentle"), STE @30%, 19940-block pool. @500 fp 3.12 (best pre-STE ever) → plateau 3.73–3.77 → STE snap shock (fp 5.68) → recovery extrapolating to ~4.5 (≈ ppl 90). Two-part diagnosis: (1) the delayed-STE phase walks past a near-zero loss floor (self-distill init KL ≈ 0.03) and buys nothing; (2) on-grid tax ~0.85 CE persists. Killed at 50%. Consequence: STE from step 1 in all later runs.

## RUN 5 — STE@0 + mixed grid + L2-SP: 34.3 (after export fix)

STE from step 1, mixed-precision grid (embed/lm_head/first+last layers off-grid, ~3.2 bpw class), L2-SP anchor w=0.1. Final eval: fp 3.77 / q3 3.77 / dg +0.00 — best yet, but the first export scored **144.9**: `llama-quantize` had re-gridded the fp-kept layers (the `--custom-q`-only-applies-to-quantized-defaults behavior, later filed as issue #2415). Correct invocation:

```
llama-quantize --custom-q 'blk\.0\..*=f16,blk\.23\..*=f16' \
  --output-tensor-type f16 --token-embedding-type f16 \
  trained-f16.gguf trained-mixed.gguf Q3_0_G128
```

→ 466 MB, **34.26 ± 0.30**. Same weights, 4x score difference from the export chain alone.

## RUN 6 — asymmetric G128 grid: 25.7

New type Q3_1_G128 (8-level asymmetric, §TYPES). Smoke already told the story: 4-step fp 4.42 vs RUN 5's 6.29 (grid starts ~1.9 CE closer to stock, as the sweep predicted). Best checkpoint (2000, power-off cut the run at ~2550):

| step | KL | fp | q3 | dg |
|---|---|---|---|---|
| 500 | 0.60 | 3.67 | 3.68 | +0.007 |
| 1000 | 0.52 | 3.58 | 3.58 | +0.003 |
| 1500 | 0.49 | 3.54 | 3.54 | +0.002 |
| 2000 | 0.48 | 3.52 | 3.52 | +0.001 |

GGUF: F16 25.67 ± 0.22 → mixed 25.69 ± 0.22 (dg +0.02, lossless) vs pure-grid 31.64 (mixed saves ~6 ppl). On-grid tax cut to +0.39 CE over stock (was +0.64). Follow-ups: drift is uniform ~23% on all mid-layer groups (no freezable subset); checkpoint averaging loses on a descending trajectory; constant-rate anchor ablated **worse** by 0.03–0.05 (rejected).

## RUN 8 — asymmetric G64, proper run: 21.9

GRID G64 + 49861-block pool + RUN 5/6 recipe, 3300 steps (275 min). Monotone descent, no snap (STE@0):

| step | KL | fp | q3 | dg |
|---|---|---|---|---|
| 500 | 0.41 | 3.51 | 3.51 | +0.003 |
| 1000 | 0.36 | 3.44 | 3.44 | +0.002 |
| 1500 | 0.34 | 3.40 | 3.40 | +0.001 |
| 2000–3300 | 0.32 | 3.39 → 3.38 | = fp | +0.000 |

GGUF: F16 **21.88 ± 0.18** → mixed G64 **21.88 ± 0.18** (dg +0.0002). Best local ever; beats stock-G64 PTQ (35.87) by 14 ppl. Gap to bar: +0.135 CE.

## Cloud V6 — batch-8 on 0.5B diverged twice (aborted)

Same recipe on L4, batch 8, LR 1e-5 then 5e-6: fp 5.06 @500 → 8.17 @1000 (train KL climbing too), vs batch-1 local 3.67 @500. Per-step drift scales with batch while no anchor counteracts it; bf16-master rounding compounds (updates ~30x smaller than bf16 spacing at typical |w|). 1.5B with batch 6 was stable (best fp 3.77 @1000, wall at ~3.9 @5000) — the failure is 0.5B+batch specific. Rule: never scale batch on 0.5B without scaling LR down and passing a 500-step gate (fp ~3.6–3.8, dg < 0.3) first.

## 1.5B scale check (L4, symmetric grid): stable, same wall

Batch 6, 8334-step schedule, stopped @5000: best fp 3.77 @1000, climbing +0.056/1000 after — same drift wall as 0.5B, one scale up. Grid adaptation (dg ~0.07) works faster at 1.5B; base drift is the wall at both scales.

## 1.5B G64 asym (L4, the 14.10 artifact): the clean run

`modal_train_15b_g64.py` on L4 — batch 2 (not 8, per the V6 lesson), LR 5e-6, 8000 steps over the 79792-block pool, GS64, peakVRAM ~13.5 GiB:

| step | KL | fp | q3 | dg |
|---|---|---|---|---|
| 500 | 0.47 | 3.22 | 3.23 | +0.001 |
| 1000–3000 | 0.45 | 3.21 → 3.20 | = fp | ~+0.001 |
| 3500–8000 | 0.45 | 3.20 flat | = fp | +0.000 |

No drift, no snap, no climb — fp locks at 3.20 from step ~2500 to the end. GGUF (`qwen15_g64_8000-mixed-Q3_1_G64`): **14.10**. This is the run the scale math rests on: same grid class as 0.5B's 21.88, one scale up, gap to bar nearly closed (14.10 vs 16.5 bar vs 10.43 stock-F16).

## Series

V1 1028.5 → R3 60.1 → R5-sym-mixed 34.26 → R6-asym-G128-mixed 25.69 → R8-asym-G64-mixed **21.88** → bar 16.5. Every step's cause is named; nothing regressed silently.
