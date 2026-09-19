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

V1 1028.5 → R3 60.1 → R5-sym-mixed 34.26 → R6-asym-G128-mixed 25.69 → R8-asym-G64-mixed **21.88** → Q3KS-codebook-mixed **21.25** → bar 16.5. Every step's cause is named; nothing regressed silently.

## Controls — recipe vs grid (0.5B, exact R8 recipe, grid bit-width only)

- **ctrlq4** (int4 sym gs128, 3300 steps, 299 min): smoke fp 3.4407 dg +0.0011 → trajectory below → FINAL fp **3.2314** / q3 3.2314 / dg +0.0000 (kl 0.1344):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| fp | 3.289 | 3.261 | 3.247 | 3.236 | 3.232 | 3.232 |

Verdict: PUBLISHED-CLASS (0.0014 above the pre-registered ≤3.23 line — within 0.04% CE): the pipeline converges in the w4 regime (+0.099 CE over stock at 1.7M tokens; published runs use ~100x more), transfer lossless (dg 0.0000 — 5th consecutive run). Q4 vs int3 (R8 3.381): −0.149 CE.
- **ctrlq8** (int8, USER-STOPPED @1043/3300 by early evidence): @500 fp 3.132, @1000 fp 3.136 — both AT STOCK (3.1334), kl flat 0.0025. Verdict: RECIPE-CLEAN by user decision (formal 3300-band not run; two at-stock evals settle it).
- Interpretation: the recipe recovers int4 to ~+0.1 CE with dg=0. The int3 wall (R8 +0.25) is grid-driven (3-bit capacity at 0.5B), not recipe-driven.

## Staging Q4→Q3 — NO-GAIN, premise dead

q3stage (int4-asym init → Q3, 2000 steps): @500 3.511 → @1000 3.445 → @1500 3.423 → @2000 3.414 (kl 0.3362, dg +0.000), geometric flattening projecting ~3.41: NO-GAIN band (≥3.37) AND worse than R8-from-scratch (3.3815). Warm-starting from the int4 basin does not beat plain Q3 training. Killed @2000, ckpt-1800 banked. Proper-init variants (same-QMAX curriculum) queued but deprioritized — grid, not init, is the wall.

## 1.5B fp32-master (Modal L4, the 12.39 champion)

Old bf16-master run: final fp 3.20, only 0.06 below its G64 PTQ floor 3.2625 (~13% gap recovery, KL flat 0.45 for 5500 steps) → diagnosed bf16 update starvation (5e-6 updates ~25x below bf16 ULP). fp32-master rerun (5000 steps, batch 2, bf16 checkpoints, 112.8 min, ~$1.8), dg 0.000–0.002 throughout:

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 5000 |
|---|---|---|---|---|---|---|---|
| kl | 0.3065 | 0.2787 | 0.2669 | 0.2594 | 0.2536 | 0.2506 | 0.2481 |
| fp | 3.020 | 2.981 | 2.964 | 2.952 | 2.947 | 2.944 | 2.942 |

~67% gap recovery (local frame: stock 2.7828, floor 3.2625). Local verify (bf16, modal-matched 100-row): fp **2.9592** / q3 2.9592 / dg +0.0001 (63.2% recovery; +0.017 modal-vs-local eval drift). GGUF (`fp32-final-mixed-Q3_1_G64`, 1.66G, blk0+blk27+emb+output F16): **12.3921 ± 0.094** vs stock 10.43 (+18.8%), old bf16 champ 14.10 (−1.71). Transfer lossless (ln ratio 0.172 vs local 0.176). Note: checkpoint bug found — torch.compile-wrapped save wrote `_orig_mod.`-prefixed keys + separate lm_head; stripped on load (future runs: save via inner model).

## 10k token-scaling probe (1.5B, from-0, CANCELLED by user, verdict by evidence)

Evals @4000–7000: fp 2.936–2.940 (7 evals in band; 2x tokens bought ~−0.005 CE vs the 5k plateau). Verdict: token scaling WEAK — plateau is grid/capacity-bound; 5x extrapolates ~2.91 (not near-lossless) → full 5x run (~$8–9) NOT bought. Cloud spend total ~$5 of $10. (A resume-from-ckpt design was rejected first: warm-restart confounds schedule-shape with token count.)

## Q3KS pilot — WIN, first run to beat R8 (0.5B, IQ3_KS codebook math sb128/gs32)

Same trainer/seed/data as R8, 2900 steps (vs R8's 3300): FINAL kl **0.2832** / fp **3.3671** / q3 3.3671 / dg −0.0000. Eval series: @500 3.452 (R8 3.509) → @1000 3.400 (3.436) → @1500 3.381 (3.403) → @2000 3.371 (3.389) → @2500 3.367 (3.383) → FINAL 3.3671. Gap decayed 0.057→0.0144 (QAT compensation asymptote ≈ the codebook's structural KL advantage, 0.2832 vs 0.321). 6th consecutive lossless transfer. Export needs the K1 GGUF type (§TYPES) — no fork type for sb128/gs32 exists. Stock-type IQ3_KS pilot SKIPPED: 896 % 256 ≠ 0, the fork itself falls back to IQ4_NL for all 896-dim rows (only 24/168 tensors quantize as IQ3_KS); redundant with this win. 1.5B pilot (1536 % 256 == 0, fully compatible) waits on compute.

## K1 PPL battery (0.5B stock, wikitext2, -ngl 99)

F16 **14.7435** | IQ3_KS 17.7457 | K1-PTQ 23.3438 | K1+imatrix **18.3903** (64-chunk wiki imatrix). Champion (Q3KS pilot weights): champ-f16 21.2775 vs champ-K1-mixed **21.2464** — K1 quantization LOSSLESS vs its own source (−0.03). Champ absolute vs stock F16 is a QAT-data gap, not a quantization gap. GPU generation token-identical CPU↔GPU. Three real bugs fixed en route (odd-stride CUDA misalign; vec_dot Q8_0 + fp16-d; char sign-extension NaN) — see TYPES.md kernel notes.

## Referee + deployment ladder (1.5B)

- **IQ4_KS referee** (stock Qwen2.5-1.5B, `llama-quantize` IQ4_KS default profile with auto type overrides, 827.57 MB vs F16 2944.68 MB ⇒ ~4.49bpw file, wikitext2 512 chunks GPU): **10.9436 ± 0.08361** (+4.9% vs stock; ln gap +0.049 CE). Table: F16 10.43 | Q4_K 10.7508 (+3.1%) | IQ4_KS 10.9436 (+4.9%) | QAT champ 12.3921 (+18.8%). The composite gate: beat IQ4_KS at ≤3.8bpw, i.e. land +≤0.049 CE.
- Champion descent prices (all GPU, same corpus, vs 12.3921):

| variant | ppl | cost | eff. bpw |
|---|---|---|---|
| mixed (tied f16 emb) | 12.3921 ± 0.094 | — | 6.19 |
| tied Q8 emb | 12.3939 | +0.002 | 5.06 |
| emb Q6_K | 12.4764 ± 0.095 | +0.084 | ~4.72 |
| emb IQ4_NL | 12.9261 ± 0.100 | +0.534 | ~4.55 |
| full-grid (layers 0/27 + emb/output on Q3) | 16.2500 ± 0.130 | **+3.86** | ~4.3 |

Reading: embeddings fall off a cliff below Q6 (rare-token rows + every-token lookup exposure) — stay tied-Q8 or QAT them; the +3.86 full-grid cost proves the mixed-precision boundary (first/last layers fp) is load-bearing, not conservative. Path to ≤3.8bpw cannot go through naive full-grid.
- Embeddings were never in the training grid (`_keep_fp` excludes emb/lm_head) — ladder prices are naive-PTQ, not the floor (emb-QAT queued: freeze-except repair on champion first, joint training only if needed).

## Lever kills (PTQ harness, asym-gs64 grid = Q3_1_G64)

0.5B 40-row frame (stock 3.1155, RTN 3.9504) and 1.5B 24-row frame (stock 2.8228, RTN 3.2732):

| lever | 0.5B CE | 1.5B CE | verdict |
|---|---|---|---|
| RTN baseline (3.50bpw) | 3.9504 | 3.2732 | reference |
| chunk-128 Hadamard (+0.00) | 3.8005 | 3.2384 | kept (17.9% / 7.7% gap recovery) |
| full-padded FHT (+0.00) | — | 3.1553 | kept (26.3% on 1.5B; 38.5% on 0.5B) |
| SpQR top-1 (+0.34) | 3.6612 | 3.2097 | kept for raw; retired from stack |
| SpQR top-2/top-3 (+0.69/+1.03) | 47.8% / 55.6% gap recovery | — | measured, superseded by SVD |
| scale-search (clip) | in stacks | in E2 | kept |
| ss+fullrot+spqr1 (3.84bpw) | 3.5670 | 3.0267 | old ceiling |
| **ss+fullrot+svd16 (~3.86bpw)** | **3.4989** | — | **new ceiling** (SVD replaces SpQR in-stack, −0.068 at matched rate; rotation+ss Gaussianize weights so dense directions beat single-outlier pinning) |
| LDLQ α=1.0/0.5/0.25 | 5.20/4.17/3.99 | — | rejected (needs MSE-type grids) |
| CEC variants (×6) | best 4.0367, folds ~10.7 | — | falsified (drifted-scale crush + overfit) |
| Cayley learned-rot | 13.08 unguarded → 3.9504 guarded, 1760/1760 fallbacks | — | harness-side dead (chunks at snap floor); random Hadamard keeps slot |
