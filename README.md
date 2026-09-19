# 3-bit QAT on Qwen2.5 (0.5B/1.5B): types that work, training short of the bar

Goal: a ~3–3.5 bpw quantization whose grid is learned by training (QAT) instead of fitted post-hoc (PTQ), so deployment *is* the training grid. Base models Qwen2.5-0.5B (fast loop) and 1.5B (scale check) — the largest a 6 GB card can train through quantization with a CPU-offload AdamW trainer.

Status: **the format/runtime half is done; the training half is not at its bar.** Pre-registered bar: wikitext2 ppl of the Q3 file ≤ 16.5 (stock F16 14.74, stock IQ3_K+imatrix 15.80 — must beat PTQ at fewer bits). Best 0.5B: **21.25** (codebook-grid champion, §2); best 1.5B: **12.39** (+18.8% over stock 10.43, fp32-master run).

Separate project: [q38-6gb](https://github.com/TitanMamo/q38-6gb) (serving Qwen3.8-Flash-Next on 6 GB VRAM). Different model, different question — kept apart deliberately.

## 1. The types (done, verified)

Three custom GGML types, CPU + CUDA (dp4a) kernels, generation token-identical to F16:

| type | grid | bpw | PTQ ppl, stock 0.5B (wikitext2) |
|---|---|---|---|
| Q3_0_G128 | symmetric 3-bit, group 128 | 3.125 | 223.9 |
| Q3_1_G128 | asymmetric 8-level, group 128 | 3.25 | 74.8 |
| Q3_1_G64 | asymmetric 8-level, group 64 | 3.50 | 35.9 |
| Q3KS_G128 (local-only, §TYPES) | dual-codebook sb128/gs32, IQ3_KS math | 3.205 | 23.3 (18.4 with imatrix) |

Grid sweep (100-row CE, stock 3.13): sym-gs128 6.33 → sym-gs64 5.74 → asym-gs128 5.22 → asym-gs64 4.44. Asymmetry + halved group size buy **−1.9 CE for +0.375 bpw**. All three beat IQ3_K at fewer bits before any training. Sustained kernel throughput ~205 tok/s (dp4a parity class).

## 2. Training series (mixed-export ppl)

Self-distillation only (frozen teacher, full-vocab KL, STE@0, filtered grid keeping embed/lm_head/first+last layers fp, wikitext corpus with 0.3% eval-contaminated blocks dropped — wikitext-2 test is partially inside wikitext-103 train):

| run | ppl | what changed |
|---|---|---|
| V1 | 1028.5 | top-K=32 renormalized targets — tail erosion, failed as designed |
| R3 | 60.1 | full-vocab KL + general corpus + teacher-context bugfix |
| R5 (sym) | 34.3 | mixed-grid export matching the training grid |
| R6 (asym G128) | 25.7 | better grid |
| R8 (asym G64) | **21.9** | best uniform-grid; transfer gap dg +0.0002 (lossless by construction) |
| Q3KS pilot (codebook sb128) | **21.25 champ-mixed** | first run to beat R8: fp 3.3671 vs 3.3815 at 3.20 vs 3.50 bpw, dg −0.0000 |

**Transfer is solved** (the trained weights survive export exactly); **the weights themselves are +0.13 CE over the bar** at 0.5B/data scale. Scale math: closing needs ~100–170k blocks and/or 1.5B scale (1.5B fp32-master QAT sits at 12.39 on the same grid class, §RUNS). On-grid drift is uniform (~23% relative on all mid-layer groups — no freezable subset, measured, anchor ablated and rejected).

## 3. Failed runs, kept for the lessons

- **Cloud batch-8 on 0.5B diverged twice** (fp 5.06 → 8.17 while batch-1 converges): the batch-1 LR recipe does not transfer; LR must scale with batch and a 500-step gate must pass before committing a full run.
- **Mixed-grid export silently re-gridded fp layers** (144.9 vs true 34.3): `--custom-q` only applies when the default ftype is quantized — found in `llama-quantize.cpp`, filed upstream as ik_llama.cpp issue #2415.
- **Teacher-context bug**: chunked teacher scoring fed context-free slices (corrupted half the targets); a full-context pass halved ppl alone.
- **V1 top-K**: renormalized 32-top targets carry zero mass on ~20% of true tokens — guaranteed tail erosion, ~1000 ppl.
- **Constant-rate anchor**: consistently 0.03–0.05 CE worse than unanchored — rejected by A/B.
- **Checkpoint averaging**: midpoint loses on a still-descending trajectory — rejected.
- **Grid controls (recipe vs grid verdict)**: int4-control reaches 3.2314 (published-class, dg 0) and int8-control sits at stock — the recipe is clean, the 0.5B int3 wall is grid capacity, not training.
- **Staged Q4→Q3 warm start**: NO-GAIN band, worse than from-scratch — the grid, not the init, is the wall. Staging dead.
- **2x token scaling (1.5B)**: ~−0.005 CE for 2x tokens — plateau is capacity-bound, not token-bound. Dead.
- **bf16-master starvation (1.5B)**: bf16 run recovered ~13% of its grid gap; fp32-master rerun ~67% → 12.39 champion. Updates at LR 5e-6 sit ~25x below bf16 ULP — found and fixed, not hypothesized.
- **CEC drift-aware calibration**: 6 variants, all lose to plain RTN (folds explode to CE ~10.7) — falsified with mechanism.
- **Per-chunk learned rotation (Cayley)**: 1760/1760 chunks fall back to identity — real chunks sit at the snap floor. Harness-side learned rotation dead; random Hadamard keeps the slot.
- **LDLQ error feedback**: worse than RTN at α = 1.0/0.5/0.25 on the min-max grid — rejected (consistent with literature: needs MSE-type grids).

## 4. Side result: IQ3_KT (upstream #2414)

Reported the shipped IQ3_KT codebook's ~22% weight error and 0.5B collapse with repro + workarounds; maintainer showed it holds at 1.5B, and I verified his claim myself (F16 3.15 / iq3_k 3.33 / iq3_kt 3.49 — no collapse, conceded with numbers). Lesson kept: 0.5B fragility does not generalize; size-match every claim to its evidence.

## Contents

- [RUNS.md](RUNS.md) — complete per-run record (configs, trajectories, verdicts).
- [TYPES.md](TYPES.md) — grid/type spec, block layouts, kernel notes.
- [DATA.md](DATA.md) — pools, contamination control, why general text.
- [REPRODUCE.md](REPRODUCE.md) — PPL verify + mixed-grid export.

## 5. Artifacts

https://huggingface.co/TitanMamo10/qwen25-qat-q3-poc — stock F16 reference (14.74), 0.5B QAT (21.88), 1.5B QAT (14.10), README with verify commands. QAT files need a build with the Q3 kernels; the F16 verifies anywhere (see [REPRODUCE.md](REPRODUCE.md)).

## 6. Upstream

- Issue ikawrakow/ik_llama.cpp#2414 (IQ3_KT): filed with diagnosis, conceded with 1.5B numbers — resolved as non-bug.
- Issue #2415 (`--custom-q` silently ignored): declined by design, warning-only patch offered.
- Discussion #2417 (Q3 types RFC): whether Q3_0_G128 / Q3_1_G128 / Q3_1_G64 belong in mainline — open, artifacts published, cleanup gated on the answer.
- Trellis rebuild watch (Sep 2026): upstream is extending sub-256-row coverage — issue #2422 (block-64 trellis request), PR #2480 (IQ4_KS_R16, new repacked type, owner calls it "first of a planned series"), PR #2481 (IQ3_KT/IQ4_KT sub-256 tails, review pending). Direction is reach (coverage), not new codebooks. Our interest: a sub-256 *codebook* type (IQ3_KS-R16 class) — 0.5B's 896-dim rows have no stock codebook home today.
- Literature now on the table (all verified from papers/code, none yet tested by us): **GSQ** (IST-DASLab, Gumbel-Softmax assignment optimization, 3-bit ≈ QTIP on 70B, GGUF K-Quant refine pipeline); **SLQ** (task-lossless at 3.3 bits via non-uniform asymmetric grids + γ² law mandating asymmetry — our grids already satisfy it); **RCO** (exact-budget discrete allocation on true loss — the principled replacement for heuristic mixed-rate).

## How this was built (human + AI)

Ideas, direction, and verification standards are mine; implementation is AI-assisted. I set the questions (grids to try, bars to beat, ablations to run), the AI writes the trainers, kernels, and drafts — I review, catch mistakes, and redirect. The grid-sweep pivot, the teacher-context bugfix, the contamination filter, and the batch-scaling postmortem all came out of that loop. Nothing here was accepted on the AI's say-so: every number is a measured artifact, every failed run is kept with its lesson, and the unmet bar is stated with its number — falsifiable claims over optimism.
