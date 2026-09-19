# Grid/type spec

Three custom GGML types (IDs 42–44, verified free on public main at the time of writing), implemented across the full stack: `ggml.h` (enum + ftype), `ggml-common.h` (block layouts), `iqk_quantize.cpp/.h` (quantize/dequant/vec-dot AVX2 + scalar), `ggml.c` (traits + quantize-chunk dispatch), `ggml-quants.c` (row validation), `llama.h`, `llama-model.cpp` / `-loader.cpp` / `-quantize.cpp`, `examples/quantize`, CUDA `dmmv.cu` / `mmvq.cu` (dp4a + f16 W3A16 variants) / `getrows.cu` / `convert.cu` + dispatch gates.

## Block layouts

| type | id | bpw | layout per group |
|---|---|---|---|
| Q3_0_G128 | 42 | 3.125 | scale fp16 + 48 B codes (symmetric q = round(w/s), s = amax/3, group 128) |
| Q3_1_G128 | 43 | 3.25 | min fp16 + scale fp16 + 48 B codes (asymmetric q = round((w−mn)/s), s = (mx−mn)/7, dequant = code·s + mn, group 128) |
| Q3_1_G64 | 44 | 3.50 | min fp16 + scale fp16 + 24 B codes (same asymmetric formula, group 64) |
| Q3KS_G128 (K1, local-only) | 45 | 3.205 | row fp16 d + per-128-superblock: qs[32] (2 low index bits) + qh[16] (high index bit) + scales[2] (4x ul nibble) + extra[1] (ul bit4 + codebook select). Dequant: w = d·(ul−16)·iq3nl[cb][idx], dual 8-entry non-uniform codebooks, gs32. Row = 2 B meta + nbl·51 B. |

Training fake-quant is the exact deployment formula (symmetric: `q=round(w/s).clamp(-3,3)`; asymmetric: `q=round((w-mn)/s).clamp(0,7)`; K1: torch port of the fork's IQ3_KS quantizer mirror, ntry=5 46-candidate search + WLS row refit, validated vs fork bytes at rel 3e-4), including torch.round half-to-even parity — training grid = deployment grid, which is why export transfer is lossless (dg +0.0002; K1 champ −0.0000).

## Kernel notes (earned the hard way)

- Fused up-gate MoE path must **exclude** the new types (same treatment as Q3_0) or `GGML_ABORT` fires in `iqk_moe_fused_up_gate`.
- Row-validation needs an explicit case per type or quantize aborts with "invalid type".
- GPU path took the asymmetric type from 218 s/pass (scalar fallback, graph split into 266 classes) to 3.43 s/pass (~64x); the q8_1 dp4a dot needs a second dp4a against `0x01010101` for the `m·sum(q8)` term.
- Pure-PTQ perplexity ladder on stock 0.5B (same corpus, GPU): Q3_0 223.9 → Q3_1_G128 74.8 → Q3_1_G64 35.9 — roughly 3x per grid improvement, matching the sweep's CE deltas.
- Generation from all three types is token-identical to F16 on probe prompts.

## K1 (Q3KS_G128) kernel + validation evidence

- Full skeleton: ggml.h (type 45 + ftype 33), ggml-common.h block, ggml.c traits + 7 case lists + quantize_chunk, iqk_quantize.cpp/.h (ntry=5 quantizer mirror + dequant + scalar vec_dot), model/loader/quantize plumbing, fused-up-gate exclusion (pre-empted August bug), CUDA convert + dmmv + **MMVQ (W3A16 + q8_1 dp4a)** + dispatch gates. Build 100% clean, 0 debug strings in libggml.so.
- Three real bugs found by validation and fixed: (1) CUDA misaligned address — odd 359 B row stride → byte-wise fp16 assembly via `__ushort_as_half`; (2) CPU vec_dot asserted n%QK_K==0 on 128-granular rows AND read block_q8_0.d as float (it is fp16) → new Q8_0 vec_dot (symptom: NaN); (3) GPU NaN — `(uint16_t)cx[0]` sign-extends (char signed; 0xAE→0xFFAE = fp16 NaN) → `(uint8_t)` casts. Lesson kept: the standalone convert test was NaN-blind (NaN comparisons are always false) — fixed to check isfinite.
- Validation tools (canonical copies in qat_tools/k1_tests/): k1_convert_test (full tensors, all widths, NaN-aware), k1_vdot_test (672 rows, 0 bad), k1_mmvq_test (1/2/4 cols, 64+66 rows, rel 2e-4 class), k1_mmvq_bench (ffn_down 896×4864×1: 57.1→50.7 µs, 34.7 GB/s).
- Speed path: scalar 123.6 → float4+branchless 189.38 → PRMT register-resident codebook **204.29 tok/s** (target 200+ hit; key insight: divergent constant/global table loads serialize — hoist 16 B codebook to regs). pp512 unchanged (~830, cublas). Q3_0 same-model reference 286 (no codebook — remaining gap structural). MMQ Tensor-Core deliberately skipped: all gates are ppl batteries (cublas path, vec kernels never run there).
- PPL battery (0.5B stock, wikitext2, -ngl 99): F16 14.7435 | IQ3_KS 17.7457 | K1-PTQ 23.3438 | K1+imatrix 18.3903. Champion mixed (blk0+blk23+output+embd F16, 469 MB): 21.2464 vs own F16 21.2775 — lossless (−0.03).
- Canonical byte rule (fixed a qh-overflow bug found by round-trip): weight (ib·32+j) → qs[j] bit-pair 2·ib | qh[j>>1] bit (ib+4·(j&1)) | ul nibbles in scales[0..1] with bit4 in extra[0..3] | cb-select extra[4..7]. Round-trip vs fakequant: max 1.3e-05, 0/896 bad cols.
