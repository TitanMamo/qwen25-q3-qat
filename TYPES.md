# Grid/type spec

Three custom GGML types (IDs 42–44, verified free on public main at the time of writing), implemented across the full stack: `ggml.h` (enum + ftype), `ggml-common.h` (block layouts), `iqk_quantize.cpp/.h` (quantize/dequant/vec-dot AVX2 + scalar), `ggml.c` (traits + quantize-chunk dispatch), `ggml-quants.c` (row validation), `llama.h`, `llama-model.cpp` / `-loader.cpp` / `-quantize.cpp`, `examples/quantize`, CUDA `dmmv.cu` / `mmvq.cu` (dp4a + f16 W3A16 variants) / `getrows.cu` / `convert.cu` + dispatch gates.

## Block layouts

| type | id | bpw | layout per group |
|---|---|---|---|
| Q3_0_G128 | 42 | 3.125 | scale fp16 + 48 B codes (symmetric q = round(w/s), s = amax/3, group 128) |
| Q3_1_G128 | 43 | 3.25 | min fp16 + scale fp16 + 48 B codes (asymmetric q = round((w−mn)/s), s = (mx−mn)/7, dequant = code·s + mn, group 128) |
| Q3_1_G64 | 44 | 3.50 | min fp16 + scale fp16 + 24 B codes (same asymmetric formula, group 64) |

Training fake-quant is the exact deployment formula (symmetric: `q=round(w/s).clamp(-3,3)`; asymmetric: `q=round((w-mn)/s).clamp(0,7)`), including torch.round half-to-even parity — training grid = deployment grid, which is why export transfer is lossless (dg +0.0002).

## Kernel notes (earned the hard way)

- Fused up-gate MoE path must **exclude** the new types (same treatment as Q3_0) or `GGML_ABORT` fires in `iqk_moe_fused_up_gate`.
- Row-validation needs an explicit case per type or quantize aborts with "invalid type".
- GPU path took the asymmetric type from 218 s/pass (scalar fallback, graph split into 266 classes) to 3.43 s/pass (~64x); the q8_1 dp4a dot needs a second dp4a against `0x01010101` for the `m·sum(q8)` term.
- Pure-PTQ perplexity ladder on stock 0.5B (same corpus, GPU): Q3_0 223.9 → Q3_1_G128 74.8 → Q3_1_G64 35.9 — roughly 3x per grid improvement, matching the sweep's CE deltas.
- Generation from all three types is token-identical to F16 on probe prompts.
