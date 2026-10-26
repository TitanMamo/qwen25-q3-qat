# Reproduce (QAT / PPL verify)

Reference PPL (any stock llama-perplexity build):

```bash
llama-perplexity -m stock-f16.gguf -f ppl_wikitext2.txt -ngl 99 -t 6  # expect ~14.74
```

QAT files need a build with the Q3_0_G128 / Q3_1_G128 / Q3_1_G64 kernels (see artifacts repo README). Mixed-grid files are the training grid (mid blocks quantized, embed/first/last/output fp); pure files quantize everything.

Mixed-grid export (the correct invocation — `--custom-q` requires a quantized default ftype, cf. ik_llama.cpp issue #2415; type names lowercase):

```bash
llama-quantize --custom-q 'blk\.0\..*=f16,blk\.23\..*=f16' \
  --output-tensor-type f16 --token-embedding-type f16 \
  trained-f16.gguf trained-mixed.gguf Q3_1_G64
```

Artifacts: https://huggingface.co/TitanMamo10/qwen25-qat-q3-poc

## K2 trellis (t16)

Runtime: the K2 branch of the fork (`TitanMamo/ik_llama.cpp`, branch `k2-t16`) — the op, the CUDA
GEMV/f16-MMA prefill, the loader path and the graph build. Without it the file does not load.

Verify the published artifact (no encoder needed):

```bash
md5sum qwen2.5-1.5b-k2-3.34bpw.gguf            # expect 584a86cb105e9771e6b30d3c7f08f52d
llama-perplexity -m qwen2.5-1.5b-k2-3.34bpw.gguf -f ppl_wikitext2.txt -ngl 99 -t 6   # expect ~10.59
```

Re-encode from scratch (19.4 min measured on one RTX 3060):

```bash
# exllamav3 + the K2 writer; -b is the allocator BUDGET, not a per-tensor rate
python exllamav3/convert.py -i <hf Qwen2.5-1.5B snapshot> -o out -w out_work \
    -hb 6 -b 3.34 -d 0
```

Read the realized per-tensor rates back out of the GGUF (this is how the 61.1/18.8/5.0/15.1% mix in
RUNS.md was obtained, and why notes about "3.34 bpw" should always be re-checked against the file):

```python
import gguf, collections
r = gguf.GGUFReader("qwen2.5-1.5b-k2-3.34bpw.gguf")
dist = collections.Counter()
for t in r.tensors:
    ne = list(t.shape)
    if len(ne) == 3 and ne[0] % 16 == 0 and ne[0] // 16 <= 8:
        dist[ne[0] // 16] += 256 * ne[1] * ne[2]      # K -> weights
print(dist, sum(k*w for k, w in dist.items()) / sum(dist.values()))
```

Evidence for every K2 number: https://huggingface.co/TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw/blob/main/evidence/k2_evidence.tar.gz

Encoder measurements live in `bench_scripts/` (profiler attribution, TF32 A/B, occupancy A/B); the
occupancy run must build inside the `cuda-box` distrobox, because the host has no CUDA compiler and
`build_ext` needs `nvcc`.
