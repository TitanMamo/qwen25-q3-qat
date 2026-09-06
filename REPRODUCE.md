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
