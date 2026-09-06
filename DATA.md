# Data

Training corpus is wikitext-103-train, packed to 512-token blocks (teacher targets computed on the fly — blocks store input tokens only).

| pool | blocks | tokens | used by |
|---|---|---|---|
| distillv2_3000blk | 3,000 | 1.5M | RUN 2 (teacher-generated text — the corpus-drift failure) |
| wiki_3000blk | 3,000 | 1.5M | RUN 3 |
| wiki_20000blk_clean | 19,940 (19,920 train + 20 heldout) | 10.2M | RUN 4/6 |
| wiki_50000blk_clean | 49,861 | 25.6M | RUN 8, cloud runs |
| wiki_80000blk_clean | 79,792 | ~41M | staged next scale-up |

## Contamination control

wikitext-2 test is partially inside wikitext-103 train. Every pool is filtered by 32-token shingle overlap against the eval corpus (`ppl_wikitext2.txt`, 262k tokens): the 20k pool dropped 60/20000 blocks (0.3%). Any run without this filter risks a dishonestly low bar score.

Eval corpus (`ppl_wikitext2.txt`) is never trained on; internal eval uses 100 fixed rows + 20 heldout blocks from the same pool family.

## Why general text, not self-generated

RUN 2 distilled teacher-generated text: KL → 0.28 on-distribution while wikitext CE rose past 5 — the student learned the teacher's output manifold and lost generality. All runs since use general text; the KL/fp/dg triple (distribution match / generality / transfer) is monitored separately at every eval so the three failure modes can't hide behind one loss number again.
