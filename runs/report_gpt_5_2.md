# Protocol A Perplexity (BPB) Report

**Verdict:** `PASS`  
**Conservative standard spend:** `$0.352988`  
**Valid responses:** `120` | **Strict adherence:** `98.3%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **103 / 118 (87.3%)** | Match breakdown: `{'exact_multitoken_top20': 3, 'exact_top20': 100, 'miss_censored': 8, 'subtoken_censored': 7}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **0.7371** | [0.5714, 0.9171] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **0.8147** | [0.6028, 1.0375] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **0.8251** | [0.6111, 1.0523] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.21, miss=15.6%) | **0.7267** | [0.5926, 0.8736] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.12) | **0.5311** | [0.4453, 0.6248] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`47.5%`** / `71.2%` / `80.5%` / `94.1%`
- MRR: `0.6186` | Mean `log2(rank)`: `1.193` | Median rank: `2.0`
- Mean Top-20 head entropy: **`0.919` bits** | Median Top-1 surprisal: **`0.199` bits**
