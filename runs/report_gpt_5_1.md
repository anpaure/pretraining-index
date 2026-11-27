# Protocol A Perplexity (BPB) Report

**Verdict:** `PASS`  
**Conservative standard spend:** `$0.329972`  
**Valid responses:** `113` | **Strict adherence:** `98.2%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **95 / 111 (85.6%)** | Match breakdown: `{'exact_multitoken_top20': 1, 'exact_top20': 94, 'miss_censored': 12, 'subtoken_censored': 4}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **0.9174** | [0.6559, 1.2231] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **0.9413** | [0.6782, 1.2571] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **0.9546** | [0.6896, 1.2732] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.32, miss=15.7%) | **0.6642** | [0.5343, 0.8145] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.19) | **0.5924** | [0.4580, 0.7507] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`49.5%`** / `73.0%` / `82.0%` / `90.1%`
- MRR: `0.6327` | Mean `log2(rank)`: `1.190` | Median rank: `2.0`
- Mean Top-20 head entropy: **`0.752` bits** | Median Top-1 surprisal: **`0.153` bits**
