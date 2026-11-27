# Protocol A Perplexity (BPB) Report

**Verdict:** `PASS`  
**Conservative standard spend:** `$0.314748`  
**Valid responses:** `120` | **Strict adherence:** `98.3%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **103 / 117 (88.0%)** | Match breakdown: `{'exact_multitoken_top20': 2, 'exact_top20': 101, 'miss_censored': 10, 'subtoken_censored': 4}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **1.0365** | [0.7791, 1.3321] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **1.0580** | [0.7907, 1.3617] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **1.0856** | [0.8101, 1.4077] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.93, miss=14.1%) | **0.6906** | [0.5631, 0.8322] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.76) | **0.5706** | [0.4559, 0.6996] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`50.4%`** / `70.9%` / `80.3%` / `93.2%`
- MRR: `0.6299` | Mean `log2(rank)`: `1.199` | Median rank: `1.0`
- Mean Top-20 head entropy: **`0.491` bits** | Median Top-1 surprisal: **`0.028` bits**
