# Protocol A Perplexity (BPB) Report

**Verdict:** `PASS`  
**Conservative standard spend:** `$0.315608`  
**Valid responses:** `120` | **Strict adherence:** `95.0%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **92 / 114 (80.7%)** | Match breakdown: `{'exact_multitoken_top20': 2, 'exact_top20': 90, 'miss_censored': 19, 'subtoken_censored': 3}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **1.0258** | [0.7677, 1.3201] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **1.0825** | [0.8123, 1.3772] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **1.1080** | [0.8360, 1.4049] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.76, miss=21.4%) | **0.8330** | [0.6792, 0.9931] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.94) | **0.6185** | [0.5115, 0.7276] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`39.5%`** / `56.1%` / `66.7%` / `84.2%`
- MRR: `0.5219` | Mean `log2(rank)`: `1.721` | Median rank: `2.0`
- Mean Top-20 head entropy: **`0.899` bits** | Median Top-1 surprisal: **`0.223` bits**
