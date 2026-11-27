# Protocol A Perplexity (BPB) Report

**Verdict:** `PASS`  
**Conservative standard spend:** `$0.315608`  
**Valid responses:** `120` | **Strict adherence:** `97.5%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **105 / 117 (89.7%)** | Match breakdown: `{'exact_multitoken_top20': 2, 'exact_top20': 103, 'miss_censored': 7, 'subtoken_censored': 5}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **0.7413** | [0.5547, 0.9286] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **0.7582** | [0.5662, 0.9514] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **0.7674** | [0.5724, 0.9648] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.24, miss=12.4%) | **0.7123** | [0.5808, 0.8499] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.22) | **0.5501** | [0.4522, 0.6561] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`47.0%`** / `73.5%` / `82.1%` / `94.9%`
- MRR: `0.6165` | Mean `log2(rank)`: `1.227` | Median rank: `2.0`
- Mean Top-20 head entropy: **`1.131` bits** | Median Top-1 surprisal: **`0.359` bits**
