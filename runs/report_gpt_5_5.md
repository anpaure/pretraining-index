# Protocol A Perplexity (BPB) Report

**Verdict:** `PASS`  
**Conservative standard spend:** `$0.353528`  
**Valid responses:** `120` | **Strict adherence:** `98.3%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **107 / 118 (90.7%)** | Match breakdown: `{'exact_multitoken_top20': 3, 'exact_top20': 104, 'miss_censored': 7, 'subtoken_censored': 4}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **0.6020** | [0.4567, 0.7530] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **0.6248** | [0.4709, 0.7820] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **0.6383** | [0.4778, 0.8044] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=1.91, miss=12.3%) | **0.6307** | [0.4980, 0.7562] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=1.93) | **0.4658** | [0.3809, 0.5571] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`56.8%`** / `74.6%` / `86.4%` / `94.9%`
- MRR: `0.6800` | Mean `log2(rank)`: `0.999` | Median rank: `1.0`
- Mean Top-20 head entropy: **`0.921` bits** | Median Top-1 surprisal: **`0.225` bits**
