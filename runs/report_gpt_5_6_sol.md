# Protocol A Perplexity (BPB) Report

**Verdict:** `FAIL_QA_GATE`  
**Conservative standard spend:** `$0.311488`  
**Valid responses:** `120` | **Strict adherence:** `98.3%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **105 / 112 (93.8%)** | Match breakdown: `{'exact_multitoken_top20': 3, 'exact_top20': 102, 'miss_censored': 6, 'subtoken_censored': 1}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **0.9304** | [0.6785, 1.2330] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **0.9424** | [0.6870, 1.2531] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **0.9513** | [0.6913, 1.2651] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.99, miss=9.4%) | **0.6142** | [0.4871, 0.7540] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.46) | **0.5946** | [0.4691, 0.7476] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`53.6%`** / `75.9%` / `82.1%` / `95.5%`
- MRR: `0.6619` | Mean `log2(rank)`: `1.059` | Median rank: `1.0`
- Mean Top-20 head entropy: **`0.529` bits** | Median Top-1 surprisal: **`0.035` bits**
