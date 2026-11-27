# Protocol A Perplexity (BPB) Report

**Verdict:** `FAIL_QA_GATE`  
**Conservative standard spend:** `$0.352948`  
**Valid responses:** `120` | **Strict adherence:** `76.7%` (`MARGINAL`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **80 / 92 (87.0%)** | Match breakdown: `{'exact_multitoken_top20': 1, 'exact_top20': 79, 'miss_censored': 10, 'subtoken_censored': 2}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **0.7630** | [0.5759, 1.0327] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **0.7929** | [0.5998, 1.0838] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **0.8168** | [0.6093, 1.1221] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.10, miss=14.6%) | **0.7542** | [0.5989, 0.9721] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.06) | **0.5882** | [0.4838, 0.7486] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`42.4%`** / `65.2%` / `73.9%` / `89.1%`
- MRR: `0.5695` | Mean `log2(rank)`: `1.465` | Median rank: `2.0`
- Mean Top-20 head entropy: **`1.138` bits** | Median Top-1 surprisal: **`0.372` bits**
