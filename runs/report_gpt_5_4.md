# Protocol A Perplexity (BPB) Report

**Verdict:** `PASS`  
**Conservative standard spend:** `$0.315128`  
**Valid responses:** `120` | **Strict adherence:** `96.7%` (`CLEAN`) | **Reasoning leaks:** `0`

## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)

Observed exact tokens: **101 / 116 (87.1%)** | Match breakdown: `{'exact_multitoken_top20': 2, 'exact_top20': 99, 'miss_censored': 10, 'subtoken_censored': 5}`

| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |
|---|---:|---|---|
| Certified Hard Lower Bound (`BPB_LB`, raw T=1) | **0.8327** | [0.6482, 1.0138] | — |
| Zero-Proxy Geometric Tail (raw T=1) | **0.8570** | [0.6651, 1.0448] | — |
| Zero-Proxy Zipf(200k) Tail (raw T=1) | **0.8854** | [0.6815, 1.0912] | — |
| Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail) (T*=2.61, miss=15.0%) | **0.7234** | [0.5926, 0.8654] | — |
| Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV) (T*=2.30) | **0.5557** | [0.4613, 0.6527] | — |

### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics

- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`50.0%`** / `72.4%` / `84.5%` / `93.1%`
- MRR: `0.6387` | Mean `log2(rank)`: `1.142` | Median rank: `1.5`
- Mean Top-20 head entropy: **`0.808` bits** | Median Top-1 surprisal: **`0.093` bits**
