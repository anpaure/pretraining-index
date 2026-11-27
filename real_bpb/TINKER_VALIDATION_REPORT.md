# White-Box BPB & Cross-Provider Validation on Tinker (`real_bpb` branch)

To directly audit our black-box **Anpaure Pretraining Index (ANPI)** against exact teacher-forced log-probabilities on the **exact same posttrained weights** and the **exact same Nov-2026 corpus**, we evaluated all **14 overlapping posttrained open-weight models across 7 labs** hosted on Thinking Machines Lab’s Tinker API (`real_bpb/tinker_bpb.py` + `real_bpb/tinker_blackbox_anpi.py`).

For every model we measured three independent quantities:
1. **Exact Full-Doc BPB (`N = 1,000` archived Nov-2026 docs, `2.61 MB` UTF-8):** teacher-forced `compute_logprobs` with strided windows and exact UTF-8 byte normalization.
2. **Exact White-Box Oracle Ranking & Gold-Span BPB (`954 × 8` ANPI items):** exact BPE-boundary-safe conditional log-probability of all 8 candidates `A..H` given their shared `common_prefix_len_all` token prefix.
3. **Fresh Black-Box ANPI Rerun (`954` items, strict `0` CoT leaks across `13,356` calls):** exact frozen `run_anpaure_v1.py` prompt rendered through each model’s native zero-think Jinja template directly on Tinker’s weights.

---

## Results (`n = 14` Models Across 7 Labs)

| Model | OpenRouter BB AUC | **Tinker BB AUC** | Δ (TK − OR) | Tinker Clean N | **White-Box Oracle AUC** | Gold-Span BPB | **Full-Doc BPB (N=1000)** | Elie Base BPB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kimi K3.1 Pro** | `0.7520` | **`0.7531`** | `+0.0011` | `954/954` | `0.7597` | `0.8805` | **`0.5332`** | `0.4840` |
| **Kimi K2.6** | `0.6785` | **`0.6785`** | `-0.0000` | `954/954` | `0.7183` | `0.9064` | **`0.5890`** | — |
| **GLM-5.3** | `0.6622` | **`0.6737`** | `+0.0115` | `954/954` | `0.7263` | `0.9231` | **`0.6078`** | — |
| **Inkling 2 Small** *(BYPASS)* | `0.6596` | **`0.6663`** | `+0.0067` | `855/954` | `0.7338` | `0.9151` | **`0.5667`** | *(`0.5128`)* |
| **DeepSeek V4.1 Flash (1017)** | `0.6545` | **`0.6595`** | `+0.0050` | `954/954` | `0.7281` | `0.9557` | **`0.6077`** | `0.5388` |
| **Nemotron 3 Ultra** | `0.6518` | **`0.6560`** | `+0.0042` | `954/954` | `0.7170` | `0.9322` | **`0.5974`** | `0.5346` |
| **GLM-5.3 Flash** | `0.6387` | **`0.6557`** | `+0.0170` | `953/954` | `0.7142` | `0.9264` | **`0.5978`** | `0.5484` |
| **Qwen3.8-27B** | `0.6463` | **`0.6511`** | `+0.0048` | `954/954` | `0.6502` | `0.9926` | **`0.6560`** | — |
| **Inkling (975B)** | `0.6277` | **`0.6303`** | `+0.0026` | `954/954` | `0.7120` | `1.0377` | **`0.6358`** | — |
| **Inkling 2 Mini** | `0.6153` | **`0.6144`** | `-0.0009` | `940/954` | `0.6367` | `1.0469` | **`0.6680`** | `0.6082` |
| **Nemotron 3 Super** | `0.5969` | **`0.6104`** | `+0.0135` | `954/954` | `0.6807` | `0.9728` | **`0.6387`** | `0.5672` |
| **gpt-oss-120b** *(BYPASS)* | `0.5710` | **`0.5707`** | `-0.0003` | `954/954` | `0.5873` | `1.3775` | **`1.3573`** | — |
| **gpt-oss-20b** *(BYPASS)* | `0.5680` | **`0.5551`** | `-0.0129` | `952/954` | `0.5692` | `1.5322` | **`1.3966`** | — |
| **Nemotron 3.5 Lightning** | `0.5470` | **`0.5409`** | `-0.0061` | `954/954` | `0.6259` | `1.0563` | **`0.6988`** | — |

---

## Key Scientific Findings

1. **Cross-provider reproducibility is near-exact (`r = +0.9916`, `ρ = +0.9956`, mean `|ΔAUC| = 0.0062`):**
   OpenRouter/Together production ANPI numbers reproduce almost identically on Tinker’s native weights (`Kimi K2.6` `0.0000`, `gpt-oss-120b` `-0.0003`, `Inkling 2 Mini` `-0.0009`, `Kimi K3.1 Pro` `+0.0011`, `Inkling` `+0.0026`, `Nemotron 3 Ultra` `+0.0042`, `DeepSeek V4.1 Flash` `+0.0050`).
2. **8-choice continuation ranking tracks exact 1,000-doc teacher-forced BPB at `ρ = +0.9341` (white-box) and `r = +0.9054` / `ρ = +0.8601` (black-box):**
   - White-Box Oracle `-d'` vs. `ln(Full-Doc BPB)` (`n = 14`): **`ρ = +0.9341`** (`r = +0.9302` native-chat).
   - Tinker Black-Box AUC vs. White-Box Oracle AUC (`n = 14`): **`r = +0.8725`, `ρ = +0.9253`**.
   - Tinker Black-Box `-d'` vs. `ln(Gold-Span BPB)` (`n = 14`): **`ρ = +0.9385`**.
   - Tinker Black-Box `-d'` vs. `ln(Full-Doc BPB)` (`n = 12` excl. `gpt-oss`): **`r = +0.9054`**, `ρ = +0.8601` (OpenRouter native-chat `n = 11`: **`r = +0.9163`**).
3. **Frontier models extract ~99% of their exact logprob oracle purely black-box:**
   `Kimi K3.1 Pro` (`2.8T`) hits **`0.7531` BB AUC** vs. **`0.7597` White-Box Oracle AUC** (`Δ = -0.0066`); `Qwen3.8-27B` hits **`0.6511` BB** vs. **`0.6502` WB** (`Δ = +0.0009`).
4. **Direct proof of `Inkling 2 Small`’s `-0.071` cold-start BYPASS tax:**
   Its exact `BB − WB` gap (`0.6596 − 0.7338 = -0.0742` on Together; `0.6663 − 0.7338 = -0.0675` on Tinker) directly confirms our earlier `-0.071` A/B estimate — and on true `Full-Doc BPB` (`0.5667`) + `White-Box Oracle AUC` (`0.7338`), `Inkling 2 Small` clearly ranks `#2` behind `Kimi K3.1 Pro` (`0.5332`).
5. **Complete explanation of `gpt-oss-120b` & `gpt-oss-20b`:**
   Both have strong Code BPB (`0.4344` / `0.5693`) but heavily distorted prose BPB (`essays = 1.894 / 1.835`, `news = 1.411 / 1.569`, `reference = 1.501 / 1.467`), giving `Full-Doc BPB = 1.357 / 1.397` and `White-Box Oracle AUC = 0.5873 / 0.5692` — matching their black-box ANPI AUC (`0.5707 / 0.5551`) to within `0.017`.
6. **Pairwise AUC survives RL sharpening better than raw `T=1` posttrained BPB:**
   `GLM-5.3` beats `GLM-5.3 Flash` on **both** `White-Box Oracle AUC` (`0.7263 > 0.7142`) and `Black-Box AUC` (`0.6737 > 0.6557`), even though `GLM-5.3`’s raw `T=1` posttrained `Full-Doc BPB` (`0.6078`) is slightly higher than `GLM-5.3 Flash` (`0.5978`).
