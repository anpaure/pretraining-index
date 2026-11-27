# Anpaure Pretraining Index (ANPI) & Black-Box Perplexity Harness

**Live Interactive Leaderboard:** https://anpaure.github.io/pretraining-index/ · **Technical Writeup:** https://anpaure.github.io/pretraining-index/writeup.html (`WRITEUP.md`)

A reproducible, pure black-box pretraining benchmark (**Rank-Coded Continuation Ranking / Rank-NCE**) covering **105 live strict zero-CoT models across 18 labs** (`37` open-weight models—including `14` native-chat anchors + `3` visible `BYPASS` models matched to [Elie Bakouch’s Pre-training Index](https://eliebak.com/pretraining-index) and `20` additional open endpoints such as `Qwen3.8 2.4T-A95B`, `MiMo-V2.6-Pro`, `Mistral Large 3`, `Llama 4 Maverick`, and `GLM-5.x`—and `68` closed frontier endpoints across OpenAI, Anthropic, Google, xAI, Meta, Qwen, Zhipu, Tencent, Mistral, MiniMax, and StepFun, including all 10 published OpenAI Nov-22 base-chart models) on **`N = 954` Nov-2026 post-cutoff items across 5 domains**, plus the earlier **Protocol A** Direct OpenAI Responses logprob BPB harness.

- **Interactive Dashboard:** `anpaure_pretraining_index.html` (mirrored at `docs/index.html`)
- **Technical Writeup:** `WRITEUP.md` (rendered at `docs/writeup.html`)
- **Independent Audit Verifier:** `python3 audit_v1.py` (`20/20` checks: byte-identical rebuild, `99,921`-row trace replay across `105` live models, strict permutation check, runner-dashboard parity, EXTRAP & log-BPB LOLO verification, 15-metric answer-pattern health check, secret scan)
- **Unit Test Suite:** `python3 -m unittest discover -s tests -v` (`19/19` passing, including `tests/test_trace_health.py`)

---

## Headline Results (`N = 954` Nov-2026 Multi-Domain Items, `908` Unique Source URLs)

Primary native zero-fit score is **Pairwise Rank AUC** ($\text{AUC} = (8 - \overline{r})/7 = \mathbb{P}(S(y_{\text{true}}) > S(y_{\text{decoy}}))$, chance $= 0.500$) on 8-choice continuation ranking against neutral 24B cloud decoys (`mistral-small-24b-instruct-2501`, `T = 1.0`), with `MRR` and `Top-1%` reported alongside. Across all `99,921` clean rank rows, candidate scores follow **Thurstone Case V** (Gaussian sums of token log-probs, `6.8×` lower KL than Plackett-Luce), and tail ranks (`r ≥ 2`) retain strong partial correlation with true BPB after controlling for Top-1. External translations onto OpenAI’s Nov-22 base scale and Elie Bakouch’s Oct-2026 Uncheatable Eval use the 2-parameter log-BPB Thurstone ruler $\ln(\text{BPB}) = a - b \cdot d'$ ($d' = \sqrt{2}\,\Phi^{-1}(\text{AUC})$, derived independently from Thurstone Case V and [Ruan et al., NeurIPS 2024](https://arxiv.org/abs/2405.10938)), with exact exponentiated Student-$t$ outer intervals and source-URL cluster-bootstrapped inner intervals.

| Model | Lab | Clean N | CoT Items (`r_tok>0`) | Native AUC (`↑`) | Native MRR (`↑`) | Top-1% | OpenAI Scale (`↓`) | Elie Unch BPB (`↓`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Claude Fable 5.2** | Anthropic | 954/954 | **`0`** (`0.0%`) | **0.8552** | 0.7353 | 59.5% | 0.749 *(extrap)* | 0.391 *(extrap)* |
| **Gemini 4 Carbon** | Google | 954/954 | **`0`** (`0.0%`) | **0.8482** | 0.7300 | 59.2% | 0.756 *(extrap)* | 0.397 *(extrap)* |
| **GPT-6.2 Astra** | OpenAI | 954/954 | **`0`** (`0.0%`) | **0.8444** | 0.7114 | 55.8% | **0.761** *(pub)* | 0.400 *(extrap)* |
| **Gemini 4 Argon** | Google | 954/954 | **`0`** (`0.0%`) | **0.8432** | 0.7206 | 58.0% | 0.760 | 0.401 *(extrap)* |
| **Gemini 4 Neon** | Google | 954/954 | **`0`** (`0.0%`) | **0.8291** | 0.7073 | 56.1% | 0.772 | 0.412 *(extrap)* |
| **Claude Opus 5.5** | Anthropic | 953/954 | **`0`** (`0.0%`) | **0.8180** | 0.6859 | 53.0% | 0.781 | 0.421 *(extrap)* |
| **GPT-6 Astra** | OpenAI | 954/954 | **`0`** (`0.0%`) | **0.8169** | 0.6823 | 52.8% | **0.783** *(pub)* | 0.422 *(extrap)* |
| **GPT-6.1 Sol** | OpenAI | 954/954 | **`0`** (`0.0%`) | **0.8119** | 0.6703 | 50.6% | 0.786 | 0.426 *(extrap)* |
| **Claude Fable 5.1** | Anthropic | 933/954 | **`0`** (`0.0%`) | **0.8080** | 0.6770 | 52.7% | 0.789 | 0.429 *(extrap)* |
| **Claude Fable 5** | Anthropic | 940/954 | **`0`** (`0.0%`) | **0.7929** | 0.6444 | 47.2% | 0.801 | 0.441 *(extrap)* |
| **Gemini 3.1 Pro** | Google | 950/954 | **`0`** (`0.0%`) | **0.7904** | 0.6406 | 46.2% | 0.803 | 0.443 *(extrap)* |
| **GPT-6.2 Sol** | OpenAI | 954/954 | **`0`** (`0.0%`) | **0.7893** | 0.6398 | 47.0% | 0.804 | 0.443 *(extrap)* |
| **Kimi K3.1 Pro** *(#1 Open)* | Moonshot | 954/954 | **`0`** (`0.0%`) | **0.7520** | 0.5875 | 39.9% | 0.832 | **0.484** *(exact)* |

### Key Source-URL Cluster-Bootstrapped Head-to-Head Comparisons *(All Zero-CoT, Pairwise AUC)*
- **Claude Fable 5.2 vs. Claude Opus 5.5** (`n=951`, `905` clusters): `+0.0373` AUC `[+0.023, +0.051]`, `+6.73%` Top-1, **`p < 0.0001`**
- **Claude Fable 5.2 vs. Gemini 4 Neon** (`n=953`, `907` clusters): `+0.0261` AUC `[+0.013, +0.040]`, `+3.57%` Top-1, **`p = 0.0004`**
- **Claude Fable 5.2 vs. GPT-6.2 Astra** (`n=953`, `907` clusters): `+0.0106` AUC `[-0.002, +0.024]`, `p = 0.116` *(MRR `+0.0234`, `p = 0.024`)*
- **Claude Fable 5.2 vs. Gemini 4 Carbon** (`n=953`, `907` clusters): `+0.0069` AUC `[-0.006, +0.020]`, `p = 0.304` *(tie at #1)*
- **GPT-6.2 Astra vs. Gemini 4 Neon** (`n=954`, `908` clusters): `+0.0153` AUC `[+0.002, +0.029]`, **`p = 0.024`**
- **GPT-6.2 Astra vs. Claude Opus 5.5** (`n=951`, `905` clusters): `+0.0266` AUC `[+0.012, +0.041]`, **`p < 0.0001`**
- **GPT-6.2 Astra vs. GPT-6 Astra** (`n=954`, `908` clusters): `+0.0276` AUC `[+0.018, +0.038]`, **`p < 0.0001`**

---

## Validation & Honest Methodological Caveats

1. **External agreement (2-parameter Thurstone $d' \to \ln\text{BPB}$, #1 of 24 zero-extra-parameter transforms across 4 independent datasets):**
   - **OpenAI Nov-22 Base Chart** (`10` models, `GPT-4o → GPT-6.2 Astra`): **`r = +0.990`**, **`ρ = +0.988`**, **`0.85%` within-OpenAI LOO error** (`1.44%` blind upward extrapolation from `GPT-4o..5.6-Sol` to `GPT-6/6.2 Astra`).
   - **Elie Bakouch Oct-2026 Uncheatable Eval** (`14` native `/chat/completions` open models across `7` labs): **`r = +0.9113`**, **`ρ = +0.8418`**, **`2.87%` LOO**, **`3.41%` leave-one-lab-out (LOLO)**.
   - **Mandatory-reasoning `BYPASS` endpoints (kept visible, excluded from fit by default via `INCLUDE_TEMPLATE_BYPASS_ANCHORS = False`):** `Muse Spark Open`, `Qwen3.8 2.4T-A95B`, `Inkling 2 Small`, `Reflection Beam 1.0`, `gpt-oss-120b`, and `gpt-oss-20b` required raw `/completions` template bypass—imposing a verified cold-start tax (`+3.6%` to `+6.3%` BPB vs. Elie base) relative to native chat endpoints.
   - **Cross-decoy & cross-corpus invariance (`n = 24` shared models):** `OLMo-2-1B-Base` 40-doc Hansard vs. `Mistral-Small-24B` 954-doc 5-domain gives **`ρ = +0.950`, `r = +0.919`** (zero statistically significant ceiling curvature across all 4 datasets, `mean |t_2| = 1.09`).
2. **Transparent limitations (verified by independent review in `runs/anpaure_v1_scientific_review.md`):**
   - **Ranks do not uniquely identify absolute generative loss:** Global monotone sharpening preserves the ordering of candidate likelihoods `p(y|x)` (not likelihood ratios `p(y|x)/q(y|x)` when `q` is nonuniform). Absolute BPB values depend on empirical calibration and extrapolate (`extrap`) for `Claude Fable 5.2` and `Gemini 4 Carbon` on OpenAI’s scale and the top closed tier on Elie’s scale.
   - **Code median-length centering artifact (`22.7%` vs. `12.5%` chance):** Because v1 decoys were filtered symmetrically around `target_len`, picking the candidate closest to median length achieves `22.7%` Top-1 in Code (`12.1%–14.1%` chance in Papers/News/Reference/Essays). Removing all 162 Code items preserves **`r = +0.9979, ρ = +0.9956`** across all `105` live models and leaves the top-9 AUC set & top-9 MRR ordering 100% invariant.
   - **Coverage & CoT telemetry:** All `105` live endpoints run at strict zero chain-of-thought across **`0 / 99,921`** clean items (`Muse Wonder` on Google Vertex bills Meta’s fixed empty-thought ATEM channel header, `raw_r_tok ≤ 7, rd_bytes = 672`, identical byte-for-byte to `"Reply OK"`). Exact per-model clean counts, `0/N` CoT telemetry, and 15 automated answer-pattern health checks (`tests/test_trace_health.py`) are audited on every build.

---

## Reproducing Locally ($0 — From Archived Traces)

```bash
# Rebuild HTML dashboard + calibrated JSON byte-for-byte from archived JSONL traces
python3 build_anpaure_index_v1.py

# Run full independent verification audit (trace replay, parity, EXTRAP, LOLO, secret scan)
python3 audit_v1.py

# Run Protocol A + NCE unit test suite (14 tests)
python3 -m unittest discover -s tests -v
```

### Repository Layout
- `anpaure_pretraining_index.html` / `index.html`: Self-contained interactive v1 dashboard
- `build_corpus_v1.py`, `topup_corpus_v1.py`: Fresh Nov-2026 5-domain corpus builders (`data/anpaure_v1_corpus_1000.jsonl`)
- `generate_decoys_v1.py`: Cloud `/api/v1/completions` neutral 24B decoy generator (`runs/anpaure_v1_n1000.json`)
- `run_anpaure_v1.py`: Checkpointed multi-provider OpenRouter evaluation runner (`runs/anpaure_v1_traces/*.jsonl`)
- `build_anpaure_index_v1.py`: Calibration, source-URL cluster bootstrap, and HTML dashboard builder
- `audit_v1.py`: Independent end-to-end verifier
- `pplx_protocol_a/` & `tests/`: Protocol A Direct OpenAI Responses logprob BPB harness, NCE scoring utilities, and unit tests
- `runs/anpaure_v1_scientific_review.md`: Independent review whose 5 findings were audited and addressed in this build
