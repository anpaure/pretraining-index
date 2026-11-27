# Measuring Pretraining Perplexity Through a Black Box

**Live Leaderboard:** [anpaure.github.io/pretraining-index](https://anpaure.github.io/pretraining-index/) · **Elie & BPBench Diff:** [comparison.html](https://anpaure.github.io/pretraining-index/comparison.html) · **Code & Raw Traces:** [github.com/anpaure/pretraining-index](https://github.com/anpaure/pretraining-index)

*November 26, 2026*

---

## TL;DR

On November 22, 2026, OpenAI’s chief scientist Jakub Pachocki argued in [*Perplexity and Intelligence*](https://openai.com/index/perplexity-and-intelligence/) that base-model log-perplexity is the cleanest single-number measure of general pretraining progress—but released no public evaluation set. Two days later, Mercor proposed averaging six public multiple-choice exams as a proxy, showing a `−0.98` rank correlation on nine OpenAI models, while raising the obvious open question: *does a downstream exam proxy actually measure pretraining compression across labs, or just post-training and time trends?*

We built the **Anpaure Pretraining Index**: a pure black-box, zero-logprob **8-choice continuation-ranking benchmark** over **`N = 954` provably post-cutoff Nov-2026 items** (`908` unique source URLs, 5 domains) against neutral 24B cloud decoys, evaluated at strict **`0 / 43,417` reasoning tokens** across **46 live open and closed endpoints** (`$46.40` total API spend for all 46 models combined).

Scoring each model by zero-fit **Pairwise Rank AUC** $= (8 - \overline{r})/7$ and mapping Thurstone Case V discriminability $d' = \sqrt{2}\,\Phi^{-1}(\text{AUC})$ linearly onto $\ln(\text{BPB})$ yields:

| Ground-Truth Test | **Anpaure** *(0-CoT AUC)* | **`AA-Omniscience-Acc`** *(full 222-model table)* | **`SimpleQA-Verified`** *(Epoch)* | **Mercor** *(6-MCQ)* |
|---|---:|---:|---:|---:|
| **OpenAI 10-model `r` / `ρ`** (`GPT-4o → 6.2 Astra`) | **`+0.990` / `+0.997`** | `+0.880` / `+0.997` | `+0.913` / `+0.900` | `+0.852` / `+0.930` |
| **OpenAI 10-model blind LOO error** | **`0.85%`** *(max `1.8%`)* | `3.27%` *(max `6.4%`)* | `2.69%` *(max `6.0%`)* | `3.61%` *(max `7.0%`)* |
| **OpenAI 11-model LOO** *(incl. `o3 = 0.902`)* | **`0.88%`** *(max `1.9%`)* | `3.14%` *(max `6.5%`)* | `2.95%` *(max `6.4%`)* | `3.73%` *(max `7.6%`)* |
| **OpenAI blind upward extrap** (`4o..5.6 → GPT-6 / 6.2`) | **`1.44%`** | `8.55%` | `7.27%` | `8.68%` |
| **Same-base invariance: `o3` vs. `GPT-4.1`** *(both `0.902`)* | **`0.76%` BPB gap** | `3.44%` *(`38.6%` vs `27.8%`)* | `5.18%` *(`49.4%` vs `31.1%`)* | `6.89%` *(`69.1%` vs `48.1%`)* |
| **Cross-lab Elie open BPB** *(13 shared native-chat, 6 labs)* | **`r=+0.907, LOLO=3.70%`** *(full 14 / 7 labs: **`3.46%`**)* | `r=+0.858, LOLO=4.72%` *(**`ρ=+0.901`** vs `+0.835`)* | *(only 6 shared)* | *(untested)* |
| **Reasoning / CoT purity** | **Strict `0 / 43,417` CoT** | `+8–10` pt effort drift | Leaks CoT *(`o3 +18` pts)* | Leaks CoT *(`o3 +21` pts)* |
| **Frontier headroom (`Claude Fable 5.2`)** | **`59.5%` Top-1 / `0.855` AUC** | `79.1%` | **`94.0%`** *(saturated)* | **`90.4%`** *(saturated)* |
| **Cost per frontier model** | **`$1–$3`** | — | — | `$35–$88` *(full set)* |

On our strict zero-CoT benchmark, **`Claude Fable 5.2`** ranks `#1` (`0.8550` AUC, `59.5%` Top-1), followed closely by **`Gemini 4 Carbon`** (`0.8482`), **`GPT-6.2 Astra`** (`0.8444`), and **`Gemini 4 Argon`** (`0.8433`), with **`Kimi K3.1 Pro`** (`0.7518`) leading all open-weight models—and stripping leaked reasoning tokens from **`Gemini 4 Neon`** (`953/953` items) drops it significantly behind `GPT-6.2 Astra` (`p = 0.024`).

---

## 1. The Problem: You Can’t Measure Perplexity on Closed APIs Directly

Teacher-forced bits-per-byte (BPB) is easy to measure on open-weight base checkpoints—[Elie Bakouch’s Pre-training Index](https://eliebak.com/pretraining-index) does this cleanly across 27 open models. On closed frontier APIs, direct logprob evaluation fails for two independent reasons:

1. **API lockouts:** Anthropic and Google omit token logprobs entirely on frontier endpoints, and OpenAI disabled `echo=true` prompt-token logprobs on modern chat APIs.
2. **Post-training RL sharpening destroys raw logprob BPB:** Even where top-$K$ next-token logprobs *are* returned, RLHF and reasoning post-training sharply peak the output distribution (effective temperature $T < 1$). In our early Protocol A logprob trials (`pplx_protocol_a/`), raw next-token BPB correlated *negatively* (`r = −0.136`) with ground-truth pretraining quality across RL-tuned endpoints because overconfident wrong tokens explode $-\log p$.

Meanwhile, indirect downstream exam proxies face three structural issues:
- **Contamination and ceiling saturation:** `Claude Fable 5.2` already scores `90.4%` on Mercor’s 6-MCQ average (`95.0%` MMLU-Pro, `95.5%` GPQA-Diamond, `97.6%` SimpleQA-MCQ) and `94.0%` on `SimpleQA-Verified`.
- **Reasoning / post-training leakage:** Running models at *"lowest reasoning strength available"* still injects chain-of-thought on reasoning-tuned endpoints, creating a `21.0`-point gap on Mercor between `o3` (`69.1%`) and `GPT-4.1` (`48.1%`) even though both share OpenAI’s identical `0.902` pretrained base.
- **Unverified cross-lab validity:** Inside one lab (`GPT-4o` 2024 $\to$ `GPT-6.2` 2026), almost every capability scales monotonically with time—so high within-OpenAI correlation does not prove a benchmark isolates pretraining compression across labs.

---

## 2. Method: Rank-Coded Continuation Ranking (Anpaure v1)

Instead of asking trivia or exam questions, we reduce next-token language modeling directly to a **zero-logprob 8-way ranking game** over unseen post-cutoff text:

1. **Provably post-cutoff 5-domain corpus (`N = 954`, `908` unique source URLs):** Every source document carries a verified November 2026 timestamp across five domains:
   - **News / Parliamentary Speech (`199`):** UK Hansard debates and BBC articles.
   - **Research Papers (`198`):** New arXiv and bioRxiv preprints.
   - **Essays & Commentary (`198`):** Guardian opinion and LessWrong posts.
   - **Encyclopedia (`197`):** Newly created November 2026 Wikipedia articles.
   - **Systems Code (`162`):** Newly added single-line blocks from `116` distinct November 2026 commits across mature repositories (CPython, Linux, PostgreSQL, Go, LLVM, PyTorch, ClickHouse, FFmpeg, etc.).
2. **Neutral 24B cloud decoys (`K = 8`):** At a natural cut point in each document, we take the authentic next snippet (~6 words of prose or ~36 bytes of code) and sample **7 distinct $T = 1.0$ raw continuations** from `mistralai/mistral-small-24b-instruct-2501` via `/api/v1/completions` (fixing the trailing-space BPE boundary bug so prefixes never split subwords). The decoy generator has zero family overlap with any ranked frontier model or calibration anchor.
3. **Strict 8-letter permutation prompt:** Each evaluated model receives the document prefix and the 8 randomly shuffled candidates `A..H` and must output an exact 8-distinct-letter permutation (`C,A,F,B,H,D,E,G`) from most likely authentic continuation to least likely. Both the runner (`run_anpaure_v1.py`) and independent verifier (`audit_v1.py`) reject any malformed, truncated, or duplicate-letter output.
4. **Strict Zero-CoT enforcement (`0 / 43,417` items):** We suppress hidden reasoning tokens via explicit no-scratchpad system prompting (validated on a 100-item `Gemini 4 Argon` control with zero prompt shift, `p = 0.85`), plus forced `tool_choice` with `role: "developer"` on `o3`, `GLM-5.3 Flash`, and `Inkling`. Pre-suppression CoT traces are archived in `runs/anpaure_v1_traces_cot_archive/`.

**Why ranking survives global RL sharpening (and an exact caveat):** Any global monotone sharpening transformation $p(y \mid x) \mapsto p(y \mid x)^\beta / Z$ leaves the ranking of candidate likelihoods $p(y_k \mid x)$ strictly invariant—unlike raw cross-entropy. However, as our independent review (`runs/anpaure_v1_scientific_review.md`) correctly noted, monotone sharpening does *not* preserve likelihood-ratio ordering $p(y \mid x) / q(y \mid x)$ when the decoy distribution $q$ is nonuniform, and ordinal rankings alone do not mathematically identify absolute generative BPB without empirical anchor calibration.

---

## 3. Why Pairwise AUC + Thurstone $d' \to \ln(\text{BPB})$

Early in v1 we scored models with Mean Reciprocal Rank ($\text{MRR} = \text{mean}(1/r)$) and calibrated linearly on $-\ln(\text{MRR})$. Testing 24 zero-extra-parameter linking functions and three 3-parameter scaling laws across **four independent datasets** (our `Mistral-24B` 5-domain suite plus our earlier `OLMo-2-1B` Hansard suite, crossed against both Elie and OpenAI ground truth) uncovered three cleaner mathematical facts:

### A. Real continuation rankings are Gaussian (Thurstone Case V), not Plackett-Luce
Each candidate snippet is ~8–12 tokens long, so its log-likelihood $S_m(y) = \sum_t \log p_m(y_t \mid x, y_{<t})$ is a sum of token log-probabilities—**Gaussian by the Central Limit Theorem**, not Gumbel. Matching a single scalar (`AUC`) for each live model and predicting its entire 8-bin rank histogram $P(r = 1 \dots 8)$ with zero free shape parameters across all `41,510+` observations:

- **Plackett-Luce (Gumbel / softmax choice):** mean $\mathrm{KL} = 0.0391$ nats, mean $|\text{MRR}_{\text{pred}} - \text{MRR}_{\text{emp}}| = 0.0352$ *(predicts `45.7%` Top-1 for `Fable 5.2` vs. `59.5%` actual)*.
- **Thurstone Case V (Gaussian scores):** mean $\mathrm{KL} = \mathbf{0.0057}$ nats (**`6.8×` better fit**), mean $|\text{MRR}_{\text{pred}} - \text{MRR}_{\text{emp}}| = \mathbf{0.0096}$.

Under Thurstone Case V ($S_{\text{true}} \sim \mathcal{N}(\mu_{\text{true}}, \sigma^2)$, decoys $\sim \mathcal{N}(\mu_{\text{decoy}}, \sigma^2)$), the exact pairwise win probability is the **Mann–Whitney U / ROC AUC**:
$$\text{AUC}_m = \frac{8 - \overline{r}_m}{7} = \mathbb{P}\!\big(S_m(y_{\text{true}}) > S_m(y_{\text{decoy}})\big) = \Phi\!\left(\frac{d'_m}{\sqrt{2}}\right)$$
whose exact inverse is Signal Detection Theory’s dimensionless discriminability:
$$d'_m = \sqrt{2}\,\Phi^{-1}(\text{AUC}_m)$$
Any constant temperature multiplier $\beta_m$ in $S_m(y) = \beta_m \log p_m(y \mid x)$ cancels out identically in $d'_m = (\mu_{\text{true}} - \mu_{\text{decoy}}) / \sigma$.

### B. Tail ranks (`r ≥ 2`) carry massive independent pretraining signal
Checking whether a model’s **conditional mean rank among misses ($r \ge 2$)** still predicts true BPB *after controlling for Top-1 accuracy*:

- `24B-OpenAI (10 models)`: partial $r = \mathbf{+0.730}$
- `1B-Elie (13 models)`: partial $r = \mathbf{+0.761}$
- `1B-OpenAI (10 models)`: partial $r = \mathbf{+0.734}$
- `24B-Elie (14 models)`: partial $r = \mathbf{+0.280}$

Because `MRR` puts ~75% of its variance on `r = 1` vs. `r ≥ 2`—and binary MCQ throws away ranks `2–8` completely—switching to pairwise `AUC` immediately jumps our 10-model OpenAI Spearman rank correlation from **`+0.891` $\to$ `+0.997`** (putting every generation from `GPT-4o` to `GPT-6.2 Astra` in exact monotonic order) and cuts OpenAI LOO error from **`1.18%` $\to$ `0.85%`**.

### C. Two-parameter log-BPB scaling beats 3-parameter curves out-of-sample
In [Ruan, Maddison & Hashimoto (NeurIPS 2024)](https://arxiv.org/abs/2405.10938) (*Observational Scaling Laws*), $\ln(\text{BPB}_m) \approx \beta_1 \ln C_m + \alpha_1$ and $\sigma^{-1}(\text{Acc}_m) \approx \beta_2 \ln C_m + \alpha_2$. Eliminating compute $\ln C_m$ gives the exact 2-parameter log-BPB Thurstone ruler:
$$\ln(\text{BPB}_m) = a + b \cdot (-d'_m) = a - b \cdot \sqrt{2}\,\Phi^{-1}(\text{AUC}_m)$$

Across all four independent datasets (including `OLMo-1B` where frontier models hit `AUC = 0.94`, stress-testing ceiling saturation):
- **`probit_auc [logY]` ranked `#1` of 24 zero-extra-parameter transforms** (`3.11%` 4-dataset average vs. `4.02%` for `-ln(MRR)`).
- It had **zero statistically significant curvature on all four datasets** (`mean |t_2| = 1.09`, whereas `-ln(MRR)` and linear `mean_rank` bent sharply at ceiling saturation with `t_2 = −3.4` to `−3.8`).
- Every 3-parameter curve we tested—[Gadre et al. (ICLR 2025)](https://arxiv.org/abs/2403.08540) Eq. 5, Ruan floor-sigmoid, and Chinchilla $E_0$-offset—**overfitted** on `n = 10–14` anchors (`24B-Elie` LOLO worsened to `4.10%–4.66%`).
- Computing Student-$t$ (`df = n − 2`) outer intervals and URL-cluster bootstrap inner intervals in log-BPB space and exponentiating back guarantees strictly positive BPB bounds even under upward extrapolation.

---

## 4. Empirical Validation: Within-Lab, Cross-Lab, and Discriminant Validity

### Within-OpenAI Ground Truth (`10` and `11` Published Models)
Across all 10 OpenAI Nov-22 base-chart checkpoints (`GPT-4o` through `GPT-6.2 Astra`), our 2-parameter ruler recovers published base log-perplexity with **`r = +0.9896`, `ρ = +0.9970`, and `0.85%` mean blind LOO error** (`1.79%` worst single miss):

| OpenAI Model | Native AUC | Thurstone $d'$ | OpenAI Published | Blind LOO Pred | LOO `% Err` |
|---|---:|---:|---:|---:|---:|
| **GPT-6.2 Astra** | `0.8444` | `+1.432` | **`0.761`** | `0.7566` | **`0.58%`** |
| **GPT-6 Astra** | `0.8169` | `+1.278` | **`0.783`** | `0.7820` | **`0.13%`** |
| **GPT-5.6 Sol** | `0.7397` | `+0.909` | **`0.828`** | `0.8428` | `1.79%` |
| **GPT-5.5** | `0.7323` | `+0.876` | **`0.848`** | `0.8464` | **`0.19%`** |
| **GPT-5.4** | `0.6857` | `+0.684` | **`0.871`** | `0.8803` | `1.07%` |
| **GPT-5.2** | `0.6827` | `+0.672` | **`0.888`** | `0.8803` | `0.87%` |
| **GPT-5** | `0.6801` | `+0.662` | **`0.893`** | `0.8816` | `1.28%` |
| **GPT-5.1** | `0.6697` | `+0.621` | **`0.893`** | `0.8897` | **`0.37%`** |
| **GPT-4.1** | `0.6626` | `+0.594` | **`0.902`** | `0.8936` | `0.93%` |
| **GPT-4o** | `0.6358` | `+0.491` | **`0.904`** | `0.9158` | `1.31%` |

- **True blind upward extrapolation:** Training *only* on `GPT-4o..GPT-5.6 Sol` (zero GPT-6 models seen) predicts **`GPT-6 Astra` = `0.774`** (`−1.20%`) and **`GPT-6.2 Astra` = `0.748`** (`−1.67%`).
- **Same-base invariance:** Models sharing an identical pretrained base stay tightly aligned—`o3` (`0-CoT`) vs. `GPT-4.1` (`0.888` vs. `0.895`, **`0.76%` gap**, and `0.6479` vs. `0.6503` on the 792 prose items, `z = +0.26`), `GPT-5` vs. `GPT-5.1` (`0.80%` gap), and `GPT-5.3 Codex` vs. `GPT-5.4` (`0.28%` gap).
- **Discriminant validity over general capability indexes:** Controlling for Epoch’s ECI across OpenAI models (`n = 9`), Anpaure retains partial correlation **`r(Anpaure, log OAI | ECI) = +0.967`** (vs. `r(ECI, log OAI | Anpaure) = +0.564`); on Elie’s open models, controlling for the Artificial Analysis Intelligence Index gives partial **`r(Anpaure, Elie | AA Index) = +0.758`** vs. `r(AA Index, Elie | Anpaure) = −0.049`.

### Cross-Lab Open-Model Ground Truth (`14` Native-Chat Models Across `7` Labs)
On Elie Bakouch’s Oct-2026 Uncheatable Eval BPB across 14 non-OpenAI open models from 7 independent labs (`Moonshot`, `DeepSeek`, `NVIDIA`, `Zhipu`, `Google`, `RicursiveAI`, `Thinking Machines`), the same 2-parameter ruler achieves **`r = +0.9097`, `LOO = 2.91%`, and `7`-lab Leave-One-Lab-Out `LOLO = 3.46%`** (macro-averaged across families: `3.45%`; max `7.70%`).

Decomposing total residual variance into **finite-$N$ (`954` items) sampling noise** vs. **irreducible structural mismatch** explains the gap between within-lab and cross-lab precision:

| Scale | Total Residual SD | Finite-$N$ (`954`) Sampling Noise | Structural Mismatch |
|---|---:|---:|---:|
| **Within-OpenAI (`10` models)** | `0.94%` | **`0.79%`** (`71.1%` of variance) | `0.50%` (`28.9%` of variance) |
| **Cross-Lab Elie (`14` models, `7` labs)** | `3.43%` | `1.38%` (`16.2%` of variance) | **`3.14%`** (`83.8%` of variance) |

Across labs, residual signs cluster systematically by family (`DeepSeek` `+3.9%`, `Google Gemma` `−2.6%`, `Moonshot` `−1.4%`; between-family bias SD = **`3.16%`** vs. within-family scatter `1.96%`). That `~3.2%` lab-level offset reflects genuine differences in pretraining domain mixtures and post-training styles—and sets an honest error bar (`~±0.015–0.020` BPB) whenever comparing two closed labs head-to-head.

---

## 5. Two Methodological Traps Exposed by Our Experiments

### Trap 1: "Minimal Reasoning" Still Leaks CoT—and Flips Frontier Rankings
When we initially ran `Gemini 4 Neon`, `o3`, and `GLM-5.3 Flash` with OpenRouter’s lowest exposed reasoning effort, hidden reasoning tokens still fired on `89%–100%` of items. Enforcing strict `r_tok = 0` produced statistically significant drops on all three:

- **`Gemini 4 Neon`** (`n = 953`, old `953/953` had `r_tok > 0`): `AUC 0.8483 → 0.8299` (**`−0.0184`, `p = 0.0004`**), `MRR 0.7339 → 0.7078` (`−0.0261`, `p = 0.0004`).
- **`o3`** (`n = 951`, old `950/951` had `r_tok > 0`): `AUC 0.6924 → 0.6721` (**`−0.0203`, `p = 0.029`**).
- **`GLM-5.3 Flash`** (`n = 939`, old `837/939` had `r_tok > 0`): `AUC 0.6580 → 0.6399` (**`−0.0181`, `p = 0.046`**), dropping its blind Elie error to `1.20%`.

Notably, *with* leaked CoT, `Gemini 4 Neon` (`0.8483`) sat above `GPT-6.2 Astra` (`0.8444`)—matching Mercor’s ordering (`85.8%` vs. `85.5%`). Once CoT is stripped to strict zero, **`GPT-6.2 Astra` cleanly beats `Gemini 4 Neon` (`+0.0153` AUC, `p = 0.024`)**.

### Trap 2: Why Optimizing Benchmark Combinations on OpenAI (SOCP / `|I| ≤ 3`) Collapses Cross-Lab
A tempting alternative to designing a continuation benchmark is scraping a large public leaderboard table and solving a second-order cone program (SOCP) or sparse subset search (`|I| ≤ 3`) to maximize correlation with OpenAI’s published perplexity scores.

We ran that exact experiment on Epoch AI’s Nov-25-2026 `benchmark_data.zip` (`90` public CSVs; `8` complete-case benchmarks covering all `11/11` OpenAI models and `7–10` Elie open models):

1. **Time-trend confounding:** On OpenAI’s 11 models (lowest effort), the exact global `|I| ≤ 3` optimizer picked **`ARC-AGI-2` alone** (`r = +0.9809`, `LOO = 1.12%`), and the exact nonnegative SOCP picked **`0.74·ARC-AGI-2 + 0.18·SimpleQA + 0.08·Chess Puzzles`** (`r = +0.9889`). Yet when evaluated cross-lab on Elie’s exact teacher-forced BPB, **`ARC-AGI-2` collapsed from `r = +0.981` $\to$ `r = +0.088`** (`r = −0.210` at best effort), **`Chess Puzzles` went negative (`r = −0.126`)**, **`OTIS Mock AIME` went negative (`r = −0.251`)**, and **`GPQA-Diamond` collapsed to `r = +0.221`**. Inside one lab, any hard RL benchmark that scales over time shares OpenAI’s time trend without measuring pretraining compression.
2. **Selection bias on small `n`:** Even with a "simple" `|I| ≤ 3` cardinality cap on `n = 11` OpenAI targets, searching over `m = 25` or `40` **pure random Gaussian noise** columns produces mean fake in-sample correlations of **`r = +0.806`** (95th pct `+0.916`) and **`r = +0.883`** (95th pct `+0.947`)—and continuous SOCP hits **`r = +0.991`** on just 25 noise vectors.

The only two Epoch benchmarks that genuinely survived cross-lab transfer onto Elie were **`SimpleQA-Verified`** (`Elie r = +0.845`) and **`LMCA`** (`Elie r = +0.868`)—reinforcing that factual/conceptual compression tracks pretraining, whereas RL search puzzles do not.

---

## 6. Headline Frontier Results (`N = 954`, Strict `0 / 43,417` CoT)

| # | Model | Lab | Clean N | Native AUC (`↑`) | Native MRR (`↑`) | Top-1 | OpenAI Scale (`↓`) | Elie Unch BPB (`↓`) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **Claude Fable 5.2** | Anthropic | 953 | **0.8550** | 0.7350 | 59.5% | 0.749 *(EXTRAP)* | 0.391 `[0.344, 0.444]` *(EXTRAP)* |
| 2 | **Gemini 4 Carbon** | Google | 954 | **0.8482** | 0.7300 | 59.2% | 0.756 *(EXTRAP)* | 0.396 `[0.350, 0.449]` *(EXTRAP)* |
| 3 | **GPT-6.2 Astra** | OpenAI | 954 | **0.8444** | 0.7114 | 55.8% | **0.761** *(PUB; est `0.759`)* | 0.400 `[0.354, 0.451]` *(EXTRAP)* |
| 4 | **Gemini 4 Argon** | Google | 952 | **0.8433** | 0.7208 | 58.0% | 0.760 | 0.400 `[0.354, 0.452]` *(EXTRAP)* |
| 5 | **Gemini 4 Neon** | Google | 954 | **0.8291** | 0.7073 | 56.1% | 0.772 | 0.412 `[0.367, 0.463]` *(EXTRAP)* |
| 6 | **Claude Opus 5.5** | Anthropic | 951 | **0.8176** | 0.6852 | 52.9% | 0.782 | 0.421 `[0.377, 0.471]` *(EXTRAP)* |
| 7 | **GPT-6 Astra** | OpenAI | 954 | **0.8169** | 0.6823 | 52.8% | **0.783** *(PUB; est `0.782`)* | 0.422 `[0.378, 0.471]` *(EXTRAP)* |
| 8 | **GPT-6.1 Sol** | OpenAI | 954 | **0.8119** | 0.6703 | 50.6% | 0.786 | 0.426 `[0.382, 0.475]` *(EXTRAP)* |
| 9 | **GPT-6.2 Sol** | OpenAI | 951 | **0.7901** | 0.6410 | 47.1% | 0.804 | 0.443 `[0.400, 0.490]` *(EXTRAP)* |
| 10 | **GPT-6.1 Terra** | OpenAI | 954 | **0.7755** | 0.6215 | 44.5% | 0.815 | 0.454 `[0.412, 0.500]` *(EXTRAP)* |
| 11 | **Kimi K3.1 Pro** *(#1 Open)* | Moonshot | 951 | **0.7518** | 0.5875 | 40.0% | 0.832 | **0.484** *(EXACT)* |
| 12 | **GPT-6 Sol** | OpenAI | 954 | **0.7432** | 0.5798 | 39.5% | 0.839 | 0.478 `[0.437, 0.522]` |

**Key paired head-to-head tests** (`5,000` source-URL cluster resamples across `905–908` clusters):
- `Claude Fable 5.2` vs. `Gemini 4 Carbon`: `+0.0069` AUC (`p = 0.304`, statistical tie at `#1`)
- `Claude Fable 5.2` vs. `GPT-6.2 Astra`: `+0.0106` AUC (`p = 0.116`; `+0.0234` MRR, `p = 0.024`)
- `Claude Fable 5.2` vs. `Gemini 4 Neon`: `+0.0261` AUC (**`p = 0.0004`**)
- `Claude Fable 5.2` vs. `Claude Opus 5.5`: `+0.0373` AUC (**`p < 0.0001`**)
- `GPT-6.2 Astra` vs. `Gemini 4 Neon`: `+0.0153` AUC (**`p = 0.024`**)
- `GPT-6.2 Astra` vs. `GPT-6 Astra`: `+0.0276` AUC (**`p < 0.0001`**)

---

## 7. Honest Limitations & Independent Audit Disclosures

We ran an independent scientific review (`runs/anpaure_v1_scientific_review.md`) and tested over 30 alternative statistical/psychometric models (1PL/2PL/Probit Rasch, SVD PC1, full 8-letter decoy consensus ordering, `A..H` position-bias debiasing, 5-domain separate Elie fits) before freezing v1. Five real limitations remain:

1. **Code median-length artifact (`22.7%` vs. `12.5%` chance):** Because v1 decoys were symmetrically length-filtered around gold length, a zero-LLM closest-to-median-length heuristic scores `22.7%` Top-1 in Code (while all four prose domains sit at `12.1%–14.1%` chance). Removing all 162 Code items preserves **`r = +0.9986, ρ = +0.9927`** across all 46 live models, leaves the top-10 AUC set unchanged, and leaves the top-9 MRR ordering 100% identical.
2. **Mandatory-reasoning zero-think cold-start tax (`BYPASS` endpoints):** Two open Elie models (`Inkling 2 Small`, `0.5128`; `Reflection Beam 1.0`, `0.5157`) rejected zero-CoT on `/api/v1/chat/completions` and required raw `/api/v1/completions` zero-think template bypasses. A 50-item controlled check on `Inkling 2 Small` verified a **`−0.071` AUC cold-start tax** (`z = 2.07`), pushing both `+5.9% / +6.4%` harsher than Elie Uncheatable BPB (mirroring their weaker prose scores on Elie’s own News/Fiction splits). Both remain visible on the 46-model leaderboard (badged `BYPASS`) and are excluded from the 14-anchor native-chat OLS ruler by default (`INCLUDE_TEMPLATE_BYPASS_ANCHORS = False`; including them gives `3.94%` LOLO across 8 labs).
3. **Extrapolation above anchor ceilings:** `Claude Fable 5.2` and `Gemini 4 Carbon` exceed top OpenAI anchor `GPT-6.2 Astra` (`0.8444`), and the top closed tier exceeds top open anchor `Kimi K3.1 Pro` (`0.7518`). All such projections carry explicit `EXTRAP` badges and wide leverage-adjusted Student-$t$ intervals.
4. **Cross-lab rank order vs. `AA-Omniscience-Accuracy`:** While Anpaure beats `AA-Omniscience-Acc` on predictive error across both OpenAI (`0.85%` vs. `3.27%` LOO) and Elie (`3.70%` vs. `4.72%` LOLO on the 13 shared native-chat models), `AA-Omniscience-Acc` slightly edges Anpaure on Elie cross-lab Spearman rank order (`ρ = +0.901` vs. `+0.835`) because cold factual recall does not penalize heavy-reasoning MoEs.
5. **Physical ceiling of v1 data:** Testing 30+ modeling refinements proved we have extracted essentially 100% of the non-overfit signal from v1’s `43,417` traces; further gains require new v2 data.

---

## 8. Roadmap for v2 & Reproducing Everything Locally (`$0`)

Four concrete upgrades for **Anpaure v2** follow directly from our variance decomposition:
1. **Scale to `N = 2,500`** (1 item per URL) to cut the `71%` finite-$N$ noise floor (`~0.55%` OpenAI LOO) and separate `#1–#4` at `p < 0.01`.
2. **Remove the Code length filter** so all domains sit at exact `12.5%` length-heuristic chance.
3. **Multi-family *base*-model decoy ensemble** (3–4 mid-tier base `/completions` generators instead of a single instruct model) plus **fresh Fiction and multilingual (`ZH/DE/FR/JA`) slices**.
4. **Monthly rolling post-cutoff snapshots** to maintain a permanent, un-Goodhartable cross-lab perplexity-over-time curve.

Every number, chart, and trace in this writeup reproduces byte-for-byte from archived JSONL traces at zero API cost:

```bash
git clone https://github.com/anpaure/pretraining-index.git
cd pretraining-index
python3 build_anpaure_index_v1.py   # deterministic HTML + calibrated JSON rebuild
python3 audit_v1.py                 # 43,417-row raw trace replay, parity, EXTRAP, LOLO, secret scan
python3 -m unittest discover -s tests -v
```
