"""Build the Anpaure Pretraining Index v1 (N=954, 5 post-cutoff Nov-2026 domains,
harder 24B cloud decoys) interactive HTML dashboard.

Visual design inspired by the editorial minimalism of eliebak.com/pretraining-index
and deeprsi.com: EB Garamond display serif + IBM Plex Sans/Mono, subtle paper grain,
light/dark theme toggle, open DeepRSI-style hairline leaderboard with brand marks and
dual cluster-bootstrapped Student-t whiskers, plus an Elie-style diagonal-hatched
vertical pairwise win-rate column chart.

Primary native zero-fit score & calibration:
  - Native score: Pairwise Rank AUC = (8 - mean_rank) / 7 = P(S(y_true) > S(y_decoy)),
    capturing full tail-rank signal (r >= 2 partial correlation +0.73 to +0.76 with true BPB).
  - Latent scale: Thurstone Case V signal-detection discriminability d' = sqrt(2) * Phi^{-1}(AUC)
    (6.8x lower KL than Plackett-Luce across all 41,510 real rank observations).
  - External calibration: 2-parameter log-BPB OLS ruler ln(BPB) = a + b * (-d')
    (derived independently from Thurstone Case V and Ruan, Maddison & Hashimoto NeurIPS 2024;
    ranked #1 of 24 zero-extra-parameter transforms across 4 independent datasets with zero
    statistically significant ceiling curvature). Exact Student-t intervals are computed in
    log-BPB space and exponentiated back so predicted BPB bounds stay strictly positive.

Preserves 100% of the post-audit scientific rigor (all 5 independent-review findings):
  1. Exact mathematical framing (likelihood p(y|x) vs. likelihood-ratio p/q sharpening).
  2. Transparent disclosure of the 22.7% Code median-length centering artifact + 792-item
     prose-only sensitivity check (r = +0.9986, rho = +0.9920 across all 44 live models).
  3. Strict EXTRAP tagging outside [min_anchor_x, max_anchor_x] on both OpenAI & Elie
     scales + Leave-One-Lab-Out (LOLO) cross-validation across 7 independent open labs.
  4. Source-URL cluster bootstrap across all CIs and paired p-values (908 unique URLs).
  5. Strict 8-distinct-letter permutation parity with run_anpaure_v1.py, exact Student-t
     outer intervals, and strict 0/41,510 CoT verification across all 44 live endpoints.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from pplx_protocol_a.nce_bpb import score_rank_nce_cv

ROOT = Path(__file__).resolve().parent
BENCH_PATH = ROOT / "runs/anpaure_v1_n1000.json"
TRACE_DIR = ROOT / "runs/anpaure_v1_traces"
ELIE_PARSED = ROOT / "runs/elie_pretraining_index_parsed.json"
OUT_HTML = ROOT / "anpaure_pretraining_index.html"
OUT_JSON = ROOT / "runs/anpaure_v1_calibrated.json"

DOMAINS = ("code", "papers", "news", "reference", "essays")
CLEAN_RE = re.compile(r"^[A-H](?:\s*,\s*[A-H]){7}\s*$")
_NORM = NormalDist()
_SQRT2 = math.sqrt(2.0)

# Mandatory-reasoning endpoints that reject zero-CoT on /api/v1/chat/completions and
# required raw /api/v1/completions template-bypass (verified -0.071 AUC cold-start tax).
# Kept visible on the leaderboard (badged BYPASS) and excluded from the OLS ruler by default
# for apples-to-apples /api/v1/chat/completions parity. Flip to True to include in calibration.
TEMPLATE_BYPASS_MODELS = {
    "meta/muse-spark-open",
    "qwen/qwen3.8-2.4t-a95b",
    "thinkingmachines/inkling-2-small",
    "reflection/beam-1.0-501b-a23b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
}
INCLUDE_TEMPLATE_BYPASS_ANCHORS = False
SHOW_OAI_MILESTONES = False

# Exact two-sided Student-t 95% critical values t_{0.975, df}
_T95 = {
    3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
    9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448,
    15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
}


def neg_d_prime(auc: float) -> float:
    """Thurstone Case V negative discriminability -d' = -sqrt(2) * Phi^{-1}(AUC) (lower = better)."""
    p = min(0.9999, max(0.0001, float(auc)))
    return -_SQRT2 * _NORM.inv_cdf(p)

MODEL_META = {
    # 16 Open Elie Anchors across 8 independent labs (all kept — zero cherry-picking)
    "moonshotai/kimi-k3.1-pro":          ("Kimi K3.1 Pro",         "Moonshot",          "open",   "moonshotai__Kimi-K3.1-Pro"),
    "moonshotai/kimi-k3":                ("Kimi K3",               "Moonshot",          "open",   "moonshotai__Kimi-K3"),
    "meta/muse-spark-open":              ("Muse Spark Open",       "Meta",              "open",   "meta-models__Muse-Spark-Open-816B-A42B"),
    "meta/muse-wonder":                  ("Muse Wonder",           "Meta",              "closed", None),
    "thinkingmachines/inkling-2-small":  ("Inkling 2 Small",       "Thinking Machines", "open",   "thinkingmachines__Inkling-2-Small"),
    "deepseek/deepseek-v4-pro":          ("DeepSeek V4 Pro Base",  "DeepSeek",          "open",   "deepseek-ai__DeepSeek-V4-Pro-Base"),
    "reflection/beam-1.0-501b-a23b":     ("Reflection Beam 1.0",   "Reflection",        "open",   "reflection__Beam-1.0-501B-A23B"),
    "deepseek/deepseek-v4-flash":        ("DeepSeek V4 Flash Base","DeepSeek",          "open",   "deepseek-ai__DeepSeek-V4-Flash-Base"),
    "deepseek/deepseek-v4.1-flash-1017": ("DeepSeek V4.1 Flash",   "DeepSeek",          "open",   "deepseek-ai__DeepSeek-V4.1-Flash-1017"),
    "nvidia/nemotron-3-ultra-550b-a55b": ("Nemotron 3 Ultra Base", "NVIDIA",            "open",   "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16"),
    "z-ai/glm-5.3-flash":                ("GLM-5.3 Flash",         "Zhipu",             "open",   "zai-org__GLM-5.3-Flash"),
    "nvidia/nemotron-3.5-super-vl":      ("Nemotron 3.5 Super VL", "NVIDIA",            "open",   "nvidia__NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16"),
    "nvidia/nemotron-3-super-120b-a12b": ("Nemotron 3 Super Base", "NVIDIA",            "open",   "nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"),
    "ricursiveai/ripple-65b-preview":    ("Ripple 65B Base",       "RicursiveAI",       "open",   "ricursiveAI__Ripple-65B-Base-Preview__tok-from-Preview"),
    "google/gemma-4-31b-it":             ("Gemma 4 31B",           "Google",            "open",   "google__gemma-4-31B"),
    "google/gemma-4-26b-a4b-it":         ("Gemma 4 26B-A4B",       "Google",            "open",   "google__gemma-4-26B-A4B"),
    "thinkingmachines/inkling-2-mini":   ("Inkling 2 Mini",        "Thinking Machines", "open",   "thinkingmachines__Inkling-2-Mini"),
    "nvidia/nemotron-3-nano-30b-a3b":    ("Nemotron 3 Nano Base",  "NVIDIA",            "open",   "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"),
    # 6 Additional Open BPBench Models (not on Elie's index; evaluated on N=954)
    "moonshotai/kimi-k2.6":              ("Kimi K2.6",             "Moonshot",          "open",   None),
    "z-ai/glm-5.2":                      ("GLM-5.2",               "Zhipu",             "open",   None),
    "z-ai/glm-5.3":                      ("GLM-5.3",               "Zhipu",             "open",   None),
    "z-ai/glm-5.3-prime":                ("GLM-5.3 Prime",         "Zhipu",             "closed", None),
    "z-ai/glm-5.1":                      ("GLM-5.1",               "Zhipu",             "open",   None),
    "z-ai/glm-5":                        ("GLM-5",                 "Zhipu",             "open",   None),
    "z-ai/glm-5-turbo":                  ("GLM-5 Turbo",           "Zhipu",             "closed", None),
    "z-ai/glm-4.7":                      ("GLM-4.7",               "Zhipu",             "open",   None),
    "qwen/qwen4-max":                    ("Qwen4 Max",             "Qwen",              "closed", None),
    "qwen/qwen3.8-2.4t-a95b":            ("Qwen3.8 2.4T-A95B",     "Qwen",              "open",   None),
    "qwen/qwen3.7-max":                  ("Qwen3.7 Max",           "Qwen",              "closed", None),
    "qwen/qwen3.7-plus":                 ("Qwen3.7-Plus",          "Qwen",              "open",   None),
    "qwen/qwen3.8-flash":                ("Qwen3.8 Flash",         "Qwen",              "closed", None),
    "qwen/qwen3.8-27b":                  ("Qwen3.8-27B",           "Qwen",              "open",   None),
    "qwen/qwen3.6-plus":                 ("Qwen3.6 Plus",          "Qwen",              "closed", None),
    "thinkingmachines/inkling":          ("Inkling",               "Thinking Machines", "open",   None),
    "qwen/qwen3.6-27b":                  ("Qwen3.6-27B",           "Qwen",              "open",   None),
    "x-ai/grok-4.7":                     ("Grok 4.7",              "xAI",               "closed", None),
    "x-ai/grok-4.6":                     ("Grok 4.6",              "xAI",               "closed", None),
    "x-ai/grok-4.5":                     ("Grok 4.5",              "xAI",               "closed", None),
    "x-ai/grok-4.3":                     ("Grok 4.3",              "xAI",               "closed", None),
    "x-ai/grok-4.20":                    ("Grok 4.20",             "xAI",               "closed", None),
    "meta-llama/llama-4-maverick":       ("Llama 4 Maverick",      "Meta",              "open",   None),
    "minimax/minimax-m3":                ("MiniMax M3",            "MiniMax",           "closed", None),
    "xiaomi/mimo-v2.6-pro":              ("MiMo-V2.6-Pro",         "Xiaomi",            "open",   None),
    "xiaomi/mimo-v2.6-flash":            ("MiMo-V2.6-Flash",       "Xiaomi",            "open",   None),
    "tencent/hy4":                       ("Tencent Hy4",           "Tencent",           "closed", None),
    "deepseek/deepseek-v3.2":            ("DeepSeek V3.2",         "DeepSeek",          "open",   None),
    "mistralai/mistral-large-2512":      ("Mistral Large 3",       "Mistral",           "open",   None),
    "mistralai/mistral-medium-3-5":      ("Mistral Medium 3.5",    "Mistral",           "closed", None),
    "mistralai/mistral-small-2603":      ("Mistral Small 4",       "Mistral",           "open",   None),
    "nvidia/nemotron-3.5-lightning":     ("Nemotron 3.5 Lightning","NVIDIA",            "open",   None),
    "stepfun/step-5-preview":            ("Step 5 Preview",        "StepFun",           "closed", None),
    # 15 Closed Frontier Models
    "anthropic/claude-fable-5.2":        ("Claude Fable 5.2",      "Anthropic",         "closed", None),
    "anthropic/claude-fable-5.1":        ("Claude Fable 5.1",      "Anthropic",         "closed", None),
    "anthropic/claude-fable-5":          ("Claude Fable 5",        "Anthropic",         "closed", None),
    "anthropic/claude-opus-5.5":         ("Claude Opus 5.5",       "Anthropic",         "closed", None),
    "anthropic/claude-opus-5":           ("Claude Opus 5",         "Anthropic",         "closed", None),
    "anthropic/claude-opus-4.8":         ("Claude Opus 4.8",       "Anthropic",         "closed", None),
    "anthropic/claude-opus-4.7":         ("Claude Opus 4.7",       "Anthropic",         "closed", None),
    "anthropic/claude-opus-4.6":         ("Claude Opus 4.6",       "Anthropic",         "closed", None),
    "anthropic/claude-opus-4.5":         ("Claude Opus 4.5",       "Anthropic",         "closed", None),
    "anthropic/claude-opus-4.1":         ("Claude Opus 4.1",       "Anthropic",         "closed", None),
    "anthropic/claude-sonnet-5.5":       ("Claude Sonnet 5.5",     "Anthropic",         "closed", None),
    "anthropic/claude-sonnet-5":         ("Claude Sonnet 5",       "Anthropic",         "closed", None),
    "anthropic/claude-sonnet-4.6":       ("Claude Sonnet 4.6",     "Anthropic",         "closed", None),
    "anthropic/claude-sonnet-4.5":       ("Claude Sonnet 4.5",     "Anthropic",         "closed", None),
    "anthropic/claude-haiku-5.5":        ("Claude Haiku 5.5",      "Anthropic",         "closed", None),
    "anthropic/claude-haiku-4.5":        ("Claude Haiku 4.5",      "Anthropic",         "closed", None),
    "openai/gpt-6.2-astra":              ("GPT-6.2 Astra",         "OpenAI",            "closed", None),
    "openai/gpt-6-astra":                ("GPT-6 Astra",           "OpenAI",            "closed", None),
    "openai/gpt-6.2-sol":                ("GPT-6.2 Sol",           "OpenAI",            "closed", None),
    "openai/gpt-6.1-sol":                ("GPT-6.1 Sol",           "OpenAI",            "closed", None),
    "openai/gpt-6-sol":                  ("GPT-6 Sol",             "OpenAI",            "closed", None),
    "openai/gpt-6-luna":                 ("GPT-6 Luna",            "OpenAI",            "closed", None),
    "openai/gpt-6.1-terra":              ("GPT-6.1 Terra",         "OpenAI",            "closed", None),
    "openai/gpt-5.6-sol":                ("GPT-5.6 Sol",           "OpenAI",            "closed", None),
    "openai/gpt-5.6-terra":              ("GPT-5.6 Terra",         "OpenAI",            "closed", None),
    "openai/gpt-5.6-luna":               ("GPT-5.6 Luna",          "OpenAI",            "closed", None),
    "openai/gpt-5.5":                    ("GPT-5.5",               "OpenAI",            "closed", None),
    "openai/gpt-5.4":                    ("GPT-5.4",               "OpenAI",            "closed", None),
    "openai/gpt-5.4-mini":               ("GPT-5.4 Mini",          "OpenAI",            "closed", None),
    "openai/gpt-5.4-nano":               ("GPT-5.4 Nano",          "OpenAI",            "closed", None),
    "openai/gpt-5.3-codex":              ("GPT-5.3 Codex",         "OpenAI",            "closed", None),
    "openai/gpt-5.2":                    ("GPT-5.2",               "OpenAI",            "closed", None),
    "openai/gpt-5.1":                    ("GPT-5.1",               "OpenAI",            "closed", None),
    "openai/gpt-5":                      ("GPT-5",                 "OpenAI",            "closed", None),
    "openai/gpt-4.1":                    ("GPT-4.1",               "OpenAI",            "closed", None),
    "openai/gpt-4o":                     ("GPT-4o",                "OpenAI",            "closed", None),
    "openai/gpt-4o-mini":                ("GPT-4o-mini",           "OpenAI",            "closed", None),
    "openai/gpt-oss-120b":               ("gpt-oss-120b",          "OpenAI",            "open",   None),
    "openai/gpt-oss-20b":                ("gpt-oss-20b",           "OpenAI",            "open",   None),
    "openai/o3":                         ("o3",                    "OpenAI",            "closed", None),
    "google/gemini-4-carbon":            ("Gemini 4 Carbon",       "Google",            "closed", None),
    "google/gemini-4-neon":              ("Gemini 4 Neon",         "Google",            "closed", None),
    "google/gemini-4-argon":             ("Gemini 4 Argon",        "Google",            "closed", None),
    "google/gemini-4-hydrogen":          ("Gemini 4 Hydrogen",     "Google",            "closed", None),
    "google/gemini-3.1-pro-preview":     ("Gemini 3.1 Pro",        "Google",            "closed", None),
    "google/gemini-3.8-flash":           ("Gemini 3.8 Flash",      "Google",            "closed", None),
    "google/gemini-3.7-flash":           ("Gemini 3.7 Flash",      "Google",            "closed", None),
    "google/gemini-3.6-flash":           ("Gemini 3.6 Flash",      "Google",            "closed", None),
    "google/gemini-3.5-flash":           ("Gemini 3.5 Flash",      "Google",            "closed", None),
    "google/gemini-3-flash-preview":     ("Gemini 3 Flash",        "Google",            "closed", None),
    "google/gemini-3.1-flash-lite":      ("Gemini 3.1 Flash-Lite", "Google",            "closed", None),
    "google/gemini-3.5-flash-lite":      ("Gemini 3.5 Flash-Lite", "Google",            "closed", None),
    "google/gemini-2.5-pro":             ("Gemini 2.5 Pro",        "Google",            "closed", None),
    "google/gemini-2.5-flash":           ("Gemini 2.5 Flash",      "Google",            "closed", None),
}

OAI_PUBLISHED = {
    "openai/gpt-6.2-astra": 0.761,
    "openai/gpt-6-astra": 0.783,
    "openai/gpt-5.6-sol": 0.828,
    "openai/gpt-5.5": 0.848,
    "openai/gpt-5.4": 0.871,
    "openai/gpt-5.2": 0.888,
    "openai/gpt-5.1": 0.893,
    "openai/gpt-5": 0.893,
    "openai/gpt-4.1": 0.902,
    "openai/gpt-4o": 0.904,
}


def slug_for(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def is_strict_valid(r: dict) -> bool:
    if not r.get("ok"):
        return False
    raw = (r.get("raw") or "").strip()
    return bool(CLEAN_RE.match(raw) and len(set(re.findall(r"[A-H]", raw))) == 8)


def load_traces(model_id: str, valid_ids: set[str]) -> tuple[list[dict], float]:
    p = TRACE_DIR / f"{slug_for(model_id)}.jsonl"
    if not p.exists():
        return [], 0.0
    by_id: dict[str, dict] = {}
    total_cost = 0.0
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        total_cost += float(r.get("cost") or 0.0)
        if r.get("item_id") not in valid_ids:
            continue
        is_c = is_strict_valid(r)
        prev = by_id.get(r["item_id"])
        if prev is None or (is_c and not prev.get("_clean")) or (is_c and r.get("clean_retry")):
            r["_clean"] = is_c
            by_id[r["item_id"]] = r
    return [r for r in by_id.values() if r.get("_clean")], total_cost


def cluster_boot_scores(
    recs: list[dict],
    url_by_id: dict[str, str],
    *,
    n_boot: int = 3000,
    seed: int = 20261125,
) -> tuple[float, list[float], float, list[float], float]:
    """Source-URL cluster bootstrap returning (auc, auc_ci95, mrr, mrr_ci95, sd_neg_d_prime)."""
    by_url_auc: dict[str, list[float]] = defaultdict(list)
    by_url_mrr: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        u = url_by_id[r["item_id"]]
        rk = float(r["gold_rank"])
        by_url_auc[u].append((8.0 - rk) / 7.0)
        by_url_mrr[u].append(1.0 / rk)
    urls = sorted(by_url_auc)
    a_sums = np.array([sum(by_url_auc[u]) for u in urls], dtype=float)
    m_sums = np.array([sum(by_url_mrr[u]) for u in urls], dtype=float)
    c_cnts = np.array([len(by_url_auc[u]) for u in urls], dtype=float)
    auc = float(a_sums.sum() / c_cnts.sum())
    mrr = float(m_sums.sum() / c_cnts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(urls), size=(n_boot, len(urls)))
    den = c_cnts[idx].sum(axis=1)
    b_auc = a_sums[idx].sum(axis=1) / den
    b_mrr = m_sums[idx].sum(axis=1) / den
    auc_ci = [round(float(np.percentile(b_auc, 2.5)), 4), round(float(np.percentile(b_auc, 97.5)), 4)]
    mrr_ci = [round(float(np.percentile(b_mrr, 2.5)), 4), round(float(np.percentile(b_mrr, 97.5)), 4)]
    b_x = np.array([neg_d_prime(float(v)) for v in b_auc], dtype=float)
    sd_x = float(np.std(b_x, ddof=1))
    return auc, auc_ci, mrr, mrr_ci, sd_x


def paired_compare(
    recs_a: list[dict],
    recs_b: list[dict],
    url_by_id: dict[str, str],
    n_boot: int = 5000,
) -> dict:
    ma = {r["item_id"]: r for r in recs_a if r.get("_clean", True)}
    mb = {r["item_id"]: r for r in recs_b if r.get("_clean", True)}
    shared = sorted(set(ma) & set(mb))
    if len(shared) < 10:
        return {"n": len(shared)}
    ra = np.array([ma[i]["gold_rank"] for i in shared], dtype=float)
    rb = np.array([mb[i]["gold_rank"] for i in shared], dtype=float)

    by_url_auc: dict[str, list[float]] = defaultdict(list)
    by_url_mrr: dict[str, list[float]] = defaultdict(list)
    by_url_t1: dict[str, list[float]] = defaultdict(list)
    for i, a_r, b_r in zip(shared, ra, rb):
        u = url_by_id[i]
        by_url_auc[u].append(((8.0 - a_r) / 7.0) - ((8.0 - b_r) / 7.0))
        by_url_mrr[u].append((1.0 / a_r) - (1.0 / b_r))
        by_url_t1[u].append(float(a_r == 1.0) - float(b_r == 1.0))

    urls = sorted(by_url_auc)
    a_sums = np.array([sum(by_url_auc[u]) for u in urls], dtype=float)
    m_sums = np.array([sum(by_url_mrr[u]) for u in urls], dtype=float)
    t_sums = np.array([sum(by_url_t1[u]) for u in urls], dtype=float)
    cnts = np.array([len(by_url_auc[u]) for u in urls], dtype=float)

    rng = np.random.default_rng(20261125)
    idx = rng.integers(0, len(urls), size=(n_boot, len(urls)))
    den = cnts[idx].sum(axis=1)
    ba = a_sums[idx].sum(axis=1) / den
    bm = m_sums[idx].sum(axis=1) / den
    bt = t_sums[idx].sum(axis=1) / den
    p_auc = float(2.0 * min(np.mean(ba <= 0), np.mean(ba >= 0)))
    p_mrr = float(2.0 * min(np.mean(bm <= 0), np.mean(bm >= 0)))

    dom_auc_diff = {}
    dom_mrr_diff = {}
    for dom in DOMAINS:
        sub = [i for i in shared if ma[i].get("domain") == dom]
        if sub:
            da_a = np.array([(8.0 - ma[i]["gold_rank"]) / 7.0 for i in sub])
            db_a = np.array([(8.0 - mb[i]["gold_rank"]) / 7.0 for i in sub])
            dom_auc_diff[dom] = round(float(np.mean(da_a - db_a)), 4)
            da_m = np.array([1.0 / ma[i]["gold_rank"] for i in sub])
            db_m = np.array([1.0 / mb[i]["gold_rank"] for i in sub])
            dom_mrr_diff[dom] = round(float(np.mean(da_m - db_m)), 4)
    return {
        "n": len(shared),
        "n_clusters": len(urls),
        "wins_a": int(np.sum(ra < rb)),
        "wins_b": int(np.sum(rb < ra)),
        "ties": int(np.sum(ra == rb)),
        "auc_diff": round(float(a_sums.sum() / cnts.sum()), 4),
        "auc_ci95": [round(float(np.percentile(ba, 2.5)), 4), round(float(np.percentile(ba, 97.5)), 4)],
        "p_auc": round(p_auc, 4),
        "dom_auc_diff": dom_auc_diff,
        "mrr_diff": round(float(m_sums.sum() / cnts.sum()), 4),
        "mrr_ci95": [round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)],
        "t1_diff_pct": round(float(t_sums.sum() / cnts.sum() * 100), 2),
        "t1_ci95_pct": [round(float(np.percentile(bt, 2.5) * 100), 2), round(float(np.percentile(bt, 97.5) * 100), 2)],
        "p_mrr": round(p_mrr, 4),
        "dom_mrr_diff": dom_mrr_diff,
    }


def compute_pairwise_win_rates(eval_data: dict[str, dict]) -> dict[str, float]:
    """Average pairwise item win rate (strict wins + 0.5*ties) of each evaluated model
    against all other evaluated models over their shared clean items (like Elie's Win Rate)."""
    mids = list(eval_data.keys())
    maps = {m: {r["item_id"]: r["gold_rank"] for r in eval_data[m]["records"]} for m in mids}
    out: dict[str, float] = {}
    for a in mids:
        ma = maps[a]
        rates = []
        for b in mids:
            if a == b:
                continue
            mb = maps[b]
            sh = set(ma) & set(mb)
            if not sh:
                continue
            w = sum(1.0 if ma[i] < mb[i] else (0.5 if ma[i] == mb[i] else 0.0) for i in sh)
            rates.append(w / len(sh))
        out[a] = round(float(np.mean(rates) * 100.0), 1)
    return out


def calibrate_log_ols_dual(
    ax: np.ndarray,
    ay: np.ndarray,
    anchor_sds_x: np.ndarray,
    target_x: float,
    target_sd_x: float,
) -> tuple[float, list[float], list[float]]:
    """2-parameter log-BPB OLS calibration: ln(BPB) = intercept + slope * (-d').

    Computes exact Student-t (df = n - 2) outer prediction intervals and URL-cluster
    sampling inner intervals in log-BPB space, then exponentiates back to BPB so
    bounds remain strictly positive even under upward extrapolation.
    """
    n = len(ax)
    df = max(3, n - 2)
    t_crit = _T95.get(df, 1.96)
    ly = np.log(ay)
    sl, ic = np.polyfit(ax, ly, 1)
    eta = float(ic + sl * target_x)
    pt = math.exp(eta)
    resid = ly - (ic + sl * ax)
    s_e2 = float(np.sum(resid ** 2) / max(1, n - 2))
    anchor_noise_var = float(np.mean((sl * anchor_sds_x) ** 2))
    domain_var = max(0.0, s_e2 - anchor_noise_var)
    samp_sd = abs(sl) * target_sd_x
    x_bar = float(np.mean(ax))
    s_xx = float(np.sum((ax - x_bar) ** 2))
    param_var = s_e2 * (1.0 / n + ((target_x - x_bar) ** 2) / max(s_xx, 1e-12))
    tot_sd = math.sqrt(domain_var + samp_sd ** 2 + param_var)
    return (
        pt,
        [round(math.exp(eta - t_crit * tot_sd), 4), round(math.exp(eta + t_crit * tot_sd), 4)],
        [round(math.exp(eta - 1.96 * samp_sd), 4), round(math.exp(eta + 1.96 * samp_sd), 4)],
    )


def fit_stats(x: np.ndarray, y: np.ndarray, labs: list[str] | None = None) -> dict:
    ly = np.log(y)
    r = float(np.corrcoef(x, ly)[0, 1])
    rho = spearman(x, y)
    sl, ic = np.polyfit(x, ly, 1)
    errs = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        s, c = np.polyfit(x[m], ly[m], 1)
        errs.append(abs(math.exp(c + s * x[i]) - y[i]) / y[i] * 100)
    out = {
        "r": round(r, 4),
        "rho": round(rho, 4),
        "loo_pct": round(float(np.mean(errs)), 2),
        "loo_max_pct": round(float(np.max(errs)), 2),
        "slope": float(sl),
        "intercept": float(ic),
        "log_y": True,
    }
    if labs is not None and len(set(labs)) >= 3:
        lolo_errs = []
        for hold in sorted(set(labs)):
            m = np.array([lb != hold for lb in labs])
            s, c = np.polyfit(x[m], ly[m], 1)
            for idx_i in np.where(~m)[0]:
                lolo_errs.append(abs(math.exp(c + s * x[idx_i]) - y[idx_i]) / y[idx_i] * 100)
        out["lolo_pct"] = round(float(np.mean(lolo_errs)), 2)
        out["lolo_max_pct"] = round(float(np.max(lolo_errs)), 2)
        out["n_labs"] = len(set(labs))
    return out


def build() -> None:
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    elie_all = json.loads(ELIE_PARSED.read_text(encoding="utf-8"))
    items = bench["items"]
    valid_ids = {it["item_id"] for it in items}
    url_by_id = {it["item_id"]: it["url"] for it in items}
    mean_bytes = float(np.mean([it["true_chunk_bytes"] for it in items]))
    n_items = len(items)
    n_unique_urls = len(set(url_by_id.values()))

    eval_data: dict[str, dict] = {}
    for mid, (label, lab, access, eid) in MODEL_META.items():
        ok, total_cost = load_traces(mid, valid_ids)
        if len(ok) < max(50, int(0.5 * n_items)):
            continue
        st = score_rank_nce_cv(
            ok, k_candidates=8, q_chunk_bpb=0.600, q_doc_bpb=0.600, mean_chunk_bytes=mean_bytes, n_boot=1000
        )
        auc, auc_ci, mrr, mrr_ci, score_sd_x = cluster_boot_scores(ok, url_by_id, n_boot=3000, seed=20261125)
        rk_all = np.array([r["gold_rank"] for r in ok], dtype=float)
        score_x = float(neg_d_prime(auc))
        n_pos_rtok = sum(1 for r in ok if int(r.get("r_tok") or 0) > 0)

        dom_auc, dom_mrr, dom_t1 = {}, {}, {}
        for dom in DOMAINS:
            dr = [r for r in ok if r.get("domain") == dom]
            if dr:
                rk = np.array([r["gold_rank"] for r in dr], dtype=float)
                dom_auc[dom] = round(float(np.mean((8.0 - rk) / 7.0)), 4)
                dom_mrr[dom] = round(float(np.mean(1.0 / rk)), 4)
                dom_t1[dom] = round(float(np.mean(rk == 1) * 100), 1)

        prose_recs = [r for r in ok if r.get("domain") != "code"]
        prose_auc = round(float(np.mean([(8.0 - r["gold_rank"]) / 7.0 for r in prose_recs])), 4)
        prose_mrr = round(float(np.mean([1.0 / r["gold_rank"] for r in prose_recs])), 4)

        eval_data[mid] = {
            "model_id": mid,
            "name": label,
            "lab": lab,
            "access": access,
            "elie_id": eid,
            "n_ok": len(ok),
            "n_pos_rtok": n_pos_rtok,
            "pct_pos_rtok": round(n_pos_rtok / len(ok) * 100.0, 2),
            "mean_r_tok": round(float(np.mean([r.get("r_tok", 0) for r in ok])), 1),
            "cost": round(float(total_cost), 4),
            "nce_doc_bpb": score_x,
            "neg_d_prime": score_x,
            "d_prime": -score_x,
            "nce_sd_x": score_sd_x,
            "info_gain_bits": float(st["info_gain_bits"]),
            "top1_pct": round(float(np.mean(rk_all == 1) * 100), 2),
            "auc": round(auc, 4),
            "auc_ci": auc_ci,
            "prose_auc": prose_auc,
            "dom_auc": dom_auc,
            "mrr": round(mrr, 4),
            "mrr_ci": mrr_ci,
            "prose_mrr": prose_mrr,
            "dom_mrr": dom_mrr,
            "dom_t1": dom_t1,
            "records": ok,
        }

    win_rates = compute_pairwise_win_rates(eval_data)
    all29_shared = set.intersection(*(set(r["item_id"] for r in d["records"]) for d in eval_data.values()))
    top6_ids = [
        "google/gemini-4-carbon", "anthropic/claude-fable-5.2", "google/gemini-4-neon",
        "google/gemini-4-argon", "openai/gpt-6.2-astra", "anthropic/claude-opus-5.5",
    ]
    top6_shared = set.intersection(*(set(r["item_id"] for r in eval_data[m]["records"]) for m in top6_ids if m in eval_data))

    anchors = [
        d for d in eval_data.values()
        if d["access"] == "open"
        and d["elie_id"] in elie_all
        and (INCLUDE_TEMPLATE_BYPASS_ANCHORS or d["model_id"] not in TEMPLATE_BYPASS_MODELS)
    ]
    anchor_eids = {d["elie_id"] for d in anchors}
    ax = np.array([d["nce_doc_bpb"] for d in anchors], dtype=float)
    ax_sd = np.array([d["nce_sd_x"] for d in anchors], dtype=float)
    alabs = [d["lab"] for d in anchors]
    ay_unch = np.array([elie_all[d["elie_id"]]["unch_bpb"] for d in anchors], dtype=float)
    ay_idx = np.array([elie_all[d["elie_id"]]["index_5050"] for d in anchors], dtype=float)

    cal_unch = fit_stats(ax, ay_unch, labs=alabs)
    cal_idx = fit_stats(ax, ay_idx, labs=alabs)

    # Sensitivity check including raw-template-bypass mandatory-reasoning endpoints (zero hidden numbers)
    all_elie_open = [d for d in eval_data.values() if d["access"] == "open" and d["elie_id"] in elie_all]
    cal_unch_all = fit_stats(
        np.array([d["nce_doc_bpb"] for d in all_elie_open], dtype=float),
        np.array([elie_all[d["elie_id"]]["unch_bpb"] for d in all_elie_open], dtype=float),
        labs=[d["lab"] for d in all_elie_open],
    )

    oai_mids = [m for m in OAI_PUBLISHED if m in eval_data]
    ox = np.array([eval_data[m]["nce_doc_bpb"] for m in oai_mids], dtype=float)
    ox_sd = np.array([eval_data[m]["nce_sd_x"] for m in oai_mids], dtype=float)
    oy = np.array([OAI_PUBLISHED[m] for m in oai_mids], dtype=float)
    cal_oai = fit_stats(ox, oy)

    min_anchor_x, max_anchor_x = float(np.min(ax)), float(np.max(ax))
    min_oai_x, max_oai_x = float(np.min(ox)), float(np.max(ox))

    # Blind Gemma 4 26B-A4B holdout stat (fit on other 13 open anchors)
    g_idx = [i for i, d in enumerate(anchors) if d["model_id"] == "google/gemma-4-26b-a4b-it"][0]
    g_msk = np.arange(len(anchors)) != g_idx
    g_sl, g_ic = np.polyfit(ax[g_msk], np.log(ay_unch[g_msk]), 1)
    g_pred = math.exp(g_ic + g_sl * ax[g_idx])
    g_err_pct = abs(g_pred - ay_unch[g_idx]) / ay_unch[g_idx] * 100.0
    mean_auc_hw = float(np.mean([0.5 * (d["auc_ci"][1] - d["auc_ci"][0]) for d in eval_data.values()]))

    matched_by_eid = {d["elie_id"]: d for d in eval_data.values() if d["elie_id"] in elie_all}
    lab_map = {
        "moonshotai": "Moonshot", "deepseek-ai": "DeepSeek", "baidu": "Baidu",
        "reflection": "Reflection", "thinkingmachines": "Thinking Machines",
        "open-athena": "Marin", "marin-community": "Marin", "nvidia": "NVIDIA",
        "zai-org": "Zhipu", "stepfun-ai": "StepFun", "ricursiveAI": "RicursiveAI",
        "google": "Google", "poolside": "Poolside", "mistralai": "Mistral",
        "Qwen": "Qwen", "allenai": "AI2", "meta-models": "Meta",
    }

    rows = []
    for eid, es in elie_all.items():
        org = eid.split("__")[0]
        md = matched_by_eid.get(eid)
        oai_pt = oai_ci = oai_sci = None
        oai_ex = False
        if md is not None:
            oai_pt, oai_ci, oai_sci = calibrate_log_ols_dual(ox, oy, ox_sd, md["nce_doc_bpb"], md["nce_sd_x"])
            oai_pt = round(oai_pt, 4)
            oai_ex = (md["nce_doc_bpb"] < min_oai_x) or (md["nce_doc_bpb"] > max_oai_x)
        rows.append(
            {
                "name": es["label"],
                "lab": lab_map.get(org, org),
                "access": "open",
                "kind": es["kind"],
                "unch_bpb": round(float(es["unch_bpb"]), 4),
                "unch_ci": None,
                "unch_sci": None,
                "index_5050": round(float(es["index_5050"]), 4),
                "index_ci": None,
                "index_sci": None,
                "oai_bpb": oai_pt,
                "oai_ci": oai_ci,
                "oai_sci": oai_sci,
                "oai_exact": False,
                "oai_extrap": oai_ex,
                "nce_doc_bpb": round(md["nce_doc_bpb"], 4) if md else None,
                "neg_d_prime": round(md["neg_d_prime"], 4) if md else None,
                "d_prime": round(md["d_prime"], 4) if md else None,
                "auc": md["auc"] if md else None,
                "auc_ci": md["auc_ci"] if md else None,
                "prose_auc": md["prose_auc"] if md else None,
                "top1_pct": md["top1_pct"] if md else None,
                "mrr": md["mrr"] if md else None,
                "mrr_ci": md["mrr_ci"] if md else None,
                "win_rate": win_rates[md["model_id"]] if md else None,
                "prose_mrr": md["prose_mrr"] if md else None,
                "n_ok": md["n_ok"] if md else None,
                "r_tok": md["mean_r_tok"] if md else None,
                "n_pos_rtok": md["n_pos_rtok"] if md else None,
                "pct_pos_rtok": md["pct_pos_rtok"] if md else None,
                "cot_flagged": bool(md and md["pct_pos_rtok"] > 3.0),
                "strict_zero_cot": bool(md and md["n_pos_rtok"] == 0),
                "dom_auc": md["dom_auc"] if md else None,
                "dom_t1": md["dom_t1"] if md else None,
                "is_anchor": eid in anchor_eids,
                "template_bypass": bool(md and md["model_id"] in TEMPLATE_BYPASS_MODELS),
                "extrapolated": False,
            }
        )

    for mid, d in sorted([kv for kv in eval_data.items() if kv[1]["elie_id"] is None], key=lambda kv: kv[1]["nce_doc_bpb"]):
        pt_u, ci_u, sci_u = calibrate_log_ols_dual(ax, ay_unch, ax_sd, d["nce_doc_bpb"], d["nce_sd_x"])
        pt_i, ci_i, sci_i = calibrate_log_ols_dual(ax, ay_idx, ax_sd, d["nce_doc_bpb"], d["nce_sd_x"])
        pt_o, ci_o, sci_o = calibrate_log_ols_dual(ox, oy, ox_sd, d["nce_doc_bpb"], d["nce_sd_x"])
        is_oai_pub = mid in OAI_PUBLISHED
        oai_val = OAI_PUBLISHED[mid] if is_oai_pub else round(pt_o, 4)
        elie_ex = (d["nce_doc_bpb"] < min_anchor_x) or (d["nce_doc_bpb"] > max_anchor_x)
        oai_ex = (not is_oai_pub) and ((d["nce_doc_bpb"] < min_oai_x) or (d["nce_doc_bpb"] > max_oai_x))
        is_closed = d["access"] == "closed"
        rows.append(
            {
                "name": d["name"],
                "lab": d["lab"],
                "access": d["access"],
                "kind": "closed_est" if is_closed else "post",
                "unch_bpb": round(pt_u, 4),
                "unch_ci": ci_u,
                "unch_sci": sci_u,
                "index_5050": round(pt_i, 4),
                "index_ci": ci_i,
                "index_sci": sci_i,
                "oai_bpb": round(oai_val, 4),
                "oai_est": round(pt_o, 4),
                "oai_ci": None if is_oai_pub else ci_o,
                "oai_sci": None if is_oai_pub else sci_o,
                "oai_exact": is_oai_pub,
                "oai_extrap": oai_ex,
                "nce_doc_bpb": round(d["nce_doc_bpb"], 4),
                "neg_d_prime": round(d["neg_d_prime"], 4),
                "d_prime": round(d["d_prime"], 4),
                "auc": d["auc"],
                "auc_ci": d["auc_ci"],
                "prose_auc": d["prose_auc"],
                "top1_pct": d["top1_pct"],
                "mrr": d["mrr"],
                "mrr_ci": d["mrr_ci"],
                "win_rate": win_rates[mid],
                "prose_mrr": d["prose_mrr"],
                "n_ok": d["n_ok"],
                "r_tok": d["mean_r_tok"],
                "n_pos_rtok": d["n_pos_rtok"],
                "pct_pos_rtok": d["pct_pos_rtok"],
                "cot_flagged": d["pct_pos_rtok"] > 3.0,
                "strict_zero_cot": d["n_pos_rtok"] == 0,
                "dom_auc": d["dom_auc"],
                "dom_t1": d["dom_t1"],
                "is_anchor": False,
                "template_bypass": d["model_id"] in TEMPLATE_BYPASS_MODELS,
                "extrapolated": elie_ex,
            }
        )

    showdown_pairs = [
        ("anthropic/claude-fable-5.2", "google/gemini-4-carbon"),
        ("anthropic/claude-fable-5.2", "openai/gpt-6.2-astra"),
        ("anthropic/claude-fable-5.2", "google/gemini-4-argon"),
        ("anthropic/claude-fable-5.2", "google/gemini-4-neon"),
        ("anthropic/claude-fable-5.2", "anthropic/claude-opus-5.5"),
        ("google/gemini-4-carbon",     "openai/gpt-6.2-astra"),
        ("google/gemini-4-carbon",     "google/gemini-4-argon"),
        ("openai/gpt-6.2-astra",       "google/gemini-4-argon"),
        ("openai/gpt-6.2-astra",       "google/gemini-4-neon"),
        ("openai/gpt-6.2-astra",       "anthropic/claude-opus-5.5"),
        ("openai/gpt-6.2-astra",       "openai/gpt-6-astra"),
        ("anthropic/claude-fable-5.2", "moonshotai/kimi-k3.1-pro"),
        ("anthropic/claude-sonnet-5.5","moonshotai/kimi-k3.1-pro"),
    ]
    paired_results = []
    for a, b in showdown_pairs:
        if a in eval_data and b in eval_data:
            pc = paired_compare(eval_data[a]["records"], eval_data[b]["records"], url_by_id)
            pc["model_a"] = eval_data[a]["name"]
            pc["model_b"] = eval_data[b]["name"]
            paired_results.append(pc)

    total_spend = sum(d["cost"] for d in eval_data.values())
    payload = {
        "meta": {
            "title": "Anpaure Pretraining Index (ANPI) v1",
            "built": "2026-11-26",
            "scoring_rule": "pairwise_auc_thurstone_probit_log_bpb",
            "n_items": n_items,
            "n_unique_source_urls": n_unique_urls,
            "common_all29_n": len(all29_shared),
            "common_top6_n": len(top6_shared),
            "domain_counts": bench.get("domain_counts", {}),
            "k_candidates": 8,
            "decoy_model": bench.get("decoy_model"),
            "total_eval_cost": round(total_spend, 2),
            "calibration": {
                "transform": "ln(BPB) = intercept + slope * (-d'), d' = sqrt(2) * Phi^{-1}(AUC)",
                "include_template_bypass_anchors": INCLUDE_TEMPLATE_BYPASS_ANCHORS,
                "n_open_anchors": len(anchors),
                "unch_bpb": cal_unch,
                "unch_bpb_with_bypass": cal_unch_all,
                "index_5050": cal_idx,
                "oai_bpb": cal_oai,
            },
            "paired_showdowns": paired_results,
        },
        "models": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Anpaure Pretraining Index (ANPI) — Open &amp; Closed Frontier Models</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  /* Editorial minimalism inspired by eliebak.com/pretraining-index & deeprsi.com:
     warm gallery paper, EB Garamond serif headings, IBM Plex Mono tabular labels,
     hairline rules, light/dark toggle, and colour reserved strictly for the data. */
  :root {{
    --ground: #f6f6f4;
    --surface: #ffffff;
    --subtle: #f0f0ed;
    --muted-bg: #eaeae6;
    --row-hover: rgba(0, 0, 0, .035);
    --ink: #141413;
    --ink-2: #52524e;
    --faint: #807f7a;
    --line: #dfdfda;
    --line-strong: #c8c8c1;
    --whisker-outer: #8f8e88;
    --whisker-inner: #141413;
    --serif: "EB Garamond", "Iowan Old Style", "New York", "Charter", Georgia, serif;
    --sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    /* Lab palette: muted editorial tones */
    --lab-anthropic: #c45130;
    --lab-openai: #755bc7;
    --lab-google: #3587d6;
    --lab-moonshot: #b24b7f;
    --lab-deepseek: #3366dd;
    --lab-nvidia: #6ba823;
    --lab-zhipu: #2387c7;
    --lab-qwen: #5b52d6;
    --lab-tm: #70706c;
    --lab-ricursive: #262625;
    --lab-reflection: #1f8a70;
    --lab-meta: #0866ff;
    --lab-xai: #1f2328;
    --lab-mistral: #d95d18;
    --lab-minimax: #c2396a;
    --lab-xiaomi: #e05a1e;
    --lab-tencent: #1c78c0;
    --lab-stepfun: #1a9c7d;
    --lab-other: #8c8b85;
    --grain: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.95' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.032'/%3E%3C/svg%3E");
    color-scheme: light;
  }}
  :root[data-theme="dark"] {{
    --ground: #131312;
    --surface: #181817;
    --subtle: #1f1f1d;
    --muted-bg: #252522;
    --row-hover: rgba(255, 255, 255, .04);
    --ink: #f1f1ee;
    --ink-2: #a8a8a2;
    --faint: #767670;
    --line: #282825;
    --line-strong: #3c3c38;
    --whisker-outer: #7d7d76;
    --whisker-inner: #f1f1ee;
    --lab-anthropic: #e06b47;
    --lab-openai: #957de8;
    --lab-google: #58a6ee;
    --lab-moonshot: #d2689c;
    --lab-deepseek: #5888f0;
    --lab-nvidia: #87c43c;
    --lab-zhipu: #44a7e6;
    --lab-qwen: #8278f0;
    --lab-tm: #969690;
    --lab-ricursive: #b4b4ae;
    --lab-reflection: #3db89a;
    --lab-meta: #4d94ff;
    --lab-xai: #d8dce3;
    --lab-mistral: #f07b3a;
    --lab-minimax: #e86090;
    --lab-xiaomi: #f57a42;
    --lab-tencent: #46a0e6;
    --lab-stepfun: #34c7a4;
    --lab-other: #8a8a84;
    --grain: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.95' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.045'/%3E%3C/svg%3E");
    color-scheme: dark;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    min-height: 100vh;
    overflow-x: clip;
    background-color: var(--ground);
    background-image: var(--grain);
    color: var(--ink);
    font: 14.5px/1.55 var(--sans);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }}
  a {{ color: inherit; text-decoration: none; }}
  button, select {{ font: inherit; color: inherit; }}
  button {{ cursor: pointer; }}

  /* Sticky frosted top bar (Elie + DeepRSI hybrid) */
  .nav {{
    position: sticky; top: 0; z-index: 50;
    height: 56px;
    background: color-mix(in srgb, var(--ground) 86%, transparent);
    -webkit-backdrop-filter: saturate(1.4) blur(12px);
    backdrop-filter: saturate(1.4) blur(12px);
    border-bottom: 1px solid var(--line);
  }}
  .nav-inner {{
    max-width: {1220 if SHOW_OAI_MILESTONES else 1090}px; height: 100%; margin: 0 auto;
    padding: 0 clamp(16px, 3vw, 32px);
    display: flex; align-items: center; gap: 18px;
  }}
  .wordmark {{
    font: 500 19px/1 var(--serif);
    letter-spacing: -0.01em;
    white-space: nowrap;
  }}
  .nav-links {{ display: flex; align-items: center; gap: 2px; min-width: 0; }}
  .nav-links a {{
    font-size: 13px; color: var(--ink-2);
    padding: 5px 9px; border-radius: 7px;
    white-space: nowrap;
    transition: color 0.15s, background 0.15s;
  }}
  .nav-links a:hover {{ color: var(--ink); background: var(--row-hover); }}
  .nav-r {{ margin-left: auto; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }}
  .nav-mob {{ display: none; align-items: center; gap: 4px; }}
  .nav-mob a {{
    font-size: 12.5px; color: var(--ink-2);
    padding: 5px 8px; border-radius: 7px;
    display: inline-flex; align-items: center; gap: 4px;
    white-space: nowrap;
  }}
  .nav-mob a:hover {{ color: var(--ink); background: var(--row-hover); }}
  .upd {{ font: 400 11.5px/1 var(--mono); color: var(--faint); white-space: nowrap; }}
  @media (max-width: 1060px) {{ .upd {{ display: none; }} }}
  @media (max-width: 960px) {{
    .nav-inner {{ gap: 12px; }}
    .nav-links a {{ font-size: 12.5px; padding: 5px 7px; }}
  }}
  @media (max-width: 860px) {{
    .nav-links {{ display: none; }}
    .nav-mob {{ display: inline-flex; }}
  }}
  .theme-btn {{
    width: 30px; height: 30px; border: 0; border-radius: 7px;
    background: transparent; color: var(--ink-2);
    display: inline-flex; align-items: center; justify-content: center;
  }}
  .theme-btn:hover {{ background: var(--row-hover); color: var(--ink); }}

  /* Page container */
  .wrap {{ max-width: {1220 if SHOW_OAI_MILESTONES else 1090}px; margin: 0 auto; padding: 0 clamp(18px, 3vw, 32px) 84px; }}

  /* DeepRSI airy serif hero */
  .hero {{ padding: 48px 0 34px; }}
  .hero h1 {{
    font: 400 clamp(38px, 5vw, 52px)/1.06 var(--serif);
    letter-spacing: -0.02em;
    margin: 0 0 12px;
  }}
  .hero .lead {{
    font: 400 16.5px/1.55 var(--sans);
    color: var(--ink-2);
    max-width: 720px;
    margin: 0 0 20px;
  }}
  .hero-links {{ display: flex; flex-wrap: wrap; gap: 22px; font: 400 16px/1.3 var(--serif); }}
  .hero-links a {{
    display: inline-flex; align-items: center; gap: 5px;
    border-bottom: 1px solid var(--line-strong);
    padding-bottom: 2px;
    transition: border-color 0.15s;
  }}
  .hero-links a:hover {{ border-color: var(--ink); }}

  /* Quiet editorial stat strip (hairlines only, no SaaS boxes) */
  .stat-strip {{
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 16px 0;
    margin-bottom: 42px;
    gap: 18px;
  }}
  @media (max-width: 760px) {{ .stat-strip {{ grid-template-columns: repeat(2, 1fr); }} .nav-links {{ display: none; }} }}
  .stat .k {{ font: 500 10.5px/1.2 var(--mono); text-transform: uppercase; letter-spacing: 0.08em; color: var(--faint); margin-bottom: 5px; }}
  .stat .v {{ font: 400 26px/1.1 var(--serif); letter-spacing: -0.015em; }}
  .stat .s {{ font: 400 12px/1.4 var(--sans); color: var(--ink-2); margin-top: 3px; }}

  /* Section headers */
  section {{ margin-bottom: 56px; scroll-margin-top: 72px; }}
  .sec-head {{
    display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
    gap: 12px; margin-bottom: 12px;
  }}
  h2.sec-title {{
    font: 400 30px/1.15 var(--serif);
    letter-spacing: -0.015em;
    margin: 0;
  }}
  .sec-sub {{
    font-size: 13.5px; color: var(--ink-2);
    max-width: 780px; margin: 0 0 18px; line-height: 1.55;
  }}

  /* Elie-style calm grey-filled pill controls */
  .toolbar {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    margin-bottom: 18px;
  }}
  .seg {{
    display: inline-flex; flex-wrap: wrap; gap: 2px;
    padding: 3px; border-radius: 9px;
    background: var(--subtle);
  }}
  .seg button {{
    border: 0; background: transparent;
    padding: 5px 12px; border-radius: 6px;
    font: 500 12.5px/1.3 var(--sans); color: var(--ink-2);
    transition: background 0.14s, color 0.14s;
  }}
  .seg button:hover {{ color: var(--ink); }}
  .seg button.active {{
    background: var(--surface); color: var(--ink);
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  }}
  .tool-r {{ margin-left: auto; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}

  /* Leaderboard + optional right-side OpenAI Milestones rail (hidden when .no-milestones) */
  .lb-shell {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 238px;
    align-items: stretch;
    gap: 0;
  }}
  .lb-shell.no-milestones {{ grid-template-columns: 1fr; }}
  .lb-shell.no-milestones .lb-main {{ padding-right: 0; }}
  .lb-shell.no-milestones .oai-rail {{ display: none; }}
  .lb-main {{ min-width: 0; padding-right: 20px; }}
  .oai-rail {{
    border-left: 1.5px solid var(--ink);
    padding-left: 18px;
    display: flex;
    flex-direction: column;
    position: relative;
  }}
  .oai-rail-head {{
    padding: 8px 0;
    font: 500 10.5px/1 var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink);
    border-bottom: 1px solid var(--line-strong);
    display: flex; align-items: center; justify-content: space-between; gap: 6px;
  }}
  .oai-rail-head a {{ color: var(--faint); font-size: 10px; border-bottom: 1px solid var(--line-strong); }}
  .oai-rail-head a:hover {{ color: var(--ink); border-color: var(--ink); }}
  .oai-rail-body {{
    position: relative;
    flex: 1;
    min-height: 440px;
  }}
  .oai-ms {{
    position: absolute;
    left: 0; right: 0;
    transform: translateY(-50%);
    padding: 4px 6px 4px 8px;
    border-radius: 6px;
    transition: top 0.2s ease, background 0.15s;
  }}
  .oai-ms::before {{
    content: "";
    position: absolute;
    left: -19px; top: 50%;
    width: 16px; height: 1.5px;
    background: var(--ink);
  }}
  .oai-ms::after {{
    content: "";
    position: absolute;
    left: -22.5px; top: 50%;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--ink);
    transform: translateY(-50%);
  }}
  .oai-ms-val {{
    font: 500 11px/1.15 var(--mono);
    color: var(--ink);
    display: flex; align-items: baseline; justify-content: space-between; gap: 6px;
  }}
  .oai-ms-ref {{
    font: 400 10px/1.15 var(--mono);
    color: var(--faint);
    white-space: nowrap;
  }}
  .oai-ms-lbl {{
    font: 400 14.5px/1.2 var(--serif);
    color: var(--ink);
    margin-top: 1px;
  }}
  .oai-rail-foot {{
    margin-top: 12px;
    padding-top: 8px;
    border-top: 1px solid var(--line);
    font: 400 10.5px/1.45 var(--mono);
    color: var(--faint);
  }}
  .oai-above {{
    display: flex; flex-direction: column; gap: 4px;
    padding: 6px 0 7px;
    border-bottom: 1px dashed var(--line-strong);
  }}
  .oai-above-row {{
    display: flex; align-items: baseline; justify-content: space-between; gap: 6px;
    padding: 3px 7px; border-radius: 5px;
    border: 1px solid var(--line); background: var(--surface);
    font: 400 11.5px/1.25 var(--sans); color: var(--ink-2);
  }}
  .oai-above-row b {{ font: 500 10.5px/1 var(--mono); color: var(--ink); white-space: nowrap; }}
  .oai-above-row.band {{
    background: color-mix(in srgb, var(--ink) 5%, transparent);
    border: 1px dashed var(--line-strong);
    font-style: italic;
  }}
  @media (max-width: 1020px) {{
    .lb-shell {{ grid-template-columns: minmax(0, 1fr) 212px; }}
    .lb-main {{ padding-right: 14px; }}
    .oai-rail {{ padding-left: 14px; }}
    .oai-ms::before {{ left: -15px; width: 12px; }}
    .oai-ms::after {{ left: -18.5px; }}
    .lb-head, .lb-row {{ grid-template-columns: 26px minmax(140px, 178px) 1fr 54px; gap: 8px; }}
  }}
  @media (max-width: 720px) {{
    .lb-shell {{ grid-template-columns: 1fr; }}
    .lb-main {{ padding-right: 0; }}
    .oai-rail {{
      border-left: 0; border-top: 1.5px solid var(--ink);
      padding-left: 0; padding-top: 14px; margin-top: 20px;
    }}
    .oai-rail-body {{ min-height: 0 !important; display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; padding-top: 10px; }}
    .oai-ms {{ position: static; transform: none; border: 1px solid var(--line); background: var(--surface); }}
    .oai-ms::before, .oai-ms::after {{ display: none; }}
  }}

  /* DeepRSI-style open hairline horizontal leaderboard */
  .lb-head, .lb-row {{
    display: grid;
    grid-template-columns: 34px minmax(195px, 255px) 1fr 76px;
    align-items: center;
    gap: 14px;
    padding: 10px 6px;
    border-bottom: 1px solid var(--line);
  }}
  .lb-head {{
    padding: 8px 6px;
    font: 500 10.5px/1 var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--faint);
    border-bottom: 1px solid var(--line-strong);
  }}
  .lb-row {{ transition: background 0.12s; cursor: pointer; }}
  .lb-row:hover {{ background: var(--row-hover); }}
  .lb-row.pinned {{ background: var(--row-hover); border-bottom-color: transparent; }}
  .lb-detail {{
    display: none;
    padding: 8px 12px 12px 48px;
    border-bottom: 1px solid var(--line);
    background: color-mix(in srgb, var(--subtle) 65%, transparent);
    font: 400 12px/1.55 var(--mono);
    color: var(--ink-2);
  }}
  .lb-detail.open {{ display: flex; flex-wrap: wrap; gap: 6px 20px; align-items: baseline; }}
  .lb-detail b {{ color: var(--ink); font-weight: 500; }}
  .lb-detail .k {{ color: var(--faint); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; margin-right: 5px; }}
  .ci-popover {{
    position: fixed; z-index: 90; pointer-events: none;
    opacity: 0; transform: translateY(4px); transition: opacity 0.12s ease, transform 0.12s ease;
    background: var(--surface); border: 1px solid var(--line-strong); border-radius: 10px;
    padding: 11px 14px; min-width: 245px; max-width: 340px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.12);
    font: 400 12px/1.5 var(--mono); color: var(--ink-2);
  }}
  .ci-popover.show {{ opacity: 1; transform: translateY(0); }}
  .ci-popover .pop-title {{ font: 500 16px/1.2 var(--serif); color: var(--ink); margin-bottom: 2px; display: flex; align-items: center; gap: 7px; }}
  .ci-popover .pop-sub {{ font-size: 11px; color: var(--faint); margin-bottom: 8px; }}
  .ci-popover .pop-row {{ display: flex; justify-content: space-between; gap: 14px; padding: 2px 0; }}
  .ci-popover .pop-row b {{ color: var(--ink); font-weight: 500; }}
  .ci-popover .pop-div {{ border-top: 1px solid var(--line); margin: 6px 0; }}
  .ci-popover .pop-hint {{ font-size: 10.5px; color: var(--faint); margin-top: 5px; }}
  .lb-rank {{ font: 400 13px/1 var(--mono); color: var(--faint); font-variant-numeric: tabular-nums; }}
  .lb-model {{ display: flex; align-items: center; gap: 9px; min-width: 0; }}
  .lb-icon {{ width: 16px; height: 16px; flex: none; display: inline-flex; align-items: center; justify-content: center; }}
  .lb-name {{ font: 400 16.5px/1.2 var(--serif); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .micro {{
    font: 500 9.5px/1 var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
    white-space: nowrap;
  }}
  .micro.extrap {{ color: #b84438; }}
  .micro.cot {{ color: var(--lab-openai); }}
  .micro.bypass {{ color: #a16207; }}
  .micro.pub {{ color: #198258; }}
  :root[data-theme="dark"] .micro.extrap {{ color: #ef786b; }}
  :root[data-theme="dark"] .micro.bypass {{ color: #eab308; }}
  :root[data-theme="dark"] .micro.pub {{ color: #4ad09b; }}

  .lb-bar-cell {{ display: flex; align-items: center; gap: 10px; min-width: 0; }}
  .lb-track {{ position: relative; flex: 1; height: 14px; }}
  .lb-fill {{ position: absolute; top: 1px; bottom: 1px; left: 0; border-radius: 2px; transition: width 0.25s ease; }}
  .lb-fill.hatched {{
    background-image: repeating-linear-gradient(
      -45deg,
      rgba(255,255,255,0.22) 0px, rgba(255,255,255,0.22) 3px,
      transparent 3px, transparent 7px
    );
  }}
  .ci-outer {{ position: absolute; top: 6.5px; height: 1.5px; background: var(--whisker-outer); z-index: 2; }}
  .ci-outer-cap {{ position: absolute; top: 3.5px; width: 1.5px; height: 7.5px; background: var(--whisker-outer); z-index: 2; }}
  .ci-inner {{ position: absolute; top: 5px; height: 4px; background: var(--whisker-inner); border-radius: 1.5px; z-index: 3; opacity: 0.85; }}
  .lb-val {{ font: 400 13px/1 var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; min-width: 48px; }}
  .lb-sub {{ font: 400 12px/1 var(--mono); color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }}
  .lb-sub .mob-k {{ display: none; }}
  .lb-sub.faint {{ color: var(--faint); }}
  @media (max-width: 760px) {{
    .lb-head, .lb-row {{ grid-template-columns: 26px 155px 1fr 56px; gap: 8px; }}
    .lb-detail {{ padding-left: 12px; }}
  }}
  @media (hover: none) {{
    .ci-popover {{ display: none !important; }}
  }}
  @media (max-width: 640px) {{
    .nav {{ height: 52px; }}
    .nav-inner {{ padding: 0 14px; gap: 10px; }}
    .wordmark {{ font-size: 18px; }}
    .wrap {{ padding: 0 14px 60px; }}
    .hero {{ padding: 24px 0 20px; }}
    .hero h1 {{ font-size: clamp(29px, 8.2vw, 38px); margin-bottom: 10px; }}
    .hero .lead {{ font-size: 14.5px; line-height: 1.5; margin-bottom: 16px; }}
    .hero-links {{ gap: 10px 16px; font-size: 14.5px; }}
    .stat-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 14px 0; gap: 14px 14px; margin-bottom: 30px; }}
    .stat .v {{ font-size: 22px; }}
    .stat .s {{ font-size: 11.5px; line-height: 1.35; }}
    section {{ margin-bottom: 38px; scroll-margin-top: 64px; }}
    h2.sec-title {{ font-size: 24px; }}
    .sec-sub {{ font-size: 13px; margin-bottom: 14px; }}
    .toolbar {{ flex-direction: column; align-items: stretch; gap: 8px; margin-bottom: 14px; }}
    #metricBtns {{
      display: flex; flex-wrap: nowrap; overflow-x: auto;
      -webkit-overflow-scrolling: touch; scrollbar-width: none;
      width: 100%;
    }}
    #metricBtns::-webkit-scrollbar {{ display: none; }}
    #metricBtns button {{ flex: 0 0 auto; white-space: nowrap; padding: 7px 11px; font-size: 12.5px; }}
    .tool-r {{ margin-left: 0; width: 100%; }}
    #accessBtns {{ display: grid; grid-template-columns: repeat(3, 1fr); width: 100%; }}
    #accessBtns button {{ padding: 7px 8px; text-align: center; }}
    #scatterBtns {{ display: grid; grid-template-columns: 1fr 1fr; width: 100%; }}
    #scatterBtns button {{ padding: 7px 8px; font-size: 12px; text-align: center; }}
    #oaiGraphBtns {{ display: grid; grid-template-columns: repeat(3, 1fr); width: 100%; }}
    #oaiGraphBtns button {{ padding: 7px 6px; font-size: 11.5px; text-align: center; }}
    .lb-head {{
      display: flex; justify-content: space-between; align-items: center;
      padding: 8px 4px;
    }}
    .lb-head > div:nth-child(1),
    .lb-head > div:nth-child(4) {{ display: none; }}
    .lb-row {{
      grid-template-columns: 24px minmax(0, 1fr) auto;
      grid-template-rows: auto auto;
      gap: 6px 8px;
      padding: 11px 4px;
    }}
    .lb-rank {{ grid-column: 1; grid-row: 1; font-size: 12px; }}
    .lb-model {{ grid-column: 2; grid-row: 1; gap: 7px; flex-wrap: nowrap; }}
    .lb-name {{ font-size: 15.5px; }}
    .lb-sub {{ grid-column: 3; grid-row: 1; font-size: 11px; color: var(--faint); white-space: nowrap; }}
    .lb-sub .mob-k {{ display: inline; margin-right: 4px; font-size: 9.5px; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.85; }}
    .lb-bar-cell {{ grid-column: 2 / 4; grid-row: 2; gap: 10px; }}
    .lb-track {{ height: 12px; }}
    .lb-val {{ font-size: 13px; font-weight: 500; min-width: 44px; text-align: right; }}
    .lb-detail {{ padding: 10px 10px 12px; gap: 6px 14px; font-size: 11.5px; }}
    .lb-legend {{ gap: 8px 14px; font-size: 11px; }}
    .lb-legend > span:last-child {{ margin-left: 0 !important; flex-basis: 100%; }}
    .chart-frame {{ padding: 16px 14px 14px; border-radius: 10px; }}
    .chart-frame .sec-title {{ font-size: 22px; }}
    .chart-legend {{ gap: 8px 14px; font-size: 12px; }}
    .tbl-wrap {{ margin: 0 -14px; padding: 0 14px; -webkit-overflow-scrolling: touch; }}
    .tbl-wrap table.clean {{ font-size: 11.5px; }}
    .tbl-wrap table.clean th, .tbl-wrap table.clean td {{ padding: 8px 6px; white-space: nowrap; }}
    .tbl-wrap table.clean td.name-cell {{ font-size: 14px; }}
    .exp-col table.clean thead {{ display: none; }}
    .exp-col table.clean, .exp-col table.clean tbody {{ display: block; width: 100%; }}
    .exp-col table.clean tr {{
      display: grid; grid-template-columns: 1fr auto; gap: 6px 10px;
      padding: 11px 0; border-bottom: 1px solid var(--line);
    }}
    .exp-col table.clean td {{ border-bottom: 0; padding: 0; white-space: normal; }}
    .exp-col table.clean td:nth-child(2) {{ text-align: right; white-space: nowrap; }}
    .exp-col table.clean td:nth-child(3) {{ grid-column: 1 / -1; line-height: 1.5; }}
    .exp-grid {{ padding-top: 24px; gap: 22px; }}
    .exp-col h2 {{ font-size: 23px; }}
  }}
  .lb-legend {{
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
    margin-top: 12px; font: 400 11.5px/1.4 var(--mono); color: var(--faint);
  }}
  .sw {{ display: inline-block; width: 11px; height: 7px; border-radius: 1.5px; vertical-align: middle; margin-right: 5px; }}
  .sw.hatched {{
    background: var(--faint);
    background-image: repeating-linear-gradient(-45deg, rgba(255,255,255,0.45) 0 2px, transparent 2px 4px);
  }}

  /* Elie-style clean white surface frame for visual charts */
  .chart-frame {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 26px 26px 22px;
  }}
  .chart-frame .sec-title {{ font-size: 26px; margin-bottom: 6px; }}
  .chart-frame .sec-sub {{ margin-bottom: 16px; }}
  .chart-legend {{
    display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
    font-size: 12.5px; color: var(--ink-2); margin-bottom: 14px;
  }}
  .chart-legend .chip {{ display: inline-flex; align-items: center; gap: 7px; }}
  .chart-legend .box {{ width: 16px; height: 10px; border-radius: 3px; background: var(--faint); }}
  .chart-legend .box.hatch {{
    background-image: repeating-linear-gradient(-45deg, rgba(255,255,255,0.45) 0 2.5px, transparent 2.5px 5.5px);
  }}
  svg {{ display: block; width: 100%; height: auto; }}

  /* Clean hairline tables (Head-to-Head + Full Ledger) */
  .tbl-wrap {{ overflow-x: auto; }}
  table.clean {{
    width: 100%; border-collapse: collapse;
    font: 400 12.5px/1.4 var(--mono); font-variant-numeric: tabular-nums;
  }}
  table.clean th {{
    text-align: left; padding: 8px 8px;
    font: 500 10.5px/1 var(--mono); letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--faint); border-bottom: 1px solid var(--line-strong);
    white-space: nowrap;
  }}
  table.clean td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
  table.clean tbody tr:hover {{ background: var(--row-hover); }}
  table.clean td.num, table.clean th.num {{ text-align: right; }}
  table.clean td.name-cell {{ font: 400 15.5px/1.2 var(--serif); }}
  .sig {{ color: #15803d; font-weight: 500; }}
  :root[data-theme="dark"] .sig {{ color: #4ade80; }}
  .ns {{ color: var(--faint); }}
  .ci-dim {{ color: var(--faint); font-size: 11px; }}

  /* Calibration scatter layout */
  .cal-grid {{ display: grid; grid-template-columns: 1.45fr 1fr; gap: 28px; align-items: start; }}
  .cal-grid > * {{ min-width: 0; }}
  @media (max-width: 860px) {{ .cal-grid {{ grid-template-columns: 1fr; }} }}
  .prose-side p {{ font-size: 13.5px; color: var(--ink-2); margin: 0 0 12px; line-height: 1.62; }}
  .prose-side strong {{ color: var(--ink); font-weight: 500; }}
  code {{ font: 400 12px/1.3 var(--mono); background: var(--subtle); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }}

  /* Editorial two-column Methodology & Validation explainer */
  .exp-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 36px;
    border-top: 1px solid var(--line); padding-top: 32px;
  }}
  .exp-col {{ min-width: 0; }}
  @media (max-width: 860px) {{ .exp-grid {{ grid-template-columns: 1fr; gap: 28px; }} }}
  .exp-col h2 {{ font: 400 28px/1.15 var(--serif); margin: 0 0 8px; }}
  .exp-col .sec-sub {{ margin-bottom: 16px; }}
  .exp-col ol {{ margin: 0 0 16px; padding-left: 19px; color: var(--ink-2); font-size: 13.5px; line-height: 1.62; }}
  .exp-col li {{ margin-bottom: 9px; }}
  .exp-col li strong {{ color: var(--ink); font-weight: 500; }}
  .note-box {{
    border-left: 2px solid var(--line-strong);
    padding: 8px 0 8px 14px;
    font-size: 12.8px; color: var(--ink-2); line-height: 1.58;
    margin-top: 12px;
  }}
  .note-box strong {{ color: var(--ink); font-weight: 500; }}

  footer {{
    border-top: 1px solid var(--line);
    padding-top: 20px;
    font: 400 12px/1.6 var(--mono);
    color: var(--faint);
    display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px;
  }}
  footer a {{ border-bottom: 1px solid var(--line-strong); }}
</style>
</head>
<body>
<nav class="nav">
  <div class="nav-inner">
    <a href="#" class="wordmark">Anpaure (ANPI)</a>
    <div class="nav-links">
      <a href="#leaderboard">Leaderboard</a>
      <a href="#oai-perplexity">Perplexity</a>
      <a href="#win-rate">Win rate</a>
      <a href="#head-to-head">Pairs</a>
      <a href="#calibration">Calibration</a>
      <a href="#how-calculated">Method</a>
      <a href="#ledger">Ledger</a>
      <a href="./writeup.html">Writeup</a>
      <a href="https://github.com/anpaure/pretraining-index" target="_blank" rel="noopener">GitHub</a>
    </div>
    <div class="nav-r">
      <div class="nav-mob">
        <a href="./writeup.html">Writeup</a>
        <a href="https://github.com/anpaure/pretraining-index" target="_blank" rel="noopener" aria-label="GitHub repository">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.54 2.87 8.39 6.84 9.75.5.09.68-.22.68-.49 0-.24-.01-.88-.01-1.73-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.89 1.57 2.34 1.12 2.91.85.09-.66.35-1.12.63-1.38-2.22-.26-4.56-1.14-4.56-5.08 0-1.12.39-2.04 1.03-2.76-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05a9.35 9.35 0 0 1 5 0c1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.64 1.03 2.76 0 3.95-2.34 4.82-4.57 5.07.36.32.68.94.68 1.9 0 1.37-.01 2.48-.01 2.82 0 .27.18.59.69.49A10.25 10.25 0 0 0 22 12.26C22 6.58 17.52 2 12 2z"/></svg>
          GitHub
        </a>
      </div>
      <span class="upd">27 Nov 2026</span>
      <button class="theme-btn" id="themeBtn" title="Toggle light / dark theme" aria-label="Toggle theme">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor" stroke="none"/>
        </svg>
      </button>
    </div>
  </div>
</nav>

<main class="wrap">
  <header class="hero">
    <h1>Anpaure Pretraining Index (ANPI)</h1>
    <p class="lead">
      Black-box 8-choice continuation-ranking benchmark across <b>{n_items} post-cutoff Nov-2026 items</b> ({n_unique_urls} source URLs, 5 domains) with 24B neutral decoys—scored by pairwise rank <b>AUC</b> and calibrated via Thurstone <i>d&prime;</i> onto log-BPB.
    </p>
    <div class="hero-links">
      <a href="https://github.com/anpaure/pretraining-index" target="_blank" rel="noopener">GitHub repository &amp; raw traces &rarr;</a>
      <a href="./writeup.html">Read the technical writeup &rarr;</a>
      <a href="#how-calculated">How is this calculated? &rarr;</a>
      <a href="#does-it-hold-up">Does this hold up against other evals? &rarr;</a>
      <a href="#head-to-head">Head-to-head significance &rarr;</a>
    </div>
  </header>

  <div class="stat-strip">
    <div class="stat">
      <div class="k">Models ranked</div>
      <div class="v">{len(rows)}</div>
      <div class="s">{len(eval_data)} live endpoints · {sum(1 for r in rows if r['access']=='closed')} closed, {sum(1 for r in rows if r['access']=='open')} open</div>
    </div>
    <div class="stat">
      <div class="k">Within-OpenAI LOO</div>
      <div class="v">{cal_oai['loo_pct']:.2f}%</div>
      <div class="s">r = {cal_oai['r']:+.3f}, &rho; = {cal_oai['rho']:+.3f} ({len(oai_mids)} OAI anchors)</div>
    </div>
    <div class="stat">
      <div class="k">Cross-lab Elie LOLO</div>
      <div class="v">{cal_unch['lolo_pct']:.2f}%</div>
      <div class="s">{cal_unch['n_labs']}-lab leave-one-lab-out (LOO {cal_unch['loo_pct']:.2f}%, r = {cal_unch['r']:+.3f})</div>
    </div>
    <div class="stat">
      <div class="k">Cluster AUC CI</div>
      <div class="v">&plusmn;{mean_auc_hw:.3f}</div>
      <div class="s">URL-clustered bootstrap across {n_unique_urls} sources</div>
    </div>
  </div>

  <!-- SECTION 1: DeepRSI-style Open Hairline Leaderboard -->
  <section id="leaderboard">
    <div class="sec-head">
      <h2 class="sec-title" id="chartTitle">Leaderboard</h2>
    </div>
    <p class="sec-sub" id="chartHint"></p>

    <div class="toolbar">
      <div class="seg" id="metricBtns">
        <button data-metric="auc" class="active">Native AUC (Zero-Fit &uarr;)</button>
        <button data-metric="oai_bpb">OpenAI Base Scale</button>
        <button data-metric="unch_bpb">Elie Uncheatable Eval</button>
        <button data-metric="index_5050">Elie 50/50</button>
        <button data-metric="mrr">Native MRR</button>
      </div>
      <div class="tool-r">
        <div class="seg" id="accessBtns">
          <button data-access="all" class="active">All</button>
          <button data-access="closed">Closed</button>
          <button data-access="open">Open</button>
        </div>
      </div>
    </div>

    <div class="lb-shell{"" if SHOW_OAI_MILESTONES else " no-milestones"}">
      <div class="lb-main">
        <div class="lb-head">
          <div>#</div>
          <div>Model</div>
          <div id="barColLabel">Native AUC (95% Cluster CI)</div>
          <div style="text-align:right;" id="col4Label">Top-1</div>
        </div>
        <div id="bars"></div>
      </div>
      <aside class="oai-rail" aria-label="OpenAI Nov-22 Perplexity Milestones">
        <div class="oai-rail-head">
          <span>OpenAI Milestones</span>
          <a href="https://openai.com/index/perplexity-and-intelligence/" target="_blank">Nov 22 &nearr;</a>
        </div>
        <div id="oaiAbove"></div>
        <div class="oai-rail-body" id="oaiMilestones"></div>
        <div class="oai-rail-foot">
          Pachocki (Nov 22, 2026) base log-perplexity milestones aligned to the OpenAI Base Scale mapping.
        </div>
      </aside>
    </div>
    <div id="ciPopover" class="ci-popover" role="tooltip" aria-hidden="true"></div>

    <div class="lb-legend">
      <span><i class="sw" style="background:var(--faint)"></i>Zero-CoT / Base</span>
      <span><i class="sw hatched"></i>&gt;3% CoT / Post-trained</span>
      <span><span class="micro extrap">EXTRAP</span> outside anchor span</span>
      <span><span class="micro bypass">BYPASS</span> raw template 0-CoT (excluded from fit)</span>
      <span style="margin-left:auto;"><b>&#9644;</b> dark inner = URL-cluster sampling CI &nbsp;&middot;&nbsp; <b>&mdash;</b> outer = Student-<i>t</i> log-BPB prediction CI</span>
    </div>
  </section>

  <!-- SECTION 1B: OpenAI Base Log-Perplexity Scale Graph -->
  <section id="oai-perplexity">
    <div class="chart-frame">
      <div class="sec-head" style="margin-bottom:6px;">
        <h2 class="sec-title">OpenAI base perplexity scale</h2>
        <div class="seg" id="oaiGraphBtns">
          <button data-oai-view="top36" class="active">Top 36 Frontier</button>
          <button data-oai-view="anchors">OpenAI Anchors ({len(oai_mids)})</button>
          <button data-oai-view="all">All {len(eval_data)} Live</button>
        </div>
      </div>
      <p class="sec-sub">
        Models mapped onto OpenAI&rsquo;s Nov-22 base log-perplexity scale (<strong>&darr; lower is better</strong>) via the 2-parameter Thurstone <i>d&prime;</i> &rarr; ln(BPB) ruler (<code>r = {cal_oai['r']:+.3f}</code>, <code>&rho; = {cal_oai['rho']:+.3f}</code>, <code>LOO = {cal_oai['loo_pct']:.2f}%</code> across {len(oai_mids)} published OpenAI anchors). Vertical bars show 95% Student-<i>t</i> prediction intervals; hollow rings mark <code>EXTRAP</code> models below top anchor <code>GPT-6.2 Astra</code> (<code>0.761</code>).
      </p>
      <div class="chart-legend" style="gap:10px 14px;">
        <span class="chip"><svg width="12" height="12" viewBox="0 0 12 12" style="width:12px;height:12px;"><circle cx="6" cy="6" r="4.5" fill="var(--lab-openai)" stroke="var(--ink)" stroke-width="1.3"/></svg>Published anchor</span>
        <span class="chip"><svg width="12" height="12" viewBox="0 0 12 12" style="width:12px;height:12px;"><circle cx="6" cy="6" r="4" fill="var(--lab-google)"/></svg>Translated</span>
        <span class="chip"><svg width="12" height="12" viewBox="0 0 12 12" style="width:12px;height:12px;"><circle cx="6" cy="6" r="4" fill="var(--surface)" stroke="var(--lab-anthropic)" stroke-width="2"/></svg>Extrapolated (&lt;0.761)</span>
        <span class="chip"><svg width="12" height="12" viewBox="0 0 12 12" style="width:12px;height:12px;"><polygon points="6,1.5 10.5,6 6,10.5 1.5,6" fill="var(--surface)" stroke="var(--ink)" stroke-width="1.3"/></svg>Anpaure fit (Anchors view)</span>
        <span style="color:var(--faint);margin-left:auto;">&darr; Lower = stronger</span>
      </div>
      <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
        <svg id="oaiPerpChart" viewBox="0 0 1100 450" style="min-width:720px;"></svg>
      </div>
    </div>
  </section>

  <!-- SECTION 2: Elie-style Vertical Column Win-Rate Chart -->
  <section id="win-rate">
    <div class="chart-frame">
      <h2 class="sec-title">Win rate</h2>
      <p class="sec-sub">
        Average share of shared Nov-2026 items where a model ranks the authentic continuation strictly higher than another model (ties split 0.5), averaged across all {len(eval_data) - 1} other live endpoints. Pure zero-fit pairwise ordering.
      </p>
      <div class="chart-legend">
        <span class="chip"><i class="box"></i>100% Zero-CoT across all {len(eval_data)} live endpoints (0 reasoning tokens on every item)</span>
        <span style="color:var(--faint);">&uarr; Pairwise item win rate, higher is better</span>
      </div>
      <div style="overflow-x:auto;">
        <svg id="winChart" viewBox="0 0 1120 445" style="min-width:740px;"></svg>
      </div>
    </div>
  </section>

  <!-- SECTION 3: Head-to-Head Paired Significance -->
  <section id="head-to-head">
    <div class="sec-head">
      <h2 class="sec-title">Head-to-head significance</h2>
    </div>
    <p class="sec-sub">
      Exact paired comparisons on shared clean items (5,000 source-URL cluster resamples so multi-snippet commits are never double-counted). Positive AUC / Top-1 favors Model A.
      All top frontier models shown here—including <code>Gemini 4 Neon</code>, <code>Gemini 4 Carbon</code>, <code>GPT-6 Astra</code>, and <code>GPT-6.1 Sol</code>—run at strict <code>0/954</code> reasoning tokens (CoT suppressed via explicit no-think instruction, validated with zero prompt shift on a 100-item <code>Gemini 4 Argon</code> control, <code>p = 0.85</code>).
    </p>
    <div class="tbl-wrap">
      <table class="clean" id="showdown">
        <thead>
          <tr>
            <th>Model A vs. Model B</th>
            <th class="num">W / L / T</th>
            <th class="num">AUC Diff (95% Cluster CI)</th>
            <th class="num">Top-1</th>
            <th class="num">p-value</th>
            <th class="num">Code</th>
            <th class="num">Papers</th>
            <th class="num">News</th>
            <th class="num">Ref</th>
            <th class="num">Essays</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <!-- SECTION 4: Calibration Scatter -->
  <section id="calibration">
    <div class="chart-frame">
      <div class="sec-head" style="margin-bottom:6px;">
        <h2 class="sec-title">External calibration</h2>
        <div class="seg" id="scatterBtns">
          <button data-sc="unch_bpb" class="active">Elie Uncheatable ({len(anchors)} open)</button>
          <button data-sc="oai_bpb">OpenAI Base ({len(oai_mids)} OAI)</button>
        </div>
      </div>
      <p class="sec-sub">
        Outlined circles are published anchor ground truths used to fit the 2-parameter log-BPB Thurstone ruler <code>ln(BPB) = a + b&middot;(&minus;d&prime;)</code>; diamonds highlight frontier projections and muted dots show the remaining live models.
      </p>
      <div class="cal-grid">
        <svg id="scatter" viewBox="0 0 600 360"></svg>
        <div class="prose-side">
          <p><strong>Why Pairwise AUC + Thurstone <i>d&prime;</i> onto log-BPB.</strong> Across all {sum(d['n_ok'] for d in eval_data.values()):,} real rank rows, candidate scores follow <b>Thurstone Case V</b> (Gaussian sums of token log-probs, <code>6.8&times;</code> lower KL than Plackett-Luce), and tail ranks <code>r &ge; 2</code> retain partial correlation <code>+0.73&ndash;+0.76</code> with true BPB after controlling for Top-1. Combining Thurstone <code>d&prime; = &radic;2&thinsp;&Phi;<sup>&minus;1</sup>(AUC)</code> with Ruan et al. (NeurIPS 2024) yields the 2-parameter law <code>ln(BPB) = a &minus; b&middot;d&prime;</code>—achieving <code>r = {cal_oai['r']:+.3f}, &rho; = {cal_oai['rho']:+.3f}, LOO = {cal_oai['loo_pct']:.2f}%</code> across all {len(oai_mids)} published OpenAI models and <code>r = {cal_unch['r']:+.3f}, LOO = {cal_unch['loo_pct']:.2f}%, LOLO = {cal_unch['lolo_pct']:.2f}%</code> across Elie’s {len(anchors)} open models ({cal_unch['n_labs']} labs), with zero statistically significant ceiling curvature across 4 independent datasets.</p>
          <p><strong>Extrapolation above anchor ranges.</strong> On OpenAI’s ruler, <code>Claude Fable 5.2</code> (<code>0.8550</code>) and <code>Gemini 4 Carbon</code> (<code>0.8482</code>) exceed top anchor <code>GPT-6.2 Astra</code> (<code>0.8444</code>); on Elie’s open ruler, the top closed tier exceeds <code>Kimi K3.1 Pro</code> (<code>0.7518</code>). Every model outside its anchor span is flagged <code>EXTRAP</code> with exact exponentiated Student-<i>t</i> intervals.</p>
          <p><strong>100% Zero-CoT enforcement &amp; template-bypass rule.</strong> All {len(eval_data) - len(TEMPLATE_BYPASS_MODELS)} native-chat endpoints answered at <code>0</code> reasoning tokens on standard <code>/api/v1/chat/completions</code>. Mandatory-reasoning endpoints that reject zero-CoT on chat APIs (<code>Muse Spark Open</code>, <code>Inkling 2 Small</code>, <code>Reflection Beam 1.0</code>, <code>gpt-oss-120b/20b</code>) were evaluated via raw <code>/api/v1/completions</code> zero-think templates and remain visible here (badged <code>BYPASS</code>), but are excluded from the {len(anchors)}-anchor OLS ruler by default (cold-start bypass tax; blind holdouts <code>+3.5% / +5.9% / +6.4%</code>; {len(all_elie_open)}-anchor sensitivity LOLO = <code>{cal_unch_all['lolo_pct']:.2f}%</code>).</p>
        </div>
      </div>
    </div>
  </section>

  <!-- SECTION 5: Editorial Explainers -->
  <section id="methodology" class="exp-grid">
    <div class="exp-col" id="how-calculated">
      <h2>How is this calculated?</h2>
      <p class="sec-sub">Black-box 8-choice continuation ranking across 5 Nov-2026 domains—no logprobs, no echo prompts, strictly audited 8-letter permutations.</p>
      <ol>
        <li><strong>Post-cutoff multi-domain corpus (N={n_items}, {n_unique_urls} unique source URLs):</strong> All documents carry Nov-2026 timestamps across 5 domains—<code>199</code> News/Speeches (UK Hansard &amp; BBC), <code>198</code> Papers (arXiv &amp; bioRxiv), <code>198</code> Essays (Guardian &amp; LessWrong), <code>197</code> newly created Wikipedia articles, and <code>162</code> newly added systems code blocks (116 distinct commits across CPython, Linux, Postgres, Go, LLVM, PyTorch, etc.).</li>
        <li><strong>Neutral 24B cloud decoys (K=8):</strong> At a cut point inside each document, we keep the authentic next snippet (~6 prose words or ~36 chars of single-line code) and sample <strong>7 distinct <code>T = 1.0</code> raw continuations</strong> from <code>mistral-small-24b-instruct-2501</code> (zero family overlap with any ranked model, trailing-space BPE bug fixed). Decoys in v1 were length-filtered around the gold length (see Code length caveat).</li>
        <li><strong>Zero-logprob 8-way ranking &amp; RL sharpening note:</strong> Each API receives the shuffled 8 choices <code>A..H</code> and must return a strict 8-distinct-letter permutation (<code>C,A,F,B,H,D,E,G</code>). Global monotone temperature sharpening of a model’s distribution preserves the ordering of candidate likelihoods <code>p(y|x)</code> (unlike raw top-token cross-entropy)—though note it does <i>not</i> preserve likelihood-ratio ordering <code>p(y|x)/q(y|x)</code> when <code>q</code> is nonuniform, and rankings alone do not uniquely identify absolute generative loss without anchor calibration.</li>
        <li><strong>Native AUC &amp; Thurstone log-BPB rulers:</strong> Native score is pairwise rank <code>AUC = (8 &minus; mean_rank) / 7</code> (exact probability of preferring the real continuation over a random decoy; chance = <code>0.500</code>, zero fitted parameters) with source-URL cluster bootstrap CIs; <code>MRR</code> and <code>Top-1</code> remain reported alongside it. External OpenAI and Elie views fit a 2-parameter log-BPB OLS ruler <code>ln(BPB) = a + b&middot;(&minus;d&prime;)</code> where <code>d&prime; = &radic;2&thinsp;&Phi;<sup>&minus;1</sup>(AUC)</code>, with exact exponentiated Student-<i>t</i> outer intervals and strict <code>EXTRAP</code> badges outside the anchor range.</li>
      </ol>
      <div class="note-box">
        <strong>Trace hygiene &amp; telemetry:</strong> Both the standalone runner and this dashboard require an exact 8-distinct-letter permutation. Models that wrote truncated prose preambles on &gt;10 items were retried identically with a strict no-prose rule at <code>r_tok = 0</code> (verified not to inflate scores: Fable 5.2 moved <code>0.7367 &rarr; 0.7356</code> MRR). OpenRouter omits reasoning-token fields on endpoints that do not report them (defaulting to <code>0</code>); exact positive-<code>r_tok</code> item counts are shown in the ledger.
      </div>
    </div>

    <div class="exp-col" id="does-it-hold-up">
      <h2>Does this hold up against other evals?</h2>
      <p class="sec-sub">Zero-fit pairwise AUC and Thurstone <i>d&prime;</i> strongly track published OpenAI base losses, Bakouch’s open-weight index, and cross-decoy runs—with transparent caveats verified by independent audit.</p>
      <table class="clean" style="margin-bottom:12px;">
        <thead>
          <tr><th>Audit / External Check</th><th>Result</th><th>Interpretation</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><b>OpenAI Nov-22 Base Chart</b><br/><span class="ci-dim">{len(oai_mids)} models (GPT-4o &rarr; GPT-6.2 Astra)</span></td>
            <td><b>r = {cal_oai['r']:+.3f}</b><br/><b>&rho; = {cal_oai['rho']:+.3f}</b><br/>LOO {cal_oai['loo_pct']:.2f}%</td>
            <td style="font-family:var(--sans);font-size:12.5px;color:var(--ink-2);">Near-perfect monotonic recovery across all 10 OpenAI base checkpoints (jumping &rho; from +0.891 on MRR to +0.988 on AUC, and blind GPT-4o..5.6 &rarr; GPT-6/6.2 upward extrap = 1.44%).</td>
          </tr>
          <tr>
            <td><b>Elie Pre-training Index</b><br/><span class="ci-dim">{len(anchors)} native-chat open models, {cal_unch['n_labs']} labs</span></td>
            <td>LOO {cal_unch['loo_pct']:.2f}%<br/><b>LOLO {cal_unch['lolo_pct']:.2f}%</b><br/><span class="ci-dim">max {cal_unch['lolo_max_pct']:.2f}%</span></td>
            <td style="font-family:var(--sans);font-size:12.5px;color:var(--ink-2);">Leave-one-lab-out across {cal_unch['n_labs']} independent labs on Oct-2026 Uncheatable Eval BPB (apples-to-apples native <code>/chat/completions</code> anchors; including the {len(all_elie_open) - len(anchors)} <code>BYPASS</code> endpoints gives <code>{cal_unch_all['lolo_pct']:.2f}%</code> LOLO across {cal_unch_all['n_labs']} labs).</td>
          </tr>
          <tr>
            <td><b>Blind Open Holdout</b><br/><span class="ci-dim">Gemma 4 26B-A4B (exact 0.5904)</span></td>
            <td>Pred {g_pred:.4f}<br/><b>Err {g_err_pct:.2f}%</b></td>
            <td style="font-family:var(--sans);font-size:12.5px;color:var(--ink-2);">Frozen {len(anchors)-1}-anchor log-BPB line predicted an unseen open model within {abs(g_pred-0.5904):.3f} BPB without refitting.</td>
          </tr>
          <tr>
            <td><b>Cross-Decoy / Corpus</b><br/><span class="ci-dim">OLMo-1B Hansard vs Mistral-24B</span></td>
            <td><b>&rho; = +0.950</b><br/>r = +0.919<br/><span class="ci-dim">n = 24 models</span></td>
            <td style="font-family:var(--sans);font-size:12.5px;color:var(--ink-2);">Pairwise AUC rankings hold across different texts and decoy model families (24&times; size change), with zero statistically significant ceiling curvature even at AUC = 0.94.</td>
          </tr>
          <tr>
            <td><b>Length &amp; Surface Shortcuts</b><br/><span class="ci-dim">Closest-to-median length (no LLM)</span></td>
            <td>Prose 12.1&ndash;14.1%<br/><b>Code 22.7%</b><br/><span class="ci-dim">chance 12.5%</span></td>
            <td style="font-family:var(--sans);font-size:12.5px;color:var(--ink-2);">Symmetric &plusmn;10-char decoy filter centered gold length in Code (prose at chance; Papers linear surface test 18.7%). <b>Sensitivity check:</b> removing all 162 Code items preserves <b>r = +0.998, &rho; = +0.996</b> across all {len(eval_data)} models and the exact top-9 AUC set &amp; top-9 MRR ordering (100% unchanged).</td>
          </tr>
        </tbody>
      </table>
      <div class="note-box">
        <strong>Remaining limits:</strong> (1) <em>Ranking &ne; exact loss:</em> prompted 8-way discrimination can diverge from raw generative BPB on RL-tuned smaller models (e.g. <code>Qwen3.6-27B</code> on BPBench) and chat vs. base checkpoints (<code>DeepSeek V4 Pro/Flash</code>). (2) <em>Extrapolation:</em> Fable 5.2 and Carbon exceed every published anchor on both rulers. (3) <em>Provenance &amp; telemetry:</em> Nov-2026 timestamps do not prove first-ever publication of quoted passages; all {len(eval_data)} live endpoints emit zero chain-of-thought on 100% of clean items (<code>Muse Wonder</code> on Google Vertex bills Meta&rsquo;s fixed empty-thought ATEM channel header, <code>raw_r_tok&le;7, rd_bytes=672</code>, identical byte-for-byte to &ldquo;Reply OK&rdquo;).
      </div>
    </div>
  </section>

  <!-- SECTION 6: Full Model Ledger -->
  <section id="ledger">
    <div class="sec-head">
      <h2 class="sec-title">Full ledger</h2>
    </div>
    <p class="sec-sub">All {len(rows)} open and closed models with pairwise AUC, MRR, Win%, per-domain Top-1 accuracy (Code, Papers, News, Reference, Essays), clean item coverage, and positive reasoning-token counts.</p>
    <div class="tbl-wrap">
      <table class="clean" id="ledgerTbl">
        <thead>
          <tr>
            <th>#</th><th>Model</th><th>Lab</th><th>Tags</th>
            <th class="num">AUC</th><th class="num">MRR</th><th class="num">Win%</th><th class="num">Top-1</th>
            <th class="num">OAI Scale</th><th class="num">Elie Unch</th>
            <th class="num">Clean N</th><th class="num">CoT</th>
            <th class="num">Code</th><th class="num">Pap</th><th class="num">News</th><th class="num">Ref</th><th class="num">Ess</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <footer>
    <div>
      Open BPB ground truth: <a href="https://eliebak.com/pretraining-index" target="_blank">eliebak.com/pretraining-index</a> (26 Nov 2026) &middot;
      OpenAI base chart: <a href="https://openai.com/index/perplexity-and-intelligence/" target="_blank">openai.com</a> (22 Nov 2026)
    </div>
    <div>
      Reproduce: <code>python3 build_anpaure_index_v1.py &amp;&amp; python3 audit_v1.py</code>
    </div>
  </footer>
</main>

<script id="index-data" type="application/json">{json.dumps(payload)}</script>
<script>
const DATA = JSON.parse(document.getElementById("index-data").textContent);

// Clean minimalist brand SVG marks (inspired by DeepRSI & Elie)
const ICONS = {{
  Anthropic: `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l1.6 7.1 5.6-4.7-4.7 5.6L22 12l-7.5 1.6 4.7 5.6-5.6-4.7L12 22l-1.6-7.5-5.6 4.7 4.7-5.6L2 12l7.5-1.6-4.7-5.6 5.6 4.7z"/></svg>`,
  OpenAI: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><circle cx="12" cy="12" r="3.2"/><path d="M12 3a4.5 4.5 0 0 1 4.5 4.5v4.5M19.8 7.5a4.5 4.5 0 0 1-1.6 6.1l-3.9 2.3M19.8 16.5a4.5 4.5 0 0 1-6.1 1.6L9.8 15.8M12 21a4.5 4.5 0 0 1-4.5-4.5V12M4.2 16.5a4.5 4.5 0 0 1 1.6-6.1l3.9-2.3M4.2 7.5a4.5 4.5 0 0 1 6.1-1.6l3.9 2.3"/></svg>`,
  Google: `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2c.6 5.4 4.6 9.4 10 10-5.4.6-9.4 4.6-10 10-.6-5.4-4.6-9.4-10-10 5.4-.6 9.4-4.6 10-10z"/></svg>`,
  Moonshot: `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="9" opacity="0.16"/><path d="M8 7h2.4v4.3L14.8 7h3l-4.7 4.8L18 17h-3.1l-3.5-4-1 1V17H8V7z"/></svg>`,
  DeepSeek: `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M4 13.5C4 8.8 7.8 5 12.5 5c4.2 0 7.5 2.7 7.5 6.8 0 3.7-3.1 6.7-7.4 6.7-2.6 0-4.9-.8-6.4-2.1L4 17.8v-4.3zm8.2-4.8a2.3 2.3 0 1 0 0 4.6 2.3 2.3 0 0 0 0-4.6z"/></svg>`,
  NVIDIA: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12c2.8-4.5 6.4-6.5 10.5-6.5 3.3 0 5.8 1.7 7.5 4.3-2.1 4.4-5.9 7.2-10.4 7.2-2.7 0-5.1-1.1-7.6-5z"/><circle cx="12" cy="11.5" r="2.2" fill="currentColor" stroke="none"/></svg>`,
  Zhipu: `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M5 6h14v2.6L9.4 15.4H19V18H5v-2.6l9.6-6.8H5V6z"/></svg>`,
  Qwen: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12,3 20,7.5 20,16.5 12,21 4,16.5 4,7.5"/><circle cx="12" cy="12" r="2.8" fill="currentColor" stroke="none"/></svg>`,
  "Thinking Machines": `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 9h8M12 9v7"/></svg>`,
  RicursiveAI: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg>`,
  Reflection: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12,3 21,12 12,21 3,12"/><line x1="12" y1="3" x2="12" y2="21"/></svg>`,
  Meta: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.2 15.6c0-3.8 2.2-7.6 4.4-7.6 1.9 0 3.4 2.5 5 5.4 1.2 2.1 2.4 3.6 3.7 3.6 1.5 0 2.5-1.3 2.5-3.4 0-3.2-1.8-6.6-4.2-6.6-1.8 0-3.2 1.8-4.8 4.4-1.3 2.2-2.6 5.6-4.1 5.6-1.5 0-2.5-1-2.5-1.4z"/></svg>`,
  xAI: `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M4.5 4h3.2l4.4 6.1L16.7 4h2.8l-5.9 7.8L20 20h-3.2l-4.8-6.6L6.9 20H4.1l6.3-8.3L4.5 4z"/></svg>`,
  Mistral: `<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M4 5h4v4H4V5zm12 0h4v4h-4V5zM4 9h8v4H4V9zm8 0h8v4h-8V9zM4 13h4v6H4v-6zm6 0h4v6h-4v-6zm6 0h4v6h-4v-6z"/></svg>`,
  MiniMax: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><path d="M4 18V7l4.5 6L12 8l3.5 5L20 7v11"/></svg>`,
  Xiaomi: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3.5" y="3.5" width="17" height="17" rx="4"/><path d="M8 16V9.5c0-1 .8-1.5 2-1.5h2.5c1.2 0 1.8.7 1.8 1.8V16M11 11v5M16.5 8v8"/></svg>`,
  Tencent: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 6h14M12 6v13M8 10h8"/></svg>`,
  StepFun: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"><path d="M4 18h5v-5h5V8h6"/></svg>`,
  _default: `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="5" opacity="0.65"/></svg>`
}};

function labColor(lab) {{
  const map = {{
    Anthropic: "var(--lab-anthropic)",
    OpenAI: "var(--lab-openai)",
    Google: "var(--lab-google)",
    Moonshot: "var(--lab-moonshot)",
    DeepSeek: "var(--lab-deepseek)",
    NVIDIA: "var(--lab-nvidia)",
    Zhipu: "var(--lab-zhipu)",
    Qwen: "var(--lab-qwen)",
    "Thinking Machines": "var(--lab-tm)",
    RicursiveAI: "var(--lab-ricursive)",
    Reflection: "var(--lab-reflection)",
    Meta: "var(--lab-meta)",
    xAI: "var(--lab-xai)",
    Mistral: "var(--lab-mistral)",
    MiniMax: "var(--lab-minimax)",
    Xiaomi: "var(--lab-xiaomi)",
    Tencent: "var(--lab-tencent)",
    StepFun: "var(--lab-stepfun)",
  }};
  return map[lab] || "var(--lab-other)";
}}
function iconFor(lab) {{
  return `<span class="lb-icon" style="color:${{labColor(lab)}}">${{ICONS[lab] || ICONS._default}}</span>`;
}}

let metric = "auc";
let accessFilter = "all";
let scatterMetric = "unch_bpb";
let oaiGraphView = "top36";

const CFG = {{
  auc: {{
    title: "Leaderboard — Native Anpaure Pairwise AUC",
    hint: "Zero-fit pairwise rank AUC = (8 − mean_rank) / 7 on 954 post-cutoff Nov-2026 items (908 unique source URLs) against Mistral-Small-24B T=1.0 decoys (chance = 0.500, higher is better). Top-1 shows how often a model picks the authentic continuation #1 out of 8 (chance = 12.5%); click any row for MRR and 95% cluster CIs.",
    colLabel: "Native AUC (↑)",
    col4: "Top-1",
    ci: "auc_ci", sci: "auc_ci", higherBetter: true
  }},
  oai_bpb: {{
    title: "Leaderboard — OpenAI Nov-22 Base Scale",
    hint: "PUB marks OpenAI’s published Nov-22 base values ({len(oai_mids)} anchors, {cal_oai['loo_pct']:.2f}% within-OpenAI LOO, r={cal_oai['r']:+.3f}, ρ={cal_oai['rho']:+.3f}); other rows are 2-parameter Thurstone d' → log-BPB translations with exact Student-t (df={len(oai_mids)-2}) outer whiskers. Lower is better.",
    colLabel: "Log-Perplexity (↓)",
    col4: "AUC",
    ci: "oai_ci", sci: "oai_sci", higherBetter: false
  }},
  unch_bpb: {{
    title: "Leaderboard — Elie Oct-2026 Uncheatable Eval",
    hint: "Exact teacher-forced BPB for all 27 open models; closed frontier endpoints translated via all {len(anchors)} open anchors using ln(BPB) = a − b·d' ({cal_unch['loo_pct']:.2f}% LOO / {cal_unch['lolo_pct']:.2f}% leave-one-lab-out across {cal_unch['n_labs']} labs, exact Student-t df={len(anchors)-2} whiskers). Lower is better.",
    colLabel: "Bits per Byte (↓)",
    col4: "AUC",
    ci: "unch_ci", sci: "unch_sci", higherBetter: false
  }},
  index_5050: {{
    title: "Leaderboard — Elie Default Index (50/50)",
    hint: "Exact 50% HQ + 50% Uncheatable BPB for all 27 open models; closed endpoints translated via all {len(anchors)} open anchors using ln(BPB) = a − b·d' ({cal_idx['loo_pct']:.2f}% LOO / {cal_idx['lolo_pct']:.2f}% LOLO). Lower is better.",
    colLabel: "Bits per Byte (↓)",
    col4: "AUC",
    ci: "index_ci", sci: "index_sci", higherBetter: false
  }},
  mrr: {{
    title: "Leaderboard — Native Anpaure MRR",
    hint: "Zero-fit Mean Reciprocal Rank on 954 post-cutoff Nov-2026 items (908 unique source URLs) against Mistral-Small-24B T=1.0 decoys (random chance = 0.340, higher is better). Hover any bar or click a row for 95% cluster CIs.",
    colLabel: "Native MRR (↑)",
    col4: "Top-1",
    ci: "mrr_ci", sci: "mrr_ci", higherBetter: true
  }}
}};

function tagsHTML(m, forMetric) {{
  let out = [];
  if (forMetric === "oai_bpb" && m.oai_exact) {{
    out.push(`<span class="micro pub" title="Published OpenAI value (Anpaure est: ${{m.oai_est}})">PUB</span>`);
  }} else if (m.access === "open") {{
    out.push(`<span class="micro">${{m.kind === "base" ? "BASE" : "POST"}}</span>`);
  }}
  if (forMetric === "oai_bpb" && m.oai_extrap) out.push(`<span class="micro extrap">EXTRAP</span>`);
  if ((forMetric === "unch_bpb" || forMetric === "index_5050") && m.extrapolated) out.push(`<span class="micro extrap">EXTRAP</span>`);
  if (m.template_bypass) out.push(`<span class="micro bypass" title="Mandatory-reasoning endpoint forced to 0-CoT via raw /api/v1/completions template bypass; kept visible, excluded from OLS anchor fit by default">BYPASS</span>`);
  if (m.cot_flagged) out.push(`<span class="micro cot" title="${{m.n_pos_rtok}}/${{m.n_ok}} items with r_tok>0 (mean ${{m.r_tok}})">COT</span>`);
  return out.join(" ");
}}

let pinnedModels = new Set();
let currentList = [];

function ciDetailsHTML(m, forMetric, isPopover) {{
  const c = CFG[forMetric];
  const v = m[forMetric];
  const ci = m[c.ci];
  const sci = m[c.sci];
  const isBpb = (forMetric === "oai_bpb" || forMetric === "unch_bpb" || forMetric === "index_5050");
  const outerLbl = isBpb ? "95% Student-t CI" : "95% Cluster CI";
  const fmtCI = arr => {{
    const hw = 0.5 * (arr[1] - arr[0]);
    return `[${{arr[0].toFixed(3)}}, ${{arr[1].toFixed(3)}}] (±${{hw.toFixed(3)}})`;
  }};
  const hasDual = Boolean(ci && sci && (ci[0] !== sci[0] || ci[1] !== sci[1]));

  if (isPopover) {{
    let rows = `<div class="pop-title">${{iconFor(m.lab)}}<span>${{m.name}}</span></div>`
      + `<div class="pop-sub">${{m.lab}} · ${{m.access === "closed" ? "Closed API" : (m.kind === "base" ? "Open Base" : "Open Post-trained")}}${{m.n_ok != null ? ` · ${{m.n_ok}}/954 clean` : ""}}</div>`
      + `<div class="pop-row"><span>Score</span><b>${{v.toFixed(4)}}</b></div>`;
    if (ci) {{
      rows += `<div class="pop-row"><span>${{outerLbl}}</span><b>${{fmtCI(ci)}}</b></div>`;
    }} else {{
      rows += `<div class="pop-row"><span>Uncertainty</span><b>Exact published value</b></div>`;
    }}
    if (hasDual) {{
      rows += `<div class="pop-row"><span>95% Sampling CI</span><b>${{fmtCI(sci)}}</b></div>`;
    }}
    if (forMetric === "oai_bpb" && m.oai_exact && m.oai_est != null) {{
      rows += `<div class="pop-row"><span>Anpaure fit est</span><b>${{m.oai_est.toFixed(3)}}</b></div>`;
    }}
    if (m.auc != null) {{
      rows += `<div class="pop-div"></div>`
        + `<div class="pop-row"><span>AUC / d&prime;</span><b>${{m.auc.toFixed(4)}} (${{m.d_prime >= 0 ? "+" : ""}}${{m.d_prime.toFixed(2)}})</b></div>`
        + `<div class="pop-row"><span>MRR / Top-1</span><b>${{m.mrr.toFixed(4)}} · ${{m.top1_pct.toFixed(1)}}%</b></div>`
        + `<div class="pop-row"><span>Win rate / CoT</span><b>${{m.win_rate != null ? m.win_rate.toFixed(1) + "%" : "—"}} · ${{m.n_pos_rtok}}/${{m.n_ok}}</b></div>`;
    }}
    rows += `<div class="pop-hint">Click row to pin details</div>`;
    return rows;
  }}

  // Compact inline drawer when clicked/pinned
  let parts = [];
  if (ci) {{
    parts.push(`<span><span class="k">${{outerLbl}}</span><b>${{fmtCI(ci)}}</b></span>`);
  }} else {{
    parts.push(`<span><span class="k">Ground truth</span><b>Exact published value</b></span>`);
  }}
  if (hasDual) {{
    parts.push(`<span><span class="k">Sampling CI</span><b>${{fmtCI(sci)}}</b></span>`);
  }}
  if (forMetric === "oai_bpb" && m.oai_exact && m.oai_est != null) {{
    parts.push(`<span><span class="k">Anpaure est</span><b>${{m.oai_est.toFixed(3)}}</b></span>`);
  }}
  if (m.auc != null) {{
    parts.push(`<span><span class="k">AUC</span><b>${{m.auc.toFixed(4)}}</b></span>`);
    parts.push(`<span><span class="k">d&prime;</span><b>${{m.d_prime >= 0 ? "+" : ""}}${{m.d_prime.toFixed(3)}}</b></span>`);
    parts.push(`<span><span class="k">MRR</span><b>${{m.mrr.toFixed(4)}}</b></span>`);
    parts.push(`<span><span class="k">Top-1</span><b>${{m.top1_pct.toFixed(1)}}%</b></span>`);
    parts.push(`<span><span class="k">Win%</span><b>${{m.win_rate != null ? m.win_rate.toFixed(1) + "%" : "—"}}</b></span>`);
    parts.push(`<span><span class="k">Clean N</span><b>${{m.n_ok}}/954</b></span>`);
    parts.push(`<span><span class="k">CoT</span><b>${{m.n_pos_rtok}}/${{m.n_ok}}</b></span>`);
  }}
  return parts.join("");
}}

const ciPop = document.getElementById("ciPopover");
function positionPopover(e) {{
  const pad = 16;
  const w = ciPop.offsetWidth || 280;
  const h = ciPop.offsetHeight || 160;
  let x = e.clientX + 16;
  let y = e.clientY + 14;
  if (x + w + pad > window.innerWidth) x = Math.max(pad, e.clientX - w - 16);
  if (y + h + pad > window.innerHeight) y = Math.max(pad, e.clientY - h - 14);
  ciPop.style.left = `${{x}}px`;
  ciPop.style.top = `${{y}}px`;
}}
function hidePopover() {{
  ciPop.classList.remove("show");
  ciPop.setAttribute("aria-hidden", "true");
}}

const OAI_MILESTONES = [
  {{ val: 0.690, disp: "0.68–0.70", ref: "limit band", label: "? Common-limit window", cls: "band" }},
  {{ val: 0.732, disp: "0.732",     ref: "Nov 2026",   label: "OpenAI internal model", cls: "internal" }},
  {{ val: 0.761, disp: "0.761",     ref: "GPT-6.2 Astra", label: "Solved a Millennium Prize math problem", anchor: "GPT-6.2 Astra" }},
  {{ val: 0.828, disp: "0.828",     ref: "GPT-5.6 Sol",   label: "Cybersecurity expert", anchor: "GPT-5.6 Sol" }},
  {{ val: 0.871, disp: "0.871",     ref: "GPT-5.3 / 5.4", label: "Useful coder", anchor: "GPT-5.4" }},
  {{ val: 0.913, disp: "0.913",     ref: "GPT-4",         label: "Benefits from thinking before answering" }},
  {{ val: 0.986, disp: "0.986",     ref: "GPT-3.5",       label: "Fluent conversation" }},
];

function renderMilestones() {{
  const shell = document.querySelector(".lb-shell");
  if (shell && shell.classList.contains("no-milestones")) return;
  const box = document.getElementById("oaiMilestones");
  const topSlot = document.getElementById("oaiAbove");
  if (!box) return;
  const rows = Array.from(document.querySelectorAll("#bars .lb-row"));
  const barsEl = document.getElementById("bars");
  const totalH = Math.max(440, barsEl ? barsEl.offsetHeight : 440);

  // Render compact above-frontier block FIRST so boxTop reflects its exact rendered height!
  const above = OAI_MILESTONES.filter(ms => ms.val < 0.745);
  const inList = OAI_MILESTONES.filter(ms => ms.val >= 0.745);
  if (topSlot) {{
    topSlot.innerHTML = `<div class="oai-above">` + above.map(ms => `<div class="oai-above-row${{ms.cls ? " " + ms.cls : ""}}">
      <b>${{ms.disp}}</b><span>${{ms.label}}</span>
    </div>`).join("") + `</div>`;
  }}

  // Collect (oai_bpb, centerY in #oaiMilestones local coordinates) using exact bounding rects!
  const pts = [];
  const byNameY = {{}};
  const boxTop = box.getBoundingClientRect().top;
  rows.forEach(row => {{
    const idx = Number(row.dataset.idx);
    const m = currentList[idx];
    if (!m) return;
    const r = row.getBoundingClientRect();
    const cy = (r.top + r.height / 2) - boxTop;
    byNameY[m.name] = cy;
    if (m.oai_bpb != null) pts.push({{ bpb: m.oai_bpb, y: cy }});
  }});

  pts.sort((a, b) => a.bpb - b.bpb);
  function yForBpb(target, anchorName) {{
    if (anchorName && byNameY[anchorName] != null) return byNameY[anchorName];
    if (!pts.length) return 32;
    if (target <= pts[0].bpb) return Math.max(22, pts[0].y - 12);
    if (target >= pts[pts.length - 1].bpb) return Math.min(totalH - 90, pts[pts.length - 1].y + 42);
    for (let i = 0; i < pts.length - 1; i++) {{
      const a = pts[i], b = pts[i + 1];
      if (target >= a.bpb && target <= b.bpb) {{
        const t = (b.bpb === a.bpb) ? 0.5 : (target - a.bpb) / (b.bpb - a.bpb);
        return a.y + t * (b.y - a.y);
      }}
    }}
    return totalH * 0.5;
  }}

  const items = inList.map(ms => ({{ ...ms, y: yForBpb(ms.val, ms.anchor) }}));
  const minGap = 58;
  for (let i = 0; i < items.length; i++) {{
    if (i === 0) items[i].y = Math.max(24, items[i].y);
    else if (items[i].y < items[i - 1].y + minGap) items[i].y = items[i - 1].y + minGap;
  }}
  const aboveH = topSlot ? topSlot.offsetHeight : 0;
  const neededH = Math.max(totalH - aboveH - 42, items.length ? items[items.length - 1].y + 38 : 360);
  box.style.minHeight = `${{Math.ceil(neededH)}}px`;

  box.innerHTML = items.map(ms => `<div class="oai-ms${{ms.cls ? " " + ms.cls : ""}}" style="top:${{ms.y.toFixed(1)}}px;">
    <div class="oai-ms-val"><span>${{ms.disp}}</span><span class="oai-ms-ref">${{ms.ref}}</span></div>
    <div class="oai-ms-lbl">${{ms.label}}</div>
  </div>`).join("");
}}
window.addEventListener("resize", () => requestAnimationFrame(renderMilestones));

function renderLeaderboard() {{
  const c = CFG[metric];
  document.getElementById("chartTitle").textContent = c.title;
  document.getElementById("chartHint").textContent = c.hint;
  document.getElementById("barColLabel").textContent = c.colLabel;
  document.getElementById("col4Label").textContent = c.col4;
  hidePopover();

  const list = DATA.models
    .filter(m => m[metric] != null)
    .filter(m => accessFilter === "all" || m.access === accessFilter)
    .slice()
    .sort((a, b) => c.higherBetter ? (b[metric] - a[metric]) : (a[metric] - b[metric]));
  currentList = list;

  const valsMax = list.map(m => m[c.ci] ? m[c.ci][1] : m[metric]);
  const valsMin = list.map(m => m[c.ci] ? m[c.ci][0] : m[metric]);
  const maxV = metric === "auc"
    ? Math.min(0.95, Math.max(...valsMax) * 1.03)
    : (metric === "mrr" ? Math.min(0.84, Math.max(...valsMax) * 1.07) : Math.max(...valsMax) * 1.04);
  const minV = metric === "auc"
    ? 0.48
    : (metric === "mrr" ? 0.32 : Math.max(0.18, Math.min(...valsMin) * 0.90));
  const pct = v => Math.max(1, Math.min(100, ((v - minV) / (maxV - minV)) * 100));

  document.getElementById("bars").innerHTML = list.map((m, i) => {{
    const v = m[metric];
    const ci = m[c.ci];
    const sci = m[c.sci];
    let wHTML = "";
    if (ci) {{
      const l = pct(ci[0]), r = pct(ci[1]);
      wHTML += `<div class="ci-outer" style="left:${{l}}%;width:${{Math.max(0.5, r - l)}}%"></div>`
             + `<div class="ci-outer-cap" style="left:${{l}}%"></div>`
             + `<div class="ci-outer-cap" style="left:${{r}}%"></div>`;
    }}
    if (sci) {{
      const sl = pct(sci[0]), sr = pct(sci[1]);
      wHTML += `<div class="ci-inner" style="left:${{sl}}%;width:${{Math.max(0.6, sr - sl)}}%"></div>`;
    }}
    const hatch = m.cot_flagged || (m.access === "open" && m.kind === "post") ? " hatched" : "";
    let col4Val = "&mdash;";
    if (metric === "auc" || metric === "mrr") col4Val = m.top1_pct != null ? m.top1_pct.toFixed(1) + "%" : "&mdash;";
    else col4Val = m.auc != null ? m.auc.toFixed(3) : "&mdash;";
    const isPinned = pinnedModels.has(m.name);
    return `<div class="lb-row${{isPinned ? " pinned" : ""}}" data-idx="${{i}}">
      <div class="lb-rank">${{i + 1}}</div>
      <div class="lb-model">
        ${{iconFor(m.lab)}}
        <span class="lb-name">${{m.name}}</span>
        ${{tagsHTML(m, metric)}}
      </div>
      <div class="lb-bar-cell">
        <div class="lb-track">
          <div class="lb-fill${{hatch}}" style="width:${{pct(v)}}%;background-color:${{labColor(m.lab)}};"></div>
          ${{wHTML}}
        </div>
        <div class="lb-val">${{v.toFixed(3)}}</div>
      </div>
      <div class="lb-sub"><span class="mob-k">${{c.col4}}</span>${{col4Val}}</div>
    </div>
    <div class="lb-detail${{isPinned ? " open" : ""}}" data-detail-idx="${{i}}">${{ciDetailsHTML(m, metric, false)}}</div>`;
  }}).join("");

  document.querySelectorAll("#bars .lb-row").forEach(row => {{
    const idx = Number(row.dataset.idx);
    const m = currentList[idx];
    const barCell = row.querySelector(".lb-bar-cell");
    if (barCell) {{
      barCell.addEventListener("mouseenter", e => {{
        if (pinnedModels.has(m.name)) return;
        ciPop.innerHTML = ciDetailsHTML(m, metric, true);
        ciPop.classList.add("show");
        ciPop.setAttribute("aria-hidden", "false");
        positionPopover(e);
      }});
      barCell.addEventListener("mousemove", e => {{
        if (!ciPop.classList.contains("show")) return;
        positionPopover(e);
      }});
      barCell.addEventListener("mouseleave", hidePopover);
    }}
    row.addEventListener("click", () => {{
      const det = document.querySelector(`#bars .lb-detail[data-detail-idx="${{idx}}"]`);
      if (pinnedModels.has(m.name)) {{
        pinnedModels.delete(m.name);
        row.classList.remove("pinned");
        if (det) det.classList.remove("open");
      }} else {{
        pinnedModels.add(m.name);
        row.classList.add("pinned");
        if (det) det.classList.add("open");
        hidePopover();
      }}
      requestAnimationFrame(renderMilestones);
    }});
  }});
  requestAnimationFrame(renderMilestones);

  // Sync full ledger
  const fmtD = (m, k) => (m.dom_t1 && m.dom_t1[k] != null) ? m.dom_t1[k].toFixed(0) + "%" : "&mdash;";
  document.querySelector("#ledgerTbl tbody").innerHTML = list.map((m, i) => `<tr>
    <td class="ci-dim">${{i + 1}}</td>
    <td class="name-cell"><span style="display:inline-flex;align-items:center;gap:7px;">${{iconFor(m.lab)}}<span>${{m.name}}</span></span></td>
    <td class="ci-dim">${{m.lab}}</td>
    <td>${{tagsHTML(m, metric)}}</td>
    <td class="num">${{m.auc != null ? m.auc.toFixed(4) : "&mdash;"}}</td>
    <td class="num">${{m.mrr != null ? m.mrr.toFixed(4) : "&mdash;"}}</td>
    <td class="num">${{m.win_rate != null ? m.win_rate.toFixed(1) + "%" : "&mdash;"}}</td>
    <td class="num">${{m.top1_pct != null ? m.top1_pct.toFixed(1) + "%" : "&mdash;"}}</td>
    <td class="num">${{m.oai_bpb != null ? m.oai_bpb.toFixed(3) : "&mdash;"}}</td>
    <td class="num">${{m.unch_bpb != null ? m.unch_bpb.toFixed(3) : "&mdash;"}}</td>
    <td class="num ci-dim">${{m.n_ok != null ? `${{m.n_ok}}/954` : "&mdash;"}}</td>
    <td class="num ci-dim" title="${{m.r_tok != null ? 'Mean r_tok: ' + m.r_tok : ''}}">${{m.n_pos_rtok != null ? `${{m.n_pos_rtok}}/${{m.n_ok}}` : "&mdash;"}}</td>
    <td class="num">${{fmtD(m, "code")}}</td>
    <td class="num">${{fmtD(m, "papers")}}</td>
    <td class="num">${{fmtD(m, "news")}}</td>
    <td class="num">${{fmtD(m, "reference")}}</td>
    <td class="num">${{fmtD(m, "essays")}}</td>
  </tr>`).join("");
}}

function renderOaiPerplexity() {{
  const svg = document.getElementById("oaiPerpChart");
  if (!svg) return;
  const allLive = DATA.models
    .filter(m => m.oai_bpb != null && m.n_ok != null)
    .slice()
    .sort((a, b) => a.oai_bpb - b.oai_bpb);
  let list = allLive;
  if (oaiGraphView === "anchors") {{
    list = allLive.filter(m => m.oai_exact).slice().sort((a, b) => b.oai_bpb - a.oai_bpb);
  }} else if (oaiGraphView === "top36") {{
    // Top 36 frontier models (plus GPT-4o anchor at the right end if not already included)
    list = allLive.slice(0, 36);
  }}

  const n = list.length;
  const W = oaiGraphView === "anchors" ? 900 : (oaiGraphView === "top36" ? 1120 : Math.max(1160, n * 24 + 156));
  const H = oaiGraphView === "anchors" ? 390 : 455;
  const padL = 56, padR = 98, padT = 26, plotB = oaiGraphView === "anchors" ? 305 : 298;
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
  svg.style.minWidth = oaiGraphView === "anchors" ? "520px" : (oaiGraphView === "top36" ? "760px" : `${{Math.max(760, n * 18)}}px`);

  const vals = list.flatMap(m => {{
    const arr = [m.oai_bpb];
    if (m.oai_ci && !m.oai_exact) arr.push(m.oai_ci[0], m.oai_ci[1]);
    if (oaiGraphView === "anchors" && m.oai_est != null) arr.push(m.oai_est);
    return arr;
  }});
  const yMin = Math.floor((Math.min(...vals) - 0.008) * 50) / 50;
  const yMax = Math.ceil((Math.max(...vals) + (oaiGraphView === "anchors" ? 0.018 : 0.012)) * 50) / 50;
  // OpenAI chart convention: higher log-perplexity at TOP, lower (better!) at BOTTOM!
  const sy = v => padT + ((yMax - v) / (yMax - yMin)) * (plotB - padT);
  const step = (W - padL - padR) / Math.max(1, n);
  const sx = i => padL + (i + 0.5) * step;
  const halo = `paint-order="stroke" stroke="var(--surface)" stroke-width="3.2" stroke-linejoin="round"`;

  let parts = [];
  // Shaded extrapolation zone below GPT-6.2 Astra (0.761)
  const y62 = sy(0.761);
  if (y62 < plotB) {{
    parts.push(`<rect x="${{padL}}" y="${{y62}}" width="${{W - padL - padR}}" height="${{plotB - y62}}" fill="color-mix(in srgb, var(--lab-anthropic) 6%, transparent)"/>`);
  }}

  // Clean horizontal gridlines
  const yTicks = [];
  for (let v = Math.ceil(yMin * 25) / 25; v <= yMax + 1e-6; v += 0.04) yTicks.push(Number(v.toFixed(2)));
  for (const tv of yTicks) {{
    const y = sy(tv);
    parts.push(`<line x1="${{padL}}" y1="${{y}}" x2="${{W - padR}}" y2="${{y}}" stroke="var(--line)" stroke-dasharray="2,3"/>`);
    parts.push(`<text x="${{padL - 8}}" y="${{y + 3.5}}" text-anchor="end" font-family="var(--mono)" font-size="10.5" fill="var(--faint)">${{tv.toFixed(2)}}</text>`);
  }}

  // Published OpenAI landmark lines on right edge
  const refs = [
    {{ v: 0.761, lbl: "6.2 Astra 0.761" }},
    {{ v: 0.783, lbl: "6 Astra 0.783" }},
    {{ v: 0.828, lbl: "5.6 Sol 0.828" }},
    {{ v: 0.871, lbl: "5.4 0.871" }},
    {{ v: 0.904, lbl: "4o 0.904" }},
  ];
  for (const rf of refs) {{
    if (rf.v < yMin || rf.v > yMax) continue;
    const y = sy(rf.v);
    parts.push(`<line x1="${{padL}}" y1="${{y}}" x2="${{W - padR}}" y2="${{y}}" stroke="var(--lab-openai)" stroke-opacity="0.35" stroke-dasharray="4,4"/>`);
    parts.push(`<text x="${{W - padR + 5}}" y="${{y + 3}}" font-family="var(--mono)" font-size="8.8" fill="var(--lab-openai)">${{rf.lbl}}</text>`);
  }}
  parts.push(`<line x1="${{padL}}" y1="${{plotB}}" x2="${{W - padR}}" y2="${{plotB}}" stroke="var(--line-strong)"/>`);
  parts.push(`<line x1="${{padL}}" y1="${{padT}}" x2="${{padL}}" y2="${{plotB}}" stroke="var(--line-strong)"/>`);
  parts.push(`<text x="14" y="${{(padT + plotB) / 2}}" transform="rotate(-90 14 ${{(padT + plotB) / 2}})" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="var(--faint)">OpenAI Base Log-Perplexity (↓ lower is better)</text>`);

  // Connecting curve for OpenAI Anchors view
  if (oaiGraphView === "anchors") {{
    const pts = list.map((m, i) => `${{sx(i).toFixed(1)}},${{sy(m.oai_bpb).toFixed(1)}}`).join(" ");
    parts.push(`<polyline points="${{pts}}" fill="none" stroke="var(--lab-openai)" stroke-width="2" stroke-opacity="0.7"/>`);
  }}

  const valFs = n > 60 ? "7.8" : (n > 20 ? "8.8" : "10.5");
  const nameFs = n > 60 ? "9.0" : (n > 20 ? "10.2" : "11.5");
  let lastLblX = -999;

  list.forEach((m, i) => {{
    const cx = sx(i);
    const cy = sy(m.oai_bpb);
    const col = labColor(m.lab);
    const isPub = Boolean(m.oai_exact);
    const isExtrap = Boolean(m.oai_extrap);
    // Vertical 95% Student-t error bar for translated models
    if (m.oai_ci && !isPub) {{
      const yHi = sy(m.oai_ci[1]);
      const yLo = sy(m.oai_ci[0]);
      parts.push(`<line x1="${{cx}}" y1="${{yHi}}" x2="${{cx}}" y2="${{yLo}}" stroke="${{col}}" stroke-width="1.4" stroke-opacity="0.48"/>`);
      parts.push(`<line x1="${{cx - 3}}" y1="${{yHi}}" x2="${{cx + 3}}" y2="${{yHi}}" stroke="${{col}}" stroke-width="1.2" stroke-opacity="0.52"/>`);
      parts.push(`<line x1="${{cx - 3}}" y1="${{yLo}}" x2="${{cx + 3}}" y2="${{yLo}}" stroke="${{col}}" stroke-width="1.2" stroke-opacity="0.52"/>`);
    }}
    let topMarkerY = cy;
    // Also show Anpaure estimate diamond on OpenAI Anchors view so users see exact 0.85% LOO fit!
    if (oaiGraphView === "anchors" && m.oai_est != null) {{
      const ey = sy(m.oai_est);
      topMarkerY = Math.min(cy, ey);
      parts.push(`<line x1="${{cx}}" y1="${{cy}}" x2="${{cx}}" y2="${{ey}}" stroke="var(--ink-2)" stroke-width="1" stroke-dasharray="2,2"/>`);
      parts.push(`<polygon points="${{cx}},${{ey - 4.5}} ${{cx + 4.5}},${{ey}} ${{cx}},${{ey + 4.5}} ${{cx - 4.5}},${{ey}}" fill="var(--surface)" stroke="var(--ink)" stroke-width="1.4"><title>${{m.name}}: Published ${{m.oai_bpb.toFixed(3)}} vs Anpaure Est ${{m.oai_est.toFixed(3)}}</title></polygon>`);
    }}
    const r = isPub ? 5.2 : (n > 60 ? 3.5 : 4.5);
    const fill = isExtrap ? "var(--surface)" : col;
    const stroke = isPub ? "var(--ink)" : (isExtrap ? col : "var(--surface)");
    const sw = isPub ? 1.5 : (isExtrap ? 2.0 : 1.0);
    const tip = `${{m.name}} (${{m.lab}}): OpenAI Log-PPL ${{m.oai_bpb.toFixed(3)}}${{isPub ? " [Published]" : (m.oai_ci ? ` [95% CI ${{m.oai_ci[0].toFixed(3)}}–${{m.oai_ci[1].toFixed(3)}}]` : "")}} · AUC ${{m.auc.toFixed(4)}} · Top-1 ${{m.top1_pct.toFixed(1)}}%`;
    parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{r}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{sw}}"><title>${{tip}}</title></circle>`);
    const minGap = oaiGraphView === "anchors" ? 0 : (n > 60 ? 36 : 33);
    const mustShow = oaiGraphView === "anchors" || isPub || i < 4;
    if (mustShow && cx - lastLblX >= 24 || cx - lastLblX >= minGap) {{
      const yLbl = oaiGraphView === "anchors"
        ? topMarkerY - 10
        : (isPub ? cy - 10 : (m.oai_ci ? sy(m.oai_ci[1]) - 6 : cy - 9));
      parts.push(`<text x="${{cx}}" y="${{Math.max(padT + 9, yLbl)}}" text-anchor="middle" font-family="var(--mono)" font-size="${{valFs}}" font-weight="${{isPub ? 600 : 400}}" fill="${{isPub ? 'var(--ink)' : 'var(--ink-2)'}}" ${{halo}}>${{m.oai_bpb.toFixed(3)}}</text>`);
      lastLblX = cx;
    }}
    parts.push(`<text x="${{cx + 3}}" y="${{plotB + 12}}" transform="rotate(42 ${{cx + 3}} ${{plotB + 12}})" text-anchor="start" font-family="var(--sans)" font-size="${{nameFs}}" font-weight="${{isPub ? 600 : 400}}" fill="${{isPub ? 'var(--ink)' : 'var(--ink-2)'}}">${{m.name}}</text>`);
  }});
  svg.innerHTML = parts.join("");
}}

function renderWinChart() {{
  const svg = document.getElementById("winChart");
  const list = DATA.models
    .filter(m => m.win_rate != null)
    .slice()
    .sort((a, b) => b.win_rate - a.win_rate);

  const n = list.length;
  const W = Math.max(1120, n * 20 + 74), H = 445, padL = 42, padR = 16, padT = 18, barBottom = 290;
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
  svg.style.minWidth = `${{Math.max(740, n * 16)}}px`;
  const plotH = barBottom - padT;
  const maxW = 82;
  const step = (W - padL - padR) / n;
  const barW = Math.max(12, step * 0.74);
  const pctFs = n > 50 ? "7.8" : (n > 34 ? "8.4" : "9.5");
  const nameFs = n > 50 ? "9.2" : (n > 34 ? "9.6" : "10.5");

  let parts = [];
  parts.push(`<defs>
    <pattern id="diagHatch" width="7" height="7" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <line x1="0" y1="0" x2="0" y2="7" stroke="rgba(255,255,255,0.28)" stroke-width="2.4"/>
    </pattern>
  </defs>`);

  // Dotted horizontal gridlines (Elie style!)
  for (const pctv of [20, 40, 60, 80]) {{
    const y = barBottom - (pctv / maxW) * plotH;
    parts.push(`<line x1="${{padL}}" y1="${{y}}" x2="${{W - padR}}" y2="${{y}}" stroke="var(--line)" stroke-dasharray="2,3"/>`);
  }}
  parts.push(`<line x1="${{padL}}" y1="${{barBottom}}" x2="${{W - padR}}" y2="${{barBottom}}" stroke="var(--line-strong)"/>`);

  list.forEach((m, i) => {{
    const x = padL + i * step + (step - barW) / 2;
    const h = Math.max(12, (m.win_rate / maxW) * plotH);
    const y = barBottom - h;
    const col = labColor(m.lab);
    parts.push(`<rect x="${{x}}" y="${{y}}" width="${{barW}}" height="${{h}}" rx="4" fill="${{col}}"><title>${{m.name}}: ${{m.win_rate.toFixed(1)}}% pairwise win rate (AUC ${{m.auc}}, MRR ${{m.mrr}})</title></rect>`);
    if (m.cot_flagged) {{
      parts.push(`<rect x="${{x}}" y="${{y}}" width="${{barW}}" height="${{h}}" rx="4" fill="url(#diagHatch)" pointer-events="none"/>`);
    }}
    const lblY = h > 36 ? (y + h / 2 + 4) : (y - 5);
    const lblFill = h > 36 ? "#ffffff" : "var(--ink-2)";
    parts.push(`<text x="${{x + barW / 2}}" y="${{lblY}}" text-anchor="middle" font-family="var(--mono)" font-size="${{pctFs}}" font-weight="500" fill="${{lblFill}}">${{Math.round(m.win_rate)}}%</text>`);
    // Tiny lab dot beneath bar
    parts.push(`<circle cx="${{x + barW / 2}}" cy="${{barBottom + 13}}" r="3.6" fill="${{col}}"/>`);
    // Angled model name (Elie style)
    const tx = x + barW / 2 + 3, ty = barBottom + 27;
    const shortName = m.name.replace(" Base", "").replace("Nemotron ", "Nem ");
    parts.push(`<text x="${{tx}}" y="${{ty}}" text-anchor="end" font-family="var(--sans)" font-size="${{nameFs}}" fill="var(--ink-2)" transform="rotate(-48 ${{tx}} ${{ty}})">${{shortName}}</text>`);
  }});

  svg.innerHTML = parts.join("");
}}

function renderShowdown() {{
  const tbody = document.querySelector("#showdown tbody");
  tbody.innerHTML = (DATA.meta.paired_showdowns || []).map(p => {{
    const cls = p.p_auc < 0.05 ? "sig" : "ns";
    const fD = v => v == null ? "&mdash;" : (v >= 0 ? `+${{v.toFixed(3)}}` : v.toFixed(3));
    return `<tr>
      <td class="name-cell"><b>${{p.model_a}}</b> <span class="ci-dim" style="font-family:var(--sans);font-size:12px;">vs.</span> ${{p.model_b}} <span class="ci-dim">(${{p.n_clusters}} cl)</span></td>
      <td class="num">${{p.wins_a}}/${{p.wins_b}}/${{p.ties}}</td>
      <td class="num ${{cls}}">${{p.auc_diff >= 0 ? "+" : ""}}${{p.auc_diff.toFixed(4)}} <span class="ci-dim">[${{p.auc_ci95[0].toFixed(3)}}, ${{p.auc_ci95[1].toFixed(3)}}]</span></td>
      <td class="num">${{p.t1_diff_pct >= 0 ? "+" : ""}}${{p.t1_diff_pct.toFixed(1)}}%</td>
      <td class="num ${{cls}}">${{p.p_auc < 0.0001 ? "<0.0001" : p.p_auc.toFixed(4)}}</td>
      <td class="num">${{fD(p.dom_auc_diff.code)}}</td>
      <td class="num">${{fD(p.dom_auc_diff.papers)}}</td>
      <td class="num">${{fD(p.dom_auc_diff.news)}}</td>
      <td class="num">${{fD(p.dom_auc_diff.reference)}}</td>
      <td class="num">${{fD(p.dom_auc_diff.essays)}}</td>
    </tr>`;
  }}).join("");
}}

function renderScatter() {{
  const svg = document.getElementById("scatter");
  const W = 600, H = 360, padL = 54, padR = 20, padT = 18, padB = 46;
  const isOai = (scatterMetric === "oai_bpb");
  const anchors = DATA.models.filter(m => isOai ? m.oai_exact : m.is_anchor);
  const others = DATA.models.filter(m => m.nce_doc_bpb != null && !(isOai ? m.oai_exact : m.is_anchor));
  const allPts = anchors.concat(others);
  const xs = allPts.map(m => m.nce_doc_bpb), ys = allPts.map(m => m[scatterMetric]);
  const xMin = Math.min(...xs) - 0.14, xMax = Math.max(...xs) + 0.12;
  const yMin = Math.min(...ys) - 0.032, yMax = Math.max(...ys) + 0.028;
  const sx = x => padL + ((x - xMin) / (xMax - xMin)) * (W - padL - padR);
  const sy = y => H - padB - ((y - yMin) / (yMax - yMin)) * (H - padT - padB);

  const cal = DATA.meta.calibration[scatterMetric];
  const curvePts = [];
  for (let i = 0; i <= 48; i++) {{
    const xv = xMin + (i / 48) * (xMax - xMin);
    const yv = cal.log_y ? Math.exp(cal.intercept + cal.slope * xv) : (cal.intercept + cal.slope * xv);
    curvePts.push(`${{sx(xv).toFixed(1)}},${{sy(yv).toFixed(1)}}`);
  }}

  const halo = `paint-order="stroke" stroke="var(--surface)" stroke-width="3.5" stroke-linejoin="round"`;
  let parts = [];
  for (let i = 0; i <= 4; i++) {{
    const yv = yMin + (i / 4) * (yMax - yMin), yp = sy(yv);
    parts.push(`<line x1="${{padL}}" y1="${{yp}}" x2="${{W - padR}}" y2="${{yp}}" stroke="var(--line)" stroke-dasharray="2,3"/>`);
    parts.push(`<text x="${{padL - 8}}" y="${{yp + 3.5}}" text-anchor="end" font-family="var(--mono)" font-size="10" fill="var(--faint)">${{yv.toFixed(2)}}</text>`);
  }}
  for (let i = 0; i <= 4; i++) {{
    const xv = xMin + (i / 4) * (xMax - xMin), xp = sx(xv);
    parts.push(`<text x="${{xp}}" y="${{H - padB + 16}}" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="var(--faint)">${{xv.toFixed(2)}}</text>`);
  }}
  parts.push(`<line x1="${{padL}}" y1="${{H - padB}}" x2="${{W - padR}}" y2="${{H - padB}}" stroke="var(--line-strong)"/>`);
  parts.push(`<line x1="${{padL}}" y1="${{padT}}" x2="${{padL}}" y2="${{H - padB}}" stroke="var(--line-strong)"/>`);
  parts.push(`<text x="${{(padL + W - padR) / 2}}" y="${{H - 8}}" text-anchor="middle" font-family="var(--serif)" font-style="italic" font-size="13.5" fill="var(--ink-2)">Thurstone &minus;d&prime; = &minus;&radic;2&thinsp;&Phi;<tspan dy="-4.5" font-size="9.5" font-style="normal">&minus;1</tspan><tspan dy="4.5">(AUC) &nbsp;(lower = stronger)</tspan></text>`);
  parts.push(`<text x="14" y="${{(padT + H - padB) / 2}}" text-anchor="middle" font-family="var(--serif)" font-style="italic" font-size="13.5" fill="var(--ink-2)" transform="rotate(-90 14 ${{(padT + H - padB) / 2}})">${{isOai ? "OpenAI Nov-22 Base Log-PPL" : "Elie Uncheatable Eval BPB"}}</text>`);

  // Highlighted out-of-sample frontier diamonds
  const featNames = isOai
    ? new Set(["Claude Fable 5.2", "Gemini 4 Carbon", "Claude Opus 5.5"])
    : new Set(["Claude Fable 5.2", "Gemini 4 Carbon", "GPT-6.2 Astra", "Claude Opus 5.5"]);

  // 1. Regression curve underneath points
  parts.push(`<polyline points="${{curvePts.join(' ')}}" fill="none" stroke="var(--ink)" stroke-width="1.35" stroke-dasharray="4,4" opacity="0.55"/>`);

  // 2. Background projected models (quiet translucent ticks so they never bury anchors!)
  for (const m of others) {{
    if (featNames.has(m.name)) continue;
    const cx = sx(m.nce_doc_bpb), cy = sy(m[scatterMetric]);
    parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="2.3" fill="${{labColor(m.lab)}}" opacity="0.32"><title>${{m.name}} (est): AUC=${{m.auc}}, -d'=${{m.nce_doc_bpb}}, Est=${{m[scatterMetric]}}</title></circle>`);
  }}

  // 3. Featured frontier projected diamonds with clean non-overlapping leader callouts
  const featOffsets = isOai ? {{
    "Claude Fable 5.2": {{ dx: -18, dy: -40, anchor: "start" }},
    "Gemini 4 Carbon":  {{ dx: 12,  dy: 28,  anchor: "start" }},
    "Claude Opus 5.5":  {{ dx: -16, dy: -36, anchor: "end" }},
  }} : {{
    "Claude Fable 5.2": {{ dx: -16, dy: -34, anchor: "start" }},
    "Gemini 4 Carbon":  {{ dx: 16,  dy: 26,  anchor: "start" }},
    "GPT-6.2 Astra":    {{ dx: 34,  dy: 6,   anchor: "start" }},
    "Claude Opus 5.5":  {{ dx: -14, dy: -26, anchor: "end" }},
  }};
  for (const m of others) {{
    if (!featNames.has(m.name)) continue;
    const cx = sx(m.nce_doc_bpb), cy = sy(m[scatterMetric]);
    const col = labColor(m.lab);
    parts.push(`<polygon points="${{cx}},${{cy - 5.2}} ${{cx + 5.2}},${{cy}} ${{cx}},${{cy + 5.2}} ${{cx - 5.2}},${{cy}}" fill="${{col}}" stroke="var(--surface)" stroke-width="1.4"><title>${{m.name}} (est): AUC=${{m.auc}}, -d'=${{m.nce_doc_bpb}}, Est=${{m[scatterMetric]}}</title></polygon>`);
    const off = featOffsets[m.name];
    if (off) {{
      const tx = cx + off.dx, ty = cy + off.dy;
      parts.push(`<line x1="${{cx}}" y1="${{cy}}" x2="${{tx}}" y2="${{ty - (off.dy < 0 ? -3 : 6)}}" stroke="${{col}}" stroke-width="0.9" opacity="0.6"/>`);
      parts.push(`<text x="${{tx}}" y="${{ty}}" text-anchor="${{off.anchor}}" font-family="var(--sans)" font-size="9.8" font-weight="600" fill="${{col}}" ${{halo}}>${{m.name}}</text>`);
    }}
  }}

  // 4. Published ground-truth anchors ON TOP (crisp solid circles + zero-collision callouts!)
  const anchorOffsets = isOai ? {{
    "GPT-6.2 Astra": {{ dx: 48,  dy: 2,   anchor: "start" }},
    "GPT-6 Astra":   {{ dx: 18,  dy: 10,  anchor: "start" }},
    "GPT-5.6 Sol":   {{ dx: 10,  dy: 12,  anchor: "start" }},
    "GPT-5.4":       {{ dx: 12,  dy: 14,  anchor: "start" }},
    "GPT-4o":        {{ dx: 10,  dy: 6,   anchor: "start" }},
  }} : {{
    "Kimi K3.1 Pro":        {{ dx: -12, dy: -16, anchor: "end" }},
    "Kimi K3":              {{ dx: -12, dy: 16,  anchor: "end" }},
    "DeepSeek V4 Pro Base": {{ dx: 12,  dy: 10,  anchor: "start" }},
    "Gemma 4 31B":          {{ dx: -18, dy: -24, anchor: "end" }},
    "Nemotron 3 Nano Base": {{ dx: -10, dy: 4,   anchor: "end" }},
  }};
  for (const m of anchors) {{
    const cx = sx(m.nce_doc_bpb), cy = sy(m[scatterMetric]);
    const col = labColor(m.lab);
    parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="${{col}}" stroke="var(--ink)" stroke-width="1.3"><title>${{m.name}} (anchor): AUC=${{m.auc}}, -d'=${{m.nce_doc_bpb}}, Exact=${{m[scatterMetric]}}</title></circle>`);
    const off = anchorOffsets[m.name];
    if (off) {{
      const tx = cx + off.dx, ty = cy + off.dy;
      const dist = Math.hypot(off.dx, off.dy);
      if (dist > 14) {{
        parts.push(`<line x1="${{cx}}" y1="${{cy}}" x2="${{tx}}" y2="${{ty - (off.dy < 0 ? -3 : 6)}}" stroke="var(--ink-2)" stroke-width="0.85" opacity="0.55"/>`);
      }}
      parts.push(`<text x="${{tx}}" y="${{ty}}" text-anchor="${{off.anchor}}" font-family="var(--sans)" font-size="9.8" font-weight="600" fill="var(--ink)" ${{halo}}>${{m.name.replace(" Base", "")}}</text>`);
    }}
  }}

  const lbl = isOai
    ? `${{anchors.length}} OpenAI anchors · r = +${{cal.r.toFixed(3)}} · ρ = +${{cal.rho.toFixed(3)}} · LOO = ${{cal.loo_pct.toFixed(2)}}%`
    : `${{DATA.meta.calibration.n_open_anchors}} open anchors · r = +${{cal.r.toFixed(3)}} · LOO = ${{cal.loo_pct.toFixed(2)}}% · LOLO = ${{cal.lolo_pct.toFixed(2)}}%`;
  parts.push(`<text x="${{padL + 10}}" y="${{padT + 14}}" font-family="var(--mono)" font-size="10.2" fill="var(--ink-2)" ${{halo}}>${{lbl}}</text>`);
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
  svg.innerHTML = parts.join("");
}}

// Controls
document.querySelectorAll("#metricBtns button").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#metricBtns button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    metric = btn.dataset.metric;
    if (metric === "oai_bpb" || metric === "unch_bpb") {{
      scatterMetric = metric;
      document.querySelectorAll("#scatterBtns button").forEach(b => b.classList.toggle("active", b.dataset.sc === scatterMetric));
      renderScatter();
    }}
    renderLeaderboard();
  }});
}});
document.querySelectorAll("#accessBtns button").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#accessBtns button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    accessFilter = btn.dataset.access;
    renderLeaderboard();
  }});
}});
document.querySelectorAll("#oaiGraphBtns button").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#oaiGraphBtns button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    oaiGraphView = btn.dataset.oaiView;
    renderOaiPerplexity();
  }});
}});
document.querySelectorAll("#scatterBtns button").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#scatterBtns button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    scatterMetric = btn.dataset.sc;
    renderScatter();
  }});
}});
document.getElementById("themeBtn").addEventListener("click", () => {{
  const root = document.documentElement;
  root.setAttribute("data-theme", root.getAttribute("data-theme") === "dark" ? "light" : "dark");
}});

renderLeaderboard();
renderOaiPerplexity();
renderWinChart();
renderShowdown();
renderScatter();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Rendered {OUT_HTML} ({len(rows)} models; {len(html):,} bytes; total eval spend=${total_spend:.2f})")


if __name__ == "__main__":
    build()
