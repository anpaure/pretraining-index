"""High-concurrency, resumable OpenRouter runner for Anpaure Pretraining Index v1
(N=1,000 multi-domain harder-decoy Rank-NCE benchmark).

Zero local compute — 100% OpenRouter API, strict frozen prompt across all models,
per-item JSONL checkpointing under `runs/anpaure_v1_traces/<slug>.jsonl`, exact
reasoning-token (`r_tok`) and cost telemetry on every single call.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from pplx_protocol_a.nce_bpb import LETTERS_MAX, parse_letter_ranking, score_rank_nce_cv

CLEAN_RE = re.compile(r"^[A-H](?:\s*,\s*[A-H]){7}\s*$")


def is_strict_valid_record(r: dict) -> bool:
    if not r.get("ok"):
        return False
    raw = (r.get("raw") or "").strip()
    return bool(CLEAN_RE.match(raw) and len(set(re.findall(r"[A-H]", raw))) == 8)

ROOT = Path(__file__).resolve().parent
BENCH_PATH = ROOT / "runs/anpaure_v1_n1000.json"
TRACE_DIR = ROOT / "runs/anpaure_v1_traces"

LETTERS = LETTERS_MAX[:8]

DOMAIN_LABELS = {
    "news": "news / parliamentary speech",
    "papers": "academic research paper",
    "code": "source code",
    "reference": "encyclopedia reference article",
    "essays": "essay / narrative prose",
}

SYS_PROMPT = (
    "You are evaluating which short continuation is the authentic next text from an unseen document "
    "(news, academic paper, source code, encyclopedia, or essay). "
    "Rank ALL 8 candidates A through H from MOST likely authentic continuation to LEAST likely. "
    "Output ONLY the 8 distinct uppercase letters separated by commas (example: C,A,F,B,H,D,E,G). "
    "No explanations."
)


def make_user_prompt(it: dict) -> str:
    dlbl = DOMAIN_LABELS.get(it["domain"], it["domain"])
    pref = it["prefix_text"][-900:]
    opts = "\n".join(
        f"{LETTERS[i]}: {json.dumps(it['candidates'][i], ensure_ascii=False)}"
        for i in range(8)
    )
    return (
        f"Document excerpt ({dlbl}) ending:\n...\n{pref}\n\n"
        f"Candidate continuations (exact next snippet):\n{opts}\n\n"
        "Rank A through H from most likely authentic to least likely (comma-separated letters only):"
    )


# Full Model Registry: (model_id, role, elie_id_or_None, reasoning_cfg, max_tokens, workers)
MODEL_SPECS: list[tuple[str, str, str | None, dict | None, int, int]] = [
    # --- 16 Open Elie Bakouch Pre-training Index Anchors (8 independent labs; zero OpenAI/Anthropic/Google-closed) ---
    ("moonshotai/kimi-k3.1-pro",          "open_anchor", "moonshotai__Kimi-K3.1-Pro",                                      {"enabled": False},    48,  16),
    ("moonshotai/kimi-k3",                "open_anchor", "moonshotai__Kimi-K3",                                            {"enabled": False},    48,  16),
    ("meta/muse-spark-open",              "open_anchor", "meta-models__Muse-Spark-Open-816B-A42B",                         {"enabled": False},    24,  18),
    ("thinkingmachines/inkling-2-small",  "open_anchor", "thinkingmachines__Inkling-2-Small",                              {"enabled": False},    24,  18),
    ("deepseek/deepseek-v4-pro",          "open_anchor", "deepseek-ai__DeepSeek-V4-Pro-Base",                              {"enabled": False},    48,  16),
    ("reflection/beam-1.0-501b-a23b",     "open_anchor", "reflection__Beam-1.0-501B-A23B",                                 {"enabled": False},    24,  18),
    ("deepseek/deepseek-v4-flash",        "open_anchor", "deepseek-ai__DeepSeek-V4-Flash-Base",                            {"enabled": False},    48,  20),
    ("deepseek/deepseek-v4.1-flash-1017", "open_anchor", "deepseek-ai__DeepSeek-V4.1-Flash-1017",                          {"enabled": False},    48,  20),
    ("nvidia/nemotron-3-ultra-550b-a55b", "open_anchor", "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16",            {"enabled": False},    48,  16),
    ("z-ai/glm-5.3-flash",                "open_anchor", "zai-org__GLM-5.3-Flash",                                         {"enabled": False},   128,  16),
    ("nvidia/nemotron-3.5-super-vl",      "open_anchor", "nvidia__NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16",       {"enabled": False},    48,  20),
    ("nvidia/nemotron-3-super-120b-a12b", "open_anchor", "nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16",            {"enabled": False},    48,  20),
    ("ricursiveai/ripple-65b-preview",    "open_anchor", "ricursiveAI__Ripple-65B-Base-Preview__tok-from-Preview",         {"enabled": False},    48,  16),
    ("google/gemma-4-31b-it",             "open_anchor", "google__gemma-4-31B",                                            {"enabled": False},    48,  20),
    ("google/gemma-4-26b-a4b-it",         "open_anchor", "google__gemma-4-26B-A4B",                                        {"enabled": False},    48,  20),
    ("thinkingmachines/inkling-2-mini",   "open_anchor", "thinkingmachines__Inkling-2-Mini",                               {"enabled": False},    48,  20),
    ("nvidia/nemotron-3-nano-30b-a3b",    "open_anchor", "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",               {"enabled": False},    48,  24),
    # --- 6 Additional BPBench Open Models (r_tok=0; evaluated on N=954 for cross-eval validation) ---
    ("moonshotai/kimi-k2.6",              "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("z-ai/glm-5.2",                      "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("z-ai/glm-5.3",                      "open_bpbench", None,                                                            {"effort": "minimal"}, 128, 20),
    ("z-ai/glm-5.3-prime",                "closed",       None,                                                            {"effort": "minimal"}, 128, 16),
    ("z-ai/glm-5",                        "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("z-ai/glm-5-turbo",                  "closed",       None,                                                            {"enabled": False},    48,  18),
    ("z-ai/glm-4.7",                      "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("qwen/qwen3.7-plus",                 "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("z-ai/glm-5.1",                      "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("thinkingmachines/inkling",          "open_bpbench", None,                                                            {"enabled": False},    48,  16),
    ("qwen/qwen3.6-27b",                  "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("qwen/qwen3.8-27b",                  "open_bpbench", None,                                                            {"enabled": False},    48,  18),
    ("qwen/qwen4-max",                    "closed",       None,                                                            {"enabled": False},    48,  16),
    ("qwen/qwen3.7-max",                  "closed",       None,                                                            {"enabled": False},    48,  16),
    ("qwen/qwen3.8-flash",                "closed",       None,                                                            {"enabled": False},    48,  18),
    ("qwen/qwen3.6-plus",                 "closed",       None,                                                            {"enabled": False},    48,  18),
    ("x-ai/grok-4.3",                     "closed",       None,                                                            {"enabled": False},    48,  16),
    ("x-ai/grok-4.20",                    "closed",       None,                                                            {"enabled": False},    48,  16),
    # --- 14 Closed Frontier Models (interleaved across Anthropic / OpenAI / Google for zero rate-limit contention) ---
    ("anthropic/claude-fable-5.2",        "closed",      None,                                                             {"effort": "minimal"}, 64,   12),
    ("anthropic/claude-fable-5.1",        "closed",      None,                                                             {"effort": "minimal"}, 64,   12),
    ("anthropic/claude-fable-5",          "closed",      None,                                                             {"effort": "minimal"}, 64,   12),
    ("anthropic/claude-opus-5.5",         "closed",      None,                                                             {"effort": "minimal"}, 64,   12),
    ("anthropic/claude-opus-5",           "closed",      None,                                                             {"enabled": False},    64,   12),
    ("anthropic/claude-opus-4.8",         "closed",      None,                                                             {"enabled": False},    64,   12),
    ("anthropic/claude-opus-4.7",         "closed",      None,                                                             {"enabled": False},    64,   12),
    ("anthropic/claude-opus-4.6",         "closed",      None,                                                             {"enabled": False},    64,   12),
    ("anthropic/claude-opus-4.5",         "closed",      None,                                                             {"enabled": False},    64,   12),
    ("anthropic/claude-opus-4.1",         "closed",      None,                                                             {"enabled": False},    64,   12),
    ("anthropic/claude-sonnet-5.5",       "closed",      None,                                                             {"effort": "minimal"}, 64,   14),
    ("anthropic/claude-sonnet-5",         "closed",      None,                                                             {"enabled": False},    64,   14),
    ("anthropic/claude-sonnet-4.6",       "closed",      None,                                                             {"enabled": False},    64,   14),
    ("anthropic/claude-sonnet-4.5",       "closed",      None,                                                             {"enabled": False},    64,   14),
    ("anthropic/claude-haiku-5.5",        "closed",      None,                                                             {"enabled": False},    64,   16),
    ("anthropic/claude-haiku-4.5",        "closed",      None,                                                             {"enabled": False},    64,   16),
    ("openai/gpt-6.2-astra",              "closed",      None,                                                             {"effort": "minimal"}, 128,  16),
    ("google/gemini-4-neon",              "closed",      None,                                                             {"effort": "minimal"}, 1536, 16),
    ("openai/gpt-6.2-sol",                "closed",      None,                                                             {"effort": "minimal"}, 128,  18),
    ("openai/gpt-6.1-sol",                "closed",      None,                                                             {"effort": "minimal"}, 256,  16),
    ("openai/gpt-6-sol",                  "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-6-luna",                 "closed",      None,                                                             {"effort": "none"},    64,   20),
    ("openai/gpt-6.1-terra",              "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-6-astra",                "closed",      None,                                                             {"effort": "minimal"}, 256,  14),
    ("google/gemini-4-argon",             "closed",      None,                                                             {"effort": "minimal"}, 512,  16),
    ("google/gemini-4-carbon",            "closed",      None,                                                             {"effort": "minimal"}, 512,  18),
    ("google/gemini-4-hydrogen",          "closed",      None,                                                             {"effort": "minimal"}, 256,  18),
    ("google/gemini-3.1-pro-preview",     "closed",      None,                                                             {"effort": "minimal"}, 128,  18),
    ("google/gemini-3.8-flash",           "closed",      None,                                                             {"effort": "minimal"}, 128,  20),
    ("google/gemini-3.7-flash",           "closed",      None,                                                             {"effort": "minimal"}, 128,  20),
    ("google/gemini-3.6-flash",           "closed",      None,                                                             {"effort": "minimal"}, 256,  20),
    ("google/gemini-3.5-flash",           "closed",      None,                                                             {"effort": "minimal"}, 256,  20),
    ("google/gemini-3-flash-preview",     "closed",      None,                                                             {"enabled": False},    48,   22),
    ("google/gemini-3.1-flash-lite",      "closed",      None,                                                             {"enabled": False},    48,   24),
    ("google/gemini-3.5-flash-lite",      "closed",      None,                                                             {"effort": "minimal"}, 256,  24),
    ("openai/gpt-5.6-sol",                "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5.6-terra",              "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5.6-luna",               "closed",      None,                                                             {"effort": "none"},    64,   20),
    ("openai/gpt-5.5",                    "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5.4",                    "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5.4-mini",               "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5.4-nano",               "closed",      None,                                                             {"effort": "none"},    64,   20),
    ("openai/gpt-5.3-codex",              "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5.2",                    "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5.1",                    "closed",      None,                                                             {"effort": "none"},    64,   18),
    ("openai/gpt-5",                      "closed",      None,                                                             {"effort": "minimal"}, 128,  18),
    ("openai/gpt-4.1",                    "closed",      None,                                                             None,                  48,   20),
    ("openai/gpt-4o",                     "closed",      None,                                                             None,                  48,   20),
    ("openai/gpt-4o-mini",                "closed",      None,                                                             None,                  48,   24),
    ("openai/gpt-oss-120b",               "open_bypass", None,                                                             {"enabled": False},    36,   18),
    ("openai/gpt-oss-20b",                "open_bypass", None,                                                             {"enabled": False},    36,   16),
    ("openai/o3",                         "closed",      None,                                                             {"effort": "low"},     1536, 16),
]


def slug_for(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def call_one(
    api_key: str,
    model_id: str,
    reasoning_cfg: dict | None,
    max_tokens: int,
    it: dict,
) -> dict:
    body: dict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": make_user_prompt(it)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if reasoning_cfg is not None:
        body["reasoning"] = dict(reasoning_cfg)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": "anpaure-v1-eval/1.0",
    }

    for att in range(5):
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=110) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                resp = json.loads(raw.decode())
            if "error" in resp:
                msg = str(resp["error"])
                if "temperature" in msg.lower() and "temperature" in body:
                    body.pop("temperature", None)
                    continue
                if "reasoning" in msg.lower() and "reasoning" in body:
                    if body["reasoning"] == {"enabled": False}:
                        body["reasoning"] = {"effort": "minimal"}
                        body["max_tokens"] = max(max_tokens, 256)
                        continue
                if att < 4:
                    time.sleep(min(12, 2 ** att))
                    continue
                return {
                    "item_id": it["item_id"],
                    "doc_id": it["doc_id"],
                    "domain": it["domain"],
                    "ok": False,
                    "cost": 0.0,
                    "r_tok": 0,
                    "err": msg[:100],
                }
            ch = (resp.get("choices") or [{}])[0]
            txt = ((ch.get("message") or {}).get("content") or "").strip()
            u = resp.get("usage") or {}
            rt = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
            cost = float(u.get("cost") or 0.0)
            if not txt:
                # If reasoning ate max_tokens, retry once with larger budget (up to 2560)
                if rt > 0 and body["max_tokens"] < 2560 and att < 2:
                    body["max_tokens"] = min(2560, int(body["max_tokens"] * 1.8) + 256)
                    continue
                return {
                    "item_id": it["item_id"],
                    "doc_id": it["doc_id"],
                    "domain": it["domain"],
                    "ok": False,
                    "complete": False,
                    "cost": cost,
                    "r_tok": rt,
                    "err": "empty_content",
                }
            rk, comp = parse_letter_ranking(txt, 8)
            gold_rank = rk.index(LETTERS[it["gold_idx"]]) + 1
            return {
                "item_id": it["item_id"],
                "doc_id": it["doc_id"],
                "domain": it["domain"],
                "ok": True,
                "complete": comp,
                "gold_rank": gold_rank,
                "raw": txt[:40],
                "r_tok": rt,
                "cost": cost,
            }
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")
            if "temperature" in err.lower() and "temperature" in body:
                body.pop("temperature", None)
                continue
            if "reasoning" in err.lower() and "reasoning" in body:
                if body["reasoning"] == {"enabled": False}:
                    body["reasoning"] = {"effort": "minimal"}
                    body["max_tokens"] = max(max_tokens, 256)
                    continue
            if e.code in (408, 429, 500, 502, 503, 504) and att < 4:
                time.sleep(min(15, 2 ** att))
                continue
            return {
                "item_id": it["item_id"],
                "doc_id": it["doc_id"],
                "domain": it["domain"],
                "ok": False,
                "cost": 0.0,
                "r_tok": 0,
                "err": f"HTTP {e.code}: {err[:80]}",
            }
        except Exception as exc:
            if att < 4:
                time.sleep(min(10, 2 ** att))
                continue
            return {
                "item_id": it["item_id"],
                "doc_id": it["doc_id"],
                "domain": it["domain"],
                "ok": False,
                "cost": 0.0,
                "r_tok": 0,
                "err": str(exc)[:80],
            }
    return {"item_id": it["item_id"], "doc_id": it["doc_id"], "domain": it["domain"], "ok": False, "cost": 0.0, "r_tok": 0, "err": "retries"}


def run_model(
    api_key: str,
    spec: tuple[str, str, str | None, dict | None, int, int],
    items: list[dict],
    trace_dir: Path,
    *,
    retry_failed: bool = False,
) -> dict:
    model_id, role, elie_id, r_cfg, max_toks, workers = spec
    trace_dir.mkdir(parents=True, exist_ok=True)
    tpath = trace_dir / f"{slug_for(model_id)}.jsonl"

    done_map: dict[str, dict] = {}
    if tpath.exists():
        for line in tpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            prev = done_map.get(rec["item_id"])
            is_c = is_strict_valid_record(rec)
            if is_c or (prev is None and not retry_failed):
                if prev is None or is_c or not is_strict_valid_record(prev):
                    done_map[rec["item_id"]] = rec

    todo = [it for it in items if it["item_id"] not in done_map]
    lock = threading.Lock()
    t0 = time.time()

    if todo:
        print(f"  [start] {model_id:38s} | todo={len(todo):4d}/{len(items)} (cached={len(done_map)}) workers={workers}", flush=True)
        with tpath.open("a", encoding="utf-8") as fh:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(call_one, api_key, model_id, r_cfg, max_toks, it) for it in todo]
                completed = 0
                for fut in concurrent.futures.as_completed(futs):
                    rec = fut.result()
                    with lock:
                        done_map[rec["item_id"]] = rec
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        completed += 1
                        if completed % 250 == 0 or completed == len(todo):
                            ok_so_far = sum(1 for r in done_map.values() if r.get("ok"))
                            cost_so_far = sum(float(r.get("cost") or 0.0) for r in done_map.values())
                            print(
                                f"    [{model_id:36s}] {completed:4d}/{len(todo)} done | ok={ok_so_far}/{len(done_map)} | ${cost_so_far:.3f} | {time.time()-t0:.0f}s",
                                flush=True,
                            )

    recs = [done_map[it["item_id"]] for it in items if it["item_id"] in done_map]
    ok_recs = [r for r in recs if is_strict_valid_record(r)]
    total_cost = sum(float(r.get("cost") or 0.0) for r in recs)
    mean_rt = float(np.mean([r.get("r_tok", 0) for r in ok_recs])) if ok_recs else 0.0
    mean_bytes = float(np.mean([it["true_chunk_bytes"] for it in items]))

    summary = None
    domain_stats: dict[str, dict] = {}
    if len(ok_recs) >= 10:
        # Use nominal q_doc_bpb=0.600 (scale cancels cleanly under linear calibration, while info_gain_bits is exact!)
        summary = score_rank_nce_cv(
            ok_recs,
            k_candidates=8,
            q_chunk_bpb=0.600,
            q_doc_bpb=0.600,
            mean_chunk_bytes=mean_bytes,
            n_boot=2000,
        )
        for dom in sorted({it["domain"] for it in items}):
            drecs = [r for r in ok_recs if r.get("domain") == dom]
            if len(drecs) >= 3:
                d_ranks = np.array([r["gold_rank"] for r in drecs], dtype=float)
                domain_stats[dom] = {
                    "n": len(drecs),
                    "top1": float(np.mean(d_ranks == 1)),
                    "mrr": float(np.mean(1.0 / d_ranks)),
                    "mean_rank": float(np.mean(d_ranks)),
                }

    top1 = summary["top1_accuracy"] * 100 if summary else 0.0
    mrr = summary["mrr"] if summary else 0.0
    ig = summary["info_gain_bits"] if summary else 0.0
    print(
        f"  [DONE ] {model_id:38s} | ok={len(ok_recs):4d}/{len(items)} | r_tok={mean_rt:5.1f} | "
        f"Top1={top1:5.1f}% | MRR={mrr:.4f} | IG={ig:.4f}b | ${total_cost:.3f}",
        flush=True,
    )
    return {
        "model_id": model_id,
        "role": role,
        "elie_id": elie_id,
        "n_total": len(items),
        "n_ok": len(ok_recs),
        "mean_r_tok": mean_rt,
        "cost": total_cost,
        "summary": summary,
        "domain_stats": domain_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(BENCH_PATH))
    ap.add_argument("--trace-dir", default=str(TRACE_DIR))
    ap.add_argument("--models", default="", help="Comma-separated substring filters (empty = all 27)")
    ap.add_argument("--role", default="", help="Filter by role: open_anchor or closed")
    ap.add_argument("--limit", type=int, default=0, help="If >0, run stratified first N items")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--parallel-models", type=int, default=1, help="Run this many models concurrently across providers")
    ap.add_argument("--out-summary", default=str(ROOT / "runs/anpaure_v1_results.json"))
    args = ap.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    bench = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    items = bench["items"]
    if args.limit > 0:
        by_d: dict[str, list[dict]] = {}
        for it in items:
            by_d.setdefault(it["domain"], []).append(it)
        per = max(1, args.limit // len(by_d))
        sub = []
        for d in sorted(by_d):
            sub.extend(by_d[d][:per])
        items = sub

    specs = MODEL_SPECS
    if args.role:
        specs = [s for s in specs if s[1] == args.role]
    if args.models:
        pats = [p.strip().lower() for p in args.models.split(",") if p.strip()]
        specs = [s for s in specs if any(p in s[0].lower() for p in pats)]

    print(f"Running {len(specs)} models on {len(items)} items (bench={args.bench}, parallel_models={args.parallel_models})...", flush=True)
    out_path = Path(args.out_summary)
    results: dict[str, dict] = {}
    if out_path.exists() and args.limit == 0:
        results = json.loads(out_path.read_text(encoding="utf-8")).get("models", {})

    grand_cost = 0.0
    save_lock = threading.Lock()

    def _run_and_save(spec: tuple[str, str, str | None, dict | None, int, int]) -> dict:
        res = run_model(api_key, spec, items, Path(args.trace_dir), retry_failed=args.retry_failed)
        with save_lock:
            results[spec[0]] = res
            if args.limit == 0:
                out_path.write_text(
                    json.dumps(
                        {
                            "bench": args.bench,
                            "decoy_model": bench.get("decoy_model"),
                            "n_items": len(items),
                            "models": results,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        return res

    if args.parallel_models <= 1:
        for spec in specs:
            res = _run_and_save(spec)
            grand_cost += res["cost"]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_models) as mpool:
            futs = [mpool.submit(_run_and_save, sp) for sp in specs]
            for fut in concurrent.futures.as_completed(futs):
                grand_cost += fut.result()["cost"]

    print(f"\nALL DONE: {len(specs)} models evaluated | batch total cost = ${grand_cost:.3f}", flush=True)


if __name__ == "__main__":
    main()
