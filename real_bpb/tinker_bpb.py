"""Tinker API teacher-forced BPB + ANPI 954x8 white-box evaluator.

Uses Thinking Machines Lab's `tinker.SamplingClient.compute_logprobs(..., prompt_logprobs_last_n=...)`
to evaluate BOTH:
  1. Exact ANPI 954-item x 8-candidate white-box testset (BPE-exact conditional logprobs,
     prefix-cached across the 8 candidates per item, reporting gold-span BPB + white-box
     oracle AUC / MRR / Top-1 alongside raw & byte-normalized rankings).
  2. Full 1,000-doc archived Nov-2026 corpus BPB (byte-weighted, macro-5 domain,
     exact `anpi_954_subset` BPB, and 2,000-draw URL-clustered 95% CIs).

Zero weight downloads, zero GPU provisioning — passes `TINKER_API_KEY` via shell env only.

Usage:
    TINKER_API_KEY=tml-... uvx --with tinker python3 real_bpb/tinker_bpb.py --preset inkling-2-mini-free --limit 30
    TINKER_API_KEY=tml-... uvx --with tinker python3 real_bpb/tinker_bpb.py --preset kimi-k3.1-pro
    TINKER_API_KEY=tml-... uvx --with tinker python3 real_bpb/tinker_bpb.py --preset glm-5.3 --preset deepseek-v4.1-flash --preset inkling-2-small
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from real_bpb.scoring import (
    LN2,
    aggregate_anpi_eval_records,
    aggregate_bpb_records,
    common_prefix_len_all,
    compute_scored_utf8_bytes,
    load_anpi_bench_doc_ids,
    load_anpi_bench_items,
    load_corpus_docs,
    longest_common_prefix_len,
    plan_strided_windows,
)

CORPUS_LOCAL = _REPO_ROOT / "data" / "anpaure_v1_corpus_1000.jsonl"
BENCH_LOCAL = _REPO_ROOT / "runs" / "anpaure_v1_n1000.json"
OUT_LOCAL_DIR = _REPO_ROOT / "runs" / "bpb_tinker"

# Verified Tinker model IDs from https://tinker-docs.thinkingmachines.ai/tinker/models.json
TINKER_PRESETS: dict[str, str] = {
    "kimi-k3.1-pro": "moonshotai/Kimi-K3.1-Pro:peft:262144",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
    "glm-5.3": "zai-org/GLM-5.3:peft:262144",
    "glm-5.3-flash": "zai-org/GLM-5.3-Flash:peft:262144",
    "deepseek-v4.1-flash": "deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144",
    "inkling-2-small": "thinkingmachines/Inkling-2-Small",
    "inkling": "thinkingmachines/Inkling",
    "inkling-small": "thinkingmachines/Inkling-Small",
    "inkling-2-mini-free": "thinkingmachines/Inkling-2-Mini:peft:65536:free",
    "nemotron-3-ultra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nemotron-3-super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nemotron-3.5-lightning": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen3.5-397b": "Qwen/Qwen3.5-397B-A17B",
    "qwen3.6-35b": "Qwen/Qwen3.6-35B-A3B",
    "qwen3.5-35b-base": "Qwen/Qwen3.5-35B-A3B-Base",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}


def sum_tinker_logprobs(logprobs: Sequence[float | None], score_from: int) -> float:
    """Return negative log-likelihood (in nats) summing `-lp` over `logprobs[score_from:]`."""
    if score_from < 1 or score_from >= len(logprobs):
        raise ValueError(f"Invalid score_from={score_from} for len={len(logprobs)}")
    nll = 0.0
    for pos in range(score_from, len(logprobs)):
        val = logprobs[pos]
        if val is None:
            raise ValueError(f"Missing Tinker logprob (None) at position {pos} (len={len(logprobs)})")
        nll -= float(val)
    return nll


def exact_conditional_logprob_tinker(
    prefix_ids: Sequence[int],
    prefix_lps: Sequence[float | None] | None,
    full_ids: Sequence[int],
    full_lps: Sequence[float | None],
) -> float:
    """Exact ln p(cand | prefix) = ln p(prefix + cand) - ln p(prefix) from Tinker logprob lists."""
    lcp = longest_common_prefix_len(prefix_ids, full_ids)
    start = max(1, lcp)
    full_tail_nll = sum_tinker_logprobs(full_lps, start) if start < len(full_ids) else 0.0
    if start < len(prefix_ids):
        if prefix_lps is None:
            raise ValueError("prefix_lps required when BPE boundary shifts prefix tokens")
        pref_tail_nll = sum_tinker_logprobs(prefix_lps, start)
    else:
        pref_tail_nll = 0.0
    return -(full_tail_nll - pref_tail_nll)


def _slug(model_id: str) -> str:
    return model_id.strip().replace("/", "__").replace(":", "_")


async def _compute_lp_with_retry(
    client: Any,
    types_mod: Any,
    token_ids: Sequence[int],
    last_n: int,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> list[float | None]:
    prompt = types_mod.ModelInput.from_ints(list(token_ids))
    # Tinker requires 1 <= prompt_logprobs_last_n <= len(prompt) - 1
    clamped = max(1, min(int(last_n), len(token_ids) - 1))
    for attempt in range(max_retries):
        async with sem:
            try:
                fut = client.compute_logprobs(prompt, prompt_logprobs_last_n=clamped)
                res = await fut
                return list(res)
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait = min(30.0, 1.5 * (2**attempt))
                print(f"  [retry {attempt + 1}/{max_retries}] {type(exc).__name__}: {exc} (sleep {wait:.1f}s)", flush=True)
                await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def evaluate_model_on_tinker(
    tinker_model_id: str,
    *,
    max_len: int = 4096,
    stride: int = 2048,
    limit: int | None = None,
    concurrency: int = 32,
    run_corpus: bool = True,
    run_anpi: bool = True,
    prepend_bos: bool = True,
) -> dict[str, Any]:
    import tinker
    from tinker import types

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY environment variable is not set.")

    t0 = time.time()
    service = tinker.ServiceClient()
    client = await service.create_sampling_client(base_model=tinker_model_id)
    try:
        tok = client.get_tokenizer()
    except Exception:
        from transformers import AutoTokenizer
        hf_repo = tinker_model_id.split(":")[0]
        tok = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
    init_sec = time.time() - t0

    bos_id = getattr(tok, "bos_token_id", None) if prepend_bos else None
    has_bos = bos_id is not None
    sem = asyncio.Semaphore(concurrency)

    print(
        f"[tinker] Connected {tinker_model_id} in {init_sec:.1f}s "
        f"(has_bos={has_bos}, bos_id={bos_id}, concurrency={concurrency})",
        flush=True,
    )

    result: dict[str, Any] = {
        "model_id": tinker_model_id,
        "backend": "tinker",
        "max_len": max_len,
        "stride": stride,
        "has_bos": has_bos,
        "bos_token_id": bos_id,
        "init_sec": round(init_sec, 2),
    }

    # ---- 1. Exact ANPI 954-item x 8-candidate white-box testset ----
    if run_anpi:
        anpi_t0 = time.time()
        items = load_anpi_bench_items(BENCH_LOCAL, limit=limit)
        per_item: list[dict[str, Any]] = []

        async def _eval_one_item(it: dict[str, Any]) -> dict[str, Any]:
            pref_text = it["prefix_text"]
            p_raw = [int(x) for x in tok.encode(pref_text, add_special_tokens=False)]
            p_ids = ([int(bos_id)] + p_raw) if has_bos else p_raw
            if len(p_ids) > max_len - 256:
                p_ids = p_ids[-(max_len - 256) :]

            cand_id_lists: list[list[int]] = []
            for cand in it["candidates"]:
                f_raw = [int(x) for x in tok.encode(pref_text + cand, add_special_tokens=False)]
                f_ids = ([int(bos_id)] + f_raw) if has_bos else f_raw
                if len(f_ids) > max_len:
                    f_ids = f_ids[-max_len:]
                cand_id_lists.append(f_ids)

            # Find the longest prefix shared identically by prefix_ids AND all 8 candidates.
            # This eliminates BPE dangling-space artifacts while conditioning all 8 candidates
            # on the exact same token history H = f_k[:start] (strictly negative logprobs!).
            start = max(1, common_prefix_len_all([p_ids, *cand_id_lists]))
            scored_cand_bytes = [
                max(1, len(tok.decode(f_ids[start:]).encode("utf-8")))
                for f_ids in cand_id_lists
            ]

            # Warm prefix cache with candidate 0, then fire candidates 1..7 concurrently!
            async def _score_cand(f_ids: list[int]) -> float:
                last_n = len(f_ids) - start
                if last_n <= 0:
                    return 0.0
                f_lps = await _compute_lp_with_retry(client, types, f_ids, last_n, sem)
                return round(-sum_tinker_logprobs(f_lps, start), 6)

            lp0 = await _score_cand(cand_id_lists[0])
            rest = await asyncio.gather(*(_score_cand(f) for f in cand_id_lists[1:]))
            g_idx = int(it["gold_idx"])
            return {
                "item_id": it["item_id"],
                "doc_id": it["doc_id"],
                "domain": it["domain"],
                "url": it.get("url"),
                "gold_idx": g_idx,
                "gold_bytes": scored_cand_bytes[g_idx],
                "cand_bytes": scored_cand_bytes,
                "cand_logprobs": [lp0, *rest],
            }

        # Process items in waves so progress prints cleanly and prefix cache stays hot
        wave_size = max(8, concurrency // 2)
        for off in range(0, len(items), wave_size):
            batch = items[off : off + wave_size]
            done = await asyncio.gather(*(_eval_one_item(it) for it in batch))
            per_item.extend(done)
            if len(per_item) % 100 < wave_size or len(per_item) == len(items):
                print(
                    f"  [anpi] {len(per_item)}/{len(items)} items ({time.time() - anpi_t0:.1f}s)",
                    flush=True,
                )

        anpi_sec = time.time() - anpi_t0
        result["anpi_eval_sec"] = round(anpi_sec, 2)
        result["anpi_eval_summary"] = aggregate_anpi_eval_records(per_item)
        result["per_item"] = per_item

    # ---- 2. Full 1,000-doc archived corpus BPB ----
    if run_corpus:
        corp_t0 = time.time()
        docs = load_corpus_docs(CORPUS_LOCAL, limit=limit)
        anpi_ids = load_anpi_bench_doc_ids(BENCH_LOCAL)

        async def _eval_one_doc(doc: dict[str, Any]) -> dict[str, Any] | None:
            text = doc["text"]
            raw_ids = [int(x) for x in tok.encode(text, add_special_tokens=False)]
            if not raw_ids:
                return None
            ids = ([int(bos_id)] + raw_ids) if has_bos else raw_ids
            first_tok_text = "" if has_bos else tok.decode([raw_ids[0]])
            n_bytes = compute_scored_utf8_bytes(text, has_bos=has_bos, first_token_text=first_tok_text)
            if n_bytes <= 0:
                return None
            windows = plan_strided_windows(ids, max_len=max_len, stride=stride, has_bos=has_bos)
            if not windows:
                return None

            win_lps = await asyncio.gather(
                *(
                    _compute_lp_with_retry(client, types, w.token_ids, w.n_scored_tokens, sem)
                    for w in windows
                )
            )
            doc_nll = sum(
                sum_tinker_logprobs(lps, w.score_from) for lps, w in zip(win_lps, windows)
            )
            scored_tokens = sum(w.n_scored_tokens for w in windows)
            return {
                "doc_id": doc["doc_id"],
                "domain": doc["domain"],
                "source": doc.get("source"),
                "url": doc.get("url"),
                "bytes": n_bytes,
                "scored_tokens": scored_tokens,
                "nll": round(doc_nll, 6),
                "bpb": round(doc_nll / (LN2 * n_bytes), 6),
            }

        per_doc: list[dict[str, Any]] = []
        wave_size = max(16, concurrency)
        for off in range(0, len(docs), wave_size):
            batch = docs[off : off + wave_size]
            done = await asyncio.gather(*(_eval_one_doc(d) for d in batch))
            per_doc.extend(x for x in done if x is not None)
            if len(per_doc) % 100 < wave_size or off + wave_size >= len(docs):
                print(
                    f"  [corpus] {len(per_doc)}/{len(docs)} docs ({time.time() - corp_t0:.1f}s)",
                    flush=True,
                )

        corp_sec = time.time() - corp_t0
        result["eval_sec"] = round(corp_sec, 2)
        result["summary"] = aggregate_bpb_records(per_doc, anpi_doc_ids=anpi_ids)
        result["per_doc"] = per_doc

    return result


def _print_summary(res: dict[str, Any], saved: Path) -> None:
    print(f"\n=== {res['model_id']} (tinker) ===")
    if "anpi_eval_summary" in res:
        ae = res["anpi_eval_summary"]
        print(
            f"  ANPI Eval Testset (N={ae['n_items']} x 8): "
            f"Gold-Span BPB={ae['gold_span_bpb']:.5f} (CI95={ae['gold_span_bpb_ci95']})  "
            f"WhiteBox AUC={ae['whitebox_auc']:.4f} (Norm={ae['whitebox_norm_auc']:.4f})  "
            f"Top-1={ae['whitebox_top1_pct']:.1f}%  MRR={ae['whitebox_mrr']:.4f}  "
            f"Margin={ae['mean_margin_nats']:+.2f} nats"
        )
    if "summary" in res:
        s = res["summary"]
        print(
            f"  Full Corpus BPB (N={s['n_docs']}): {s['bpb']:.5f}  "
            f"CI95={s['bpb_ci95']}  Macro-5={s['macro_domain_bpb']:.5f}"
        )
        if "anpi_954_subset" in s:
            s9 = s["anpi_954_subset"]
            print(f"  ANPI-954 Full-Doc BPB (N={s9['n_docs']}): {s9['bpb']:.5f}  Macro-5={s9['macro_domain_bpb']:.5f}")
        print("  Domains: " + "  ".join(f"{k}={v['bpb']:.4f}" for k, v in s["domains"].items()))
    print(
        f"  Timing: init={res['init_sec']}s, "
        f"anpi={res.get('anpi_eval_sec', 0)}s, corpus={res.get('eval_sec', 0)}s"
    )
    print(f"  Saved: {saved}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ANPI 954x8 + 1000-doc BPB evals via Tinker API.")
    ap.add_argument("--preset", action="append", default=[], help=f"Preset name(s): {sorted(TINKER_PRESETS)}")
    ap.add_argument("--model-id", action="append", default=[], help="Raw Tinker base_model ID(s)")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on docs/items for smoke tests")
    ap.add_argument("--concurrency", type=int, default=32, help="Max concurrent Tinker logprob calls")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--anpi-only", action="store_true", help="Skip 1,000-doc corpus BPB; run only 954x8 ANPI testset")
    ap.add_argument("--corpus-only", action="store_true", help="Skip 954x8 ANPI testset; run only 1,000-doc corpus BPB")
    ap.add_argument("--skip-existing", action="store_true", help="Skip models whose output JSON already exists")
    args = ap.parse_args()

    targets: list[str] = []
    for p in args.preset:
        k = p.strip().lower()
        if k not in TINKER_PRESETS:
            raise SystemExit(f"Unknown preset '{p}'. Available: {sorted(TINKER_PRESETS)}")
        targets.append(TINKER_PRESETS[k])
    targets.extend(m.strip() for m in args.model_id if m.strip())
    if not targets:
        raise SystemExit(f"Specify --preset or --model-id. Available presets: {sorted(TINKER_PRESETS)}")

    lim = args.limit if args.limit > 0 else None
    OUT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    for mid in targets:
        suffix = f"_n{lim}" if lim else ""
        if args.anpi_only:
            suffix += "_anpi"
        elif args.corpus_only:
            suffix += "_corpus"
        out_path = OUT_LOCAL_DIR / f"{_slug(mid)}__tinker{suffix}.json"
        if args.skip_existing and out_path.exists():
            print(f"[skip] Already exists: {out_path}", flush=True)
            continue
        res = asyncio.run(
            evaluate_model_on_tinker(
                mid,
                max_len=args.max_len,
                stride=args.stride,
                limit=lim,
                concurrency=args.concurrency,
                run_corpus=not args.anpi_only,
                run_anpi=not args.corpus_only,
            )
        )
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        _print_summary(res, out_path)


if __name__ == "__main__":
    main()
