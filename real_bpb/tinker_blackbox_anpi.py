"""Strict zero-CoT black-box ANPI rerun directly on Tinker's hosted weights.

Runs the exact frozen 8-choice continuation-ranking prompt from `run_anpaure_v1.py`
(`SYS_PROMPT` + `make_user_prompt`, `temperature=0.0`, `max_tokens=32`) via
`tinker.SamplingClient.sample()` with verified zero-thinking chat templates for
each of the 14 overlapping posttrained models.

Saves per-item JSONL traces to `runs/bpb_tinker_blackbox/<slug>.jsonl` and
reports side-by-side comparison against our existing OpenRouter/Together black-box
ANPI AUC and Tinker white-box oracle AUC.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from real_bpb.scoring import load_anpi_bench_items
from real_bpb.tinker_bpb import TINKER_PRESETS, _slug
from run_anpaure_v1 import CLEAN_RE, LETTERS, SYS_PROMPT, make_user_prompt

BENCH_LOCAL = _REPO_ROOT / "runs" / "anpaure_v1_n1000.json"
OUT_BB_DIR = _REPO_ROOT / "runs" / "bpb_tinker_blackbox"


def render_zero_cot_prompt(tok: Any, tinker_model_id: str, user_prompt: str) -> str:
    """Render each model's exact official zero-thinking chat prompt."""
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    mid_l = tinker_model_id.lower()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if "kimi" in mid_l:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, thinking=False)
        elif "glm-5.3" in mid_l:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="low")
            if s.endswith("<think>"):
                s = s + "</think>\n"
        elif "inkling" in mid_l:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="none")
            if not s.endswith("<|message_model|>"):
                s = s + "<|message_model|>"
            s = s + "<|content_text|>"
        elif "gpt-oss" in mid_l:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            s = s + "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
        else:
            # DeepSeek V4.1 Flash, Nemotron 3.*, Qwen3.8-27B all natively honor enable_thinking=False
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return s


def _stop_tokens_for(tok: Any, tinker_model_id: str) -> list[str]:
    mid_l = tinker_model_id.lower()
    stops = ["\n"]
    if "inkling" in mid_l:
        stops.extend(["<|end_message|>", "<|content_model_end_sampling|>"])
    elif "kimi-k3" in mid_l:
        stops.extend(["<|close|>", "<|end_of_msg|>"])
    elif "kimi-k2" in mid_l:
        stops.append("<|im_end|>")
    elif "gpt-oss" in mid_l:
        stops.extend(["<|return|>", "<|end|>"])
    elif "glm" in mid_l:
        stops.extend(["<|user|>", "<|endoftext|>"])
    elif "deepseek" in mid_l:
        stops.append("<｜end▁of▁sentence｜>")
    else:
        stops.append("<|im_end|>")
    return stops


def parse_strict_order(raw_text: str) -> tuple[list[int] | None, str]:
    s = raw_text.strip()
    # Strip harmless empty close-tag or code-fence wrapper (never hides reasoning prose)
    if s.startswith("</think>"):
        s = s[len("</think>") :].strip()
    if s.startswith("```"):
        lines = [ln.strip() for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("```")]
        if len(lines) == 1:
            s = lines[0]
    # Strip a single trailing punctuation mark (e.g. "H,F,E,B,C,D,A,G:")
    if len(s) >= 15 and s[-1] in ":.,;":
        s = s[:-1].strip()
    if not CLEAN_RE.match(s):
        return (None, raw_text.strip())
    letters = re.findall(r"[A-H]", s)
    if len(letters) != 8 or len(set(letters)) != 8:
        return (None, raw_text.strip())
    return ([LETTERS.index(ch) for ch in letters], s)


async def run_blackbox_model(
    tinker_model_id: str,
    *,
    limit: int | None = None,
    concurrency: int = 96,
    max_tokens: int = 48,
    topup: bool = False,
) -> dict[str, Any]:
    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY environment variable is not set.")

    t0 = time.time()
    service = tinker.ServiceClient()
    client = await service.create_sampling_client(base_model=tinker_model_id)
    hf_repo = tinker_model_id.split(":")[0]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tok = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
    init_sec = time.time() - t0

    items = load_anpi_bench_items(BENCH_LOCAL, limit=limit)
    stops = [s for s in _stop_tokens_for(tok, tinker_model_id) if (s != "\n" if topup else True)]
    sem = asyncio.Semaphore(concurrency)

    print(f"[tinker-bb] Connected {tinker_model_id} in {init_sec:.1f}s (N={len(items)}, conc={concurrency}, topup={topup})", flush=True)

    OUT_BB_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_n{limit}" if limit else ""
    out_jsonl = OUT_BB_DIR / f"{_slug(tinker_model_id)}__bb{suffix}.jsonl"

    # Resume support (also re-checks strict parse on any harmless trailing-colon rows)
    existing: dict[str, dict[str, Any]] = {}
    if out_jsonl.exists():
        for ln in out_jsonl.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if rec.get("ok"):
                existing[rec["item_id"]] = rec
            elif topup and not rec.get("cot_leak"):
                order, norm = parse_strict_order(rec.get("raw", ""))
                if order is not None:
                    rec["ok"] = True
                    rec["raw"] = norm
                    rec["gold_rank"] = order.index(int(rec["gold_idx"])) + 1
                    existing[rec["item_id"]] = rec

    todo = [it for it in items if it["item_id"] not in existing]
    if existing:
        print(f"  [resume] {len(existing)} already clean, {len(todo)} remaining", flush=True)

    eval_t0 = time.time()
    retry_temps = (0.2, 0.4, 0.6, 0.8) if topup else (0.0,)

    async def _eval_item(it: dict[str, Any]) -> dict[str, Any]:
        prompt_str = render_zero_cot_prompt(tok, tinker_model_id, make_user_prompt(it))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ids = [int(x) for x in tok.encode(prompt_str, add_special_tokens=False)]
        mi = types.ModelInput.from_ints(ids)
        last_rec: dict[str, Any] | None = None
        last_err = None
        for t_idx, temp in enumerate(retry_temps):
            sp = types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temp,
                seed=(t_idx + 17) if temp > 0 else None,
                stop=stops,
            )
            for attempt in range(3):
                async with sem:
                    try:
                        resp = await client.sample(prompt=mi, num_samples=1, sampling_params=sp)
                        seq_tokens = list(resp.sequences[0].tokens)
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            raw = tok.decode(seq_tokens, skip_special_tokens=True).strip()
                        for st in stops:
                            if st and st in raw:
                                raw = raw.split(st, 1)[0].strip()
                        cot_leak = any(tag in raw.lower() for tag in ("<think", "content_thinking", "analysis"))
                        order, norm_raw = (None, raw) if cot_leak else parse_strict_order(raw)
                        g_idx = int(it["gold_idx"])
                        gold_rank = (order.index(g_idx) + 1) if order is not None else None
                        last_rec = {
                            "item_id": it["item_id"],
                            "doc_id": it["doc_id"],
                            "domain": it["domain"],
                            "gold_idx": g_idx,
                            "ok": order is not None,
                            "cot_leak": cot_leak,
                            "raw": norm_raw,
                            "gold_rank": gold_rank,
                            "out_tokens": len(seq_tokens),
                        }
                        if order is not None:
                            return last_rec
                        break
                    except Exception as exc:
                        last_err = f"{type(exc).__name__}: {exc}"
                        await asyncio.sleep(min(20.0, 1.5 * (2**attempt)))
        if last_rec is not None:
            return last_rec
        return {
            "item_id": it["item_id"],
            "doc_id": it["doc_id"],
            "domain": it["domain"],
            "gold_idx": int(it["gold_idx"]),
            "ok": False,
            "cot_leak": False,
            "raw": "",
            "gold_rank": None,
            "err": last_err,
        }

    new_records: list[dict[str, Any]] = []
    if todo:
        wave = max(32, concurrency)
        for off in range(0, len(todo), wave):
            batch = todo[off : off + wave]
            done = await asyncio.gather(*(_eval_item(it) for it in batch))
            new_records.extend(done)
            n_ok = sum(1 for r in new_records if r["ok"])
            print(
                f"  [bb] {tinker_model_id.split('/')[-1]}: {len(new_records)}/{len(todo)} "
                f"(clean={n_ok}/{len(new_records)}, {time.time() - eval_t0:.1f}s)",
                flush=True,
            )

    all_by_id = dict(existing)
    for r in new_records:
        if r["ok"] or r["item_id"] not in all_by_id:
            all_by_id[r["item_id"]] = r
    ordered = [all_by_id[it["item_id"]] for it in items if it["item_id"] in all_by_id]

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok_recs = [r for r in ordered if r["ok"]]
    n_ok = len(ok_recs)
    auc = sum((8 - r["gold_rank"]) / 7.0 for r in ok_recs) / n_ok if n_ok else 0.0
    mrr = sum(1.0 / r["gold_rank"] for r in ok_recs) / n_ok if n_ok else 0.0
    top1 = 100.0 * sum(1 for r in ok_recs if r["gold_rank"] == 1) / n_ok if n_ok else 0.0
    leaks = sum(1 for r in ordered if r.get("cot_leak"))
    sample_bad = [r["raw"][:60] for r in ordered if not r["ok"]][:3]

    summary = {
        "model_id": tinker_model_id,
        "n_total": len(ordered),
        "n_ok": n_ok,
        "cot_leaks": leaks,
        "auc": round(auc, 4),
        "mrr": round(mrr, 4),
        "top1_pct": round(top1, 2),
        "eval_sec": round(time.time() - eval_t0, 2),
        "sample_bad": sample_bad,
        "out_jsonl": str(out_jsonl),
    }
    print(
        f"=== [DONE] {tinker_model_id}: clean={n_ok}/{len(ordered)} leaks={leaks} "
        f"AUC={auc:.4f} Top1={top1:.1f}% MRR={mrr:.4f} ({summary['eval_sec']}s) "
        f"{'bad=' + str(sample_bad) if sample_bad else ''} ===",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--topup", action="store_true")
    args = ap.parse_args()
    k = args.preset.strip().lower()
    if k not in TINKER_PRESETS:
        raise SystemExit(f"Unknown preset '{args.preset}'; choose from {sorted(TINKER_PRESETS)}")
    lim = args.limit if args.limit > 0 else None
    asyncio.run(run_blackbox_model(TINKER_PRESETS[k], limit=lim, concurrency=args.concurrency, topup=args.topup))


if __name__ == "__main__":
    main()
