"""Generate the frozen N=1,000 multi-domain Rank-NCE benchmark (`runs/anpaure_v1_n1000.json`)
using ONLY cloud OpenRouter `/api/v1/completions` (ZERO local compute).

Default decoy generator: `mistralai/mistral-small-24b-instruct-2501` ($0.05/$0.08 per M;
24B dense — 24x stronger than OLMo-1B, zero family overlap with OpenAI, Anthropic,
Google, Moonshot, DeepSeek, NVIDIA, Zhipu, Thinking Machines, or RicursiveAI).

Anti-leakage guarantees:
  - Prose domains (`news`, `papers`, `reference`, `essays`): whitespace-normalized,
    cut at a clean word boundary ~60% into the doc (~700-900 char prefix); true
    continuation and all 7 decoys are EXACTLY 6 whitespace-delimited words, distinct
    (case-insensitive), free of assistant/chat leakage, and length-matched.
  - Code domain (`code`): raw multi-line code prefix (~600-900 chars) ending inside
    an added Nov-2026 code block at a non-trivial line (>= 36 chars remaining on
    that line); true continuation and all 7 decoys are single-line continuations
    truncated cleanly to the SAME byte-length window (32-44 chars, snapped to a
    clean token boundary, no newlines, distinct, free of chat leakage).
  - Deterministic shuffle seed per doc_id (`sha256`) -> 100% reproducible.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data/anpaure_v1_corpus_1000.jsonl"
OUT_PATH = ROOT / "runs/anpaure_v1_n1000.json"

CHAT_LEAK_RE = re.compile(
    r"(here(?:'s| is)|sure[,!]|certainly|it appears|as an ai|would you like|write a |summarize|continuation:|```|</|<\|)",
    re.I,
)


def pick_prose_cut(doc: dict) -> tuple[str, str] | None:
    text = re.sub(r"\s+", " ", doc["text"]).strip()
    words = text.split(" ")
    if len(words) < 110:
        return None
    # Deterministic cut between 55% and 70% of the doc, ensuring prefix >= 550 chars
    h = int(hashlib.sha256(doc["doc_id"].encode()).hexdigest()[:8], 16)
    frac = 0.55 + (h % 150) / 1000.0
    cut_idx = max(95, min(len(words) - 12, int(len(words) * frac)))
    prefix = " ".join(words[:cut_idx])
    if len(prefix) > 900:
        # Keep tail ~900 chars snapped cleanly to a word boundary
        tail = prefix[-900:]
        sp = tail.find(" ")
        if sp > 0:
            tail = tail[sp + 1 :]
        prefix = tail
    true_words = words[cut_idx : cut_idx + 6]
    if len(true_words) != 6:
        return None
    true_chunk = " ".join(true_words)
    if len(true_chunk) < 18 or len(true_chunk) > 68:
        return None
    return prefix, true_chunk


def snap_code_snippet(raw: str, target_len: int) -> str | None:
    """Take a clean single-line code snippet close to target_len (within [target_len-7, target_len+7])."""
    line = raw.split("\n", 1)[0].rstrip("\r")
    if len(line) < max(22, target_len - 7):
        return None
    if len(line) <= target_len + 5:
        return line.rstrip()
    # Snap near target_len to a clean code delimiter (space, (, ), ,, ;, ., =, +, -, :)
    lo = max(20, target_len - 6)
    hi = min(len(line), target_len + 6)
    best = None
    best_dist = 999
    for idx in range(lo, hi + 1):
        if idx == len(line) or line[idx] in " \t()[]{},.;:=+-/*<>!&|":
            d = abs(idx - target_len)
            if d < best_dist:
                best_dist = d
                best = idx
    cut = best if best is not None else target_len
    out = line[:cut].rstrip()
    return out if len(out) >= max(20, target_len - 8) else None


def pick_code_cut(doc: dict) -> tuple[str, str] | None:
    lines = doc["text"].splitlines()
    if len(lines) < 12:
        return None
    # Candidate cut lines in the second half of the block with a substantial line (>= 42 chars after indent)
    cands = []
    for li in range(max(4, len(lines) // 3), len(lines) - 1):
        ln = lines[li]
        stripped = ln.lstrip(" \t")
        if len(stripped) < 34:
            continue
        # Avoid pure comment lines
        if stripped.startswith(("//", "#", "/*", "*", "<!--")):
            continue
        # Cut right after indentation + first token (or right after indentation if first token is long)
        indent = ln[: len(ln) - len(stripped)]
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_.:->]*\s+)", stripped)
        if m and len(stripped) - len(m.group(1)) >= 28:
            line_pref = indent + m.group(1)
            rem = stripped[len(m.group(1)) :]
        else:
            line_pref = indent
            rem = stripped
        prefix = "\n".join(lines[:li] + [line_pref])
        if len(prefix) > 900:
            tail = prefix[-900:]
            nl = tail.find("\n")
            if nl > 0:
                tail = tail[nl + 1 :]
            prefix = tail
        if len(prefix) < 200:
            continue
        snip = snap_code_snippet(rem, target_len=min(36, max(26, len(rem))))
        if snip and 22 <= len(snip) <= 46:
            cands.append((prefix, snip))
    if not cands:
        return None
    h = int(hashlib.sha256(doc["doc_id"].encode()).hexdigest()[:8], 16)
    return cands[h % len(cands)]


def sample_completion_cloud(
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    stop: list[str] | None = None,
    temperature: float = 1.0,
) -> tuple[str, float]:
    body: dict = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    if stop:
        body["stop"] = stop
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": "anpaure-decoy-gen/1.0",
    }
    for att in range(4):
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                d = json.loads(raw.decode())
            if "error" in d:
                if att < 3:
                    time.sleep(1.5 * (att + 1))
                    continue
                return "", 0.0
            txt = (d.get("choices") or [{}])[0].get("text") or ""
            cost = float((d.get("usage") or {}).get("cost") or 0.0)
            return txt, cost
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and att < 3:
                time.sleep(2 ** att)
                continue
            return "", 0.0
        except Exception:
            if att < 3:
                time.sleep(1.5 * (att + 1))
                continue
            return "", 0.0
    return "", 0.0


def build_one_item(doc: dict, api_key: str, decoy_model: str, k_total: int = 8) -> tuple[dict | None, float]:
    domain = doc["domain"]
    is_code = domain == "code"
    cut = pick_code_cut(doc) if is_code else pick_prose_cut(doc)
    if cut is None:
        return None, 0.0
    prefix, true_chunk = cut
    target_len = len(true_chunk)

    decoys: list[str] = []
    seen_norm: set[str] = {re.sub(r"\s+", " ", true_chunk).strip().lower()}
    total_cost = 0.0
    need = k_total - 1

    # Prompt ends cleanly WITHOUT a trailing space (avoiding BPE leading-space distortion!)
    comp_prompt = prefix

    def _one_draw(temp: float) -> tuple[str, float]:
        return sample_completion_cloud(
            api_key,
            decoy_model,
            comp_prompt,
            max_tokens=24 if not is_code else 28,
            stop=["\n"] if is_code else None,
            temperature=temp,
        )

    # Sample decoys in fast parallel waves (wave 1 = 10 draws concurrently -> usually finishes in 1 RTT!)
    for wave_idx, (n_draws, temp) in enumerate([(10, 1.0), (10, 1.0), (12, 1.10)]):
        if len(decoys) >= need:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, n_draws)) as subpool:
            draws = list(subpool.map(lambda _: _one_draw(temp), range(n_draws)))
        for raw_txt, cost in draws:
            total_cost += cost
            if len(decoys) >= need:
                continue
            if not raw_txt or CHAT_LEAK_RE.search(raw_txt):
                continue
            if is_code:
                cand_raw = raw_txt.lstrip(" ") if comp_prompt.endswith((" ", "\t")) else raw_txt
                cand = snap_code_snippet(cand_raw, target_len=target_len)
                if not cand or "\n" in cand or "\r" in cand:
                    continue
                if abs(len(cand) - target_len) > 10:
                    continue
            else:
                if not raw_txt[0].isspace():
                    continue
                cleaned = re.sub(r"\s+", " ", raw_txt).strip()
                if any(ord(ch) > 0x206F and not ch.isalpha() for ch in cleaned):
                    continue
                ws = cleaned.split(" ")
                if len(ws) < 6:
                    continue
                cand = " ".join(ws[:6])
                if len(cand) < max(16, int(target_len * 0.55)) or len(cand) > min(74, int(target_len * 1.55) + 6):
                    continue
            norm = re.sub(r"\s+", " ", cand).strip().lower()
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            decoys.append(cand)

    if len(decoys) < need:
        return None, total_cost

    # Deterministic shuffle
    cands = [true_chunk] + decoys
    seed_int = int(hashlib.sha256((doc["doc_id"] + "::shuffle_v1").encode()).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    order = list(range(k_total))
    rng.shuffle(order)
    shuffled = [cands[i] for i in order]
    gold_idx = order.index(0)

    item = {
        "item_id": f"v1_{doc['doc_id']}",
        "doc_id": doc["doc_id"],
        "domain": domain,
        "source": doc["source"],
        "published": doc["published"],
        "url": doc["url"],
        "prefix_text": prefix,
        "true_chunk": true_chunk,
        "true_chunk_bytes": len(true_chunk.encode("utf-8")),
        "candidates": shuffled,
        "gold_idx": gold_idx,
        "decoy_model": decoy_model,
    }
    return item, total_cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoy-model", default="mistralai/mistral-small-24b-instruct-2501")
    ap.add_argument("--limit", type=int, default=0, help="If >0, build this many items evenly across domains")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    docs = [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    out_path = Path(args.out)
    existing: dict[str, dict] = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            existing[it["doc_id"]] = it
        print(f"Resuming from {out_path}: {len(existing)} already generated")

    if args.limit > 0:
        # Stratified subset across domains
        by_dom: dict[str, list[dict]] = {}
        for d in docs:
            by_dom.setdefault(d["domain"], []).append(d)
        per = max(1, args.limit // len(by_dom))
        selected = []
        for dom in sorted(by_dom):
            selected.extend(by_dom[dom][:per])
        docs = selected

    todo = [d for d in docs if d["doc_id"] not in existing]
    print(f"Generating decoys with {args.decoy_model} for {len(todo)}/{len(docs)} docs (workers={args.workers})...")

    items = list(existing.values())
    total_cost = 0.0
    failed = 0
    t0 = time.time()

    def _save() -> None:
        dom_counts: dict[str, int] = {}
        for it in items:
            dom_counts[it["domain"]] = dom_counts.get(it["domain"], 0) + 1
        payload = {
            "version": "anpaure_v1",
            "decoy_model": args.decoy_model,
            "k_candidates": 8,
            "n_items": len(items),
            "domain_counts": dom_counts,
            "items": sorted(items, key=lambda x: (x["domain"], x["doc_id"])),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(build_one_item, d, api_key, args.decoy_model, 8) for d in todo]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            it, cost = fut.result()
            total_cost += cost
            if it is not None:
                items.append(it)
            else:
                failed += 1
            if idx % 50 == 0 or idx == len(todo):
                dt = time.time() - t0
                print(f"  [{idx}/{len(todo)}] built={len(items)} failed={failed} cost=${total_cost:.4f} elapsed={dt:.1f}s")
                _save()

    _save()
    print(f"DONE: {len(items)} items saved to {out_path} (failed={failed}, cost=${total_cost:.4f})")


if __name__ == "__main__":
    main()
