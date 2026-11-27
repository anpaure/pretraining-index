"""Chauvin-style Logprob Tracking (LT) API stability canary for acquisition batches.

Based on Timothée Chauvin et al., "Log Probability Tracking of LLM APIs" (ICLR 2026)
and "Token-Efficient Change Detection in LLM APIs" (ICML 2026):
  - Repeats a tiny frozen set of short prompts before, during, and after a
    perplexity acquisition batch.
  - Records request SHA-256, UTC timestamp, returned model snapshot, service
    tier, and top-20 logprobs for the first generated token.
  - Flags snapshot changes or statistically large top-token logprob drift
    between pre-batch and post-batch probes so silently altered batches are
    analyzed separately rather than pooled.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .corpus import canonical_json_bytes, sha256_hex
from .openai_responses import DEFAULT_URL, post_responses_once

# Frozen short deterministic canary prompts (never edit mid-campaign).
DEFAULT_CANARY_PROMPTS: tuple[tuple[str, str], ...] = (
    ("canary_capitals", "The capital of France is"),
    ("canary_math", "The square root of 144 is"),
    ("canary_prose", "On Tuesday morning, the committee voted to"),
    ("canary_code", "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return"),
)


def run_lt_canary_probe(
    api_key: str,
    *,
    model: str,
    phase: str,  # "pre" | "mid" | "post"
    service_tier: str | None = "flex",
    reasoning_effort: str | None = "none",
    replicates: int = 2,
    url: str = DEFAULT_URL,
    prompts: Sequence[tuple[str, str]] = DEFAULT_CANARY_PROMPTS,
    transport_fn: Callable[..., tuple[int, dict[str, Any] | None, str | None, float]] = post_responses_once,
) -> list[dict[str, Any]]:
    """Run one lightweight LT stability canary sweep (8 calls default, <$0.002)."""
    records: list[dict[str, Any]] = []
    for cid, prompt in prompts:
        body: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_output_tokens": 16,
            "top_logprobs": 20,
            "include": ["message.output_text.logprobs"],
            "store": False,
            "text": {"format": {"type": "text"}},
        }
        if reasoning_effort is not None and reasoning_effort != "omit":
            body["reasoning"] = {"effort": reasoning_effort}
        if service_tier is not None and service_tier != "omit":
            body["service_tier"] = service_tier
        req_hash = sha256_hex(canonical_json_bytes(body))

        for rep in range(replicates):
            ts = datetime.now(timezone.utc).isoformat()
            code, raw, err, dt = transport_fn(api_key, body, url=url, timeout=45.0)
            entry: dict[str, Any] = {
                "canary_id": cid,
                "phase": phase,
                "replicate": rep,
                "timestamp_utc": ts,
                "request_sha256": req_hash,
                "requested_model": model,
                "http_status": code,
                "latency_seconds": round(dt, 4),
            }
            if code == 200 and isinstance(raw, dict):
                entry["returned_model"] = raw.get("model")
                entry["returned_service_tier"] = raw.get("service_tier")
                entry["system_fingerprint"] = raw.get("system_fingerprint")
                parts = [
                    pt
                    for it in raw.get("output", [])
                    if isinstance(it, dict) and it.get("type") == "message"
                    for pt in (it.get("content") or [])
                    if isinstance(pt, dict) and pt.get("type") == "output_text"
                ]
                lps = parts[0].get("logprobs") if parts else None
                if lps:
                    pos0 = lps[0]
                    entry["pos0_top_logprobs"] = {
                        bytes(t["bytes"]).hex(): float(t["logprob"])
                        for t in pos0.get("top_logprobs", [])
                        if isinstance(t.get("bytes"), list) and t.get("logprob") is not None
                    }
            else:
                entry["error"] = err
            records.append(entry)
    return records


def evaluate_canary_drift(
    records: Sequence[dict[str, Any]],
    *,
    max_mean_abs_shift_nats: float = 0.50,
) -> dict[str, Any]:
    """Compare `pre` vs `post` canary probes for snapshot swaps or mean logprob drift."""
    ok = [r for r in records if r.get("http_status") == 200 and r.get("pos0_top_logprobs")]
    returned_models = sorted({str(r.get("returned_model")) for r in ok})
    fingerprints = sorted({str(r.get("system_fingerprint")) for r in ok if r.get("system_fingerprint")})

    pre_by_cid: dict[str, list[dict[str, float]]] = {}
    post_by_cid: dict[str, list[dict[str, float]]] = {}
    for r in ok:
        cid = str(r["canary_id"])
        if r.get("phase") == "pre":
            pre_by_cid.setdefault(cid, []).append(r["pos0_top_logprobs"])
        elif r.get("phase") == "post":
            post_by_cid.setdefault(cid, []).append(r["pos0_top_logprobs"])

    shifts: list[float] = []
    for cid in sorted(set(pre_by_cid) & set(post_by_cid)):
        # Track top-3 tokens present in pre replicates
        common_toks = set.intersection(*(set(d.keys()) for d in pre_by_cid[cid] + post_by_cid[cid]))
        if not common_toks:
            continue
        ranked = sorted(common_toks, key=lambda h: np.mean([d[h] for d in pre_by_cid[cid]]), reverse=True)[:3]
        for tok in ranked:
            m_pre = float(np.mean([d[tok] for d in pre_by_cid[cid]]))
            m_post = float(np.mean([d[tok] for d in post_by_cid[cid]]))
            shifts.append(abs(m_post - m_pre))

    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    max_shift = float(np.max(shifts)) if shifts else 0.0
    model_swapped = len(returned_models) > 1
    drift_flagged = model_swapped or (mean_shift > max_mean_abs_shift_nats)

    return {
        "ok_probes": len(ok),
        "returned_models": returned_models,
        "system_fingerprints": fingerprints,
        "model_snapshot_swapped": model_swapped,
        "tracked_token_comparisons": len(shifts),
        "mean_abs_logprob_shift_nats": round(mean_shift, 4),
        "max_abs_logprob_shift_nats": round(max_shift, 4),
        "drift_flagged": drift_flagged,
        "verdict": "DRIFT_WARNING" if drift_flagged else "STABLE",
    }
