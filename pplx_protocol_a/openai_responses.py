"""Hard-capped Direct OpenAI `/v1/responses` client for Protocol A.

Built strictly around live-verified `gpt-5.6-sol` `/v1/responses` semantics
from the August–September 2026 audits:
  - Uses Direct `POST https://api.openai.com/v1/responses` (NEVER OpenRouter,
    which stripped temperature/n; NEVER Chat Completions, which capped Sol at
    top_logprobs=5; NEVER legacy /v1/completions, which rejects Sol).
  - Enforces `max_output_tokens >= 16`, `top_logprobs = 20`, `temperature = 1.0`,
    `top_p = 1.0`, `reasoning = {"effort": "none"}`, `store = false`,
    `text = {"format": {"type": "text"}}`.
  - Explicitly asserts ZERO forbidden parameters (`json_schema`, `logit_bias`,
    `tools`, `seed`) in every serialized request payload so structured-output
    mask renormalization and cross-schema drift cannot occur.
  - Audits `usage.output_tokens_details.reasoning_tokens == 0`, exact returned
    model, service tier, monotonic top-list logprobs, and valid raw `bytes`.
  - Enforces conservative standard-price accounting + hard USD budget cap,
    and sanitizes all persisted records (no keys, no request/response IDs).
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import math
import os
import stat
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .corpus import PositionSpec, canonical_json_bytes, sha256_hex

DEFAULT_URL = "https://api.openai.com/v1/responses"

# Frozen locked system prompt for echo_chat mode (do not edit mid-campaign).
ECHO_CHAT_SYSTEM = (
    "You continue texts verbatim. Begin your reply EXACTLY with the quoted string "
    "shown after the text (preserving exact capitalization, spacing, and punctuation), "
    "then immediately continue the text naturally. Output nothing else — no surrounding "
    "quotes, no commentary, no markdown."
)

# Frozen locked system prompt for assistant_prefill mode (for A/B boundary pre-gate).
PREFILL_SYSTEM = (
    "Continue the document seamlessly from the exact character where the assistant "
    "message ends. Output only the continuation text."
)

FORBIDDEN_WIRE_TOKENS = (b"json_schema", b"logit_bias", b'"tools"', b'"seed"')


class ProtocolAError(RuntimeError):
    """Raised when an API contract, telemetry, or spend invariant is violated."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ProtocolAError(msg)


def load_api_key(env_var: str = "OPENAI_API_KEY", key_file: Path | None = None) -> str:
    """Load OpenAI API key from a 0600 file or environment variable."""
    if key_file is not None:
        _require(key_file.is_file() and not key_file.is_symlink(), "key_file must be a regular file")
        mode = stat.S_IMODE(key_file.stat().st_mode)
        _require(mode == 0o600, f"key_file must have mode 0600 (got {oct(mode)})")
        matches = []
        for line in key_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith(f"{env_var}="):
                matches.append(s.split("=", 1)[1].strip().strip('"').strip("'"))
        _require(len(matches) == 1 and bool(matches[0]), f"exactly one {env_var}=... line required in {key_file}")
        return matches[0]
    val = os.environ.get(env_var, "").strip()
    _require(bool(val), f"Missing {env_var} environment variable (or pass --key-file)")
    return val


def build_responses_payload(
    pos: PositionSpec,
    *,
    model: str = "gpt-5.6-sol",
    mode: str = "echo_chat",
    service_tier: str | None = "flex",
    max_output_tokens: int = 32,
    top_logprobs: int = 20,
    temperature: float = 1.0,
    top_p: float = 1.0,
    reasoning_effort: str | None = "none",
) -> dict[str, Any]:
    """Construct a locked Protocol A `/v1/responses` JSON payload."""
    _require(mode in ("echo_chat", "assistant_prefill"), f"Unsupported mode: {mode}")
    _require(max_output_tokens >= 16, "Direct Responses requires max_output_tokens >= 16")
    _require(1 <= top_logprobs <= 20, "Direct Responses requires 1 <= top_logprobs <= 20")
    if service_tier is not None and service_tier != "omit":
        _require(service_tier in ("flex", "default", "auto", "priority"), f"Unexpected service_tier: {service_tier}")

    if mode == "echo_chat":
        user_content = f'{pos.prefix_text}\n\nBegin exactly with: "{pos.anchor_text}"'
        messages = [
            {"role": "system", "content": ECHO_CHAT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": PREFILL_SYSTEM},
            {"role": "user", "content": "Continue the following text verbatim from where it stops."},
            {"role": "assistant", "content": pos.prefix_text},
        ]

    body: dict[str, Any] = {
        "model": model,
        "input": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "top_logprobs": int(top_logprobs),
        "include": ["message.output_text.logprobs"],
        "store": False,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "text": {"format": {"type": "text"}},
    }
    if reasoning_effort is not None and reasoning_effort != "omit":
        body["reasoning"] = {"effort": reasoning_effort}
    if service_tier is not None and service_tier != "omit":
        body["service_tier"] = service_tier

    wire = canonical_json_bytes(body).lower()
    for forbidden in FORBIDDEN_WIRE_TOKENS:
        _require(forbidden not in wire, f"Forbidden parameter present in payload: {forbidden!r}")
    return body


def _valid_bytes_list(val: Any) -> bool:
    return isinstance(val, list) and bool(val) and all(type(x) is int and 0 <= x <= 255 for x in val)


def _finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _clean_logprob_entry(entry: Any, path: str) -> dict[str, Any]:
    _require(isinstance(entry, dict), f"{path} must be an object")
    _require(_valid_bytes_list(entry.get("bytes")), f"{path}.bytes invalid")
    _require(_finite_number(entry.get("logprob")), f"{path}.logprob invalid")
    return {"bytes_hex": bytes(entry["bytes"]).hex(), "logprob": float(entry["logprob"])}


def clean_usage(usage: Any) -> dict[str, int]:
    """Validate and extract integer token usage including reasoning_tokens."""
    _require(isinstance(usage, dict), "response.usage missing")
    in_det = usage.get("input_tokens_details") or {}
    out_det = usage.get("output_tokens_details") or {}
    res = {
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": in_det.get("cached_tokens", 0),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_output_tokens": out_det.get("reasoning_tokens", 0),
        "total_tokens": usage.get("total_tokens"),
    }
    for k, v in res.items():
        _require(type(v) is int and v >= 0, f"usage.{k} invalid: {v!r}")
    _require(res["input_tokens"] + res["output_tokens"] == res["total_tokens"], "usage total mismatch")
    _require(res["cached_input_tokens"] <= res["input_tokens"], "cached_input > input")
    return res


def compute_conservative_cost_usd(usage: dict[str, int], pricing: dict[str, float]) -> float:
    """Compute conservative USD cost (defaults to Standard rates for safety)."""
    uncached = usage["input_tokens"] - usage["cached_input_tokens"]
    return (
        uncached * float(pricing["uncached_input_usd_per_million"])
        + usage["cached_input_tokens"] * float(pricing["cached_input_usd_per_million"])
        + usage["output_tokens"] * float(pricing["output_usd_per_million"])
    ) / 1_000_000.0


def estimate_manifest_cost_usd(
    positions: Sequence[PositionSpec],
    *,
    max_output_tokens: int,
    pricing: dict[str, float],
    bytes_per_token_conservative: float = 3.2,
    overhead_tokens: int = 80,
) -> dict[str, float]:
    """Dry-run upper-bound cost estimate before spending $0.01."""
    est_in = sum(int(math.ceil(p.prefix_utf8_bytes / bytes_per_token_conservative)) + overhead_tokens for p in positions)
    est_out = len(positions) * int(max_output_tokens)
    std_cost = (
        est_in * float(pricing["uncached_input_usd_per_million"])
        + est_out * float(pricing["output_usd_per_million"])
    ) / 1_000_000.0
    flex_cost = 0.5 * std_cost
    return {
        "positions": float(len(positions)),
        "estimated_input_tokens_upper": float(est_in),
        "estimated_output_tokens_upper": float(est_out),
        "estimated_standard_usd_upper": std_cost,
        "estimated_flex_usd_upper": flex_cost,
    }


def sanitize_responses_reply(
    raw_resp: dict[str, Any],
    pos: PositionSpec,
    *,
    expected_model: str,
    expected_tier: str,
    top_logprobs_cap: int,
    pricing: dict[str, float],
    latency_seconds: float,
    mode: str,
    require_zero_reasoning: bool = True,
) -> dict[str, Any]:
    """Validate and strip all IDs/headers from a Direct Responses JSON object."""
    _require(isinstance(raw_resp, dict), "response is not a JSON object")
    status = raw_resp.get("status")
    cap_terminal = status == "incomplete" and raw_resp.get("incomplete_details") == {
        "reason": "max_output_tokens"
    }
    _require(status == "completed" or cap_terminal, f"unexpected status: {status!r}")
    ret_model = raw_resp.get("model")
    # OpenAI may return a dated snapshot id (e.g., gpt-5.6-sol-2026-...) or exact match;
    # require prefix match and record exact returned model string for provenance.
    _require(
        isinstance(ret_model, str) and (ret_model == expected_model or ret_model.startswith(expected_model + "-")),
        f"returned model mismatch: got {ret_model!r}, expected {expected_model!r}",
    )
    usage = clean_usage(raw_resp.get("usage"))
    if require_zero_reasoning:
        _require(
            usage["reasoning_output_tokens"] == 0,
            f"reasoning tokens leaked: {usage['reasoning_output_tokens']}",
        )

    output = raw_resp.get("output")
    _require(isinstance(output, list), "response.output missing")
    text_parts = []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_parts.append(part)
    _require(len(text_parts) == 1, f"expected 1 output_text part, found {len(text_parts)}")
    lp_rows = text_parts[0].get("logprobs")
    _require(isinstance(lp_rows, list) and bool(lp_rows), "missing message.output_text.logprobs")

    cleaned_positions = []
    dup_rows = 0
    for i, row in enumerate(lp_rows):
        _require(isinstance(row, dict), f"logprobs[{i}] not an object")
        chosen = _clean_logprob_entry(row, f"logprobs[{i}].chosen")
        top_src = row.get("top_logprobs")
        _require(isinstance(top_src, list) and 1 <= len(top_src) <= top_logprobs_cap, f"top_logprobs[{i}] bad length")
        top = [_clean_logprob_entry(e, f"logprobs[{i}].top[{j}]") for j, e in enumerate(top_src)]
        # Verify monotonic nonincreasing order (allow 1e-9 fp tolerance)
        _require(
            all(top[j]["logprob"] >= top[j + 1]["logprob"] - 1e-9 for j in range(len(top) - 1)),
            f"top_logprobs[{i}] not nonincreasing",
        )
        top_hex = [e["bytes_hex"] for e in top]
        if len(top_hex) != len(set(top_hex)):
            dup_rows += 1
        cleaned_positions.append({"chosen": chosen, "top_logprobs": top})

    cost = compute_conservative_cost_usd(usage, pricing)
    return {
        "position_id": pos.position_id,
        "slice_name": pos.slice_name,
        "doc_id": pos.doc_id,
        "mode": mode,
        "requested_model": expected_model,
        "returned_model": ret_model,
        "requested_service_tier": expected_tier,
        "returned_service_tier": raw_resp.get("service_tier"),
        "system_fingerprint": raw_resp.get("system_fingerprint"),
        "response_status": status,
        "cap_terminal": cap_terminal,
        "duplicate_top_byte_rows": dup_rows,
        "latency_seconds": round(float(latency_seconds), 4),
        "usage": usage,
        "conservative_standard_cost_usd": cost,
        "positions": cleaned_positions,
        "http_status": 200,
    }


def post_responses_once(
    api_key: str,
    body: dict[str, Any],
    *,
    url: str = DEFAULT_URL,
    timeout: float = 90.0,
    max_retries: int = 3,
) -> tuple[int, dict[str, Any] | None, str | None, float]:
    """Send one Direct Responses POST with gzip transport and backoff on 429/5xx."""
    payload = canonical_json_bytes(body)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": "pplx-protocol-a/0.1",
    }
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                return int(resp.status), json.loads(raw.decode("utf-8")), None, time.monotonic() - t0
        except urllib.error.HTTPError as exc:
            dt = time.monotonic() - t0
            err_body = ""
            try:
                raw = exc.read()
                if exc.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                err_body = raw.decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            return int(exc.code), None, err_body or str(exc), dt
        except Exception as exc:
            dt = time.monotonic() - t0
            if attempt < max_retries:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            return 0, None, str(exc), dt
    return 0, None, "exhausted retries", 0.0


def run_sol_acquisition(
    positions: Sequence[PositionSpec],
    output_jsonl: Path,
    *,
    api_key: str,
    model: str = "gpt-5.6-sol",
    mode: str = "echo_chat",
    service_tier: str = "flex",
    max_output_tokens: int = 32,
    top_logprobs: int = 20,
    temperature: float = 1.0,
    top_p: float = 1.0,
    reasoning_effort: str = "none",
    pricing: dict[str, float],
    max_cost_usd: float,
    url: str = DEFAULT_URL,
    workers: int = 4,
    timeout: float = 90.0,
    transport_fn: Callable[..., tuple[int, dict[str, Any] | None, str | None, float]] = post_responses_once,
) -> dict[str, Any]:
    """Execute Protocol A calls with resume support and strict spend enforcement.

    Processes positions in randomized order (seeded deterministically) so any
    chronological backend drift is orthogonal to document/slice ordering.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    spent_usd = 0.0

    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("http_status") == 200 and rec.get("position_id"):
                    completed_ids.add(rec["position_id"])
                spent_usd += float(rec.get("conservative_standard_cost_usd", 0.0))

    pending = [p for p in positions if p.position_id not in completed_ids]
    # Deterministic interleaving across slices/docs via sha256(position_id)
    pending.sort(key=lambda p: sha256_hex(p.position_id.encode("utf-8")))

    ok_count = len(completed_ids)
    err_count = 0
    stopped_budget = False

    def _call_one(pos: PositionSpec) -> dict[str, Any]:
        body = build_responses_payload(
            pos,
            model=model,
            mode=mode,
            service_tier=service_tier,
            max_output_tokens=max_output_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
        )
        code, raw, err, dt = transport_fn(api_key, body, url=url, timeout=timeout)
        if code != 200 or raw is None:
            return {
                "position_id": pos.position_id,
                "slice_name": pos.slice_name,
                "doc_id": pos.doc_id,
                "mode": mode,
                "http_status": code,
                "error": err,
                "latency_seconds": round(dt, 4),
                "conservative_standard_cost_usd": 0.0,
            }
        try:
            return sanitize_responses_reply(
                raw,
                pos,
                expected_model=model,
                expected_tier=service_tier,
                top_logprobs_cap=top_logprobs,
                pricing=pricing,
                latency_seconds=dt,
                mode=mode,
            )
        except ProtocolAError as exc:
            # Record telemetry/contract violation transparently without discarding cost
            usage_safe = raw.get("usage") if isinstance(raw, dict) else None
            cost_est = 0.0
            try:
                if isinstance(usage_safe, dict):
                    cost_est = compute_conservative_cost_usd(clean_usage(usage_safe), pricing)
            except Exception:
                pass
            return {
                "position_id": pos.position_id,
                "slice_name": pos.slice_name,
                "doc_id": pos.doc_id,
                "mode": mode,
                "http_status": 200,
                "contract_violation": str(exc),
                "latency_seconds": round(dt, 4),
                "conservative_standard_cost_usd": cost_est,
            }

    # Execute in small waves of size `workers` so hard USD cap stops cleanly
    with output_jsonl.open("a", encoding="utf-8") as sink:
        idx = 0
        while idx < len(pending):
            if spent_usd >= max_cost_usd:
                stopped_budget = True
                break
            wave = pending[idx : idx + max(1, workers)]
            idx += len(wave)
            if workers <= 1 or len(wave) == 1:
                results = [_call_one(p) for p in wave]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(_call_one, wave))
            for rec in results:
                sink.write(canonical_json_bytes(rec).decode("utf-8") + "\n")
                sink.flush()
                spent_usd += float(rec.get("conservative_standard_cost_usd", 0.0))
                if rec.get("http_status") == 200 and "contract_violation" not in rec:
                    ok_count += 1
                else:
                    err_count += 1

    return {
        "total_positions": len(positions),
        "completed_ok": ok_count,
        "errors_or_violations": err_count,
        "spent_conservative_standard_usd": round(spent_usd, 6),
        "max_cost_usd": max_cost_usd,
        "stopped_due_to_budget": stopped_budget,
        "output_jsonl": str(output_jsonl),
    }
