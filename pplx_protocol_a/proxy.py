"""Open-weight teacher-forced proxy scorer (`/v1/completions` + `echo=true`).

Works with any OpenAI-compatible completions server that supports prompt logprobs:
  - vLLM (`vllm serve ...`)
  - Fireworks serverless (`/inference/v1/completions`)
  - SGLang / llama.cpp server

Serves TWO critical roles in Protocol A:
  1. Supplies the open-weight proxy `q` needed for:
       - Certified Operational Hybrid Compression (`BPB_comp`)
       - Shifted Box-Cox Tobit tail deconvolution (`BPB_tobit`)
       - Full-corpus proxy Control Variate variance reduction
  2. Enables the mandatory Open-Model Ground-Truth Gate: run Protocol A on an
     open model's chat/responses endpoint and compare against its exact
     teacher-forced BPB computed here before quoting closed-model numbers.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .corpus import PositionSpec, canonical_json_bytes

LN2 = math.log(2.0)


def post_completions_echo(
    api_key: str,
    *,
    url: str,
    model: str,
    prompt: str,
    logprobs: int = 5,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call `/v1/completions` with `echo=true, max_tokens=0` for exact prompt logprobs."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 0,
        "echo": True,
        "logprobs": int(logprobs),
        "temperature": 0.0,
    }
    data = canonical_json_bytes(body)
    headers = {"Content-Type": "application/json", "User-Agent": "pplx-protocol-a-proxy/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"Proxy HTTP {exc.code}: {detail}") from exc
        except Exception:
            if attempt < max_retries:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            raise
    raise RuntimeError("Proxy call exhausted retries")


def parse_echo_document_logprobs(
    response: dict[str, Any],
    raw_text: str,
) -> tuple[float, list[tuple[int, int, float]]]:
    """Parse `/v1/completions` `echo=true` response into exact BPB + char-offset token logprobs.

    Returns:
      (doc_bpb, token_spans) where each span is `(char_start, char_end, logprob_nats)`.
    Detects placeholder all-zero logprobs (like the July 2026 Fireworks K3 bug!).
    """
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("Proxy response has no choices")
    lp_obj = choices[0].get("logprobs")
    if not isinstance(lp_obj, dict):
        raise ValueError("Proxy response missing choices[0].logprobs")

    tokens = lp_obj.get("tokens") or []
    token_lps = lp_obj.get("token_logprobs") or []
    offsets = lp_obj.get("text_offset") or []
    if not (len(tokens) == len(token_lps) == len(offsets)) or len(tokens) < 2:
        raise ValueError("Inconsistent proxy logprobs array lengths")

    # Sanity check against all-zero placeholder logprob bugs!
    valid_lps = [float(x) for x in token_lps[1:] if x is not None]
    if not valid_lps:
        raise ValueError("Proxy returned no non-null prompt logprobs")
    if all(abs(x) < 1e-12 for x in valid_lps):
        raise ValueError("Proxy returned all-zero placeholder logprobs (endpoint cannot score prompts)")

    spans: list[tuple[int, int, float]] = []
    total_bits = 0.0
    for i in range(len(tokens)):
        lp = token_lps[i]
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < len(offsets) else len(raw_text)
        if lp is None:
            continue
        lp_f = float(lp)
        total_bits += -lp_f / LN2
        spans.append((start, end, lp_f))

    # Count bytes strictly on raw UTF-8 text (never decoded token strings!)
    # Exclude leading BOS / first-token bytes that received null logprob.
    scored_start = spans[0][0] if spans else 0
    scored_bytes = len(raw_text[scored_start:].encode("utf-8"))
    if scored_bytes <= 0:
        raise ValueError("Zero scored UTF-8 bytes in document")
    return total_bits / scored_bytes, spans


def extract_target_span_bits(
    spans: Sequence[tuple[int, int, float]],
    target_char_start: int,
    target_char_end: int,
) -> float:
    """Sum exact proxy surprisal (in bits) over tokens overlapping `[target_char_start, target_char_end)`.

    If a proxy token partially straddles a boundary (rare with leading-space
    atom alignment), attributes surprisal proportionally to character overlap so
    the sum across adjacent spans stays strictly additive.
    """
    bits = 0.0
    for s, e, lp in spans:
        if e <= target_char_start or s >= target_char_end:
            continue
        ov = min(e, target_char_end) - max(s, target_char_start)
        tok_len = max(1, e - s)
        bits += (-lp / LN2) * (ov / tok_len)
    return max(0.005, bits)


def score_proxy_corpus(
    docs: Sequence[dict[str, str]],
    positions: Sequence[PositionSpec],
    *,
    api_key: str,
    url: str,
    model: str,
    output_json: Path,
    caller_fn: Callable[..., dict[str, Any]] = post_completions_echo,
) -> dict[str, Any]:
    """Score full documents and every manifest position using an open-weight proxy."""
    doc_map = {(d["slice_name"], d["doc_id"]): d["text"] for d in docs}
    pos_by_doc: dict[tuple[str, str], list[PositionSpec]] = {}
    for p in positions:
        pos_by_doc.setdefault((p.slice_name, p.doc_id), []).append(p)

    slice_bits: dict[str, float] = {}
    slice_bytes: dict[str, int] = {}
    position_proxy_bits: dict[str, float] = {}
    doc_bpbs: dict[str, float] = {}

    for (slc, did), doc_positions in sorted(pos_by_doc.items()):
        text = doc_map[(slc, did)]
        resp = caller_fn(api_key, url=url, model=model, prompt=text)
        bpb, spans = parse_echo_document_logprobs(resp, text)
        nbytes = len(text.encode("utf-8"))
        slice_bits[slc] = slice_bits.get(slc, 0.0) + bpb * nbytes
        slice_bytes[slc] = slice_bytes.get(slc, 0) + nbytes
        doc_bpbs[f"{slc}__{did}"] = bpb

        for p in doc_positions:
            # Locate exact occurrence of prefix_text + target_text in raw text
            needle = p.prefix_text + p.target_text
            loc = text.find(needle)
            if loc < 0:
                raise ValueError(f"Could not locate position {p.position_id} inside source document")
            t_start = loc + len(p.prefix_text)
            t_end = t_start + len(p.target_text)
            position_proxy_bits[p.position_id] = extract_target_span_bits(spans, t_start, t_end)

    full_slice_bpb = {s: slice_bits[s] / slice_bytes[s] for s in sorted(slice_bits)}
    result = {
        "schema_version": "protocol_a_proxy_v1",
        "proxy_model": model,
        "proxy_url": url,
        "full_corpus_macro_bpb": float(np.mean(list(full_slice_bpb.values()))),
        "full_corpus_slice_bpb": full_slice_bpb,
        "position_proxy_bits": position_proxy_bits,
        "doc_bpbs": doc_bpbs,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(canonical_json_bytes(result))
    return result
