"""Exact tokenizer-free fixed-byte-horizon probability bounds and certified win rates.

Incorporates the rigorous interval-bounding methodology from `replay_byte_prefixes.py`:
  - Evaluates fixed-length UTF-8 byte events `remaining_text.encode()[:h]`
    (e.g., h=4 bytes) without assuming single-token words or guessing miss costs.
  - Recursively tracks exact probability mass down the sampled prefix chain while
    leaving unobserved branches explicitly unresolved inside `[p_lo, p_hi]`.
  - Applies a configurable downward logprob slack (`log_slack=0.0002` nats by
    default) to guard against wire rounding / display-precision artifacts.
  - Computes Bakouch-style document win rates with an explicit `overlap`
    (undecided) category whenever model loss intervals intersect.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def aligned_strict_offset(rows: Sequence[dict[str, Any]], anchor: bytes) -> int | None:
    """Return the first token index strictly after an exact byte-aligned echo anchor."""
    output = b""
    for idx, row in enumerate(rows):
        output += bytes.fromhex(row["chosen"]["bytes_hex"])
        if output == anchor:
            return idx + 1 if idx + 1 < len(rows) else None
        if not anchor.startswith(output):
            break
    return None


def byte_prefix_bounds(
    rows: Sequence[dict[str, Any]],
    target: bytes,
    *,
    k: int = 20,
    log_slack: float = 0.0002,
) -> tuple[float, float]:
    """Return certified probability bounds `(p_lo, p_hi)` for the exact byte prefix `target`.

    Every logged token in `top_logprobs[:k]` (plus `chosen` if outside Top-k) is
    classified cleanly into:
      - `good`: `token.startswith(suffix)` -> guaranteed to realize `suffix`
      - `bad`:  `not suffix.startswith(token)` -> guaranteed NOT to realize `suffix`
      - `unresolved`: strict prefix of `suffix`, empty bytes, or unobserved tail.
    When `chosen` follows a strict prefix of `suffix`, recurses into `rows[index + 1]`
    on `suffix[len(chosen):]` to tighten both `good` and `bad` mass.
    """
    if not rows or not target:
        raise ValueError("Nonempty rows and target bytes are required")
    if k < 1 or log_slack < 0:
        raise ValueError("Invalid top-k count or negative log_slack")

    def visit(index: int, suffix: bytes) -> tuple[float, float]:
        row = rows[index]
        entries = [(bytes.fromhex(e["bytes_hex"]), float(e["logprob"])) for e in row["top_logprobs"][:k]]
        chosen = bytes.fromhex(row["chosen"]["bytes_hex"])
        if all(tok != chosen for tok, _ in entries):
            entries.append((chosen, float(row["chosen"]["logprob"])))

        good: list[float] = []
        bad: list[float] = []
        known_mass: list[float] = []
        for token, lp in entries:
            if lp <= -9000.0:
                continue
            if not math.isfinite(lp):
                raise ValueError("Non-finite logprob in ledger")
            w = math.exp(lp - log_slack)
            known_mass.append(w)
            if not token:
                continue
            if token.startswith(suffix):
                good.append(w)
            elif not suffix.startswith(token):
                bad.append(w)

        if chosen and len(chosen) < len(suffix) and suffix.startswith(chosen) and index + 1 < len(rows):
            chosen_lp = float(row["chosen"]["logprob"])
            if chosen_lp > -9000.0:
                cg, cb = visit(index + 1, suffix[len(chosen) :])
                w = math.exp(chosen_lp - log_slack)
                good.append(w * cg)
                bad.append(w * cb)

        if math.fsum(known_mass) > 1.0 + 1e-6:
            raise ValueError(f"Reported mass {math.fsum(known_mass):.6f} exceeds 1.0 (increase log_slack)")
        gm, bm = math.fsum(good), math.fsum(bad)
        return gm, bm

    good, bad = visit(0, target)
    return good, max(good, min(1.0, 1.0 - bad))


def interval_bit_width(bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return math.log2(hi / lo) if lo > 0.0 else math.inf


def certified_document_win_rate(
    positions: Mapping[str, Any],
    first_bounds: Mapping[str, tuple[float, float]],
    second_bounds: Mapping[str, tuple[float, float]],
) -> dict[str, int]:
    """Interval-certified document win counts: `first`, `second`, `overlap`, `unaligned`.

    Model A provably wins document d iff `high_loss(A, d) < low_loss(B, d)`.
    No guessed miss cost is ever injected.
    """
    docs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, pos in positions.items():
        slc = pos.slice_name if hasattr(pos, "slice_name") else pos["slice_name"]
        did = pos.doc_id if hasattr(pos, "doc_id") else pos["doc_id"]
        docs[(slc, did)].append(pid)

    counts: Counter[str] = Counter()
    for pids in docs.values():
        if any(pid not in first_bounds or pid not in second_bounds for pid in pids):
            counts["unaligned"] += 1
            continue
        pair_losses: list[tuple[float, float]] = []
        for bmap in (first_bounds, second_bounds):
            bs = [bmap[pid] for pid in pids]
            lo_loss = sum(-math.log2(hi) if hi > 0.0 else math.inf for _, hi in bs)
            hi_loss = sum(-math.log2(lo) if lo > 0.0 else math.inf for lo, _ in bs)
            pair_losses.append((lo_loss, hi_loss))
        if pair_losses[0][1] < pair_losses[1][0]:
            counts["first"] += 1
        elif pair_losses[1][1] < pair_losses[0][0]:
            counts["second"] += 1
        else:
            counts["overlap"] += 1

    return {
        "first": int(counts["first"]),
        "second": int(counts["second"]),
        "overlap": int(counts["overlap"]),
        "unaligned": int(counts["unaligned"]),
        "total_docs": len(docs),
    }
