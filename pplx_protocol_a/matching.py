"""Exact UTF-8 byte-level alignment and Top-20 / multi-token target extraction.

Operates strictly on raw token byte arrays (`bytes_hex`) returned by OpenAI
Responses (`message.output_text.logprobs`), never on provider token IDs and
never stripping capitalization or punctuation.

Handles four real-world BPE edge cases cleanly:
  1. Echo-anchor boundary alignment: verifies strict UTF-8 byte equality of the
     anchor prefix before extracting the continuation distribution.
  2. BPE trailing-space absorption: if the anchor's final token absorbs the
     target's leading space (e.g. anchor ends with `"the"` and the emitted
     token is `b"the "`), automatically shifts the remaining target to
     `target_bytes[1:]` while logging `absorbed_leading_space=True`.
  3. Super-token & exact-atom aggregation: sums probabilities in probability
     space over all Top-20 entries that realize the exact next lexical atom
     (including super-tokens like `b" amendment,"` when `target = b" amendment"`
     ends with alphanumeric and the next byte is a non-alphanumeric boundary).
  4. Multi-token targets: if the model's actual emission follows a multi-token
     prefix chain covering `target_bytes`, sums exact token logprobs across the
     chain (including outside-top-20 real logprobs returned by Direct Responses).
     Otherwise, if a sub-token prefix appears in Top-20 without being emitted,
     marks the position explicitly as `subtoken_censored` with an exact upper
     bound rather than silently mis-scoring.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

LN2 = math.log(2.0)


@dataclass(frozen=True)
class TokenMatchResult:
    """Extracted Protocol A observable at one evaluation position."""

    position_id: str
    adherent_strict: bool
    adherent_relaxed: bool
    adherence_note: str
    absorbed_leading_space: bool
    duplicate_top_bytes: bool
    top_k_count: int
    match_kind: str  # "exact_top20" | "exact_chosen_outside" | "exact_multitoken_emitted" | "subtoken_censored" | "miss_censored" | "non_adherent"
    observed_exact: bool
    surprisal_bits: float | None  # exact -log2(p) when observed_exact=True
    censor_lb_bits: float | None  # exact lower bound on surprisal when censored
    h_20_bits: float | None       # -log2(p_20) at the target position
    m_tail: float | None          # 1 - sum_{j=1}^K exp(ell_j)
    p_20: float | None
    top1_bits: float | None
    target_utf8_bytes: int
    matched_top_entries: list[dict[str, Any]]
    top_entries_bytes_hex: list[str]
    top_entries_logprobs: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_word_byte(b: int) -> bool:
    return (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 39 or b >= 128


def _is_exact_atom_token(cand: bytes, target: bytes) -> bool:
    """True if non-empty `cand` realizes `target` as the next lexical atom."""
    if not cand or not target:
        return False
    if cand == target:
        return True
    if not cand.startswith(target) or len(cand) <= len(target):
        return False
    # Super-token (e.g., cand=b" amendment," when target=b" amendment"):
    # valid iff the atom boundary is not split mid-word.
    last_t = target[-1]
    next_c = cand[len(target)]
    if _is_word_byte(last_t) and _is_word_byte(next_c):
        # e.g., target=b" war", cand=b" warning" -> NOT a match!
        return False
    return True


def _deduplicate_top_entries(
    top_logprobs: Sequence[dict[str, Any]],
) -> tuple[list[tuple[bytes, float]], bool]:
    """Return `(entries, had_duplicates)` merging duplicate non-empty byte keys by prob sum.

    Empty-byte special/control tokens (e.g. EOS `<|im_end|>`, `bytes_hex == ""`)
    are preserved individually rather than collapsed or treated as text prefixes.
    """
    seen: dict[bytes, float] = {}
    empty_entries: list[tuple[bytes, float]] = []
    had_dup = False
    for entry in top_logprobs:
        b = bytes.fromhex(entry["bytes_hex"])
        lp = float(entry["logprob"])
        if not b:
            empty_entries.append((b, lp))
            continue
        if b in seen:
            had_dup = True
            # Sum probabilities in log space
            a, c = seen[b], lp
            m = max(a, c)
            seen[b] = m + math.log(math.exp(a - m) + math.exp(c - m))
        else:
            seen[b] = lp
    entries = sorted(list(seen.items()) + empty_entries, key=lambda kv: kv[1], reverse=True)
    return entries, had_dup


def _locate_anchor_boundary(
    positions: Sequence[dict[str, Any]],
    anchor_bytes: bytes,
    target_bytes: bytes,
    *,
    mode: str = "echo_chat",
) -> tuple[bool, bool, str, int, bytes, bool]:
    """Locate the first token index immediately after the anchor.

    Returns:
      (adherent_strict, adherent_relaxed, note, target_pos_idx, effective_target_bytes, absorbed_space)
    """
    if mode == "assistant_prefill":
        if not positions:
            return False, False, "empty_response", -1, target_bytes, False
        return True, True, "prefill_pos0", 0, target_bytes, False

    if not positions:
        return False, False, "empty_response", -1, target_bytes, False

    # Build cumulative byte offsets of emitted chosen tokens
    tok_bytes = [bytes.fromhex(p["chosen"]["bytes_hex"]) for p in positions]
    cum = b""
    offsets: list[tuple[int, int]] = []
    for tb in tok_bytes:
        start = len(cum)
        cum += tb
        offsets.append((start, len(cum)))

    # 1. Strict adherence: emitted bytes start with exact `anchor_bytes`
    adherent_strict = cum.startswith(anchor_bytes)
    prefix_skip = 0
    note = "strict_exact"

    if not adherent_strict:
        # Check relaxed adherence (diagnostic only: leading quote/newline/whitespace)
        stripped = cum.lstrip(b' "\'\n\r\t')
        if stripped.startswith(anchor_bytes):
            prefix_skip = len(cum) - len(stripped)
            note = f"relaxed_leading_trim_{prefix_skip}B"
            adherent_relaxed = True
        else:
            return False, False, "anchor_mismatch", -1, target_bytes, False
    else:
        adherent_relaxed = True

    anchor_end = prefix_skip + len(anchor_bytes)

    # Find token index where emitted byte offset reaches or crosses `anchor_end`
    for idx, (s, e) in enumerate(offsets):
        if e == anchor_end:
            if idx + 1 >= len(positions):
                return adherent_strict, adherent_relaxed, "truncated_at_anchor", -1, target_bytes, False
            return adherent_strict, adherent_relaxed, note, idx + 1, target_bytes, False
        if s < anchor_end < e:
            # Token `idx` straddled the anchor end!
            spill = cum[anchor_end:e]
            # Clean case A: token absorbed ONLY the leading space of `target_bytes`
            if target_bytes.startswith(spill) and spill == b" " and len(target_bytes) > 1:
                if idx + 1 >= len(positions):
                    return adherent_strict, adherent_relaxed, "truncated_after_space", -1, target_bytes[1:], True
                return (
                    adherent_strict,
                    adherent_relaxed,
                    note + "+absorbed_space",
                    idx + 1,
                    target_bytes[1:],
                    True,
                )
            # Clean case B: token merged anchor end + partial/full target (rare BPE merge)
            return False, adherent_relaxed, "anchor_target_bpe_merge", -1, target_bytes, False

    return False, False, "anchor_too_short", -1, target_bytes, False


def extract_position_match(
    position_id: str,
    positions: Sequence[dict[str, Any]],
    anchor_bytes: bytes,
    target_bytes: bytes,
    *,
    mode: str = "echo_chat",
    allow_relaxed: bool = False,
) -> TokenMatchResult:
    """Align a single sanitized OpenAI Responses call and score `target_bytes`."""
    strict, relaxed, note, idx, eff_target, absorbed_space = _locate_anchor_boundary(
        positions, anchor_bytes, target_bytes, mode=mode
    )

    if idx < 0 or (not strict and not (allow_relaxed and relaxed)):
        return TokenMatchResult(
            position_id=position_id,
            adherent_strict=strict,
            adherent_relaxed=relaxed,
            adherence_note=note,
            absorbed_leading_space=absorbed_space,
            duplicate_top_bytes=False,
            top_k_count=0,
            match_kind="non_adherent",
            observed_exact=False,
            surprisal_bits=None,
            censor_lb_bits=None,
            h_20_bits=None,
            m_tail=None,
            p_20=None,
            top1_bits=None,
            target_utf8_bytes=len(target_bytes),
            matched_top_entries=[],
            top_entries_bytes_hex=[],
            top_entries_logprobs=[],
        )

    row = positions[idx]
    top_entries, had_dup = _deduplicate_top_entries(row["top_logprobs"])
    top_ps = [math.exp(lp) for _, lp in top_entries]
    s_head = sum(top_ps)
    m_tail = max(1e-15, 1.0 - s_head)
    p_20 = top_ps[-1] if top_ps else 1e-15
    h_20_bits = -top_entries[-1][1] / LN2 if top_entries else None
    top1_bits = -top_entries[0][1] / LN2 if top_entries else None

    top_hex = [b.hex() for b, _ in top_entries]
    top_lps = [lp for _, lp in top_entries]

    # 1. Check exact / super-token matches in Top-20
    exact_matches: list[tuple[bytes, float]] = []
    sub_prefixes: list[tuple[bytes, float]] = []
    for b, lp in top_entries:
        if not b:
            # Empty-byte special/control token (e.g. EOS): never matches text!
            continue
        if _is_exact_atom_token(b, eff_target):
            exact_matches.append((b, lp))
        elif len(b) < len(eff_target) and eff_target.startswith(b):
            sub_prefixes.append((b, lp))

    # If exact match(es) exist AND any stray sub-prefix (e.g. a p=1e-7 standalone
    # space token b" " when target=b" private") contributes negligible mass
    # (<= 0.10 bit upper/lower bound gap), the exact match is tightly resolved!
    if exact_matches:
        p_exact = sum(math.exp(lp) for _, lp in exact_matches)
        p_sub = sum(math.exp(lp) for _, lp in sub_prefixes)
        if p_sub == 0.0 or math.log2((p_exact + p_sub) / max(p_exact, 1e-300)) <= 0.10:
            surprisal = -math.log2(max(p_exact, 1e-300))
            return TokenMatchResult(
                position_id=position_id,
                adherent_strict=strict,
                adherent_relaxed=relaxed,
                adherence_note=note,
                absorbed_leading_space=absorbed_space,
                duplicate_top_bytes=had_dup,
                top_k_count=len(row["top_logprobs"]),
                match_kind="exact_top20",
                observed_exact=True,
                surprisal_bits=surprisal,
                censor_lb_bits=surprisal,
                h_20_bits=h_20_bits,
                m_tail=m_tail,
                p_20=p_20,
                top1_bits=top1_bits,
                target_utf8_bytes=len(target_bytes),
                matched_top_entries=[{"bytes_hex": b.hex(), "logprob": lp} for b, lp in exact_matches],
                top_entries_bytes_hex=top_hex,
                top_entries_logprobs=top_lps,
            )

    # 2. Check if the actually emitted token(s) realized `eff_target` (either outside
    #    Top-20 with a real Direct Responses logprob, OR by stepping through emitted
    #    strict-prefix token(s) like standalone b" " / subword prefix and checking
    #    `top_logprobs` at the continuation position in the VERY SAME response!).
    chosen_b = bytes.fromhex(row["chosen"]["bytes_hex"])
    chosen_lp = float(row["chosen"]["logprob"])
    if chosen_lp > -9000.0 and _is_exact_atom_token(chosen_b, eff_target):
        p_sum = math.exp(chosen_lp) + sum(
            math.exp(lp) for b, lp in exact_matches if b != chosen_b
        )
        surprisal = -math.log2(max(p_sum, 1e-300))
        kind = "exact_top20" if any(b == chosen_b for b, _ in exact_matches) else "exact_chosen_outside"
        return TokenMatchResult(
            position_id=position_id,
            adherent_strict=strict,
            adherent_relaxed=relaxed,
            adherence_note=note,
            absorbed_leading_space=absorbed_space,
            duplicate_top_bytes=had_dup,
            top_k_count=len(row["top_logprobs"]),
            match_kind=kind,
            observed_exact=True,
            surprisal_bits=surprisal,
            censor_lb_bits=surprisal,
            h_20_bits=h_20_bits,
            m_tail=m_tail,
            p_20=p_20,
            top1_bits=top1_bits,
            target_utf8_bytes=len(target_bytes),
            matched_top_entries=[{"bytes_hex": chosen_b.hex(), "logprob": chosen_lp}],
            top_entries_bytes_hex=top_hex,
            top_entries_logprobs=top_lps,
        )

    # Step through emitted strict prefix chain (e.g. chosen=b" " at p=0.999 followed
    # by top_logprobs at idx+1 containing b"32" or b"times"!).
    if chosen_b and chosen_lp > -9000.0 and len(chosen_b) < len(eff_target) and eff_target.startswith(chosen_b):
        acc = chosen_b
        lp_sum = chosen_lp
        chain = [{"bytes_hex": chosen_b.hex(), "logprob": chosen_lp}]
        j = idx + 1
        while j < len(positions) and len(acc) < len(eff_target):
            rem = eff_target[len(acc) :]
            next_row = positions[j]
            next_top, _ = _deduplicate_top_entries(next_row["top_logprobs"])
            next_exact = [(b, lp) for b, lp in next_top if b and _is_exact_atom_token(b, rem)]
            next_sub = [(b, lp) for b, lp in next_top if b and len(b) < len(rem) and rem.startswith(b)]
            if next_exact:
                p_ex = sum(math.exp(lp) for _, lp in next_exact)
                p_sb = sum(math.exp(lp) for _, lp in next_sub)
                if p_sb == 0.0 or math.log2((p_ex + p_sb) / max(p_ex, 1e-300)) <= 0.10:
                    total_lp = lp_sum + math.log(max(p_ex, 1e-300))
                    # Add any direct single-token exact match mass from step 0 too
                    if exact_matches:
                        p_tot = math.exp(total_lp) + sum(math.exp(lp) for _, lp in exact_matches)
                        surprisal = -math.log2(max(p_tot, 1e-300))
                    else:
                        surprisal = -total_lp / LN2
                    chain_out = chain + [{"bytes_hex": b.hex(), "logprob": lp} for b, lp in next_exact]
                    return TokenMatchResult(
                        position_id=position_id,
                        adherent_strict=strict,
                        adherent_relaxed=relaxed,
                        adherence_note=note + "+prefix_step",
                        absorbed_leading_space=absorbed_space,
                        duplicate_top_bytes=had_dup,
                        top_k_count=len(row["top_logprobs"]),
                        match_kind="exact_multitoken_top20",
                        observed_exact=True,
                        surprisal_bits=surprisal,
                        censor_lb_bits=surprisal,
                        h_20_bits=h_20_bits,
                        m_tail=m_tail,
                        p_20=p_20,
                        top1_bits=top1_bits,
                        target_utf8_bytes=len(target_bytes),
                        matched_top_entries=chain_out,
                        top_entries_bytes_hex=top_hex,
                        top_entries_logprobs=top_lps,
                    )
            nb = bytes.fromhex(next_row["chosen"]["bytes_hex"])
            nlp = float(next_row["chosen"]["logprob"])
            if nlp <= -9000.0:
                break
            acc += nb
            lp_sum += nlp
            chain.append({"bytes_hex": nb.hex(), "logprob": nlp})
            if _is_exact_atom_token(acc, eff_target):
                surprisal = -lp_sum / LN2
                return TokenMatchResult(
                    position_id=position_id,
                    adherent_strict=strict,
                    adherent_relaxed=relaxed,
                    adherence_note=note,
                    absorbed_leading_space=absorbed_space,
                    duplicate_top_bytes=had_dup,
                    top_k_count=len(row["top_logprobs"]),
                    match_kind="exact_multitoken_emitted",
                    observed_exact=True,
                    surprisal_bits=surprisal,
                    censor_lb_bits=surprisal,
                    h_20_bits=h_20_bits,
                    m_tail=m_tail,
                    p_20=p_20,
                    top1_bits=top1_bits,
                    target_utf8_bytes=len(target_bytes),
                    matched_top_entries=chain,
                    top_entries_bytes_hex=top_hex,
                    top_entries_logprobs=top_lps,
                )
            if not eff_target.startswith(acc):
                # Emitted strict prefix at idx, then missed at j -> tighten censor_lb
                # using j's top-20 floor + lp_sum!
                next_ps = [math.exp(lp) for _, lp in next_top]
                next_p20 = next_ps[-1] if next_ps else 1e-15
                next_mt = max(1e-15, 1.0 - sum(next_ps))
                step_p_ub = math.exp(lp_sum - nlp) * min(next_p20, next_mt)
                if exact_matches:
                    step_p_ub += sum(math.exp(lp) for _, lp in exact_matches)
                censor_lb = -math.log2(max(min(1.0, step_p_ub), 1e-300))
                return TokenMatchResult(
                    position_id=position_id,
                    adherent_strict=strict,
                    adherent_relaxed=relaxed,
                    adherence_note=note + "+prefix_step_miss",
                    absorbed_leading_space=absorbed_space,
                    duplicate_top_bytes=had_dup,
                    top_k_count=len(row["top_logprobs"]),
                    match_kind="miss_censored",
                    observed_exact=False,
                    surprisal_bits=None,
                    censor_lb_bits=censor_lb,
                    h_20_bits=-math.log2(max(math.exp(lp_sum - nlp) * next_p20, 1e-300)),
                    m_tail=math.exp(lp_sum - nlp) * next_mt,
                    p_20=math.exp(lp_sum - nlp) * next_p20,
                    top1_bits=top1_bits,
                    target_utf8_bytes=len(target_bytes),
                    matched_top_entries=chain[:-1],
                    top_entries_bytes_hex=top_hex,
                    top_entries_logprobs=top_lps,
                )
            j += 1

    # 3. Censored cases
    if exact_matches or sub_prefixes:
        # Sub-token prefix appeared in Top-20 alongside (or without) an exact match.
        # Keep exact_matches and sub_prefixes STRICTLY separated so downstream rank
        # reporting never credits a partial prefix (e.g. b" elect") as a whole-target
        # hit for b" electors"!
        p_upper = min(
            1.0,
            sum(math.exp(lp) for _, lp in exact_matches)
            + sum(math.exp(lp) for _, lp in sub_prefixes)
            + m_tail,
        )
        censor_lb = -math.log2(max(p_upper, 1e-300))
        return TokenMatchResult(
            position_id=position_id,
            adherent_strict=strict,
            adherent_relaxed=relaxed,
            adherence_note=note,
            absorbed_leading_space=absorbed_space,
            duplicate_top_bytes=had_dup,
            top_k_count=len(row["top_logprobs"]),
            match_kind="subtoken_censored",
            observed_exact=False,
            surprisal_bits=None,
            censor_lb_bits=censor_lb,
            h_20_bits=h_20_bits,
            m_tail=m_tail,
            p_20=p_20,
            top1_bits=top1_bits,
            target_utf8_bytes=len(target_bytes),
            matched_top_entries=[{"bytes_hex": b.hex(), "logprob": lp, "is_exact": True} for b, lp in exact_matches]
            + [{"bytes_hex": b.hex(), "logprob": lp, "is_exact": False} for b, lp in sub_prefixes],
            top_entries_bytes_hex=top_hex,
            top_entries_logprobs=top_lps,
        )

    # 4. Clean Top-20 miss: p(target) <= min(p_20, m_tail)
    censor_lb = max(
        h_20_bits if h_20_bits is not None else 0.0,
        -math.log2(max(m_tail, 1e-300)),
    )
    return TokenMatchResult(
        position_id=position_id,
        adherent_strict=strict,
        adherent_relaxed=relaxed,
        adherence_note=note,
        absorbed_leading_space=absorbed_space,
        duplicate_top_bytes=had_dup,
        top_k_count=len(row["top_logprobs"]),
        match_kind="miss_censored",
        observed_exact=False,
        surprisal_bits=None,
        censor_lb_bits=censor_lb,
        h_20_bits=h_20_bits,
        m_tail=m_tail,
        p_20=p_20,
        top1_bits=top1_bits,
        target_utf8_bytes=len(target_bytes),
        matched_top_entries=[],
        top_entries_bytes_hex=top_hex,
        top_entries_logprobs=top_lps,
    )
