"""Pure-Python sliding-window planning, UTF-8 byte accounting, and BPB aggregation.

Zero GPU / network dependencies so every mathematical invariant is unit-tested locally.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

LN2 = math.log(2.0)
DOMAINS = ("code", "papers", "news", "reference", "essays")


@dataclass(frozen=True)
class StridedWindow:
    """One forward-pass window over a tokenized document.

    Attributes:
        token_ids: Slice of token IDs fed to the model (`len >= 2`).
        score_from: Index inside `token_ids` of the FIRST token whose
            negative log-probability `-ln p(token_ids[t] | token_ids[:t])`
            should be summed (`score_from >= 1`). Every token in
            `token_ids[score_from:]` is scored exactly once across the doc.
        doc_start: Start index in the full document token sequence.
        doc_end: Exclusive end index in the full document token sequence.
    """

    token_ids: tuple[int, ...]
    score_from: int
    doc_start: int
    doc_end: int

    @property
    def n_scored_tokens(self) -> int:
        return len(self.token_ids) - self.score_from


def plan_strided_windows(
    token_ids: Sequence[int],
    max_len: int = 4096,
    stride: int = 2048,
    has_bos: bool = True,
) -> list[StridedWindow]:
    """Partition `token_ids` into overlapping windows scoring each target token once.

    When `has_bos=True`, `token_ids[0]` is the prepended BOS token, so every real
    document token `1..N-1` has left context and is scored exactly once.
    When `has_bos=False`, `token_ids[0]` has no left context and is skipped; tokens
    `1..N-1` are scored exactly once (and callers must subtract token 0's UTF-8 byte
    length from the denominator via `compute_scored_utf8_bytes`).
    """
    if max_len < 2:
        raise ValueError(f"max_len must be >= 2, got {max_len}")
    if stride < 1 or stride > max_len - 1:
        raise ValueError(f"stride must be in [1, max_len - 1], got {stride}")

    n = len(token_ids)
    if n < 2:
        return []

    windows: list[StridedWindow] = []
    prev_end = 1  # token index 0 is never scored (either BOS or no left context)

    begin = 0
    while True:
        end = min(begin + max_len, n)
        if end > prev_end:
            score_from = prev_end - begin
            assert 1 <= score_from < (end - begin)
            windows.append(
                StridedWindow(
                    token_ids=tuple(token_ids[begin:end]),
                    score_from=score_from,
                    doc_start=begin,
                    doc_end=end,
                )
            )
            prev_end = end
        if end >= n:
            break
        begin += stride

    return windows


def compute_scored_utf8_bytes(
    text: str,
    has_bos: bool,
    first_token_text: str = "",
) -> int:
    """Return the exact number of UTF-8 bytes corresponding to scored tokens."""
    total_bytes = len(text.encode("utf-8"))
    if has_bos:
        return total_bytes
    if not first_token_text:
        return total_bytes
    first_bytes = len(first_token_text.encode("utf-8"))
    return max(0, total_bytes - first_bytes)


def sum_vllm_window_nll(
    prompt_logprobs: Sequence[dict[int, Any] | None],
    token_ids: Sequence[int],
    score_from: int,
) -> float:
    """Sum exact negative log-likelihood (in nats) from vLLM prompt_logprobs.

    Supports both vLLM `Logprob` objects (`.logprob` attr) and plain floats/dicts.
    """
    if len(prompt_logprobs) != len(token_ids):
        raise ValueError(
            f"Length mismatch: prompt_logprobs={len(prompt_logprobs)} vs tokens={len(token_ids)}"
        )
    nll = 0.0
    for pos in range(score_from, len(token_ids)):
        entry = prompt_logprobs[pos]
        if not entry:
            raise ValueError(f"Missing prompt_logprobs entry at position {pos}")
        tid = int(token_ids[pos])
        if tid in entry:
            val = entry[tid]
        elif len(entry) == 1:
            val = next(iter(entry.values()))
        else:
            raise KeyError(f"Token id {tid} not found in prompt_logprobs at pos {pos}")
        lp = float(val.logprob) if hasattr(val, "logprob") else float(val)
        nll -= lp
    return nll


def load_corpus_docs(
    corpus_path: Path | str,
    limit: int | None = None,
    domains: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Load archived ANPI corpus documents deterministically."""
    path = Path(corpus_path)
    allowed = set(domains) if domains else None
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if allowed and d.get("domain") not in allowed:
                continue
            docs.append(d)
    if limit is not None and limit > 0:
        docs = docs[:limit]
    return docs


def load_anpi_bench_doc_ids(bench_path: Path | str) -> set[str]:
    """Return the exact 954 doc_ids present in runs/anpaure_v1_n1000.json."""
    raw = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    items = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
    return {str(it["doc_id"]) for it in items if "doc_id" in it}


def cluster_bootstrap_bpb_ci(
    per_doc: Sequence[dict[str, Any]],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    nll_key: str = "nll",
) -> tuple[float, float] | None:
    """Source-URL cluster-bootstrapped percentile CI for byte-weighted BPB."""
    clusters: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for d in per_doc:
        b = int(d.get("bytes", 0))
        nll = d.get(nll_key)
        if b <= 0 or nll is None:
            continue
        key = str(d.get("url") or d.get("doc_id") or id(d))
        clusters[key].append((float(nll), b))

    cluster_sums = [
        (sum(x[0] for x in vals), sum(x[1] for x in vals))
        for vals in clusters.values()
    ]
    k = len(cluster_sums)
    if k < 2:
        return None

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        tot_nll = 0.0
        tot_b = 0
        for _ in range(k):
            cn, cb = cluster_sums[rng.randrange(k)]
            tot_nll += cn
            tot_b += cb
        if tot_b > 0:
            boots.append(tot_nll / (LN2 * tot_b))
    boots.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * len(boots))))
    hi_idx = min(len(boots) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(boots))) - 1)
    return (round(boots[lo_idx], 6), round(boots[hi_idx], 6))


def aggregate_bpb_records(
    per_doc: Sequence[dict[str, Any]],
    anpi_doc_ids: set[str] | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute byte-weighted, macro-domain, per-domain, and URL-clustered CI stats."""
    valid = [d for d in per_doc if int(d.get("bytes", 0)) > 0 and d.get("nll") is not None]
    if not valid:
        raise ValueError("No valid scored documents in per_doc")

    def _summarize(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tot_nll = sum(float(d["nll"]) for d in subset)
        tot_bytes = sum(int(d["bytes"]) for d in subset)
        tot_tokens = sum(int(d.get("scored_tokens", 0)) for d in subset)
        bpb = tot_nll / (LN2 * tot_bytes)

        by_dom: dict[str, dict[str, Any]] = {}
        dom_bpbs: list[float] = []
        for dom in sorted({str(d.get("domain", "unknown")) for d in subset}):
            sub_d = [d for d in subset if d.get("domain") == dom]
            dn = sum(float(d["nll"]) for d in sub_d)
            db = sum(int(d["bytes"]) for d in sub_d)
            dt = sum(int(d.get("scored_tokens", 0)) for d in sub_d)
            dbpb = dn / (LN2 * db)
            by_dom[dom] = {
                "n_docs": len(sub_d),
                "bytes": db,
                "scored_tokens": dt,
                "bpb": round(dbpb, 6),
            }
            if dom in DOMAINS:
                dom_bpbs.append(dbpb)

        macro_bpb = sum(dom_bpbs) / len(dom_bpbs) if dom_bpbs else bpb
        ci = cluster_bootstrap_bpb_ci(subset, n_boot=n_boot, seed=seed)
        out: dict[str, Any] = {
            "n_docs": len(subset),
            "n_unique_urls": len({str(d.get("url") or d.get("doc_id")) for d in subset}),
            "total_bytes": tot_bytes,
            "total_scored_tokens": tot_tokens,
            "bytes_per_token": round(tot_bytes / tot_tokens, 4) if tot_tokens > 0 else None,
            "bpb": round(bpb, 6),
            "bpb_ci95": list(ci) if ci else None,
            "macro_domain_bpb": round(macro_bpb, 6),
            "domains": by_dom,
        }
        return out

    summary = _summarize(valid)
    if anpi_doc_ids:
        sub_954 = [d for d in valid if str(d.get("doc_id")) in anpi_doc_ids]
        if sub_954:
            summary["anpi_954_subset"] = _summarize(sub_954)
    return summary


def load_anpi_bench_items(
    bench_path: Path | str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load ANPI 8-choice continuation-ranking items from runs/anpaure_v1_n1000.json."""
    raw = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    items = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def longest_common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Return length of the longest identical token prefix shared by `a` and `b`."""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def common_prefix_len_all(seqs: Sequence[Sequence[int]]) -> int:
    """Return length of the longest identical token prefix shared by ALL sequences."""
    if not seqs:
        return 0
    m = min(len(s) for s in seqs)
    i = 0
    while i < m:
        v = seqs[0][i]
        if any(s[i] != v for s in seqs[1:]):
            break
        i += 1
    return i
    """Return length of the longest identical token prefix shared by `a` and `b`."""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def exact_conditional_logprob_from_joint(
    prefix_token_ids: Sequence[int],
    prefix_prompt_logprobs: Sequence[dict[int, Any] | None],
    full_token_ids: Sequence[int],
    full_prompt_logprobs: Sequence[dict[int, Any] | None],
) -> float:
    """Compute exact ln p(cand | prefix) = ln p(prefix + cand) - ln p(prefix).

    Uses the longest common token prefix so shared tokens cancel identically while
    any BPE boundary re-tokenization at the junction is accounted for exactly.
    """
    lcp = longest_common_prefix_len(prefix_token_ids, full_token_ids)
    # Token 0 has no prompt_logprob; any divergence at index >= 1 is summed exactly.
    start = max(1, lcp)
    full_tail_nll = (
        sum_vllm_window_nll(full_prompt_logprobs, full_token_ids, score_from=start)
        if start < len(full_token_ids)
        else 0.0
    )
    pref_tail_nll = (
        sum_vllm_window_nll(prefix_prompt_logprobs, prefix_token_ids, score_from=start)
        if start < len(prefix_token_ids)
        else 0.0
    )
    return -(full_tail_nll - pref_tail_nll)


def aggregate_anpi_eval_records(
    per_item: Sequence[dict[str, Any]],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate exact white-box 954-item ANPI continuation-span BPB & 8-way ranking."""
    valid = [it for it in per_item if it.get("gold_bytes", 0) > 0 and it.get("cand_logprobs")]
    if not valid:
        raise ValueError("No valid ANPI eval items to aggregate")

    def _rank_1based(scores: Sequence[float], gold_idx: int) -> float:
        g = scores[gold_idx]
        strictly_better = sum(1 for i, s in enumerate(scores) if i != gold_idx and s > g)
        tied = sum(1 for i, s in enumerate(scores) if i != gold_idx and s == g)
        return 1.0 + strictly_better + 0.5 * tied

    enriched: list[dict[str, Any]] = []
    for it in valid:
        lps = [float(x) for x in it["cand_logprobs"]]
        cbytes = [max(1, int(b)) for b in it["cand_bytes"]]
        g_idx = int(it["gold_idx"])
        r_raw = _rank_1based(lps, g_idx)
        norm_scores = [lp / b for lp, b in zip(lps, cbytes)]
        r_norm = _rank_1based(norm_scores, g_idx)
        k = len(lps)
        auc_raw = (k - r_raw) / (k - 1)
        auc_norm = (k - r_norm) / (k - 1)
        g_lp = lps[g_idx]
        mean_decoy_lp = sum(lp for i, lp in enumerate(lps) if i != g_idx) / max(1, k - 1)
        enriched.append(
            {
                **it,
                "gold_nll": -g_lp,
                "bytes": int(it["gold_bytes"]),
                "rank_raw": r_raw,
                "rank_norm": r_norm,
                "auc_raw": auc_raw,
                "auc_norm": auc_norm,
                "mrr_raw": 1.0 / r_raw,
                "mrr_norm": 1.0 / r_norm,
                "top1_raw": 1.0 if r_raw == 1.0 else 0.0,
                "top1_norm": 1.0 if r_norm == 1.0 else 0.0,
                "margin_nats": g_lp - mean_decoy_lp,
            }
        )

    def _summ(sub: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tot_nll = sum(x["gold_nll"] for x in sub)
        tot_b = sum(x["bytes"] for x in sub)
        n = len(sub)
        return {
            "n_items": n,
            "gold_bytes": tot_b,
            "gold_span_bpb": round(tot_nll / (LN2 * tot_b), 6),
            "whitebox_auc": round(sum(x["auc_raw"] for x in sub) / n, 6),
            "whitebox_mrr": round(sum(x["mrr_raw"] for x in sub) / n, 6),
            "whitebox_top1_pct": round(100.0 * sum(x["top1_raw"] for x in sub) / n, 3),
            "whitebox_norm_auc": round(sum(x["auc_norm"] for x in sub) / n, 6),
            "whitebox_norm_mrr": round(sum(x["mrr_norm"] for x in sub) / n, 6),
            "whitebox_norm_top1_pct": round(100.0 * sum(x["top1_norm"] for x in sub) / n, 3),
            "mean_margin_nats": round(sum(x["margin_nats"] for x in sub) / n, 5),
        }

    out = _summ(enriched)
    ci = cluster_bootstrap_bpb_ci(enriched, n_boot=n_boot, seed=seed, nll_key="gold_nll")
    out["gold_span_bpb_ci95"] = list(ci) if ci else None
    out["domains"] = {}
    for dom in sorted({str(x.get("domain", "unknown")) for x in enriched}):
        out["domains"][dom] = _summ([x for x in enriched if x.get("domain") == dom])
    return out


def select_best_temperature(
    temps: Sequence[float],
    nll_by_temp: Sequence[float],
    total_bytes: int,
) -> dict[str, Any]:
    """Pick optimal temperature T* minimizing BPB(T) on a grid."""
    if not temps or len(temps) != len(nll_by_temp):
        raise ValueError("temps and nll_by_temp must be non-empty and equal length")
    best_i = min(range(len(temps)), key=lambda i: nll_by_temp[i])
    curve = [
        {"T": round(float(t), 4), "bpb": round(float(nll) / (LN2 * total_bytes), 6)}
        for t, nll in zip(temps, nll_by_temp)
    ]
    return {
        "t_opt": round(float(temps[best_i]), 4),
        "bpb_topt": round(float(nll_by_temp[best_i]) / (LN2 * total_bytes), 6),
        "curve": curve,
    }
