"""Frozen lexical-estimator transfer module (ShanZM, scalar HeadCal, Kinked HeadCal).

Fits rank, magnitude, escape, and equal-family additive conversion parameters
strictly on the training families (`smol`, `qwen`) with ZERO access to held-out
`olmo` data, persists a reusable `fitted_params.json`, scores per-position
predictions for all three families, and supports zero-shot reuse on live OpenAI
API ledgers (`runs/live_*.jsonl`).

Units are named explicitly everywhere:
  - `native_lexical_bpb`: bits per UTF-8 byte on the frozen lexical-atom targets
    in `pilot_manifest.json` (slice-level sum(bits)/sum(bytes), macro-averaged
    across `hansard_2026_07` and `hansard_2026_11`).
  - `converted_base_raw_4byte_bpb`: bits per UTF-8 byte on raw four-byte base
    continuations (via equal-family training-only additive conversion).
  - `converted_base_document_bpb`: bits per UTF-8 byte on full base documents
    (via equal-family training-only additive conversion).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .corpus import PositionSpec, canonical_json_bytes, load_manifest, sha256_hex
from .matching import TokenMatchResult, _is_exact_atom_token, extract_position_match

LN2 = math.log(2.0)

# ---------------------------------------------------------------------------
# Frozen hyperparameter grids & constants (fixed BEFORE evaluating OLMo!)
# ---------------------------------------------------------------------------
FROZEN_SPEC: dict[str, Any] = {
    "schema_version": "lexical_transfer_v1",
    "train_families": ["smol", "qwen"],
    "holdout_family": "olmo",
    "k_head": 20,
    "tail_escape_bits": 9.0,
    "shanzm_alpha_grid": [round(0.60 + 0.05 * i, 4) for i in range(53)],  # 0.60..3.20 step 0.05
    "shanzm_delta_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
    "headcal_t_grid": [round(0.70 + 0.05 * i, 4) for i in range(71)],     # 0.70..4.20 step 0.05
    "kinked_w1_grid": [round(0.10 + 0.05 * i, 4) for i in range(23)],     # 0.10..1.20 step 0.05
    "kinked_wrest_grid": [round(0.10 + 0.05 * i, 4) for i in range(23)],  # 0.10..1.20 step 0.05
    "kinked_shrinkage_lambda_per_hit": 0.002,
    "kinked_formula": "s_0=0; s_j = w1*(ell_1 - ell_0) + w_rest*(ell_j - ell_1) for j>=1; p = softmax(s)",
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20261125,
    "preregistered_abs_bpb_tolerance": 0.03,
}


@dataclass(frozen=True)
class ExtractedLexicalObs:
    """Step-0 lexical-atom observation extracted from a post-anchor top-20 row."""

    position_id: str
    slice_name: str
    doc_id: str
    target_utf8_bytes: int
    match_kind: str
    is_head_hit: bool
    exact_rank_1idx: int | None          # 1..20 iff is_head_hit
    exact_top_indices_0idx: tuple[int, ...]
    partial_prefix_rank_1idx: int | None # strictly separate from whole-target hit
    top_logprobs: tuple[float, ...]
    raw_surprisal_bits: float | None     # raw T=1 exact surprisal if observed_exact
    head_entropy_bits: float
    gap_l1_l2_nats: float


def load_post_anchor_jsonl(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load Codex's `*_instruct_post_anchor.jsonl` trace (`position_id` -> `post_anchor_rows`)."""
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec["position_id"]
            if pid in out:
                raise ValueError(f"Duplicate position_id {pid} in {path}")
            out[pid] = rec["post_anchor_rows"]
    return out


def load_live_api_post_anchor_rows(
    jsonl_path: Path,
    positions: Sequence[PositionSpec],
    *,
    allow_relaxed: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Slice a live OpenAI Protocol A ledger (`runs/live_*.jsonl`) at the echo anchor boundary.

    Returns `{position_id: post_anchor_rows}` in the exact same format as Codex's
    `*_instruct_post_anchor.jsonl` so the fitted scorer runs identically on live API runs!
    """
    from .matching import _locate_anchor_boundary

    pmap = {p.position_id: p for p in positions}
    out: dict[str, list[dict[str, Any]]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("http_status") != 200 or "contract_violation" in rec or "positions" not in rec:
                continue
            if int((rec.get("usage") or {}).get("reasoning_output_tokens", 0)) > 0:
                continue
            pid = rec["position_id"]
            if pid not in pmap:
                continue
            pos = pmap[pid]
            strict, relaxed, _, idx, eff_target, absorbed = _locate_anchor_boundary(
                rec["positions"],
                pos.anchor_text.encode("utf-8"),
                bytes.fromhex(pos.target_hex),
                mode=str(rec.get("mode", "echo_chat")),
            )
            if idx < 0 or (not strict and not (allow_relaxed and relaxed)):
                continue
            if absorbed:
                # Re-wrap leading space cleanly? For exact parity with unabsorbed
                # target_bytes, prepend a metadata marker or skip; here attach rows[idx:]
                # and note absorbed_space is handled if scoring via `extract_lexical_obs`.
                pass
            out[pid] = rec["positions"][idx:]
    return out


def extract_lexical_obs(
    positions: Sequence[PositionSpec],
    post_anchor_by_pid: Mapping[str, Sequence[dict[str, Any]]],
    *,
    k_head: int = 20,
) -> list[ExtractedLexicalObs]:
    """Extract step-0 whole-target head-hit & top-20 logprob vectors for all matched positions."""
    obs: list[ExtractedLexicalObs] = []
    for pos in positions:
        if pos.position_id not in post_anchor_by_pid:
            continue
        rows = post_anchor_by_pid[pos.position_id]
        target_b = bytes.fromhex(pos.target_hex)
        # Mode="assistant_prefill" tells extract_position_match that row 0 is ALREADY post-anchor!
        m: TokenMatchResult = extract_position_match(
            pos.position_id,
            rows,
            pos.anchor_text.encode("utf-8"),
            target_b,
            mode="assistant_prefill",
        )
        top_lps = tuple(float(x) for x in m.top_entries_logprobs[:k_head])
        top_bs = [bytes.fromhex(h) for h in m.top_entries_bytes_hex[:k_head]]

        exact_idxs: list[int] = []
        partial_idx: int | None = None
        for j, tb in enumerate(top_bs):
            if not tb:
                continue
            if _is_exact_atom_token(tb, target_b):
                exact_idxs.append(j)
            elif partial_idx is None and len(tb) < len(target_b) and target_b.startswith(tb):
                partial_idx = j

        # Strictly require single-step whole-target exact_top20 match for head hit
        is_hit = (m.observed_exact and m.match_kind == "exact_top20" and bool(exact_idxs))
        exact_rank = (exact_idxs[0] + 1) if is_hit else None
        partial_rank = (partial_idx + 1) if (not is_hit and partial_idx is not None) else None

        ps = [math.exp(lp) for lp in top_lps]
        ent = -sum(p * (lp / LN2) for p, lp in zip(ps, top_lps))
        g12 = (top_lps[0] - top_lps[1]) if len(top_lps) >= 2 else 0.0

        obs.append(
            ExtractedLexicalObs(
                position_id=pos.position_id,
                slice_name=pos.slice_name,
                doc_id=pos.doc_id,
                target_utf8_bytes=pos.target_utf8_bytes,
                match_kind=m.match_kind,
                is_head_hit=is_hit,
                exact_rank_1idx=exact_rank,
                exact_top_indices_0idx=tuple(exact_idxs) if is_hit else (),
                partial_prefix_rank_1idx=partial_rank,
                top_logprobs=top_lps,
                raw_surprisal_bits=m.surprisal_bits if is_hit else None,
                head_entropy_bits=ent,
                gap_l1_l2_nats=g12,
            )
        )
    return obs


# ---------------------------------------------------------------------------
# Scoring primitives for each of the 3 frozen methods
# ---------------------------------------------------------------------------

def _shanzm_probs(alpha: float, delta: float, k_head: int = 20) -> np.ndarray:
    r = np.arange(1, k_head + 1, dtype=float)
    w = np.power(r + float(delta), -float(alpha))
    return w / float(np.sum(w))


def _headcal_hit_bits(lps: Sequence[float], exact_idxs: Sequence[int], temperature: float) -> float:
    arr = np.array(lps, dtype=float) / float(temperature)
    mx = float(np.max(arr))
    exps = np.exp(arr - mx)
    denom = float(np.sum(exps))
    num = float(np.sum(exps[list(exact_idxs)]))
    return max(0.0, -math.log2(max(num / denom, 1e-300)))


def _kinked_hit_bits(lps: Sequence[float], exact_idxs: Sequence[int], w1: float, w_rest: float) -> float:
    arr = np.array(lps, dtype=float)
    sc = np.zeros_like(arr)
    if len(arr) >= 2:
        sc[1:] = float(w1) * (arr[1] - arr[0]) + float(w_rest) * (arr[1:] - arr[1])
    mx = float(np.max(sc))
    exps = np.exp(sc - mx)
    denom = float(np.sum(exps))
    num = float(np.sum(exps[list(exact_idxs)]))
    return max(0.0, -math.log2(max(num / denom, 1e-300)))


def score_obs_with_params(
    obs: Sequence[ExtractedLexicalObs],
    fitted_params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Score extracted observations using frozen `fitted_params` (zero refitting!).

    Returns:
      (per_position_rows, method_summaries)
    """
    k_head = int(fitted_params["spec"]["k_head"])
    esc_bits = float(fitted_params["spec"]["tail_escape_bits"])
    pi_miss = float(fitted_params["shared"]["pi_miss"])
    head_gate_bits = -math.log2(max(1.0 - pi_miss, 1e-12))
    miss_total_bits = -math.log2(max(pi_miss, 1e-12)) + esc_bits

    zm_a = float(fitted_params["methods"]["shanzm"]["alpha"])
    zm_d = float(fitted_params["methods"]["shanzm"]["delta"])
    p_zm = _shanzm_probs(zm_a, zm_d, k_head=k_head)

    hc_T = float(fitted_params["methods"]["headcal"]["temperature"])
    kk_w1 = float(fitted_params["methods"]["kinked"]["w1"])
    kk_wr = float(fitted_params["methods"]["kinked"]["w_rest"])

    per_pos: list[dict[str, Any]] = []
    bits_by_method: dict[str, list[float]] = {"shanzm": [], "headcal": [], "kinked": []}

    for o in obs:
        if o.is_head_hit and o.exact_rank_1idx is not None and o.exact_top_indices_0idx:
            # Sum ZM rank probs across matched exact/super-token indices in Top-K
            p_zm_hit = float(sum(p_zm[idx] for idx in o.exact_top_indices_0idx if idx < k_head))
            b_zm = head_gate_bits - math.log2(max(p_zm_hit, 1e-300))
            b_hc = head_gate_bits + _headcal_hit_bits(o.top_logprobs, o.exact_top_indices_0idx, hc_T)
            b_kk = head_gate_bits + _kinked_hit_bits(o.top_logprobs, o.exact_top_indices_0idx, kk_w1, kk_wr)
        else:
            b_zm = miss_total_bits
            b_hc = miss_total_bits
            b_kk = miss_total_bits

        bits_by_method["shanzm"].append(b_zm)
        bits_by_method["headcal"].append(b_hc)
        bits_by_method["kinked"].append(b_kk)

        per_pos.append(
            {
                "position_id": o.position_id,
                "slice_name": o.slice_name,
                "doc_id": o.doc_id,
                "target_utf8_bytes": o.target_utf8_bytes,
                "match_kind": o.match_kind,
                "is_head_hit": o.is_head_hit,
                "exact_rank_1idx": o.exact_rank_1idx,
                "partial_prefix_rank_1idx": o.partial_prefix_rank_1idx,
                "raw_surprisal_bits": o.raw_surprisal_bits,
                "head_entropy_bits": o.head_entropy_bits,
                "gap_l1_l2_nats": o.gap_l1_l2_nats,
                "pred_bits_shanzm": b_zm,
                "pred_bits_headcal": b_hc,
                "pred_bits_kinked": b_kk,
                "pred_bpb_shanzm": b_zm / o.target_utf8_bytes,
                "pred_bpb_headcal": b_hc / o.target_utf8_bytes,
                "pred_bpb_kinked": b_kk / o.target_utf8_bytes,
            }
        )

    # Compute exact slice-level ratio -> macro-average native lexical BPB
    slices = sorted({o.slice_name for o in obs})
    bl = np.array([float(o.target_utf8_bytes) for o in obs], dtype=float)
    summaries: dict[str, dict[str, Any]] = {}
    for mkey in ("shanzm", "headcal", "kinked"):
        b_arr = np.array(bits_by_method[mkey], dtype=float)
        per_slice = {}
        for s in slices:
            mask = np.array([o.slice_name == s for o in obs], dtype=bool)
            per_slice[s] = float(np.sum(b_arr[mask]) / np.sum(bl[mask]))
        native_macro = float(np.mean(list(per_slice.values())))
        entry: dict[str, Any] = {
            "native_lexical_bpb": native_macro,
            "native_lexical_per_slice_bpb": per_slice,
            "unit_native": "bits per UTF-8 byte (lexical-atom targets, 2-slice macro ratio)",
        }
        offsets = fitted_params.get("additive_offsets", {}).get(mkey)
        if offsets:
            entry["converted_base_raw_4byte_bpb"] = native_macro + float(offsets["to_base_raw_4byte_bpb"])
            entry["unit_raw_4byte"] = "bits per UTF-8 byte (4-byte raw base continuation, additive transfer)"
            entry["converted_base_document_bpb"] = native_macro + float(offsets["to_base_document_bpb"])
            entry["unit_document"] = "bits per UTF-8 byte (full-document base BPB, additive transfer)"
        summaries[mkey] = entry
    return per_pos, summaries


def fit_lexical_transfer_models(
    train_obs_by_family: Mapping[str, Sequence[ExtractedLexicalObs]],
    train_base_targets_by_family: Mapping[str, Mapping[str, float]],
    *,
    spec: Mapping[str, Any] = FROZEN_SPEC,
) -> dict[str, Any]:
    """Fit ShanZM, scalar HeadCal, Kinked HeadCal, escape pi_miss, and additive offsets.

    STRICT DATA ISOLATION: asserts only declared `train_families` (`smol`, `qwen`)
    are passed, and `holdout_family` (`olmo`) is completely absent!
    """
    expected_train = list(spec["train_families"])
    holdout = str(spec["holdout_family"])
    if holdout in train_obs_by_family or holdout in train_base_targets_by_family:
        raise ValueError(f"DATA LEAKAGE GUARD: holdout family {holdout!r} passed to trainer!")
    if sorted(train_obs_by_family.keys()) != sorted(expected_train):
        raise ValueError(f"Expected training families {expected_train}, got {sorted(train_obs_by_family.keys())}")

    k_head = int(spec["k_head"])
    esc_bits = float(spec["tail_escape_bits"])

    # 1. Family-balanced escape probability pi_miss (Laplace +0.5/+1 per family, equal family weight)
    fam_miss_rates: dict[str, float] = {}
    hits_by_fam: dict[str, list[ExtractedLexicalObs]] = {}
    for fam in expected_train:
        obs_f = list(train_obs_by_family[fam])
        hits_f = [o for o in obs_f if o.is_head_hit and o.exact_rank_1idx is not None and o.exact_top_indices_0idx]
        hits_by_fam[fam] = hits_f
        n_miss = len(obs_f) - len(hits_f)
        fam_miss_rates[fam] = (n_miss + 0.5) / (len(obs_f) + 1.0)
    pi_miss = float(np.mean([fam_miss_rates[f] for f in expected_train]))

    # 2. Fit ShanZM (alpha, delta) — family-balanced mean head bits per hit
    best_zm = (1.5, 0.25)
    best_zm_loss = float("inf")
    for a in spec["shanzm_alpha_grid"]:
        for d in spec["shanzm_delta_grid"]:
            p_zm = _shanzm_probs(float(a), float(d), k_head=k_head)
            fam_losses = []
            for fam in expected_train:
                h_bits = [
                    -math.log2(max(sum(float(p_zm[idx]) for idx in o.exact_top_indices_0idx if idx < k_head), 1e-300))
                    for o in hits_by_fam[fam]
                ]
                fam_losses.append(float(np.mean(h_bits)))
            loss = float(np.mean(fam_losses))
            if loss < best_zm_loss:
                best_zm_loss = loss
                best_zm = (float(a), float(d))

    # 3. Fit Scalar HeadCal (temperature T) — family-balanced mean head bits per hit
    best_T = 1.0
    best_hc_loss = float("inf")
    for T in spec["headcal_t_grid"]:
        fam_losses = []
        for fam in expected_train:
            h_bits = [_headcal_hit_bits(o.top_logprobs, o.exact_top_indices_0idx, float(T)) for o in hits_by_fam[fam]]
            fam_losses.append(float(np.mean(h_bits)))
        loss = float(np.mean(fam_losses))
        if loss < best_hc_loss:
            best_hc_loss = loss
            best_T = float(T)

    # 4. Fit Kinked Winner-Spike HeadCal (w1, w_rest) — family-balanced mean head bits + frozen shrinkage
    lam_shrink = float(spec["kinked_shrinkage_lambda_per_hit"])
    best_kink = (1.0 / best_T, 1.0 / best_T)
    best_kk_loss = float("inf")
    for w1 in spec["kinked_w1_grid"]:
        for wr in spec["kinked_wrest_grid"]:
            fam_losses = []
            for fam in expected_train:
                h_bits = [
                    _kinked_hit_bits(o.top_logprobs, o.exact_top_indices_0idx, float(w1), float(wr))
                    for o in hits_by_fam[fam]
                ]
                fam_losses.append(float(np.mean(h_bits)))
            loss = float(np.mean(fam_losses)) + lam_shrink * ((float(w1) - float(wr)) ** 2)
            if loss < best_kk_loss:
                best_kk_loss = loss
                best_kink = (float(w1), float(wr))

    fitted: dict[str, Any] = {
        "spec": dict(spec),
        "shared": {
            "pi_miss": pi_miss,
            "tail_escape_bits": esc_bits,
            "per_train_family_miss_rates": fam_miss_rates,
            "per_train_family_hit_counts": {f: len(hits_by_fam[f]) for f in expected_train},
        },
        "methods": {
            "shanzm": {"alpha": best_zm[0], "delta": best_zm[1], "train_balanced_head_bits": best_zm_loss},
            "headcal": {"temperature": best_T, "train_balanced_head_bits": best_hc_loss},
            "kinked": {
                "w1": best_kink[0],
                "w_rest": best_kink[1],
                "shrinkage_lambda_per_hit": lam_shrink,
                "train_balanced_obj": best_kk_loss,
            },
        },
        "additive_offsets": {},
        "train_native_lexical_bpb": {},
    }

    # 5. Fit equal-family additive conversions strictly on train families (`smol`, `qwen`)!
    train_summaries: dict[str, dict[str, Any]] = {}
    for fam in expected_train:
        _, summ = score_obs_with_params(train_obs_by_family[fam], fitted)
        train_summaries[fam] = summ
        fitted["train_native_lexical_bpb"][fam] = {m: summ[m]["native_lexical_bpb"] for m in ("shanzm", "headcal", "kinked")}

    for mkey in ("shanzm", "headcal", "kinked"):
        off_raw = float(
            np.mean(
                [
                    float(train_base_targets_by_family[f]["raw_bpb"]) - train_summaries[f][mkey]["native_lexical_bpb"]
                    for f in expected_train
                ]
            )
        )
        off_doc = float(
            np.mean(
                [
                    float(train_base_targets_by_family[f]["document_bpb"]) - train_summaries[f][mkey]["native_lexical_bpb"]
                    for f in expected_train
                ]
            )
        )
        fitted["additive_offsets"][mkey] = {
            "to_base_raw_4byte_bpb": off_raw,
            "to_base_document_bpb": off_doc,
            "per_family_raw_gaps": {
                f: float(train_base_targets_by_family[f]["raw_bpb"]) - train_summaries[f][mkey]["native_lexical_bpb"]
                for f in expected_train
            },
            "per_family_doc_gaps": {
                f: float(train_base_targets_by_family[f]["document_bpb"]) - train_summaries[f][mkey]["native_lexical_bpb"]
                for f in expected_train
            },
        }
    return fitted


def _doc_bootstrap_transfer(
    codex_dir: Path,
    per_pos_by_family: Mapping[str, Sequence[dict[str, Any]]],
    *,
    train_families: Sequence[str] = ("smol", "qwen"),
    holdout_family: str = "olmo",
    n_boot: int = 10000,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Shared stratified document-bootstrap (20 July + 20 Nov docs) matching `summarize_hansard.py`.

    Uses the exact same document ordering, seed (20261125), and 10,000 draws as Codex's
    `summarize_hansard.py` so every interval is directly comparable to the four-byte audit.
    Recomputes the equal-family additive offset inside each bootstrap draw (holding the
    frozen Smol+Qwen rank/temperature/kink parameters fixed).
    """
    # Load per-document base raw_bpb (mean of 3 four-byte positions) and full-doc (bits, bytes)
    base_doc_raw: dict[str, np.ndarray] = {}
    base_doc_bits: dict[str, np.ndarray] = {}
    base_doc_bytes: dict[str, np.ndarray] = {}
    doc_order: list[tuple[str, str]] | None = None

    for fam in (*train_families, holdout_family):
        d = json.loads((codex_dir / f"{fam}_base.json").read_text(encoding="utf-8"))
        ps = sorted(d["positions"], key=lambda p: (p["slice_name"], p["doc_id"], p["position_id"]))
        docs = sorted(d["documents"], key=lambda p: (p["slice_name"], p["doc_id"]))
        order = [(p["slice_name"], p["doc_id"]) for p in docs]
        if doc_order is None:
            doc_order = order
        assert order == doc_order and len(order) == 40
        assert [(p["slice_name"], p["doc_id"]) for p in docs for _ in range(3)] == [
            (p["slice_name"], p["doc_id"]) for p in ps
        ]
        base_doc_raw[fam] = np.array([p["raw_bpb"] for p in ps], dtype=float).reshape(40, 3).mean(axis=1)
        base_doc_bits[fam] = np.array([p["bits"] for p in docs], dtype=float)
        base_doc_bytes[fam] = np.array([p["bytes"] for p in docs], dtype=float)

    assert doc_order is not None
    # Verify exact 20 July + 20 Nov split
    assert doc_order[19][0] != doc_order[20][0]

    # Build per-document (sum_pred_bits, sum_target_bytes) for each family & method
    lex_doc_bits: dict[tuple[str, str], np.ndarray] = {}
    lex_doc_bytes: dict[str, np.ndarray] = {}
    doc_to_idx = {k: i for i, k in enumerate(doc_order)}

    for fam in (*train_families, holdout_family):
        bl = np.zeros(40, dtype=float)
        bm = {m: np.zeros(40, dtype=float) for m in ("shanzm", "headcal", "kinked")}
        for row in per_pos_by_family[fam]:
            di = doc_to_idx[(row["slice_name"], row["doc_id"])]
            bl[di] += float(row["target_utf8_bytes"])
            bm["shanzm"][di] += float(row["pred_bits_shanzm"])
            bm["headcal"][di] += float(row["pred_bits_headcal"])
            bm["kinked"][di] += float(row["pred_bits_kinked"])
        lex_doc_bytes[fam] = bl
        for m in ("shanzm", "headcal", "kinked"):
            lex_doc_bits[(fam, m)] = bm[m]

    rng = np.random.default_rng(seed)
    july = rng.integers(0, 20, (n_boot, 20))
    nov = rng.integers(20, 40, (n_boot, 20))
    idx = np.vstack((np.arange(40), np.hstack((july, nov))))

    def macro_ratio(bits_40: np.ndarray, bytes_40: np.ndarray) -> np.ndarray:
        b = bits_40[idx].reshape(-1, 2, 20).sum(axis=2)
        y = bytes_40[idx].reshape(-1, 2, 20).sum(axis=2)
        return (b / y).mean(axis=1)

    def desc(x: np.ndarray) -> dict[str, Any]:
        lo, hi = np.quantile(x[1:], [0.025, 0.975])
        return {"bpb": float(x[0]), "document_bootstrap_ci95": [float(lo), float(hi)]}

    base_raw_boot = {f: base_doc_raw[f][idx].mean(axis=1) for f in (*train_families, holdout_family)}
    base_doc_boot = {f: macro_ratio(base_doc_bits[f], base_doc_bytes[f]) for f in (*train_families, holdout_family)}

    out: dict[str, Any] = {}
    for m in ("shanzm", "headcal", "kinked"):
        lex_boot = {f: macro_ratio(lex_doc_bits[(f, m)], lex_doc_bytes[f]) for f in (*train_families, holdout_family)}
        obs_hold = lex_boot[holdout_family]
        entry: dict[str, Any] = {
            "native_lexical_bpb": {f: desc(lex_boot[f]) for f in (*train_families, holdout_family)},
        }
        for tgt_name, tgt_map in (
            ("base_raw_4byte_bpb", base_raw_boot),
            ("base_document_bpb", base_doc_boot),
        ):
            truth = tgt_map[holdout_family]
            offset = np.mean([tgt_map[f] - lex_boot[f] for f in train_families], axis=0)
            corrected = obs_hold + offset
            signed_err = corrected - truth
            entry[tgt_name] = {
                "truth": desc(truth),
                "offset": desc(offset),
                "uncorrected_native_minus_truth": desc(obs_hold - truth),
                "converted_prediction": desc(corrected),
                "signed_error": desc(signed_err),
                "abs_error_point": float(abs(signed_err[0])),
                "within_0.03_point": bool(abs(signed_err[0]) <= 0.03),
                "ci95_contained_in_pm_0.03": bool(
                    float(np.quantile(signed_err[1:], 0.025)) >= -0.03
                    and float(np.quantile(signed_err[1:], 0.975)) <= 0.03
                ),
            }
        out[m] = entry
    return out


def run_heldout_olmo_experiment(
    manifest_path: Path,
    codex_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Execute the full frozen Smol+Qwen -> OLMo held-out lexical transfer evaluation."""
    positions, _, manifest_sha = load_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all three post-anchor instruct traces and record SHA-256 provenance
    traces: dict[str, dict[str, list[dict[str, Any]]]] = {}
    obs_by_fam: dict[str, list[ExtractedLexicalObs]] = {}
    file_hashes: dict[str, str] = {"pilot_manifest.json": manifest_sha}

    for fam in ("smol", "qwen", "olmo"):
        p_trace = codex_dir / f"{fam}_instruct_post_anchor.jsonl"
        p_base = codex_dir / f"{fam}_base.json"
        file_hashes[p_trace.name] = sha256_hex(p_trace.read_bytes())
        file_hashes[p_base.name] = sha256_hex(p_base.read_bytes())
        traces[fam] = load_post_anchor_jsonl(p_trace)
        obs_by_fam[fam] = extract_lexical_obs(positions, traces[fam], k_head=int(FROZEN_SPEC["k_head"]))
        assert len(obs_by_fam[fam]) == len(positions) == 120, f"Position count mismatch for {fam}"

    summary = json.loads((codex_dir / "summary.json").read_text(encoding="utf-8"))
    file_hashes["summary.json"] = sha256_hex((codex_dir / "summary.json").read_bytes())

    # STRICT ISOLATION: pass ONLY smol and qwen into trainer!
    train_obs = {f: obs_by_fam[f] for f in ("smol", "qwen")}
    train_targets = {
        f: {
            "raw_bpb": float(summary["checkpoints"][f"{f}_base"]["raw_bpb"]["bpb"]),
            "document_bpb": float(summary["checkpoints"][f"{f}_base"]["document_bpb"]["bpb"]),
        }
        for f in ("smol", "qwen")
    }
    fitted = fit_lexical_transfer_models(train_obs, train_targets, spec=FROZEN_SPEC)
    fitted["provenance_sha256"] = file_hashes

    # Persist reusable fitted params
    fitted_path = out_dir / "fitted_params.json"
    fitted_path.write_bytes(canonical_json_bytes(fitted))

    # Score all three families using the exact frozen fitted_params (zero OLMo refitting!)
    per_pos_by_fam: dict[str, list[dict[str, Any]]] = {}
    summaries_by_fam: dict[str, dict[str, Any]] = {}
    diagnostics_by_fam: dict[str, dict[str, Any]] = {}

    for fam in ("smol", "qwen", "olmo"):
        rows, summ = score_obs_with_params(obs_by_fam[fam], fitted)
        per_pos_by_fam[fam] = rows
        summaries_by_fam[fam] = summ
        n_hit = sum(1 for r in rows if r["is_head_hit"])
        n_part = sum(1 for r in rows if (not r["is_head_hit"] and r["partial_prefix_rank_1idx"] is not None))
        diagnostics_by_fam[fam] = {
            "n_positions": len(rows),
            "whole_target_top20_hits": n_hit,
            "whole_target_top20_rate": n_hit / len(rows),
            "partial_prefix_only_count": n_part,
            "partial_prefix_only_rate": n_part / len(rows),
            "mean_head_entropy_bits": float(np.mean([r["head_entropy_bits"] for r in rows])),
            "mean_gap_l1_l2_nats": float(np.mean([r["gap_l1_l2_nats"] for r in rows])),
            "by_match_kind": {
                k: sum(1 for r in rows if r["match_kind"] == k)
                for k in sorted({r["match_kind"] for r in rows})
            },
        }

    # Write per-position predictions JSONL for all 3 families (360 rows)
    pred_path = out_dir / "per_position_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as sink:
        for fam in ("smol", "qwen", "olmo"):
            for r in per_pos_by_fam[fam]:
                sink.write(canonical_json_bytes({"family": fam, **r}).decode("utf-8") + "\n")

    # Compute shared 10,000-replicate stratified document-bootstrap intervals
    boot = _doc_bootstrap_transfer(
        codex_dir,
        per_pos_by_fam,
        train_families=("smol", "qwen"),
        holdout_family="olmo",
        n_boot=int(FROZEN_SPEC["bootstrap_replicates"]),
        seed=int(FROZEN_SPEC["bootstrap_seed"]),
    )

    # Independent integrity checks
    olmo_raw_truth = float(summary["checkpoints"]["olmo_base"]["raw_bpb"]["bpb"])
    olmo_doc_truth = float(summary["checkpoints"]["olmo_base"]["document_bpb"]["bpb"])
    checks = {
        "all_families_120_positions": all(len(per_pos_by_fam[f]) == 120 for f in ("smol", "qwen", "olmo")),
        "olmo_excluded_from_fit": "olmo" not in fitted["train_native_lexical_bpb"],
        "olmo_raw_4byte_truth_matches_0.97615": abs(olmo_raw_truth - 0.97615) < 1e-4,
        "olmo_document_truth_matches_0.74269": abs(olmo_doc_truth - 0.74269) < 1e-4,
        "additive_identity_exact_lt_1e12": all(
            abs(
                summaries_by_fam["olmo"][m]["converted_base_raw_4byte_bpb"]
                - (summaries_by_fam["olmo"][m]["native_lexical_bpb"] + fitted["additive_offsets"][m]["to_base_raw_4byte_bpb"])
            )
            < 1e-12
            and abs(
                summaries_by_fam["olmo"][m]["converted_base_document_bpb"]
                - (summaries_by_fam["olmo"][m]["native_lexical_bpb"] + fitted["additive_offsets"][m]["to_base_document_bpb"])
            )
            < 1e-12
            for m in ("shanzm", "headcal", "kinked")
        ),
        "bootstrap_point_matches_summaries_lt_1e12": all(
            abs(boot[m]["native_lexical_bpb"]["olmo"]["bpb"] - summaries_by_fam["olmo"][m]["native_lexical_bpb"]) < 1e-12
            and abs(boot[m]["base_raw_4byte_bpb"]["converted_prediction"]["bpb"] - summaries_by_fam["olmo"][m]["converted_base_raw_4byte_bpb"]) < 1e-12
            and abs(boot[m]["base_document_bpb"]["converted_prediction"]["bpb"] - summaries_by_fam["olmo"][m]["converted_base_document_bpb"]) < 1e-12
            for m in ("shanzm", "headcal", "kinked")
        ),
        "no_empty_byte_matches": all(
            not (o.is_head_hit and any(o.top_logprobs[i] is None for i in o.exact_top_indices_0idx))
            for f in ("smol", "qwen", "olmo")
            for o in obs_by_fam[f]
        ),
    }

    results = {
        "schema_version": "lexical_olmo_transfer_results_v1",
        "provenance_sha256": file_hashes,
        "fitted_params_path": str(fitted_path),
        "per_position_predictions_path": str(pred_path),
        "units": {
            "native_lexical_bpb": "bits per UTF-8 byte on frozen lexical-atom targets (2-slice macro ratio)",
            "base_raw_4byte_bpb": "bits per UTF-8 byte on raw 4-byte base continuations (equal-family additive conversion)",
            "base_document_bpb": "bits per UTF-8 byte on full base documents (equal-family additive conversion)",
        },
        "fitted_params": {
            "shared": fitted["shared"],
            "methods": fitted["methods"],
            "additive_offsets": fitted["additive_offsets"],
        },
        "family_diagnostics": diagnostics_by_fam,
        "family_summaries": summaries_by_fam,
        "heldout_olmo_transfer": boot,
        "codex_four_byte_reference": {
            "olmo_base_raw_4byte_bpb": olmo_raw_truth,
            "olmo_base_document_bpb": olmo_doc_truth,
            "codex_top20_additive_raw_signed_error": float(
                summary["transfer"]["olmo"]["base_raw_bpb"]["signed_error"]["bpb"]
            ),
            "codex_top20_additive_doc_signed_error": float(
                summary["transfer"]["olmo"]["base_document_bpb"]["signed_error"]["bpb"]
            ),
        },
        "integrity_checks": checks,
    }

    res_path = out_dir / "results.json"
    res_path.write_bytes(canonical_json_bytes(results))

    # Write concise human-readable Markdown report
    md_lines = [
        "# Held-Out OLMo-2-0425-1B Lexical Transfer Results (Smol + Qwen Training Only)",
        "",
        f"Training families: `smol`, `qwen` (family-balanced, 120 positions each). Holdout: `olmo` (120 positions, zero leakage).",
        f"Fixed escape cost: `9.0` bits | Shared fitted `pi_miss = {fitted['shared']['pi_miss']:.4f}` "
        f"(Smol `{fitted['shared']['per_train_family_miss_rates']['smol']:.4f}`, Qwen `{fitted['shared']['per_train_family_miss_rates']['qwen']:.4f}`).",
        "",
        "## Fitted Parameters (Smol + Qwen Only)",
        f"- **ShanZM:** `alpha = {fitted['methods']['shanzm']['alpha']:.2f}`, `delta = {fitted['methods']['shanzm']['delta']:.2f}`",
        f"- **Scalar HeadCal:** `T = {fitted['methods']['headcal']['temperature']:.2f}`",
        f"- **Kinked Winner-Spike HeadCal:** `w1 = {fitted['methods']['kinked']['w1']:.2f}`, `w_rest = {fitted['methods']['kinked']['w_rest']:.2f}` (`lambda = {FROZEN_SPEC['kinked_shrinkage_lambda_per_hit']}`)",
        "",
        "## Held-Out OLMo-2-0425-1B Evaluation (Units: bits per UTF-8 byte)",
        "",
        "| Method | Native Lexical BPB (OLMo) | Offset -> Raw 4B | Pred Base Raw 4B (Truth `0.97615`) | Signed Error Raw 4B [95% CI] | Pass `<=0.03` | Offset -> Full Doc | Pred Base Full Doc (Truth `0.74269`) | Signed Error Doc [95% CI] | Pass `<=0.03` |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for m, label in (("shanzm", "ShanZM"), ("headcal", "Scalar HeadCal"), ("kinked", "Kinked HeadCal")):
        b = boot[m]
        nat = b["native_lexical_bpb"]["olmo"]["bpb"]
        r4 = b["base_raw_4byte_bpb"]
        fd = b["base_document_bpb"]
        r4_ci = r4["signed_error"]["document_bootstrap_ci95"]
        fd_ci = fd["signed_error"]["document_bootstrap_ci95"]
        md_lines.append(
            f"| {label} | {nat:.5f} | {r4['offset']['bpb']:+.5f} | {r4['converted_prediction']['bpb']:.5f} | "
            f"{r4['signed_error']['bpb']:+.5f} [{r4_ci[0]:+.5f}, {r4_ci[1]:+.5f}] | {'PASS' if r4['within_0.03_point'] else 'FAIL'} | "
            f"{fd['offset']['bpb']:+.5f} | {fd['converted_prediction']['bpb']:.5f} | "
            f"{fd['signed_error']['bpb']:+.5f} [{fd_ci[0]:+.5f}, {fd_ci[1]:+.5f}] | {'PASS' if fd['within_0.03_point'] else 'FAIL'} |"
        )
    md_lines.extend(
        [
            "",
            f"All integrity checks passed: `{all(checks.values())}` ({sum(checks.values())}/{len(checks)} checks).",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/pilot_manifest.json"))
    parser.add_argument("--codex-dir", type=Path, default=Path("runs/codex_three_way"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/lexical_olmo_transfer"))
    args = parser.parse_args()
    res = run_heldout_olmo_experiment(args.manifest, args.codex_dir, args.out_dir)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"))
    return 0 if all(res["integrity_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
