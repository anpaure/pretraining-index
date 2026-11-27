"""Automated answer-pattern & bias health tests across ALL live models in anpaure/pretraining-index.

Guards against pathological zero-CoT response patterns that silently skew AUC / BPB negatively:
  1. Verbatim prompt-example copying ("C,A,F,B,H,D,E,G")
  2. Literal alphabetical ("A,B,C,D,E,F,G,H") or reverse-alphabetical ("H,G,F,E,D,C,B,A") dumping
  3. Tail-alphabetical dumping (positions 2..8 or 3..8 strictly sorted alphabetically)
  4. Single-permutation collapse (any one permutation repeated excessively)
  5. Severe slot-1 / primacy-recency bias (single letter picked #1 > 35% vs 12.5% uniform,
     or |Pearson r(gold_idx, gold_rank)| >= 0.35)
  6. Rank inversion (overall AUC <= 0.50 or tail AUC over ranks 2..8 <= 0.50)

Known, audited exceptions on weaker/older/bypass/quirky models are explicitly pinned in
KNOWN_PATHOLOGY_EXCEPTIONS so any NEW model exhibiting a bad pattern fails CI immediately.
"""

from __future__ import annotations

import json
import re
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from build_anpaure_index_v1 import BENCH_PATH, MODEL_META, load_traces

EX_PERM = "C,A,F,B,H,D,E,G"
EX_POS = [list("CAFBHDEG").index(chr(65 + i)) for i in range(8)]
PAIRS28 = list(combinations(range(8), 2))
ALPHA_PERM = "A,B,C,D,E,F,G,H"
RALPHA_PERM = "H,G,F,E,D,C,B,A"

# Clean thresholds for newly added / healthy models (14 automated pathology & paranormal checks)
THRESHOLDS = {
    "ex_copy_pct": 2.0,    # verbatim prompt example echo
    "near_ex_pct": 6.0,    # near-example echo (Kendall tau >= 0.70 to prompt example)
    "alpha_pct": 5.0,      # literal A,B,C,D,E,F,G,H
    "ralpha_pct": 2.0,     # literal H,G,F,E,D,C,B,A
    "tail28_pct": 8.0,     # ranks 2..8 strictly alphabetical
    "tail38_pct": 12.0,    # ranks 3..8 strictly alphabetical
    "max_perm_pct": 5.0,   # any single permutation repeated
    "slot1_max_pct": 35.0, # single letter picked #1 (uniform = 12.5%)
    "slot8_max_pct": 45.0, # single letter dumped #8 (uniform = 12.5%)
    "abs_pos_corr": 0.35,  # |r(gold_idx, gold_rank)|
    "abs_len_corr": 0.15,  # |r(candidate_length, assigned_rank)| (length shortcut check)
    "sticky_p": 2.0,       # % consecutive items emitting identical permutation
    "drop_pct": 2.5,       # % missing/dropped items (>2.5% = survivorship bias risk)
}

# Explicitly audited & documented exceptions (model_id -> set of allowed flagged metrics).
# Adding a new model that triggers ANY threshold above will fail unittest until investigated.
KNOWN_PATHOLOGY_EXCEPTIONS: dict[str, set[str]] = {
    "reflection/beam-1.0-501b-a23b":   {"ex_copy_pct"},                            # 4.5% example copy (template bypass)
    "nvidia/nemotron-3-nano-30b-a3b":  {"ex_copy_pct", "alpha_pct", "tail28_pct", "tail38_pct", "max_perm_pct", "slot8_max_pct"},
    "nvidia/nemotron-3.5-super-vl":    {"alpha_pct", "tail28_pct", "max_perm_pct", "abs_pos_corr"},
    "ricursiveai/ripple-65b-preview":  {"tail28_pct", "abs_pos_corr"},
    "qwen/qwen3.6-plus":               {"tail38_pct", "slot8_max_pct"},            # 17.2% ranks 3..8 sorted, 47.1% H last
    "qwen/qwen3.7-plus":               {"tail38_pct"},                             # 15.1% ranks 3..8 sorted
    "meta/muse-spark-open":            {"slot1_max_pct"},                          # 40.3% slot G #1 (ATEM bypass)
    "thinkingmachines/inkling":        {"slot1_max_pct", "slot8_max_pct", "abs_pos_corr"}, # 46.2% A #1, 48.6% H #8
    "openai/gpt-4o":                   {"slot1_max_pct"},                          # 38.9% slot A #1
    "openai/gpt-4o-mini":              {"slot1_max_pct", "abs_pos_corr"},          # 38.6% slot A #1
    "openai/gpt-oss-120b":             {"slot1_max_pct"},                          # 36.2% slot A #1 (template bypass)
    "openai/gpt-oss-20b":              {"alpha_pct", "max_perm_pct", "abs_pos_corr"},
    "minimax/minimax-m3":              {"slot1_max_pct"},                          # 59.6% slot A #1 primacy bias
    "mistralai/mistral-small-2603":    {"ralpha_pct"},                             # 3.16% reverse-alphabetical H..A
    "nvidia/nemotron-3.5-lightning":   {"alpha_pct", "max_perm_pct", "slot1_max_pct"}, # 6.9% alpha, 43.2% slot C #1
    "stepfun/step-5-preview":          {"drop_pct"},                               # 880/954 (7.8% dropped formatting)
    "qwen/qwen3.8-2.4t-a95b":          {"drop_pct"},                               # 884/954 (7.3% dropped prose on bypass)
}

KNOWN_SUBCHANCE_DOMAIN_EXCEPTIONS = {
    "openai/gpt-oss-120b",  # BYPASS model: essays=0.4899
}


def compute_model_health(model_id: str, items_by_id: dict[str, dict]) -> dict | None:
    clean, _ = load_traces(model_id, set(items_by_id))
    if not clean:
        return None
    n = len(clean)
    n_total = len(items_by_id)
    perms: list[str] = []
    top1_slots: list[str] = []
    slot8_list: list[str] = []
    gold_idxs: list[int] = []
    gold_ranks: list[int] = []
    tail28 = 0
    tail38 = 0
    near_ex = 0
    sticky = 0
    prev_p: str | None = None
    tail_aucs: list[float] = []
    len_corrs: list[float] = []
    dom_aucs: dict[str, list[float]] = {}
    for r in clean:
        it = items_by_id[r["item_id"]]
        lets = re.findall(r"[A-H]", (r.get("raw") or "").upper())
        p = ",".join(lets)
        perms.append(p)
        top1_slots.append(lets[0])
        slot8_list.append(lets[-1])
        if prev_p is not None and p == prev_p:
            sticky += 1
        prev_p = p
        pos = [0] * 8
        for rk, ch in enumerate(lets):
            pos[ord(ch) - 65] = rk + 1
        et = sum(1 if (pos[i] < pos[j]) == (EX_POS[i] < EX_POS[j]) else -1 for i, j in PAIRS28) / 28.0
        if et >= 0.70:
            near_ex += 1
        clen = np.array([len(c) for c in it["candidates"]], dtype=float)
        if np.std(clen) > 0:
            len_corrs.append(float(np.corrcoef(clen, pos)[0, 1]))
        gi = it["gold_idx"]
        gr = r["gold_rank"]
        gold_idxs.append(gi)
        gold_ranks.append(gr)
        if lets[1:] == sorted(lets[1:]):
            tail28 += 1
        if lets[2:] == sorted(lets[2:]):
            tail38 += 1
        if gr > 1:
            tail_aucs.append((8.0 - gr) / 6.0)
        dom_aucs.setdefault(r["domain"], []).append((8.0 - gr) / 7.0)

    cnt = Counter(perms)
    top_perm, top_perm_n = cnt.most_common(1)[0]
    scnt = Counter(top1_slots)
    top_slot, top_slot_n = scnt.most_common(1)[0]
    s8cnt = Counter(slot8_list)
    s8_slot, s8_n = s8cnt.most_common(1)[0]
    corr = float(np.corrcoef(gold_idxs, gold_ranks)[0, 1])
    auc = float(np.mean([(8.0 - gr) / 7.0 for gr in gold_ranks]))
    tail_auc = float(np.mean(tail_aucs)) if tail_aucs else 0.5
    min_dom = min(dom_aucs, key=lambda d: float(np.mean(dom_aucs[d])))
    min_dom_auc = float(np.mean(dom_aucs[min_dom]))
    mean_len_corr = float(np.mean(len_corrs)) if len_corrs else 0.0
    return {
        "model_id": model_id,
        "n": n,
        "drop_pct": 100.0 * (n_total - n) / n_total,
        "auc": auc,
        "tail_auc": tail_auc,
        "min_dom": min_dom,
        "min_dom_auc": min_dom_auc,
        "ex_copy_pct": 100.0 * cnt[EX_PERM] / n,
        "near_ex_pct": 100.0 * near_ex / n,
        "alpha_pct": 100.0 * cnt[ALPHA_PERM] / n,
        "ralpha_pct": 100.0 * cnt[RALPHA_PERM] / n,
        "tail28_pct": 100.0 * tail28 / n,
        "tail38_pct": 100.0 * tail38 / n,
        "max_perm_pct": 100.0 * top_perm_n / n,
        "uniq_perm_pct": 100.0 * len(cnt) / n,
        "top_perm": top_perm,
        "slot1_max_pct": 100.0 * top_slot_n / n,
        "top_slot": top_slot,
        "slot8_max_pct": 100.0 * s8_n / n,
        "slot8": s8_slot,
        "pos_corr": corr,
        "abs_pos_corr": abs(corr),
        "len_corr": mean_len_corr,
        "abs_len_corr": abs(mean_len_corr),
        "sticky_p": 100.0 * sticky / max(n - 1, 1),
    }


class TestTraceAnswerHealth(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
        cls.items_by_id = {it["item_id"]: it for it in bench["items"]}
        cls.health: dict[str, dict] = {}
        for mid in MODEL_META:
            h = compute_model_health(mid, cls.items_by_id)
            if h is not None:
                cls.health[mid] = h

    def test_all_live_models_above_chance_overall_and_tail(self) -> None:
        """No model may exhibit inverted ranking overall (AUC <= 0.50) or in the tail (ranks 2..8 <= 0.50)."""
        self.assertGreaterEqual(len(self.health), 80)
        for mid, h in self.health.items():
            self.assertGreater(h["auc"], 0.50, f"{mid} overall AUC inverted: {h['auc']:.4f}")
            self.assertGreater(h["tail_auc"], 0.50, f"{mid} tail AUC (ranks 2..8) inverted: {h['tail_auc']:.4f}")

    def test_no_unexpected_answer_pathologies(self) -> None:
        """Every live model must pass all 8 answer-pattern checks unless explicitly audited & allowlisted."""
        unexpected: list[str] = []
        for mid, h in sorted(self.health.items()):
            allowed = KNOWN_PATHOLOGY_EXCEPTIONS.get(mid, set())
            for metric, limit in THRESHOLDS.items():
                val = h[metric]
                if val >= limit and metric not in allowed:
                    unexpected.append(f"{mid}: {metric}={val:.2f} >= {limit} (top_perm={h['top_perm']}, slot1={h['top_slot']})")
        self.assertEqual(unexpected, [], "Unexpected answer-pattern pathologies detected:\n  " + "\n  ".join(unexpected))

    def test_allowlist_entries_are_current_not_stale(self) -> None:
        """Every model & metric in KNOWN_PATHOLOGY_EXCEPTIONS must still actually exceed its threshold."""
        stale: list[str] = []
        for mid, metrics in KNOWN_PATHOLOGY_EXCEPTIONS.items():
            self.assertIn(mid, self.health, f"Allowlisted model {mid} missing from live traces")
            h = self.health[mid]
            for m in metrics:
                if h[m] < THRESHOLDS[m]:
                    stale.append(f"{mid}: {m}={h[m]:.2f} < {THRESHOLDS[m]}")
        self.assertEqual(stale, [], "Stale allowlist entries (clean them up):\n  " + "\n  ".join(stale))

    def test_top25_frontier_models_are_100pct_pathology_free(self) -> None:
        """The top-25 models on the leaderboard must have ZERO answer-pattern anomalies."""
        top25 = sorted(self.health.values(), key=lambda x: -x["auc"])[:25]
        bad = []
        for h in top25:
            for metric, limit in THRESHOLDS.items():
                if h[metric] >= limit:
                    bad.append(f"{h['model_id']} (AUC={h['auc']:.4f}): {metric}={h[metric]:.2f}")
        self.assertEqual(bad, [], "Top-25 frontier model flagged for answer pathology:\n  " + "\n  ".join(bad))

    def test_no_single_domain_collapse_or_low_diversity(self) -> None:
        """Every non-allowlisted model must beat chance on ALL 5 domains individually and emit >=50% unique perms."""
        bad_dom, low_div = [], []
        for mid, h in sorted(self.health.items()):
            if h["min_dom_auc"] <= 0.50 and mid not in KNOWN_SUBCHANCE_DOMAIN_EXCEPTIONS:
                bad_dom.append(f"{mid}: {h['min_dom']}={h['min_dom_auc']:.4f}")
            if h["uniq_perm_pct"] < 50.0:
                low_div.append(f"{mid}: uniq_perm_pct={h['uniq_perm_pct']:.1f}%")
        self.assertEqual(bad_dom, [], "Single-domain collapse detected:\n  " + "\n  ".join(bad_dom))
        self.assertEqual(low_div, [], "Extreme permutation diversity collapse:\n  " + "\n  ".join(low_div))


if __name__ == "__main__":
    unittest.main()
