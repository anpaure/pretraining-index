"""Unit tests for real_bpb.scoring & real_bpb.tinker_bpb (zero GPU / network required)."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from real_bpb.scoring import (
    LN2,
    aggregate_anpi_eval_records,
    aggregate_bpb_records,
    compute_scored_utf8_bytes,
    exact_conditional_logprob_from_joint,
    load_anpi_bench_doc_ids,
    load_anpi_bench_items,
    load_corpus_docs,
    plan_strided_windows,
    select_best_temperature,
    sum_vllm_window_nll,
)

from real_bpb.tinker_bpb import (
    TINKER_PRESETS,
    exact_conditional_logprob_tinker,
    sum_tinker_logprobs,
)

ROOT = Path(__file__).resolve().parent.parent


class _FakeLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class TestRealBpbScoring(unittest.TestCase):
    def test_strided_windows_cover_every_target_token_exactly_once(self) -> None:
        for doc_len in (2, 5, 16, 17, 32, 33, 100):
            for max_len, stride in ((16, 8), (16, 5), (16, 15)):
                ids = list(range(1000, 1000 + doc_len))
                wins = plan_strided_windows(ids, max_len=max_len, stride=stride, has_bos=True)
                scored_positions: list[int] = []
                for w in wins:
                    self.assertGreaterEqual(w.score_from, 1)
                    self.assertLessEqual(len(w.token_ids), max_len)
                    self.assertEqual(list(w.token_ids), ids[w.doc_start : w.doc_end])
                    for local_pos in range(w.score_from, len(w.token_ids)):
                        scored_positions.append(w.doc_start + local_pos)
                self.assertEqual(
                    scored_positions,
                    list(range(1, doc_len)),
                    f"Failed exact coverage for doc_len={doc_len}, max_len={max_len}, stride={stride}",
                )

    def test_utf8_byte_accounting_with_and_without_bos(self) -> None:
        text = "café — naïve"
        utf8_len = len(text.encode("utf-8"))
        self.assertEqual(compute_scored_utf8_bytes(text, has_bos=True), utf8_len)
        self.assertEqual(
            compute_scored_utf8_bytes(text, has_bos=False, first_token_text="café"),
            utf8_len - len("café".encode("utf-8")),
        )

    def test_vllm_logprob_reducer_matches_exact_nll(self) -> None:
        tids = [1, 42, 99, 7]
        lps = [
            None,
            {42: _FakeLogprob(-1.5)},
            {99: _FakeLogprob(-0.5)},
            {7: _FakeLogprob(-2.0)},
        ]
        # Score from index 1 -> sums 1.5 + 0.5 + 2.0 = 4.0
        self.assertAlmostEqual(sum_vllm_window_nll(lps, tids, score_from=1), 4.0, places=9)
        # Score from index 2 (overlapping window) -> sums 0.5 + 2.0 = 2.5
        self.assertAlmostEqual(sum_vllm_window_nll(lps, tids, score_from=2), 2.5, places=9)

    def test_aggregate_bpb_and_anpi_954_subset_and_ci(self) -> None:
        per_doc = [
            {"doc_id": "d1", "domain": "news", "url": "https://a.example/1", "bytes": 100, "scored_tokens": 25, "nll": 50.0 * LN2},
            {"doc_id": "d2", "domain": "code", "url": "https://b.example/2", "bytes": 200, "scored_tokens": 50, "nll": 160.0 * LN2},
            {"doc_id": "d3", "domain": "code", "url": "https://b.example/2", "bytes": 100, "scored_tokens": 25, "nll": 80.0 * LN2},
        ]
        agg = aggregate_bpb_records(per_doc, anpi_doc_ids={"d1", "d2"}, n_boot=200, seed=7)
        # Total nll = 290 * ln2, total bytes = 400 -> BPB = 0.725
        self.assertAlmostEqual(agg["bpb"], 0.725, places=6)
        self.assertEqual(agg["n_unique_urls"], 2)
        self.assertAlmostEqual(agg["domains"]["news"]["bpb"], 0.50, places=6)
        self.assertAlmostEqual(agg["domains"]["code"]["bpb"], 0.80, places=6)
        self.assertAlmostEqual(agg["macro_domain_bpb"], 0.65, places=6)
        self.assertIsNotNone(agg["bpb_ci95"])
        self.assertLessEqual(agg["bpb_ci95"][0], agg["bpb"])
        self.assertGreaterEqual(agg["bpb_ci95"][1], agg["bpb"])
        # Subset {d1, d2}: (50 + 160) / 300 = 0.70
        self.assertAlmostEqual(agg["anpi_954_subset"]["bpb"], 0.70, places=6)

    def test_temperature_selection(self) -> None:
        res = select_best_temperature([0.8, 1.0, 1.2], [75.0 * LN2, 70.0 * LN2, 62.0 * LN2], total_bytes=100)
        self.assertEqual(res["t_opt"], 1.2)
        self.assertAlmostEqual(res["bpb_topt"], 0.62, places=6)

    def test_exact_conditional_logprob_handles_bpe_boundary_merge(self) -> None:
        # Prefix ends in trailing space token 99 (lp=-0.5); when concatenated with cand,
        # BPE merges space+word into token 200 (lp=-1.8) followed by token 201 (lp=-0.7).
        # Exact ln p(cand | prefix) = ln p(full) - ln p(prefix) = (-1.8 - 0.7) - (-0.5) = -2.0
        p_ids = [1, 10, 99]
        p_lps = [None, {10: _FakeLogprob(-1.0)}, {99: _FakeLogprob(-0.5)}]
        f_ids = [1, 10, 200, 201]
        f_lps = [None, {10: _FakeLogprob(-1.0)}, {200: _FakeLogprob(-1.8)}, {201: _FakeLogprob(-0.7)}]
        lp = exact_conditional_logprob_from_joint(p_ids, p_lps, f_ids, f_lps)
        self.assertAlmostEqual(lp, -2.0, places=9)

    def test_aggregate_anpi_eval_records(self) -> None:
        per_item = [
            {
                "item_id": "i1",
                "doc_id": "d1",
                "domain": "news",
                "url": "https://a.example/1",
                "gold_idx": 2,
                "gold_bytes": 10,
                "cand_bytes": [10, 10, 10, 10],
                "cand_logprobs": [-9.0, -8.0, -5.0 * LN2, -7.0],  # gold rank 1 -> AUC 1.0, BPB 0.5
            },
            {
                "item_id": "i2",
                "doc_id": "d2",
                "domain": "code",
                "url": "https://b.example/2",
                "gold_idx": 0,
                "gold_bytes": 10,
                "cand_bytes": [10, 10, 10, 10],
                "cand_logprobs": [-7.0 * LN2, -4.0, -9.0, -10.0],  # gold rank 2 of 4 -> AUC 2/3, BPB 0.7
            },
        ]
        agg = aggregate_anpi_eval_records(per_item, n_boot=200, seed=7)
        self.assertAlmostEqual(agg["gold_span_bpb"], 0.60, places=6)
        self.assertAlmostEqual(agg["whitebox_auc"], (1.0 + 2.0 / 3.0) / 2.0, places=6)
        self.assertAlmostEqual(agg["whitebox_top1_pct"], 50.0, places=3)
        self.assertAlmostEqual(agg["whitebox_mrr"], 0.75, places=6)

    def test_tinker_logprob_reducers_match_exact_bpe_chain_rule(self) -> None:
        # Prefix ends in token 99 (lp=-0.5); full replaces token 99 with 200 (-1.8) + 201 (-0.7)
        # Exact ln p(cand | prefix) = (-1.8 - 0.7) - (-0.5) = -2.0
        p_ids = [1, 10, 99]
        p_lps = [None, None, -0.5]
        f_ids = [1, 10, 200, 201]
        f_lps = [None, None, -1.8, -0.7]
        self.assertAlmostEqual(
            exact_conditional_logprob_tinker(p_ids, p_lps, f_ids, f_lps),
            -2.0,
            places=9,
        )
        # No BPE boundary shift -> prefix_lps not needed
        f_ids2 = [1, 10, 99, 300]
        f_lps2 = [None, None, None, -1.25]
        self.assertAlmostEqual(
            exact_conditional_logprob_tinker(p_ids, None, f_ids2, f_lps2),
            -1.25,
            places=9,
        )
        self.assertAlmostEqual(sum_tinker_logprobs([None, -1.0, -2.5], 1), 3.5, places=9)
        self.assertIn("kimi-k3.1-pro", TINKER_PRESETS)
        self.assertIn("glm-5.3", TINKER_PRESETS)
        self.assertIn("deepseek-v4.1-flash", TINKER_PRESETS)
        self.assertIn("inkling-2-small", TINKER_PRESETS)

    def test_archived_corpus_and_bench_ids_load_cleanly(self) -> None:
        docs = load_corpus_docs(ROOT / "data/anpaure_v1_corpus_1000.jsonl")
        bench_ids = load_anpi_bench_doc_ids(ROOT / "runs/anpaure_v1_n1000.json")
        items = load_anpi_bench_items(ROOT / "runs/anpaure_v1_n1000.json")
        self.assertEqual(len(docs), 1000)
        self.assertEqual(len(bench_ids), 954)
        self.assertEqual(len(items), 954)
        doc_ids = {d["doc_id"] for d in docs}
        self.assertTrue(bench_ids.issubset(doc_ids))


if __name__ == "__main__":
    unittest.main()
