"""Comprehensive unit, synthetic recovery, and live-ledger replay tests for Protocol A."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pplx_protocol_a.byte_prefix import (
    aligned_strict_offset,
    byte_prefix_bounds,
    certified_document_win_rate,
)
from pplx_protocol_a.canary import evaluate_canary_drift, run_lt_canary_probe
from pplx_protocol_a.corpus import (
    build_position_manifest,
    extract_anchor,
    load_manifest,
    save_manifest,
    split_exact_atoms,
)
from pplx_protocol_a.estimators import (
    PositionObservation,
    bpb_to_perplexity_metrics,
    cross_model_outlier_and_winrate,
    evaluate_protocol_a,
    geom_tail_surprisal,
    shannon_rank_coded_bpb_cv,
    zipf_tail_surprisal,
)
from pplx_protocol_a.lexical_transfer import (
    FROZEN_SPEC,
    fit_lexical_transfer_models,
)
from pplx_protocol_a.nce_bpb import build_rank_nce_prompt, parse_letter_ranking, score_rank_nce_cv
from pplx_protocol_a.matching import extract_position_match
from pplx_protocol_a.openai_responses import (
    ProtocolAError,
    build_responses_payload,
    run_sol_acquisition,
    sanitize_responses_reply,
)
from pplx_protocol_a.proxy import extract_target_span_bits, parse_echo_document_logprobs

LN2 = math.log(2.0)


def _mk_pos(chosen_b: bytes, chosen_lp: float, top_list: list[tuple[bytes, float]]) -> dict:
    return {
        "chosen": {"bytes_hex": chosen_b.hex(), "logprob": chosen_lp},
        "top_logprobs": [{"bytes_hex": b.hex(), "logprob": lp} for b, lp in top_list],
    }


def _mk_anchor_positions(anchor_tokens: list[bytes]) -> list[dict]:
    return [_mk_pos(tb, -0.001, [(tb, -0.001)]) for tb in anchor_tokens]


class TestCorpusAndMatching(unittest.TestCase):
    def test_exact_atom_reconstruction_unicode_and_punctuation(self) -> None:
        text = "Mr. Speaker, the committee's vote—unanimous!—cost €42.50 in café fees.\nNext line."
        atoms = split_exact_atoms(text)
        self.assertEqual("".join(a for _, _, a in atoms), text)

    def test_manifest_preserves_exact_boundary_and_roundtrips(self) -> None:
        doc_text = (
            "The Honourable Member for Northminster spoke at length about the proposed "
            "amendment to Clause 14 of the Energy Infrastructure Bill, emphasizing that "
            "independent regulators must publish quarterly reliability audits without delay. "
            "Several opposition spokespersons welcomed the transparency requirements while "
            "warning that grid interconnection queues still exceed five years across rural counties."
        )
        docs = [{"slice_name": "hansard", "doc_id": "d1", "text": doc_text}]
        positions = build_position_manifest(docs, fracs=(0.30, 0.55, 0.80), anchor_words=8)
        self.assertEqual(len(positions), 3)
        for p in positions:
            self.assertTrue(p.prefix_text.endswith(p.anchor_text))
            self.assertIn(p.prefix_text + p.target_text, doc_text)
            self.assertEqual(p.target_utf8_bytes, len(bytes.fromhex(p.target_hex)))

        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "manifest.json"
            digest = save_manifest(mp, positions, {"source": "test"})
            loaded, meta, digest2 = load_manifest(mp)
            self.assertEqual(digest, digest2)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(meta["source"], "test")

    def test_exact_case_sensitive_and_supertoken_matching(self) -> None:
        anchor = b"voted unanimously on the"
        target = b" amendment"
        anchor_pos = _mk_anchor_positions([b"voted", b" unanimously", b" on", b" the"])

        # Top-20 contains lowercase target (exact), super-token b" amendment,", and capitalized wrong-case b" Amendment"
        # Wrong-case b" Amendment" MUST NOT match! Super-token b" amendment," MUST match!
        p_exact = math.log(0.20)
        p_super = math.log(0.05)
        p_wrong_case = math.log(0.50)
        p_other = math.log(0.10)
        top = [
            (b" Amendment", p_wrong_case),
            (b" amendment", p_exact),
            (b" bill", p_other),
            (b" amendment,", p_super),
        ]
        rows = anchor_pos + [_mk_pos(b" Amendment", p_wrong_case, top)]
        m = extract_position_match("p1", rows, anchor, target, mode="echo_chat")
        self.assertTrue(m.adherent_strict)
        self.assertTrue(m.observed_exact)
        self.assertEqual(m.match_kind, "exact_top20")
        # Probability must equal 0.20 + 0.05 = 0.25 (2.0 bits), NOT including 0.50 wrong case!
        self.assertAlmostEqual(m.surprisal_bits or 0.0, -math.log2(0.25), places=6)

        # Ensure mid-word extension b" warning" does NOT match target b" war"
        top_war = [(b" warning", math.log(0.60)), (b" peace", math.log(0.20))]
        rows_war = anchor_pos + [_mk_pos(b" warning", math.log(0.60), top_war)]
        m_war = extract_position_match("p2", rows_war, anchor, b" war", mode="echo_chat")
        self.assertFalse(m_war.observed_exact)
        self.assertEqual(m_war.match_kind, "miss_censored")

    def test_bpe_absorbed_trailing_space(self) -> None:
        anchor = b"voted on the"
        target = b" amendment"
        # Final anchor token absorbed the leading space: b" the "
        anchor_pos = _mk_anchor_positions([b"voted", b" on", b" the "])
        top = [(b"amendment", math.log(0.40)), (b"proposal", math.log(0.30))]
        rows = anchor_pos + [_mk_pos(b"proposal", math.log(0.30), top)]
        m = extract_position_match("p3", rows, anchor, target, mode="echo_chat")
        self.assertTrue(m.adherent_strict)
        self.assertTrue(m.absorbed_leading_space)
        self.assertTrue(m.observed_exact)
        self.assertAlmostEqual(m.surprisal_bits or 0.0, -math.log2(0.40), places=6)

    def test_outside_top20_chosen_real_logprob_and_multitoken_chain(self) -> None:
        anchor = b"voted on the"
        anchor_pos = _mk_anchor_positions([b"voted", b" on", b" the"])

        # Case A: chosen token outside top-20 with real Direct Responses logprob (-10.0 nats)
        top_a = [(f" w{i}".encode(), math.log(0.04)) for i in range(20)]
        rows_a = anchor_pos + [_mk_pos(b" xylophone", -10.0, top_a)]
        ma = extract_position_match("p4", rows_a, anchor, b" xylophone", mode="echo_chat")
        self.assertTrue(ma.observed_exact)
        self.assertEqual(ma.match_kind, "exact_chosen_outside")
        self.assertAlmostEqual(ma.surprisal_bits or 0.0, 10.0 / LN2, places=6)

        # Case B: multi-token emitted chain b" unan" + b"imously" (resolved via step-2 top-20)
        rows_b = anchor_pos + [
            _mk_pos(b" unan", -2.0, [(b" unan", -2.0), (b" other", -0.5)]),
            _mk_pos(b"imously", -0.5, [(b"imously", -0.5)]),
        ]
        mb = extract_position_match("p5", rows_b, anchor, b" unanimously", mode="echo_chat")
        self.assertTrue(mb.observed_exact)
        self.assertIn(mb.match_kind, ("exact_multitoken_emitted", "exact_multitoken_top20"))
        self.assertAlmostEqual(mb.surprisal_bits or 0.0, 2.5 / LN2, places=6)

        # Case C: exact match in top-20 alongside tiny stray b" " sub-prefix (must stay exact_top20!)
        rows_c = anchor_pos + [
            _mk_pos(b" other", -0.2, [(b" other", -0.2), (b" private", -7.0), (b" ", -17.0)])
        ]
        mc = extract_position_match("p6", rows_c, anchor, b" private", mode="echo_chat")
        self.assertTrue(mc.observed_exact)
        self.assertEqual(mc.match_kind, "exact_top20")
        self.assertAlmostEqual(mc.surprisal_bits or 0.0, 7.0 / LN2, places=6)

        # Case D: emitted standalone b" " at idx, followed by non-emitted b"32" in top-20 at idx+1!
        rows_d = anchor_pos + [
            _mk_pos(b" ", -0.01, [(b" ", -0.01)]),
            _mk_pos(b"40", -0.5, [(b"40", -0.5), (b"32", -3.0)]),
        ]
        md = extract_position_match("p7", rows_d, anchor, b" 32", mode="echo_chat")
        self.assertTrue(md.observed_exact)
        self.assertEqual(md.match_kind, "exact_multitoken_top20")
        self.assertAlmostEqual(md.surprisal_bits or 0.0, 3.01 / LN2, places=6)

        # Case E: empty-byte EOS/special token (bytes_hex == "") must NEVER match as sub-prefix!
        rows_e = [
            _mk_pos(b" other", -0.5, [(b"", -1.1), (b" other", -0.5), (b" private", -3.0)])
        ]
        me = extract_position_match("p8", rows_e, b"", b" private", mode="assistant_prefill")
        self.assertTrue(me.observed_exact)
        self.assertEqual(me.match_kind, "exact_top20")
        self.assertAlmostEqual(me.surprisal_bits or 0.0, 3.0 / LN2, places=6)

        # Case F: holdout leakage guard in lexical_transfer
        with self.assertRaises(ValueError):
            fit_lexical_transfer_models({"smol": [], "olmo": []}, {"smol": {}, "olmo": {}}, spec=FROZEN_SPEC)


class TestOpenAIResponsesGuards(unittest.TestCase):
    def _dummy_pos(self):
        text = "One two three four five six seven eight nine ten eleven twelve " * 8
        return build_position_manifest([{"slice_name": "s", "doc_id": "d", "text": text}])[0]

    def test_payload_forbids_unsafe_fields_and_low_max_tokens(self) -> None:
        pos = self._dummy_pos()
        body = build_responses_payload(pos, model="gpt-5.6-sol", max_output_tokens=32)
        self.assertEqual(body["top_logprobs"], 20)
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertFalse(body["store"])
        with self.assertRaises(ProtocolAError):
            build_responses_payload(pos, max_output_tokens=8)

    def test_reasoning_leak_rejected_and_budget_cap_halts(self) -> None:
        pos = self._dummy_pos()
        pricing = {
            "uncached_input_usd_per_million": 4.0,
            "cached_input_usd_per_million": 0.4,
            "output_usd_per_million": 20.0,
        }
        # Reasoning leak must raise ProtocolAError
        bad_raw = {
            "id": "resp_secret_id",
            "status": "completed",
            "model": "gpt-5.6-sol",
            "service_tier": "flex",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "output_tokens_details": {"reasoning_tokens": 5},
            },
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "logprobs": [
                                {"bytes": [65], "logprob": -0.1, "top_logprobs": [{"bytes": [65], "logprob": -0.1}]}
                            ],
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(ProtocolAError):
            sanitize_responses_reply(
                bad_raw,
                pos,
                expected_model="gpt-5.6-sol",
                expected_tier="flex",
                top_logprobs_cap=20,
                pricing=pricing,
                latency_seconds=0.5,
                mode="echo_chat",
            )

        # Good reply sanitized without response id, and budget cap halts loop
        good_raw = json.loads(json.dumps(bad_raw))
        good_raw["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
        clean = sanitize_responses_reply(
            good_raw,
            pos,
            expected_model="gpt-5.6-sol",
            expected_tier="flex",
            top_logprobs_cap=20,
            pricing=pricing,
            latency_seconds=0.5,
            mode="echo_chat",
        )
        self.assertNotIn("id", clean)

        # Test hard USD budget cap via mocked transport
        calls = 0

        def fake_transport(api_key, body, url, timeout):
            nonlocal calls
            calls += 1
            return 200, good_raw, None, 0.05

        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 10
        many_pos = build_position_manifest(
            [{"slice_name": "s", "doc_id": f"d{i}", "text": text} for i in range(5)],
            fracs=(0.3, 0.6),
        )
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.jsonl"
            # Each call costs (100*4 + 20*20)/1e6 = $0.0008 -> cap at $0.0015 stops after 2 calls!
            res = run_sol_acquisition(
                many_pos,
                out_path,
                api_key="dummy",
                pricing=pricing,
                max_cost_usd=0.0015,
                workers=1,
                transport_fn=fake_transport,
            )
            self.assertTrue(res["stopped_due_to_budget"])
            self.assertEqual(calls, 2)


class TestProxyAndEstimators(unittest.TestCase):
    def test_proxy_rejects_placeholder_zeros_and_extracts_exact_spans(self) -> None:
        raw_text = "The committee voted."
        zero_resp = {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["The", " committee", " voted", "."],
                        "token_logprobs": [None, 0.0, 0.0, 0.0],
                        "text_offset": [0, 3, 13, 19],
                    }
                }
            ]
        }
        with self.assertRaises(ValueError):
            parse_echo_document_logprobs(zero_resp, raw_text)

        real_resp = {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["The", " committee", " voted", "."],
                        "token_logprobs": [None, -2.0 * LN2, -3.0 * LN2, -1.0 * LN2],
                        "text_offset": [0, 3, 13, 19],
                    }
                }
            ]
        }
        bpb, spans = parse_echo_document_logprobs(real_resp, raw_text)
        # Scored suffix is " committee voted." = 17 UTF-8 bytes; total bits = 2 + 3 + 1 = 6 bits
        self.assertAlmostEqual(bpb, 6.0 / 17.0, places=6)
        self.assertAlmostEqual(extract_target_span_bits(spans, 13, 19), 3.0, places=6)

    def test_synthetic_open_model_recovery_gate(self) -> None:
        """End-to-end synthetic recovery across 0.55 and 0.85 true BPB with Sol-like jitter."""
        rng = np.random.default_rng(1234)
        for true_bpb in (0.55, 0.85):
            M = 900
            bl = np.clip(np.round(rng.gamma(4.5, 1.05, size=M) + 1.1), 2, 18)
            z = rng.gamma(0.85, 1.0, size=M) * (0.6 + 0.4 * bl / bl.mean())
            q_bits = np.maximum(0.02, z / z.mean() * (0.65 * bl.mean()))
            q_bits *= (0.65 * bl.sum()) / q_bits.sum()
            log_q = np.log(q_bits + 0.15)
            t_bits = np.maximum(0.02, np.exp(log_q + rng.normal(0.0, 0.45 * np.std(log_q), size=M)) - 0.15)
            t_bits *= (true_bpb * bl.sum()) / t_bits.sum()

            # Sol-like BF16 rounding + 0.25-nat SD backend drift + dynamic h_20 threshold
            h_obs = np.maximum(0.01, t_bits + rng.normal(0.0, 0.25 / LN2, size=M))
            h_20 = np.clip(rng.normal(8.2, 1.8, size=M), 4.5, 14.0)
            in20 = h_obs <= h_20

            obs = []
            for i in range(M):
                p20 = 2.0 ** (-float(h_20[i]))
                mt = min(0.45, max(p20, p20 * 12.0))
                clb = max(float(h_20[i]), -math.log2(mt))
                obs.append(
                    PositionObservation(
                        position_id=f"s0__d{i // 3}__p{i}",
                        slice_name="slice_a" if i % 2 == 0 else "slice_b",
                        doc_id=f"d{i // 3}",
                        target_utf8_bytes=int(bl[i]),
                        observed_exact=bool(in20[i]),
                        surprisal_bits=float(h_obs[i]) if in20[i] else None,
                        censor_lb_bits=float(h_obs[i]) if in20[i] else clb,
                        h_20_bits=float(h_20[i]),
                        p_20=p20,
                        m_tail=mt,
                        match_kind="exact_top20" if in20[i] else "miss_censored",
                        proxy_bits=float(q_bits[i]),
                    )
                )

            res = evaluate_protocol_a(
                obs,
                proxy_full_slice_bpb={"slice_a": 0.65, "slice_b": 0.65},
                n_boot=200,
                seed=7,
            )
            lb = res["bpb_lower_bound"]["macro_bpb"]
            tobit = res["bpb_shifted_boxcox_tobit"]["macro_bpb"]
            comp = res["bpb_operational_hybrid_comp"]["macro_bpb"]
            self.assertLessEqual(lb, true_bpb + 0.02)
            self.assertLess(abs(tobit - true_bpb), 0.04, f"Tobit {tobit:.3f} far from {true_bpb:.3f}")
            self.assertGreater(comp, lb)

    def test_ppl_conversion_and_cross_model_outlier_detection(self) -> None:
        m = bpb_to_perplexity_metrics(0.60, bytes_per_token=5.36)
        self.assertAlmostEqual(m["byte_ppl"], 2.0 ** 0.60, places=6)
        self.assertAlmostEqual(m["token_ppl"], math.exp(m["nats_per_token"]), places=6)

        # Model C memorized doc 'd_mem' (loss 0.05 vs ~0.60 -> r < -3)
        panel = {
            "mA": {"d1": 0.60, "d2": 0.70, "d3": 0.80, "d_mem": 0.65},
            "mB": {"d1": 0.66, "d2": 0.77, "d3": 0.88, "d_mem": 0.72},
            "mC": {"d1": 0.58, "d2": 0.68, "d3": 0.78, "d_mem": 0.05},
        }
        diag = cross_model_outlier_and_winrate(panel, r_threshold=-2.0)
        flagged_ids = [x["doc_id"] for x in diag["flagged_outlier_docs"]]
        self.assertEqual(flagged_ids, ["d_mem"])
        self.assertIn("mC", diag["flagged_outlier_docs"][0]["flagged_models"])
        self.assertGreater(diag["avg_pairwise_win_rate"]["mC"], diag["avg_pairwise_win_rate"]["mB"])

    def test_shannon_rank_bpb_exact_monotone_rl_invariance(self) -> None:
        """Pure Shannon rank-coded BPB must be bit-identical under arbitrary monotone RL warping."""
        rng = np.random.default_rng(99)
        obs_base, obs_rl = [], []
        for i in range(60):
            # Generate 20 decreasing logprobs
            raw = -np.sort(rng.exponential(scale=1.5, size=20))
            lps = raw - math.log(float(np.sum(np.exp(raw))))
            rank = int(rng.choice(np.arange(1, 21), p=np.exp(lps) / np.sum(np.exp(lps))))
            # Arbitrary nonlinear position-specific RL sharpening warp (preserves strict order!)
            gamma = 1.5 + 2.5 * (i % 5) / 4.0
            warped_raw = lps * gamma - 0.3 * np.power(np.maximum(0.0, lps[0] - lps), 1.4)
            warped_lps = warped_raw - math.log(float(np.sum(np.exp(warped_raw))))
            for target_list, vec in ((obs_base, lps), (obs_rl, warped_lps)):
                s_bits = -float(vec[rank - 1]) / LN2
                target_list.append(
                    PositionObservation(
                        position_id=f"s__d{i // 3}__p{i}",
                        slice_name="s",
                        doc_id=f"d{i // 3}",
                        target_utf8_bytes=5,
                        observed_exact=True,
                        surprisal_bits=s_bits,
                        censor_lb_bits=s_bits,
                        h_20_bits=-float(vec[-1]) / LN2,
                        p_20=math.exp(float(vec[-1])),
                        m_tail=1e-4,
                        match_kind="exact_top20",
                        top_entries_logprobs=tuple(float(x) for x in vec),
                        target_rank_in_top=rank,
                    )
                )
        r_base = shannon_rank_coded_bpb_cv(obs_base, n_boot=100, seed=42)
        r_rl = shannon_rank_coded_bpb_cv(obs_rl, n_boot=100, seed=42)
        self.assertAlmostEqual(r_base["macro_bpb"], r_rl["macro_bpb"], places=12)

    def test_byte_prefix_bounds_and_canary_drift(self) -> None:
        anchor = b"on the"
        rows = _mk_anchor_positions([b"on", b" the"]) + [
            _mk_pos(b" ", math.log(0.40), [(b" ", math.log(0.40)), (b" bill", math.log(0.50))]),
            _mk_pos(b"amen", math.log(0.50), [(b"amen", math.log(0.50)), (b"prop", math.log(0.40))]),
        ]
        off = aligned_strict_offset(rows, anchor)
        self.assertEqual(off, 2)
        lo, hi = byte_prefix_bounds(rows[off:], b" ame", k=20, log_slack=0.0)
        # Good = 0.40 * 0.50 = 0.20; Bad = 0.50 + 0.40 * 0.40 = 0.66 -> hi = 1 - 0.66 = 0.34
        self.assertAlmostEqual(lo, 0.20, places=6)
        self.assertAlmostEqual(hi, 0.34, places=6)

        # Canary drift detection catches model swap & large shift
        def fake_canary(api_key, body, url, timeout):
            return 200, {
                "model": "gpt-5.6-sol",
                "service_tier": "flex",
                "output": [{"type": "message", "content": [{"type": "output_text", "logprobs": [{"bytes": [65], "logprob": -0.1, "top_logprobs": [{"bytes": [65], "logprob": -0.1}]}]}]}],
            }, None, 0.05

        pre = run_lt_canary_probe("k", model="gpt-5.6-sol", phase="pre", replicates=1, prompts=[("c1", "hi")], transport_fn=fake_canary)
        post = run_lt_canary_probe("k", model="gpt-5.6-sol", phase="post", replicates=1, prompts=[("c1", "hi")], transport_fn=fake_canary)
        res_ok = evaluate_canary_drift(pre + post)
        self.assertEqual(res_ok["verdict"], "STABLE")
        post[0]["returned_model"] = "gpt-5.6-sol-new"
        res_swap = evaluate_canary_drift(pre + post)
        self.assertEqual(res_swap["verdict"], "DRIFT_WARNING")

    def test_rank_nce_bpb_parser_and_variational_bound(self) -> None:
        rk, complete = parse_letter_ranking("C, A, H, B, G, D, F, E", 8)
        self.assertTrue(complete)
        self.assertEqual(rk, ["C", "A", "H", "B", "G", "D", "F", "E"])

        # Random guessing (uniform ranks 1..8) must yield ~0 info gain (NCE BPB ~ q BPB)
        rng = np.random.default_rng(7)
        unif_recs = [{"doc_id": f"d{i}", "ok": True, "gold_rank": int(rng.integers(1, 9))} for i in range(80)]
        r_unif = score_rank_nce_cv(unif_recs, k_candidates=8, q_chunk_bpb=0.71, q_doc_bpb=0.74, mean_chunk_bytes=35.0, n_boot=100)
        self.assertLess(abs(r_unif["info_gain_bits"]), 0.25)

        # Strong model (always rank 1) must achieve positive gain bounded by log2(8)=3 bits
        perf_recs = [{"doc_id": f"d{i}", "ok": True, "gold_rank": 1} for i in range(40)]
        r_perf = score_rank_nce_cv(perf_recs, k_candidates=8, q_chunk_bpb=0.71, q_doc_bpb=0.74, mean_chunk_bytes=35.0, n_boot=100)
        self.assertGreater(r_perf["info_gain_bits"], 2.0)
        self.assertLessEqual(r_perf["info_gain_bits"], math.log2(8.0) + 1e-9)
        self.assertLess(r_perf["nce_doc_bpb"], 0.74)

    def test_live_sol_t1_ledger_tail_bracketing_if_present(self) -> None:
        ledger_env = os.environ.get("ANPAURE_LIVE_SOL_LEDGER", "")
        if not ledger_env:
            self.skipTest("Optional local Sol T=1 ledger not configured (set ANPAURE_LIVE_SOL_LEDGER)")
        ledger = Path(ledger_env)
        if not ledger.exists():
            self.skipTest("Optional local Sol T=1 ledger not present")
        data = json.loads(ledger.read_text(encoding="utf-8"))
        tail_true, tail_geom, tail_zipf = [], [], []
        for rec in data["records"]:
            if rec.get("http_status") != 200:
                continue
            for pos in rec.get("positions", []):
                top = pos.get("top_logprobs", [])
                ch = pos.get("chosen")
                if not ch or len(top) < 20:
                    continue
                if ch["bytes_hex"] in {t["bytes_hex"] for t in top}:
                    continue
                top_ps = [math.exp(t["logprob"]) for t in top]
                p20 = min(top_ps)
                mt = max(1e-12, 1.0 - sum(top_ps))
                tail_true.append(-ch["logprob"] / LN2)
                tail_geom.append(geom_tail_surprisal(p20, mt))
                tail_zipf.append(zipf_tail_surprisal(p20, mt))
        self.assertGreater(len(tail_true), 800)
        m_true = float(np.mean(tail_true))
        m_geom = float(np.mean(tail_geom))
        m_zipf = float(np.mean(tail_zipf))
        # True live Sol tail mean sits cleanly bracketed between Geometric and Zipf(200k)
        self.assertLess(m_geom, m_true)
        self.assertLess(m_true, m_zipf)


if __name__ == "__main__":
    unittest.main()
