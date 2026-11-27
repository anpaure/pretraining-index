"""Command-line interface for Protocol A (Direct OpenAI Responses BPB harness).

Subcommands:
  plan        Build a frozen position manifest + conservative dry-run cost estimate ($0).
  run-sol     Execute Protocol A against Direct OpenAI `/v1/responses` (requires --execute-paid).
  run-proxy   Compute exact teacher-forced BPB + per-position proxy logprobs via `/v1/completions`.
  score       Compute all 4 BPB estimators, QA gates, bootstrap CIs, and Markdown/JSON report.
  validate    Run synthetic open-model recovery validation (or replay live Sol ledger).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .corpus import (
    PositionSpec,
    build_position_manifest,
    canonical_json_bytes,
    load_jsonl_corpus,
    load_manifest,
    save_manifest,
)
from .estimators import PositionObservation, evaluate_protocol_a
from .matching import TokenMatchResult, extract_position_match
from .openai_responses import (
    DEFAULT_URL,
    estimate_manifest_cost_usd,
    load_api_key,
    run_sol_acquisition,
)
from .proxy import score_proxy_corpus


def _load_prereg(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_slices(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            name, p = item.split("=", 1)
            out.append((name.strip(), Path(p.strip())))
        else:
            p = Path(item.strip())
            out.append((p.stem, p))
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    prereg = _load_prereg(Path(args.prereg))
    slice_specs = _parse_slices(args.slice)
    docs = load_jsonl_corpus(slice_specs, text_field=args.text_field, id_field=args.id_field)
    if not docs:
        raise SystemExit("No valid documents loaded from --slice inputs")

    cfg = prereg["protocol"]
    positions = build_position_manifest(
        docs,
        fracs=tuple(cfg.get("position_fracs", [0.30, 0.55, 0.80])),
        anchor_words=int(cfg.get("anchor_words", 8)),
        min_atoms=int(cfg.get("min_atoms_per_doc", 40)),
        max_prefix_chars=int(cfg.get("max_prefix_chars", 16000)),
        words_only=bool(cfg.get("words_only", False)),
    )
    if args.max_positions and len(positions) > args.max_positions:
        positions = positions[: args.max_positions]

    cost_est = estimate_manifest_cost_usd(
        positions,
        max_output_tokens=int(cfg["max_output_tokens"]),
        pricing=prereg["pricing_guard"],
    )
    manifest_path = Path(args.manifest)
    digest = save_manifest(
        manifest_path,
        positions,
        metadata={
            "prereg_path": str(args.prereg),
            "slices": [name for name, _ in slice_specs],
            "n_documents": len(docs),
            "cost_estimate": cost_est,
        },
    )
    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "documents": len(docs),
        "positions": len(positions),
        "slices": sorted({p.slice_name for p in positions}),
        "cost_estimate": cost_est,
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_run_sol(args: argparse.Namespace) -> int:
    prereg = _load_prereg(Path(args.prereg))
    positions, _, manifest_sha = load_manifest(Path(args.manifest))
    if args.limit and args.limit < len(positions):
        positions = positions[: args.limit]

    cfg = prereg["protocol"]
    pricing = prereg["pricing_guard"]
    model = args.model or cfg["model"]
    mode = args.mode or cfg.get("prompt_mode", "echo_chat")
    tier = args.service_tier if args.service_tier is not None else cfg.get("service_tier", "flex")
    reasoning_effort = args.reasoning_effort if args.reasoning_effort is not None else cfg.get("reasoning_effort", "none")
    max_out = int(args.max_output_tokens or cfg["max_output_tokens"])
    top_k = int(cfg["top_logprobs"])
    max_cost = float(args.max_cost_usd if args.max_cost_usd is not None else pricing["hard_cap_usd"])

    est = estimate_manifest_cost_usd(positions, max_output_tokens=max_out, pricing=pricing)
    if not args.execute_paid:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "message": "Pass --execute-paid to send live API requests.",
                    "manifest_sha256": manifest_sha,
                    "model": model,
                    "mode": mode,
                    "service_tier": tier,
                    "reasoning_effort": reasoning_effort,
                    "max_cost_usd": max_cost,
                    "estimate": est,
                },
                indent=2,
            )
        )
        return 0

    key = load_api_key(
        env_var=args.api_key_env,
        key_file=Path(args.key_file) if args.key_file else None,
    )
    res = run_sol_acquisition(
        positions,
        Path(args.output),
        api_key=key,
        model=model,
        mode=mode,
        service_tier=tier,
        max_output_tokens=max_out,
        top_logprobs=top_k,
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        reasoning_effort=reasoning_effort,
        pricing=pricing,
        max_cost_usd=max_cost,
        url=args.url or prereg.get("endpoint", DEFAULT_URL),
        workers=int(args.workers),
        timeout=float(args.timeout),
    )
    res["manifest_sha256"] = manifest_sha
    print(json.dumps(res, indent=2))
    return 0


def cmd_run_proxy(args: argparse.Namespace) -> int:
    slice_specs = _parse_slices(args.slice)
    docs = load_jsonl_corpus(slice_specs, text_field=args.text_field, id_field=args.id_field)
    positions, _, _ = load_manifest(Path(args.manifest))
    key = ""
    if args.api_key_env:
        import os

        key = os.environ.get(args.api_key_env, "").strip()
    if args.key_file:
        key = load_api_key(env_var=args.api_key_env or "OPENAI_API_KEY", key_file=Path(args.key_file))
    res = score_proxy_corpus(
        docs,
        positions,
        api_key=key,
        url=args.url,
        model=args.model,
        output_json=Path(args.output),
    )
    print(
        json.dumps(
            {
                "proxy_model": res["proxy_model"],
                "full_corpus_macro_bpb": res["full_corpus_macro_bpb"],
                "full_corpus_slice_bpb": res["full_corpus_slice_bpb"],
                "scored_positions": len(res["position_proxy_bits"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


def score_run(
    positions: list[PositionSpec],
    responses_jsonl: Path,
    *,
    prereg: dict[str, Any],
    proxy_json: Path | None = None,
    allow_relaxed_adherence: bool = False,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Audit QA gates and compute all Protocol A BPB estimators on a saved ledger."""
    pos_map = {p.position_id: p for p in positions}
    records: dict[str, dict[str, Any]] = {}
    total_lines = 0
    http_errors = 0
    contract_violations = 0
    reasoning_leaks = 0
    total_cost = 0.0
    duplicate_rows = 0
    short_top_lists = 0

    with responses_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            rec = json.loads(line)
            total_cost += float(rec.get("conservative_standard_cost_usd", 0.0))
            pid = rec.get("position_id")
            if rec.get("http_status") != 200:
                http_errors += 1
                continue
            if "contract_violation" in rec:
                contract_violations += 1
                if "reasoning" in str(rec["contract_violation"]).lower():
                    reasoning_leaks += 1
                continue
            usage = rec.get("usage") or {}
            if int(usage.get("reasoning_output_tokens", 0)) > 0:
                reasoning_leaks += 1
                continue
            if pid in pos_map:
                records[pid] = rec

    proxy_pos_bits: dict[str, float] = {}
    proxy_full_slices: dict[str, float] | None = None
    proxy_model_name: str | None = None
    if proxy_json is not None and proxy_json.exists():
        pdata = json.loads(proxy_json.read_text(encoding="utf-8"))
        proxy_pos_bits = {k: float(v) for k, v in pdata.get("position_proxy_bits", {}).items()}
        proxy_full_slices = pdata.get("full_corpus_slice_bpb")
        proxy_model_name = pdata.get("proxy_model")

    matches: list[TokenMatchResult] = []
    obs_list: list[PositionObservation] = []
    strict_adh = 0
    relaxed_adh = 0
    absorbed_spaces = 0
    expected_top_k = int(prereg["protocol"]["top_logprobs"])

    for pid, rec in records.items():
        pos = pos_map[pid]
        mode = str(rec.get("mode", prereg["protocol"].get("prompt_mode", "echo_chat")))
        m = extract_position_match(
            pid,
            rec["positions"],
            pos.anchor_text.encode("utf-8"),
            bytes.fromhex(pos.target_hex),
            mode=mode,
            allow_relaxed=allow_relaxed_adherence,
        )
        matches.append(m)
        if m.adherent_strict:
            strict_adh += 1
        if m.adherent_relaxed:
            relaxed_adh += 1
        if m.absorbed_leading_space:
            absorbed_spaces += 1
        if m.duplicate_top_bytes:
            duplicate_rows += 1
        if m.match_kind != "non_adherent":
            # OpenAI Responses legitimately omits 1-2 entries below top-20 on
            # ultra-peaked states (e.g. non-UTF8 special/EOS tokens or <1e-9 tail).
            # Flag as anomalous only if top_k_count < 10 OR head mass < 0.99 when < K.
            head_mass = 1.0 - float(m.m_tail or 0.0)
            if m.top_k_count < min(10, expected_top_k) or (m.top_k_count < expected_top_k and head_mass < 0.99):
                short_top_lists += 1
            # Strictly separate WHOLE-target exact rank from PARTIAL-prefix rank
            # (fixes cli.py:285 bug that credited partial prefix b" elect" as a
            # whole-target hit for b" electors"!).
            rank_in_top: int | None = None
            partial_prefix_rank: int | None = None
            if m.observed_exact and m.matched_top_entries:
                hx = m.matched_top_entries[0]["bytes_hex"]
                if hx in m.top_entries_bytes_hex:
                    rank_in_top = m.top_entries_bytes_hex.index(hx) + 1
            elif m.match_kind == "subtoken_censored" and m.matched_top_entries:
                exact_es = [e for e in m.matched_top_entries if e.get("is_exact")]
                part_es = [e for e in m.matched_top_entries if not e.get("is_exact")]
                if exact_es and exact_es[0]["bytes_hex"] in m.top_entries_bytes_hex:
                    rank_in_top = m.top_entries_bytes_hex.index(exact_es[0]["bytes_hex"]) + 1
                if part_es and part_es[0]["bytes_hex"] in m.top_entries_bytes_hex:
                    partial_prefix_rank = m.top_entries_bytes_hex.index(part_es[0]["bytes_hex"]) + 1
            obs_list.append(
                PositionObservation(
                    position_id=pid,
                    slice_name=pos.slice_name,
                    doc_id=pos.doc_id,
                    target_utf8_bytes=pos.target_utf8_bytes,
                    observed_exact=m.observed_exact,
                    surprisal_bits=m.surprisal_bits,
                    censor_lb_bits=float(m.censor_lb_bits or 0.0),
                    h_20_bits=float(m.h_20_bits or 0.0),
                    p_20=float(m.p_20 or 1e-15),
                    m_tail=float(m.m_tail or 1e-15),
                    match_kind=m.match_kind,
                    top_entries_logprobs=tuple(m.top_entries_logprobs),
                    target_rank_in_top=rank_in_top,
                    partial_prefix_rank_in_top=partial_prefix_rank,
                    proxy_bits=proxy_pos_bits.get(pid),
                )
            )

    n_valid = len(records)
    strict_rate = strict_adh / max(1, n_valid)
    relaxed_rate = relaxed_adh / max(1, n_valid)
    exact_k_rate = 1.0 - (short_top_lists / max(1, len(obs_list)))

    gates = prereg["qa_gates"]
    adherence_band = (
        "CLEAN"
        if strict_rate >= float(gates["adherence_clean_min"])
        else ("MARGINAL" if strict_rate >= float(gates["adherence_marginal_min"]) else "FAIL")
    )
    qa_pass = (
        reasoning_leaks == 0
        and adherence_band != "FAIL"
        and exact_k_rate >= float(gates["exact_topk_fraction_min"])
        and len(obs_list) > 0
    )

    estimates = (
        evaluate_protocol_a(
            obs_list,
            proxy_full_slice_bpb=proxy_full_slices,
            vocab_size=int(prereg["protocol"].get("zipf_vocab_size", 200_000)),
            n_boot=n_boot,
        )
        if obs_list
        else None
    )

    return {
        "schema_version": "protocol_a_report_v1",
        "verdict": "PASS" if qa_pass else "FAIL_QA_GATE",
        "proxy_model": proxy_model_name,
        "qa": {
            "total_ledger_rows": total_lines,
            "valid_responses": n_valid,
            "http_errors": http_errors,
            "contract_violations": contract_violations,
            "reasoning_token_leaks": reasoning_leaks,
            "strict_adherence_rate": round(strict_rate, 4),
            "relaxed_adherence_rate": round(relaxed_rate, 4),
            "adherence_band": adherence_band,
            "absorbed_leading_space_count": absorbed_spaces,
            "duplicate_top_byte_positions": duplicate_rows,
            "exact_topk_rate": round(exact_k_rate, 5),
            "conservative_standard_cost_usd": round(total_cost, 6),
        },
        "estimates": estimates,
        "match_notes_histogram": {
            note: sum(1 for m in matches if m.adherence_note == note)
            for note in sorted({m.adherence_note for m in matches})
        },
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    qa = report["qa"]
    est = report.get("estimates") or {}
    lines = [
        "# Protocol A Perplexity (BPB) Report",
        "",
        f"**Verdict:** `{report['verdict']}`  ",
        f"**Conservative standard spend:** `${qa['conservative_standard_cost_usd']:.6f}`  ",
        f"**Valid responses:** `{qa['valid_responses']}` | **Strict adherence:** `{qa['strict_adherence_rate']:.1%}` (`{qa['adherence_band']}`) | **Reasoning leaks:** `{qa['reasoning_token_leaks']}`",
        "",
    ]
    if est:
        cnt = est["counts"]
        lines.extend(
            [
                "## BPB Estimates (Macro-Averaged Over Slices, Document-Level 95% Bootstrap CI)",
                "",
                f"Observed exact tokens: **{cnt['observed_exact']} / {cnt['total']} ({cnt['exact_rate']:.1%})** | Match breakdown: `{cnt['by_match_kind']}`",
                "",
                "| Estimator | Macro BPB | 95% CI | Control-Variate BPB (95% CI) |",
                "|---|---:|---|---|",
            ]
        )
        rows = [
            ("Certified Hard Lower Bound (`BPB_LB`, raw T=1)", "bpb_lower_bound"),
            ("Zero-Proxy Geometric Tail (raw T=1)", "bpb_zero_proxy_geom"),
            ("Zero-Proxy Zipf(200k) Tail (raw T=1)", "bpb_zero_proxy_zipf"),
            ("Pure Shannon Rank-Coded BPB (5-fold CV, 100% monotone-RL invariant)", "bpb_shannon_rank_cv"),
            ("Head-Normalized + Empirical-Tail Calibrated Compressor (5-fold CV, zero Z_tail)", "bpb_head_calibrated_cv"),
            ("Full-Vocab Temp-Calibrated `BPB_LB` (5-fold doc CV)", "bpb_temp_calibrated_cv"),
            ("Certified Operational Hybrid Compressor (`BPB_comp`)", "bpb_operational_hybrid_comp"),
            ("Shifted Box-Cox Tobit MLE (`BPB_tobit`)", "bpb_shifted_boxcox_tobit"),
        ]
        for label, key in rows:
            if key not in est:
                continue
            r = est[key]
            ci = f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
            if "control_variate" in r:
                cv = r["control_variate"]
                cv_str = f"**{cv['macro_bpb']:.4f}** [{cv['ci95'][0]:.4f}, {cv['ci95'][1]:.4f}]"
            else:
                cv_str = "—"
            extra = ""
            if "optimal_head_temperature_mean" in r:
                extra = f" (T*={r['optimal_head_temperature_mean']:.2f}, miss={r['empirical_miss_rate_mean']:.1%})"
            elif "optimal_temperature_mean" in r:
                extra = f" (T*={r['optimal_temperature_mean']:.2f})"
            lines.append(f"| {label}{extra} | **{r['macro_bpb']:.4f}** | {ci} | {cv_str} |")

        if "rank_and_sharpening" in est:
            rs = est["rank_and_sharpening"]
            lines.extend(
                [
                    "",
                    "### Temperature-Invariant Rank & Post-Training Sharpening Diagnostics",
                    "",
                    f"- Top-1 / Top-3 / Top-5 / Top-20 accuracy: **`{rs['top1_accuracy']:.1%}`** / `{rs['top3_accuracy']:.1%}` / `{rs['top5_accuracy']:.1%}` / `{rs['top20_accuracy']:.1%}`",
                    f"- MRR: `{rs['mrr']:.4f}` | Mean `log2(rank)`: `{rs['mean_log2_rank']:.3f}` | Median rank: `{rs['median_rank']:.1f}`",
                    f"- Mean Top-20 head entropy: **`{rs['mean_top20_entropy_bits']:.3f}` bits** | Median Top-1 surprisal: **`{rs['median_top1_surprisal_bits']:.3f}` bits**",
                ]
            )

        if "bpb_shifted_boxcox_tobit" in est and "fit" in est["bpb_shifted_boxcox_tobit"]:
            f = est["bpb_shifted_boxcox_tobit"]["fit"]
            lines.extend(
                [
                    "",
                    f"Tobit fit diagnostics: `a={f['a']:.4f}`, `b={f['b']:.4f}`, `sigma={f['sigma']:.4f}`, `lambda={f['lambda']:.3f}` (censored={f['censored_count']}).",
                ]
            )
    return "\n".join(lines) + "\n"


def cmd_score(args: argparse.Namespace) -> int:
    prereg = _load_prereg(Path(args.prereg))
    positions, _, _ = load_manifest(Path(args.manifest))
    report = score_run(
        positions,
        Path(args.responses),
        prereg=prereg,
        proxy_json=Path(args.proxy) if args.proxy else None,
        allow_relaxed_adherence=bool(args.allow_relaxed),
        n_boot=int(args.n_boot),
    )
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(canonical_json_bytes(report))
    md = format_markdown_report(report)
    if args.out_md:
        p = Path(args.out_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
    print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pplx-protocol-a", description="Protocol A Direct Responses BPB Harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="Build frozen manifest and conservative dry-run cost estimate")
    sp.add_argument("--prereg", default="preregistration.json")
    sp.add_argument("--slice", action="append", required=True, help="name=path.jsonl (repeatable)")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--text-field", default="text")
    sp.add_argument("--id-field", default="id")
    sp.add_argument("--max-positions", type=int, default=0)
    sp.set_defaults(func=cmd_plan)

    sr = sub.add_parser("run-sol", help="Execute Protocol A on Direct OpenAI Responses")
    sr.add_argument("--prereg", default="preregistration.json")
    sr.add_argument("--manifest", required=True)
    sr.add_argument("--output", required=True)
    sr.add_argument("--model", default=None)
    sr.add_argument("--mode", choices=["echo_chat", "assistant_prefill"], default=None)
    sr.add_argument("--service-tier", default=None, help="flex | default | auto | priority | omit")
    sr.add_argument("--reasoning-effort", default=None, help="none | omit (use omit for gpt-4o/gpt-4.1)")
    sr.add_argument("--max-output-tokens", type=int, default=None)
    sr.add_argument("--max-cost-usd", type=float, default=None)
    sr.add_argument("--limit", type=int, default=0)
    sr.add_argument("--workers", type=int, default=4)
    sr.add_argument("--timeout", type=float, default=90.0)
    sr.add_argument("--url", default=None)
    sr.add_argument("--api-key-env", default="OPENAI_API_KEY")
    sr.add_argument("--key-file", default=None)
    sr.add_argument("--execute-paid", action="store_true", help="Required flag to actually send paid requests")
    sr.set_defaults(func=cmd_run_sol)

    spx = sub.add_parser("run-proxy", help="Compute exact teacher-forced proxy BPB via /v1/completions echo")
    spx.add_argument("--slice", action="append", required=True)
    spx.add_argument("--manifest", required=True)
    spx.add_argument("--url", required=True, help="OpenAI-compatible /v1/completions endpoint (vLLM/Fireworks)")
    spx.add_argument("--model", required=True)
    spx.add_argument("--output", required=True)
    spx.add_argument("--text-field", default="text")
    spx.add_argument("--id-field", default="id")
    spx.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    spx.add_argument("--key-file", default=None)
    spx.set_defaults(func=cmd_run_proxy)

    ss = sub.add_parser("score", help="Score saved Protocol A ledger + optional proxy JSON")
    ss.add_argument("--prereg", default="preregistration.json")
    ss.add_argument("--manifest", required=True)
    ss.add_argument("--responses", required=True)
    ss.add_argument("--proxy", default=None)
    ss.add_argument("--allow-relaxed", action="store_true")
    ss.add_argument("--n-boot", type=int, default=2000)
    ss.add_argument("--out-json", default=None)
    ss.add_argument("--out-md", default=None)
    ss.set_defaults(func=cmd_score)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
