"""Three-way validation: OpenRouter BB ANPI vs. Tinker BB ANPI vs. Tinker White-Box BPB/Oracle."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_bpb.tinker_bpb import _slug

TINKER_DIR = ROOT / "runs" / "bpb_tinker"
TINKER_BB_DIR = ROOT / "runs" / "bpb_tinker_blackbox"
CAL_JSON = ROOT / "runs" / "anpaure_v1_calibrated.json"

_NORM = NormalDist()
_SQRT2 = math.sqrt(2.0)

# Exact 1:1 mapping from Tinker model_id -> ANPI display name
TINKER_TO_ANPI: dict[str, str] = {
    "moonshotai/Kimi-K3.1-Pro:peft:262144": "Kimi K3.1 Pro",
    "moonshotai/Kimi-K2.6": "Kimi K2.6",
    "zai-org/GLM-5.3:peft:262144": "GLM-5.3",
    "zai-org/GLM-5.3-Flash:peft:262144": "GLM-5.3 Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144": "DeepSeek V4.1 Flash (1017)",
    "thinkingmachines/Inkling-2-Small": "Inkling 2 Small",
    "thinkingmachines/Inkling": "Inkling",
    "thinkingmachines/Inkling-2-Mini:peft:65536:free": "Inkling 2 Mini",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "Nemotron 3 Ultra Base",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "Nemotron 3 Super Base",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "Nemotron 3.5 Lightning",
    "Qwen/Qwen3.8-27B": "Qwen3.8-27B",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
}


def _rank(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    return ranks


def _corr(x: list[float], y: list[float]) -> tuple[float, float]:
    ax, ay = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(ax) < 3:
        return (float("nan"), float("nan"))
    r = float(np.corrcoef(ax, ay)[0, 1])
    rho = float(np.corrcoef(_rank(ax), _rank(ay))[0, 1])
    return (r, rho)


def _dprime(auc: float) -> float:
    return _SQRT2 * _NORM.inv_cdf(auc)


def main() -> None:
    cal = json.loads(CAL_JSON.read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in cal["models"]}

    rows = []
    for tid, anpi_name in TINKER_TO_ANPI.items():
        wb_path = TINKER_DIR / f"{_slug(tid)}__tinker.json"
        if not wb_path.exists():
            continue
        d = json.loads(wb_path.read_text(encoding="utf-8"))
        m = by_name.get(anpi_name)
        ae = d.get("anpi_eval_summary", {})
        s = d.get("summary", {})
        s9 = s.get("anpi_954_subset", s)

        tk_bb_auc = None
        tk_bb_n = 0
        tk_bb_top1 = None
        bb_path = TINKER_BB_DIR / f"{_slug(tid)}__bb.jsonl"
        if bb_path.exists():
            recs = [json.loads(ln) for ln in bb_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            ok = [r for r in recs if r.get("ok")]
            tk_bb_n = len(ok)
            if ok:
                tk_bb_auc = round(sum((8 - r["gold_rank"]) / 7.0 for r in ok) / len(ok), 4)
                tk_bb_top1 = round(100.0 * sum(1 for r in ok if r["gold_rank"] == 1) / len(ok), 2)

        rows.append(
            {
                "tinker_id": tid,
                "name": anpi_name,
                "bypass": bool(m.get("template_bypass")) if m else False,
                "or_bb_auc": m.get("auc") if m else None,
                "tk_bb_auc": tk_bb_auc,
                "tk_bb_n_ok": tk_bb_n,
                "tk_bb_top1": tk_bb_top1,
                "wb_auc": ae.get("whitebox_auc"),
                "wb_norm_auc": ae.get("whitebox_norm_auc"),
                "wb_top1": ae.get("whitebox_top1_pct"),
                "span_bpb": ae.get("gold_span_bpb"),
                "doc_bpb": s.get("bpb"),
                "doc954_bpb": s9.get("bpb"),
                "macro5_bpb": s.get("macro_domain_bpb"),
                "domains_bpb": {k: v["bpb"] for k, v in s.get("domains", {}).items()},
                "elie_bpb": m.get("unch_bpb") if (m and m.get("is_anchor")) else None,
            }
        )

    rows.sort(key=lambda r: -(r["tk_bb_auc"] or r["or_bb_auc"] or 0.0))
    print(f"Loaded {len(rows)} completed Tinker evaluations\n")
    hdr = (
        f"{'Model':27s} | {'OR BB':6s} | {'TK BB':6s} | {'TK-OR':7s} | {'n_ok':4s} | "
        f"{'WB AUC':6s} | {'SpanBPB':7s} | {'DocBPB':6s} | {'Elie':6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        or_s = f"{r['or_bb_auc']:.4f}" if r["or_bb_auc"] is not None else "   -  "
        tk_s = f"{r['tk_bb_auc']:.4f}" if r["tk_bb_auc"] is not None else "   -  "
        df_s = (
            f"{r['tk_bb_auc'] - r['or_bb_auc']:+7.4f}"
            if (r["tk_bb_auc"] is not None and r["or_bb_auc"] is not None)
            else "   -   "
        )
        wb_s = f"{r['wb_auc']:.4f}" if r["wb_auc"] is not None else "   -  "
        sp_s = f"{r['span_bpb']:.4f}" if r["span_bpb"] is not None else "   -   "
        db_s = f"{r['doc_bpb']:.4f}" if r["doc_bpb"] is not None else "   -  "
        el_s = f"{r['elie_bpb']:.4f}" if r["elie_bpb"] is not None else "  -   "
        print(f"{r['name']:27s} | {or_s} | {tk_s} | {df_s} | {r['tk_bb_n_ok']:4d} | {wb_s} | {sp_s:7s} | {db_s} | {el_s}")

    matched = [r for r in rows if r["or_bb_auc"] is not None and r["tk_bb_auc"] is not None and r["doc_bpb"] is not None]
    no_oss = [r for r in matched if "gpt-oss" not in r["name"]]
    native = [r for r in matched if not r["bypass"]]

    metrics: dict[str, dict[str, float]] = {}
    if len(matched) >= 3:
        r_ot, rho_ot = _corr([r["or_bb_auc"] for r in matched], [r["tk_bb_auc"] for r in matched])
        r_tw, rho_tw = _corr([r["tk_bb_auc"] for r in matched], [r["wb_auc"] for r in matched])
        r_ts, rho_ts = _corr([-_dprime(r["tk_bb_auc"]) for r in matched], [math.log(r["span_bpb"]) for r in matched])
        r_wd, rho_wd = _corr([-_dprime(r["wb_auc"]) for r in matched], [math.log(r["doc_bpb"]) for r in matched])
        r_tn, rho_tn = _corr([-_dprime(r["tk_bb_auc"]) for r in no_oss], [math.log(r["doc_bpb"]) for r in no_oss])
        r_on, rho_on = _corr([-_dprime(r["or_bb_auc"]) for r in native], [math.log(r["doc_bpb"]) for r in native])
        mad = float(np.mean([abs(r["tk_bb_auc"] - r["or_bb_auc"]) for r in matched]))
        metrics = {
            "or_bb_vs_tinker_bb_all14": {"r": round(r_ot, 4), "rho": round(rho_ot, 4), "mean_abs_diff": round(mad, 4)},
            "tinker_bb_vs_whitebox_oracle_all14": {"r": round(r_tw, 4), "rho": round(rho_tw, 4)},
            "tinker_bb_dprime_vs_ln_span_bpb_all14": {"r": round(r_ts, 4), "rho": round(rho_ts, 4)},
            "whitebox_dprime_vs_ln_doc_bpb_all14": {"r": round(r_wd, 4), "rho": round(rho_wd, 4)},
            "tinker_bb_dprime_vs_ln_doc_bpb_no_oss12": {"r": round(r_tn, 4), "rho": round(rho_tn, 4)},
            "or_bb_dprime_vs_ln_doc_bpb_native11": {"r": round(r_on, 4), "rho": round(rho_on, 4)},
        }
        print("\n=== SUMMARY CORRELATIONS ===")
        for k, v in metrics.items():
            print(f"  {k:42s}: {v}")

    summary_path = TINKER_DIR / "SUMMARY.json"
    summary_path.write_text(json.dumps({"n_models": len(rows), "correlations": metrics, "models": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
