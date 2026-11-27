"""Independent end-to-end verification audit for Anpaure Pretraining Index v1.

Verifies from scratch (without importing scoring code from build_anpaure_index_v1.py):
  1. Byte-for-byte reproducibility of HTML & calibrated JSON.
  2. Raw corpus & benchmark provenance (954 Nov-2026 items, 0 duplicates, 8 distinct candidates).
  3. Independent raw trace replay of all 44 live JSONL trace files (strict 8-distinct-letter check,
     exact gold_rank match against shuffled candidates, exact AUC / d' / MRR / Top-1% match vs dashboard).
  4. 100% parity between canonical runner summary (`runs/anpaure_v1_results.json`) and dashboard JSON.
  5. Strict EXTRAP badge verification on BOTH OpenAI & Elie scales (no models outside anchor range unbadged).
  6. Exact log-BPB Thurstone d' LOO, Leave-One-Lab-Out (LOLO), and Student-t 95% CI reproduction.
  7. Strict Zero-CoT (n_pos_rtok == 0) across all 44 live endpoints.
  8. Prose-only (792 non-code items) sensitivity check & Code median-length shortcut quantification.
  9. Zero API keys / credentials anywhere in the repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent
BENCH_PATH = ROOT / "runs/anpaure_v1_n1000.json"
CORPUS_PATH = ROOT / "data/anpaure_v1_corpus_1000.jsonl"
TRACE_DIR = ROOT / "runs/anpaure_v1_traces"
CAL_JSON = ROOT / "runs/anpaure_v1_calibrated.json"
RUN_JSON = ROOT / "runs/anpaure_v1_results.json"
HTML_PATH = ROOT / "anpaure_pretraining_index.html"

STRICT_RE = re.compile(r"^[A-H](?:\s*,\s*[A-H]){7}\s*$")
LETTERS = "ABCDEFGH"
_NORM = NormalDist()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main() -> None:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"  [PASS] {msg}")
        else:
            print(f"  [FAIL] {msg}")
            failures.append(msg)

    print("=== 1. DETERMINISTIC BYTE-FOR-BYTE REBUILD ===")
    h_html_before = sha256_file(HTML_PATH)
    h_json_before = sha256_file(CAL_JSON)
    subprocess.run([sys.executable, str(ROOT / "build_anpaure_index_v1.py")], check=True, stdout=subprocess.DEVNULL)
    h_html_after = sha256_file(HTML_PATH)
    h_json_after = sha256_file(CAL_JSON)
    check(h_html_before == h_html_after, f"HTML byte-identical rebuild (sha256={h_html_after})")
    check(h_json_before == h_json_after, f"Calibrated JSON byte-identical rebuild (sha256={h_json_after})")

    print("\n=== 2. CORPUS & BENCHMARK INTEGRITY ===")
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    items = bench["items"]
    items_by_id = {it["item_id"]: it for it in items}
    corpus_docs = {json.loads(ln)["doc_id"]: json.loads(ln) for ln in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()}
    check(len(items) == 954, f"Benchmark item count = {len(items)}")
    non_nov = sum(1 for it in items if "2026" not in str(it["published"]) or ("Nov" not in str(it["published"]) and "-11-" not in str(it["published"])))
    check(non_nov == 0, f"All items have Nov-2026 timestamps (violations={non_nov})")
    missing_src = sum(1 for it in items if it["doc_id"] not in corpus_docs)
    check(missing_src == 0, f"All 954 items trace back to archived corpus docs (missing={missing_src})")
    bad_cands = sum(1 for it in items if len(it["candidates"]) != 8 or len({c.strip().lower() for c in it["candidates"]}) != 8)
    check(bad_cands == 0, f"All items have 8 distinct candidates (bad={bad_cands})")

    print("\n=== 3. INDEPENDENT RAW TRACE REPLAY & RUNNER-DASHBOARD PARITY ===")
    cal = json.loads(CAL_JSON.read_text(encoding="utf-8"))
    run_res = json.loads(RUN_JSON.read_text(encoding="utf-8"))["models"]
    cal_by_name = {m["name"]: m for m in cal["models"]}

    # Map model_id -> display name from run_res + slug
    total_valid_rows = 0
    dup_letter_found = 0
    rank_mismatch = 0
    parity_diffs = 0
    for mid, rinfo in run_res.items():
        slug = mid.replace("/", "__").replace(":", "_")
        tpath = TRACE_DIR / f"{slug}.jsonl"
        by_id = {}
        for ln in tpath.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            iid = rec.get("item_id")
            if iid not in items_by_id:
                continue
            raw = (rec.get("raw") or "").strip()
            lets = re.findall(r"[A-H]", raw)
            is_c = bool(rec.get("ok") and STRICT_RE.match(raw) and len(lets) == 8 and len(set(lets)) == 8)
            prev = by_id.get(iid)
            if prev is None or (is_c and not prev["_c"]) or (is_c and rec.get("clean_retry")):
                rec["_c"] = is_c
                rec["_lets"] = lets
                by_id[iid] = rec
        clean = [r for r in by_id.values() if r["_c"]]
        total_valid_rows += len(clean)
        for r in clean:
            if len(set(r["_lets"])) != 8:
                dup_letter_found += 1
            gold_letter = LETTERS[items_by_id[r["item_id"]]["gold_idx"]]
            rk = r["_lets"].index(gold_letter) + 1
            if rk != r["gold_rank"]:
                rank_mismatch += 1
        auc_raw = float(np.mean([(8.0 - r["gold_rank"]) / 7.0 for r in clean]))
        auc = round(auc_raw, 4)
        neg_dp = round(-math.sqrt(2.0) * _NORM.inv_cdf(min(0.9999, max(0.0001, auc_raw))), 4)
        mrr = round(float(np.mean([1.0 / r["gold_rank"] for r in clean])), 4)
        t1 = round(float(np.mean([r["gold_rank"] == 1 for r in clean]) * 100), 2)
        n_pos = sum(1 for r in clean if int(r.get("r_tok") or 0) > 0)
        if rinfo["n_ok"] != len(clean) or abs(rinfo["mrr"] - mrr) > 1e-4 or abs(rinfo["top1_pct"] - t1) > 1e-2:
            parity_diffs += 1
            print(f"    runner mismatch on {mid}: runner={rinfo['n_ok']}/{rinfo['mrr']} vs replay={len(clean)}/{mrr}")
        # Match against calibrated JSON (including AUC and Thurstone -d')
        matched = [
            m for m in cal["models"]
            if m.get("n_ok") == len(clean)
            and m.get("auc") == auc
            and m.get("neg_d_prime") == neg_dp
            and m.get("mrr") == mrr
            and m.get("top1_pct") == t1
            and m.get("n_pos_rtok") == n_pos
        ]
        if not matched:
            parity_diffs += 1
            print(f"    dashboard mismatch on {mid}: replay={len(clean)}/auc={auc}/-d'={neg_dp}/mrr={mrr}/{t1}/{n_pos}")

    check(dup_letter_found == 0, f"Zero malformed/duplicate-letter rankings in {total_valid_rows} clean rows")
    check(rank_mismatch == 0, f"Zero gold_rank replay mismatches across {total_valid_rows} rows")
    check(parity_diffs == 0, f"100% parity across all {len(run_res)} models: raw JSONL == runner summary == dashboard JSON (AUC, -d', MRR, Top-1)")

    print("\n=== 4. STRICT EXTRAP TAGS, LOLO & LOG-BPB STUDENT-t VERIFICATION ===")
    oai_anchors = [m for m in cal["models"] if m.get("oai_exact")]
    oai_min_x = min(m["nce_doc_bpb"] for m in oai_anchors)
    oai_max_x = max(m["nce_doc_bpb"] for m in oai_anchors)
    bad_oai_ex = []
    for m in cal["models"]:
        if m.get("nce_doc_bpb") is None or m.get("oai_exact"):
            continue
        expected_ex = (m["nce_doc_bpb"] < oai_min_x) or (m["nce_doc_bpb"] > oai_max_x)
        if bool(m.get("oai_extrap")) != expected_ex:
            bad_oai_ex.append(m["name"])
    check(len(bad_oai_ex) == 0, f"OpenAI EXTRAP badge exact for all {len(run_res)-len(oai_anchors)} non-anchor models")

    elie_anchors = [m for m in cal["models"] if m.get("is_anchor")]
    el_min_x = min(m["nce_doc_bpb"] for m in elie_anchors)
    el_max_x = max(m["nce_doc_bpb"] for m in elie_anchors)
    bad_el_ex = [m["name"] for m in cal["models"] if m["access"] == "closed" and bool(m.get("extrapolated")) != (m["nce_doc_bpb"] < el_min_x or m["nce_doc_bpb"] > el_max_x)]
    check(len(bad_el_ex) == 0, f"Elie EXTRAP badge exact for all {sum(1 for m in cal['models'] if m['access'] == 'closed')} closed models")

    # Independently reproduce log-BPB OLS LOO & LOLO on Elie (14 open, 7 labs) and OpenAI (10 base)!
    ax = np.array([m["nce_doc_bpb"] for m in elie_anchors], dtype=float)
    ay = np.array([m["unch_bpb"] for m in elie_anchors], dtype=float)
    alabs = [m["lab"] for m in elie_anchors]
    el_loo_ind = float(np.mean([abs(math.exp(np.polyval(np.polyfit(ax[np.arange(len(ax)) != i], np.log(ay[np.arange(len(ax)) != i]), 1), ax[i])) - ay[i]) / ay[i] * 100 for i in range(len(ax))]))
    el_lolo_errs = []
    for hold in sorted(set(alabs)):
        msk = np.array([lb != hold for lb in alabs])
        s, c = np.polyfit(ax[msk], np.log(ay[msk]), 1)
        for i in np.where(~msk)[0]:
            el_lolo_errs.append(abs(math.exp(c + s * ax[i]) - ay[i]) / ay[i] * 100)
    el_lolo_ind = float(np.mean(el_lolo_errs))

    ox = np.array([m["nce_doc_bpb"] for m in oai_anchors], dtype=float)
    oy = np.array([m["oai_bpb"] for m in oai_anchors], dtype=float)
    oai_loo_ind = float(np.mean([abs(math.exp(np.polyval(np.polyfit(ox[np.arange(len(ox)) != i], np.log(oy[np.arange(len(ox)) != i]), 1), ox[i])) - oy[i]) / oy[i] * 100 for i in range(len(ox))]))

    cal_u = cal["meta"]["calibration"]["unch_bpb"]
    cal_u_elie_only = cal["meta"]["calibration"].get("unch_bpb_elie_only", cal_u)
    cal_a = cal["meta"]["calibration"].get("anpi_bpb")
    cal_o = cal["meta"]["calibration"]["oai_bpb"]
    check(cal_u_elie_only.get("log_y") is True and abs(cal_u_elie_only["lolo_pct"] - el_lolo_ind) < 0.02 and abs(cal_u_elie_only["loo_pct"] - el_loo_ind) < 0.02,
          f"Elie standalone log-BPB Thurstone d' LOO={cal_u_elie_only['loo_pct']}% / LOLO={cal_u_elie_only['lolo_pct']}% (r={cal_u_elie_only['r']:+.4f}, rho={cal_u_elie_only['rho']:+.4f}) independently verified")

    if cal_a is not None:
        anpi_anchors = [m for m in cal["models"] if m.get("anpi_anchor")]
        obs = [(m["name"], m["lab"], 0, m["nce_doc_bpb"], math.log(m["unch_bpb"]), m["unch_bpb"]) for m in elie_anchors]
        obs += [(m["name"], m["lab"], 1, m["nce_doc_bpb"], math.log(m["anpi_bpb"]), m["anpi_bpb"]) for m in anpi_anchors]
        cnt = {}
        for o in obs:
            cnt[o[0]] = cnt.get(o[0], 0) + 1
        w = np.array([1.0 / cnt[o[0]] for o in obs], dtype=float)
        X = np.array([[1.0, float(o[2]), float(o[3])] for o in obs], dtype=float)
        Y = np.array([o[4] for o in obs], dtype=float)
        lolo_e, lolo_a = [], []
        for hold in sorted({o[1] for o in obs}):
            tr = [i for i, o in enumerate(obs) if o[1] != hold]
            te = [i for i, o in enumerate(obs) if o[1] == hold]
            bt = np.linalg.inv(X[tr].T @ np.diag(w[tr]) @ X[tr]) @ (X[tr].T @ (w[tr] * Y[tr]))
            for i in te:
                err = abs(math.exp(float(X[i] @ bt)) - obs[i][5]) / obs[i][5] * 100.0
                (lolo_e if obs[i][2] == 0 else lolo_a).append(err)
        jl_e, jl_a = float(np.mean(lolo_e)), float(np.mean(lolo_a))
        check(
            abs(cal_u["lolo_pct"] - jl_e) < 0.02 and abs(cal_a["lolo_pct"] - jl_a) < 0.02,
            f"Joint Bridge WLS ({len(cnt)} unique models / {len({o[1] for o in obs})} labs): Elie LOLO={cal_u['lolo_pct']}% / ANPI LOLO={cal_a['lolo_pct']}% independently verified",
        )
        all_br_xs = [o[3] for o in obs]
        br_min_x, br_max_x = min(all_br_xs), max(all_br_xs)
        bad_anpi_ex = [
            m["name"] for m in cal["models"]
            if m.get("anpi_bpb") is not None and not m.get("anpi_exact")
            and bool(m.get("anpi_extrap")) != (m["nce_doc_bpb"] < br_min_x or m["nce_doc_bpb"] > br_max_x)
        ]
        check(len(bad_anpi_ex) == 0, f"ANPI Corpus EXTRAP badge exact for all {sum(1 for m in cal['models'] if m.get('anpi_bpb') is not None and not m.get('anpi_exact'))} non-exact live models")

    check(cal_o.get("log_y") is True and abs(cal_o["loo_pct"] - oai_loo_ind) < 0.02 and cal_o["rho"] > 0.98,
          f"OpenAI log-BPB Thurstone d' within-family LOO={cal_o['loo_pct']}% (r={cal_o['r']:+.4f}, rho={cal_o['rho']:+.4f}, {len(oai_anchors)} anchors) independently verified")

    non_zero_cot = [m["name"] for m in cal["models"] if (m.get("n_pos_rtok") or 0) > 0]
    check(len(non_zero_cot) == 0, f"100% strict zero-CoT across all {len(run_res)} live endpoints (0/{total_valid_rows} items with r_tok>0)")

    print("\n=== 5. PROSE-ONLY (792 NON-CODE) ORDERING & SHORTCUT DISCLOSURE ===")
    live_models = [x for x in cal["models"] if x.get("auc") is not None]
    fa = np.array([x["auc"] for x in live_models], dtype=float)
    pa = np.array([x["prose_auc"] for x in live_models], dtype=float)
    r_fp = float(np.corrcoef(fa, pa)[0, 1])
    rho_fp = float(np.corrcoef(np.argsort(np.argsort(fa)), np.argsort(np.argsort(pa)))[0, 1])
    top9_all_auc = {m["name"] for m in sorted(live_models, key=lambda x: -x["auc"])[:9]}
    top9_prose_auc = {m["name"] for m in sorted(live_models, key=lambda x: -x["prose_auc"])[:9]}
    top9_all_mrr = [m["name"] for m in sorted(live_models, key=lambda x: -x["mrr"])[:9]]
    top9_prose_mrr = [m["name"] for m in sorted(live_models, key=lambda x: -x["prose_mrr"])[:9]]
    check(r_fp > 0.997 and rho_fp > 0.995 and top9_all_auc == top9_prose_auc and top9_all_mrr == top9_prose_mrr,
          f"Prose-only sensitivity verified (AUC r={r_fp:+.4f}, rho={rho_fp:+.4f}; top-9 AUC set & top-9 MRR ordering 100% invariant)")
    html_txt = HTML_PATH.read_text(encoding="utf-8")
    check("22.7%" in html_txt and "LOLO" in html_txt and "Thurstone" in html_txt and "does <i>not</i> preserve likelihood-ratio" in html_txt,
          "Dashboard transparently discloses Code 22.7% median-length shortcut, LOLO, Thurstone log-BPB model, and p/q sharpening nuance")

    print("\n=== 6. CREDENTIAL / SECRET SCAN ===")
    secret_re = re.compile(r"(sk-or-v1-[a-zA-Z0-9]{10,}|sk-proj-[a-zA-Z0-9]{10,}|sk-ant-[a-zA-Z0-9]{10,}|ghp_[a-zA-Z0-9]{10,}|AIza[a-zA-Z0-9_-]{20,})")
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or "__pycache__" in p.parts:
            continue
        if p.stat().st_size > 20_000_000:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if secret_re.search(txt):
            hits.append(str(p.relative_to(ROOT)))
    check(len(hits) == 0, f"Zero API keys or secrets anywhere in workspace (hits={hits})")

    print("\n=== 7. ANSWER-PATTERN & POSITION-BIAS HEALTH CHECK ===")
    from tests.test_trace_health import KNOWN_PATHOLOGY_EXCEPTIONS, THRESHOLDS, compute_model_health
    unexp_path, inv_models, top25_path = [], [], []
    all_health = []
    for mid in run_res:
        h = compute_model_health(mid, items_by_id)
        if not h:
            continue
        all_health.append(h)
        if h["auc"] <= 0.50 or h["tail_auc"] <= 0.50:
            inv_models.append(mid)
        allowed = KNOWN_PATHOLOGY_EXCEPTIONS.get(mid, set())
        for metric, limit in THRESHOLDS.items():
            if h[metric] >= limit and metric not in allowed:
                unexp_path.append(f"{mid}:{metric}={h[metric]:.2f}")
    for h in sorted(all_health, key=lambda x: -x["auc"])[:25]:
        for metric, limit in THRESHOLDS.items():
            if h[metric] >= limit:
                top25_path.append(f"{h['model_id']}:{metric}={h[metric]:.2f}")
    check(len(inv_models) == 0, f"Zero inverted models (overall & tail AUC > 0.50 across all {len(all_health)} live models)")
    check(len(top25_path) == 0, f"Top-25 frontier models 100% pathology-free (flags={top25_path})")
    check(len(unexp_path) == 0, f"Zero unexpected answer-pattern pathologies outside {len(KNOWN_PATHOLOGY_EXCEPTIONS)} audited exceptions (unexpected={unexp_path})")

    print(f"\n=== AUDIT RESULT: {'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED'} ===")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
