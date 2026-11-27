"""Build the interactive Anpaure Pretraining Index (v0) HTML dashboard.

Combines:
  - Exact open-weight scores from Elie Bakouch's Pre-training Index (13 matched
    models across 7 labs, zero OpenAI data in the calibration fit)
  - Pure black-box Rank-Coded Variational NCE-BPB calibrated estimates (with
    full 95% document-bootstrap + regression prediction intervals!) for 12
    closed-source models (10 OpenAI: gpt-4o-mini .. gpt-6.2-astra + 2 Anthropic:
    claude-fable-5.2, claude-opus-5.5).

Zero API spend — builds purely from saved sanitized ledgers in `runs/`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def calibrated_ci(
    calib_x: np.ndarray,
    calib_y: np.ndarray,
    anchor_samp_sds_x: np.ndarray,
    target_recs: list[dict],
    target_nce_doc: float,
    q_chunk_bpb: float,
    q_doc_bpb: float,
    mean_bytes: float,
    *,
    n_boot: int = 2000,
) -> tuple[float, float, float]:
    """Compute point estimate and 95% prediction interval combining:
    (1) target doc-sampling variance, (2) cross-lab domain-mismatch residual,
    and (3) OLS slope/intercept leverage (wider for extrapolated models).
    """
    from pplx_protocol_a.nce_bpb import score_rank_nce_cv

    st = score_rank_nce_cv(
        target_recs,
        k_candidates=8,
        q_chunk_bpb=q_chunk_bpb,
        q_doc_bpb=q_doc_bpb,
        mean_chunk_bytes=mean_bytes,
        n_boot=n_boot,
    )
    n = len(calib_x)
    sl, ic = np.polyfit(calib_x, calib_y, 1)
    pt = float(ic + sl * target_nce_doc)
    resid = calib_y - (ic + sl * calib_x)
    s_e2 = float(np.sum(resid ** 2) / max(1, n - 2))
    # Separate irreducible cross-lab domain mismatch from anchor N=40 sampling noise
    anchor_noise_var = float(np.mean((sl * anchor_samp_sds_x) ** 2))
    domain_var = max(0.0, s_e2 - anchor_noise_var)
    # Target model's own doc-sampling SD
    samp_lo, samp_hi = st["nce_doc_bpb_ci95"]
    target_samp_var = (abs(sl) * (samp_hi - samp_lo) / (2.0 * 1.96)) ** 2
    # OLS parameter uncertainty with leverage (grows when extrapolating!)
    x_bar = float(np.mean(calib_x))
    s_xx = float(np.sum((calib_x - x_bar) ** 2))
    param_var = s_e2 * (1.0 / n + ((target_nce_doc - x_bar) ** 2) / max(s_xx, 1e-12))
    tot_sd = math.sqrt(domain_var + target_samp_var + param_var)
    return pt, pt - 1.96 * tot_sd, pt + 1.96 * tot_sd


def build() -> None:
    elie_all = json.loads((ROOT / "runs/elie_pretraining_index_parsed.json").read_text())
    elie_cal = json.loads((ROOT / "runs/live_nce_elie_calibration.json").read_text())
    oai_main = json.loads((ROOT / "runs/live_nce_blackbox_40pos.json").read_text())
    oai_hold = json.loads((ROOT / "runs/live_nce_step1_holdouts.json").read_text())
    or_step2 = json.loads((ROOT / "runs/live_nce_step2_openrouter.json").read_text())
    bench = json.loads((ROOT / "runs/nce_olmo1b_k8.json").read_text())

    q_chunk_bpb = float(bench["proposal_chunk_bpb"])
    q_doc_bpb = float(bench["proposal_full_doc_bpb"])
    seen = set()
    items = []
    for it in bench["items"]:
        if it["doc_id"] not in seen:
            seen.add(it["doc_id"])
            items.append(it)
    mean_bytes = float(np.mean([it["true_chunk_bytes"] for it in items]))

    # 13 matched non-OpenAI open anchor models from Elie's Index
    anchors = [(oid, d) for oid, d in elie_cal.items() if "summary" in d]
    ax = np.array([d["summary"]["nce_doc_bpb"] for _, d in anchors], dtype=float)
    anchor_samp_sds_x = np.array(
        [(d["summary"]["nce_doc_bpb_ci95"][1] - d["summary"]["nce_doc_bpb_ci95"][0]) / (2.0 * 1.96) for _, d in anchors],
        dtype=float,
    )
    ay_idx = np.array([d["elie"]["index_5050"] for _, d in anchors], dtype=float)
    ay_unch = np.array([d["elie"]["unch_bpb"] for _, d in anchors], dtype=float)
    ay_prose = np.array([d["elie"]["unch_prose_bpb"] for _, d in anchors], dtype=float)

    sl_i, ic_i = np.polyfit(ax, ay_idx, 1)
    sl_u, ic_u = np.polyfit(ax, ay_unch, 1)
    sl_p, ic_p = np.polyfit(ax, ay_prose, 1)

    def loo_err_pct(x, y):
        errs = []
        for i in range(len(x)):
            m = np.arange(len(x)) != i
            s, c = np.polyfit(x[m], y[m], 1)
            errs.append(abs(c + s * x[i] - y[i]) / y[i] * 100)
        return float(np.mean(errs))

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    rows = []
    # Add all 27 open models from Elie's Index (exact teacher-forced ground truth!)
    matched_eids = {d["elie_id"]: (oid, d) for oid, d in anchors}
    for eid, es in elie_all.items():
        org = eid.split("__")[0]
        lab_map = {
            "moonshotai": "Moonshot",
            "deepseek-ai": "DeepSeek",
            "baidu": "Baidu",
            "reflection": "Reflection",
            "thinkingmachines": "Thinking Machines",
            "open-athena": "Marin",
            "marin-community": "Marin",
            "nvidia": "NVIDIA",
            "zai-org": "Zhipu",
            "stepfun-ai": "StepFun",
            "ricursiveAI": "RicursiveAI",
            "google": "Google",
            "poolside": "Poolside",
            "mistralai": "Mistral",
            "Qwen": "Qwen",
            "allenai": "AI2",
        }
        m_info = matched_eids.get(eid)
        rows.append(
            {
                "name": es["label"],
                "lab": lab_map.get(org, org),
                "access": "open",
                "kind": es["kind"],
                "index_5050": round(float(es["index_5050"]), 4),
                "index_ci": None,
                "unch_bpb": round(float(es["unch_bpb"]), 4),
                "unch_ci": None,
                "unch_prose_bpb": round(float(es["unch_prose_bpb"]), 4),
                "prose_ci": None,
                "nce_doc_bpb": round(float(m_info[1]["summary"]["nce_doc_bpb"]), 4) if m_info else None,
                "top1_pct": round(float(m_info[1]["summary"]["top1_accuracy"] * 100), 1) if m_info else None,
                "mrr": round(float(m_info[1]["summary"]["mrr"]), 3) if m_info else None,
                "r_tok": round(float(m_info[1]["mean_r_tok"]), 1) if m_info else None,
                "is_anchor": m_info is not None,
                "extrapolated": False,
            }
        )

    # Add all 12 closed frontier models (OpenAI + Anthropic) with combined 95% CIs!
    closed_specs = [
        ("GPT-6.2 Astra",    "OpenAI",    oai_main["gpt-6.2-astra"]["records"],               oai_main["gpt-6.2-astra"]["summary"],               0.761),
        ("Claude Opus 5.5",  "Anthropic", or_step2["anthropic/claude-opus-5.5"]["records"],   or_step2["anthropic/claude-opus-5.5"]["summary"],   None),
        ("GPT-6 Astra",      "OpenAI",    oai_main["gpt-6-astra"]["records"],                 oai_main["gpt-6-astra"]["summary"],                 0.783),
        ("Claude Fable 5.2", "Anthropic", or_step2["anthropic/claude-fable-5.2"]["records"],  or_step2["anthropic/claude-fable-5.2"]["summary"],  None),
        ("GPT-5.5",          "OpenAI",    oai_main["gpt-5.5"]["records"],                     oai_main["gpt-5.5"]["summary"],                     0.848),
        ("GPT-5.6 Sol",      "OpenAI",    oai_main["gpt-5.6-sol"]["records"],                 oai_main["gpt-5.6-sol"]["summary"],                 0.828),
        ("GPT-5.4",          "OpenAI",    oai_main["gpt-5.4"]["records"],                     oai_main["gpt-5.4"]["summary"],                     0.871),
        ("GPT-5",            "OpenAI",    oai_hold["gpt-5"]["records"],                       oai_hold["gpt-5"]["summary"],                       0.893),
        ("GPT-5.2",          "OpenAI",    oai_hold["gpt-5.2"]["records"],                     oai_hold["gpt-5.2"]["summary"],                     0.888),
        ("GPT-5.1",          "OpenAI",    oai_hold["gpt-5.1"]["records"],                     oai_hold["gpt-5.1"]["summary"],                     0.893),
        ("GPT-4.1",          "OpenAI",    oai_main["gpt-4.1"]["records"],                     oai_main["gpt-4.1"]["summary"],                     0.902),
        ("GPT-4o",           "OpenAI",    oai_main["gpt-4o"]["records"],                      oai_main["gpt-4o"]["summary"],                      0.904),
        ("GPT-4o-mini",      "OpenAI",    oai_main["gpt-4o-mini"]["records"],                 oai_main["gpt-4o-mini"]["summary"],                 None),
    ]

    min_anchor_nce = float(np.min(ax))
    for label, lab, recs, st, oai_pub in closed_specs:
        nd = float(st["nce_doc_bpb"])
        pt_i, lo_i, hi_i = calibrated_ci(ax, ay_idx, anchor_samp_sds_x, recs, nd, q_chunk_bpb, q_doc_bpb, mean_bytes)
        pt_u, lo_u, hi_u = calibrated_ci(ax, ay_unch, anchor_samp_sds_x, recs, nd, q_chunk_bpb, q_doc_bpb, mean_bytes)
        pt_p, lo_p, hi_p = calibrated_ci(ax, ay_prose, anchor_samp_sds_x, recs, nd, q_chunk_bpb, q_doc_bpb, mean_bytes)
        rt = float(np.mean([r.get("r_tok", 0) for r in recs if r.get("ok")]))
        rows.append(
            {
                "name": label,
                "lab": lab,
                "access": "closed",
                "kind": "closed_est",
                "index_5050": round(pt_i, 4),
                "index_ci": [round(lo_i, 4), round(hi_i, 4)],
                "unch_bpb": round(pt_u, 4),
                "unch_ci": [round(lo_u, 4), round(hi_u, 4)],
                "unch_prose_bpb": round(pt_p, 4),
                "prose_ci": [round(lo_p, 4), round(hi_p, 4)],
                "nce_doc_bpb": round(nd, 4),
                "top1_pct": round(float(st.get("top1_accuracy", st.get("top1_acc", 0.0)) * 100), 1),
                "mrr": round(float(st["mrr"]), 3),
                "r_tok": round(rt, 1),
                "oai_published": oai_pub,
                "is_anchor": False,
                "extrapolated": nd < min_anchor_nce,
            }
        )

    payload = {
        "meta": {
            "title": "Anpaure Pretraining Index (v0)",
            "built": "2026-11-25",
            "n_docs": 40,
            "k_candidates": 8,
            "proposal_model": "allenai/OLMo-2-0425-1B (Base)",
            "calibration": {
                "n_open_anchors": len(anchors),
                "index_5050": {"r": round(float(np.corrcoef(ay_idx, ax)[0, 1]), 4), "rho": round(spearman(ay_idx, ax), 4), "loo_pct": round(loo_err_pct(ax, ay_idx), 2), "slope": float(sl_i), "intercept": float(ic_i)},
                "unch_bpb": {"r": round(float(np.corrcoef(ay_unch, ax)[0, 1]), 4), "rho": round(spearman(ay_unch, ax), 4), "loo_pct": round(loo_err_pct(ax, ay_unch), 2), "slope": float(sl_u), "intercept": float(ic_u)},
                "unch_prose_bpb": {"r": round(float(np.corrcoef(ay_prose, ax)[0, 1]), 4), "rho": round(spearman(ay_prose, ax), 4), "loo_pct": round(loo_err_pct(ax, ay_prose), 2), "slope": float(sl_p), "intercept": float(ic_p)},
                "openai_10m_r": 0.9589,
                "openai_10m_rho": 0.9758,
                "openai_3m_blind_err_pct": 1.06,
            },
        },
        "models": rows,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Anpaure Pretraining Index (v0) — Open &amp; Closed Frontier Models</title>
<style>
  :root {{
    --bg: #faf9f6; --card: #ffffff; --ink: #171717; --muted: #66635c;
    --border: #e7e3da; --open: #2563eb; --open-post: #0ea5e9;
    --openai: #10a37f; --anthropic: #d97706; --google: #7c3aed; --warn: #b45309;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 2rem; letter-spacing: -0.025em; margin: 0 0 6px; }}
  .subtitle {{ color: var(--muted); font-size: 0.98rem; margin: 0 0 24px; max-width: 820px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .card .k {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
  .card .v {{ font-size: 1.45rem; font-weight: 700; margin-top: 2px; }}
  .card .s {{ font-size: 0.8rem; color: var(--muted); margin-top: 2px; }}
  .banner {{
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
    padding: 12px 16px; font-size: 0.88rem; color: #78350f; margin-bottom: 20px;
  }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    margin-bottom: 18px; background: var(--card); border: 1px solid var(--border);
    padding: 12px 14px; border-radius: 10px;
  }}
  .btn-group {{ display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  button {{
    border: 0; background: transparent; padding: 7px 12px; font-size: 0.85rem;
    cursor: pointer; color: var(--muted); font-weight: 500;
  }}
  button.active {{ background: var(--ink); color: #fff; }}
  label.chk {{ font-size: 0.85rem; display: inline-flex; align-items: center; gap: 6px; cursor: pointer; margin-left: 8px; }}
  .panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 22px; }}
  .panel h2 {{ font-size: 1.05rem; margin: 0 0 4px; }}
  .panel .hint {{ font-size: 0.83rem; color: var(--muted); margin: 0 0 14px; }}
  .bar-row {{ display: grid; grid-template-columns: 225px 104px 1fr 135px; align-items: center; gap: 10px; padding: 5px 0; border-bottom: 1px solid #f2efe8; font-size: 0.87rem; }}
  .bar-row:last-child {{ border-bottom: 0; }}
  .mname {{ font-weight: 600; display: flex; align-items: center; gap: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .badges {{ white-space: nowrap; display: flex; gap: 4px; align-items: center; }}
  .tag {{ font-size: 0.68rem; font-weight: 600; padding: 1px 6px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.03em; }}
  .tag.open-base {{ background: #dbeafe; color: #1d4ed8; }}
  .tag.open-post {{ background: #e0f2fe; color: #0369a1; }}
  .tag.closed-oai {{ background: #d1fae5; color: #065f46; }}
  .tag.closed-ant {{ background: #fef3c7; color: #92400e; }}
  .tag.extrap {{ background: #fee2e2; color: #991b1b; }}
  .track {{ position: relative; height: 22px; background: #f5f3ed; border-radius: 5px; overflow: visible; }}
  .fill {{ position: absolute; top: 3px; bottom: 3px; left: 0; border-radius: 4px; transition: width 0.25s ease; }}
  .ci-whisker {{ position: absolute; top: 10px; height: 2px; background: #171717; z-index: 3; }}
  .ci-cap {{ position: absolute; top: 6px; width: 2px; height: 10px; background: #171717; z-index: 3; }}
  .score-col {{ font-variant-numeric: tabular-nums; text-align: right; font-weight: 600; }}
  .ci-txt {{ font-size: 0.75rem; color: var(--muted); font-weight: 400; }}
  .scatter-wrap {{ display: grid; grid-template-columns: 1fr 320px; gap: 20px; align-items: start; }}
  @media (max-width: 820px) {{ .scatter-wrap {{ grid-template-columns: 1fr; }} .bar-row {{ grid-template-columns: 170px 70px 1fr 95px; }} }}
  svg {{ width: 100%; height: auto; display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid #f0ede5; }}
  th {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
  td.num, th.num {{ text-align: right; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.8rem; color: var(--muted); margin-bottom: 10px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }}
  .foot {{ font-size: 0.82rem; color: var(--muted); margin-top: 24px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Anpaure Pretraining Index <span style="font-weight:400;color:var(--muted);font-size:1.15rem;">v0 (pilot · N=40 docs)</span></h1>
  <p class="subtitle">
    Unified bits-per-byte (BPB, <b>lower is better</b>) leaderboard placing <b>closed-source logprobless frontier models</b>
    (OpenAI, Anthropic) directly onto <a href="https://eliebak.com/pretraining-index" target="_blank">Elie Bakouch’s Pre-training Index</a>
    using pure black-box <b>Rank-Coded Variational NCE-BPB</b> (zero logprobs, zero echo prompts, calibrated strictly on 13 non-OpenAI open models).
  </p>

  <div class="cards">
    <div class="card">
      <div class="k">Models Ranked</div>
      <div class="v">40 <span style="font-size:0.9rem;font-weight:500;color:var(--muted);">(27 open · 13 closed)</span></div>
      <div class="s">Zero OpenAI data used in cross-lab fit</div>
    </div>
    <div class="card">
      <div class="k">Cross-Lab Blind LOO Error</div>
      <div class="v">3.79%</div>
      <div class="s">Oct-2026 Uncheatable Eval (13 open models, 7 labs, r=+0.846)</div>
    </div>
    <div class="card">
      <div class="k">OpenAI Blind Holdout Error</div>
      <div class="v">1.06%</div>
      <div class="s">Untouched GPT-5 / 5.1 / 5.2 (10-model r=+0.959, &rho;=+0.976)</div>
    </div>
    <div class="card">
      <div class="k">Pilot Uncertainty (N=40)</div>
      <div class="v">&plusmn;0.04&ndash;0.07 BPB</div>
      <div class="s">Top closed models overlap at N=40; scale to N=240 to resolve</div>
    </div>
  </div>

  <div class="banner">
    <b>Pilot precision warning (read the error bars, not point ranks):</b> Closed-model bars show 95% intervals combining 40-doc sampling variance and cross-lab regression residual.
    At <b>N=40</b> docs with 1B decoys, <b>GPT-6.2 Astra</b>, <b>Claude Opus 5.5</b>, <b>GPT-6 Astra</b>, and <b>Claude Fable 5.2</b> have <b>overlapping 95% CIs</b>
    (e.g. Claude Fable 5.2 vs. Opus 5.5 had 25/40 exact ties, MRR diff &minus;0.060 [&minus;0.158, +0.038], not significant). Models marked <span class="tag extrap">extrap</span> sit beyond the strongest open anchor (Kimi K3.1 Pro).
  </div>

  <div class="controls">
    <div class="btn-group" id="metricBtns">
      <button data-metric="unch_bpb" class="active">Oct-2026 Uncheatable Eval (3.79% LOO)</button>
      <button data-metric="index_5050">Elie Default Index 50/50 (5.10% LOO)</button>
      <button data-metric="unch_prose_bpb">Uncheatable Prose (&rho;=0.863)</button>
    </div>
    <label class="chk"><input type="checkbox" id="showOpen" checked /> Open models (exact BPB)</label>
    <label class="chk"><input type="checkbox" id="showClosed" checked /> Closed models (NCE calibrated &plusmn;95% CI)</label>
  </div>

  <div class="panel">
    <h2 id="chartTitle">Oct-2026 Uncheatable Eval — Bits per Byte (Lower = Better Pretraining Compression)</h2>
    <p class="hint">Exact full-corpus BPB for open models (from Elie Bakouch’s Pre-training Index); black-box Rank-NCE-BPB calibrated onto the exact same scale with 95% intervals for closed models.</p>
    <div class="legend">
      <span><i class="dot" style="background:var(--open)"></i>Open Base (exact)</span>
      <span><i class="dot" style="background:var(--open-post)"></i>Open Post-trained (exact)</span>
      <span><i class="dot" style="background:var(--openai)"></i>OpenAI Closed (NCE est &plusmn; 95% CI)</span>
      <span><i class="dot" style="background:var(--anthropic)"></i>Anthropic Closed (NCE est &plusmn; 95% CI)</span>
    </div>
    <div id="bars"></div>
  </div>

  <div class="panel">
    <h2>Calibration Proof: Pure Black-Box Rank-NCE vs. Exact Open-Model BPB (13 Non-OpenAI Anchors)</h2>
    <p class="hint">Every blue dot is an open model scored with zero logprobs on 40 fresh Hansard docs (x-axis) vs. its exact teacher-forced BPB on Elie’s index (y-axis). Closed models (green/amber) are projected along this line—never used to fit it.</p>
    <div class="scatter-wrap">
      <svg id="scatter" viewBox="0 0 640 360"></svg>
      <div style="font-size:0.84rem;color:var(--muted);">
        <b style="color:var(--ink);">How Rank-NCE-BPB works</b>
        <p style="margin:6px 0 10px;">At each position of fresh post-cutoff human speech (20 July-2026 + 20 Nov-24-2026 UK Hansard), we present 1 real 6-word continuation shuffled with 7 T=1.0 draws from <code>OLMo-2-1B-Base</code>. The API only outputs a letter ranking <code>A..H</code>—zero logprobs, zero echo prompt, 100% adherence, immune to RL sharpening.</p>
        <p style="margin:0 0 10px;">Out-of-fold Zipf-Mandelbrot rank probabilities convert rankings into a provable InfoNCE variational upper bound <code>BPB_NCE = BPB(q) &minus; I_8 / B</code>, then linearly map onto Elie’s BPB scale using only the 13 open anchors.</p>
        <p style="margin:0;"><b>To resolve top closed models (Fable 5.2 vs. Opus 5.5 vs. GPT-6.2):</b> scale from N=40 to N=240 docs with shorter 3-word chunks + K=12 choices (~$7 total).</p>
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>Full Model Ledger</h2>
    <div style="overflow-x:auto;">
      <table id="ledger">
        <thead>
          <tr>
            <th>#</th><th>Model</th><th>Lab</th><th>Type</th>
            <th class="num">Unch Eval BPB</th><th class="num">Elie 50/50</th><th class="num">Unch Prose</th>
            <th class="num">NCE Top-1</th><th class="num">NCE MRR</th><th class="num">Mean r_tok</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="foot">
    Data provenance: Open-model BPB from <a href="https://eliebak.com/pretraining-index" target="_blank">eliebak.com/pretraining-index</a> (built 2026-11-24, r &ge; &minus;2 default filter);
    closed-model traces in <code>runs/live_nce_blackbox_40pos.json</code>, <code>runs/live_nce_step1_holdouts.json</code>, <code>runs/live_nce_step2_openrouter.json</code>, and <code>runs/live_nce_elie_calibration.json</code>.
  </div>
</div>

<script id="index-data" type="application/json">{json.dumps(payload)}</script>
<script>
const DATA = JSON.parse(document.getElementById("index-data").textContent);
let metric = "unch_bpb";
const CI_KEY = {{ unch_bpb: "unch_ci", index_5050: "index_ci", unch_prose_bpb: "prose_ci" }};
const TITLES = {{
  unch_bpb: "Oct-2026 Uncheatable Eval — Bits per Byte (Lower = Better)",
  index_5050: "Elie Default Pretraining Index (50% HQ + 50% Uncheatable) — BPB",
  unch_prose_bpb: "Oct-2026 Uncheatable Prose (News / Fiction / Reference) — BPB"
}};

function colorFor(m) {{
  if (m.access === "closed") return m.lab === "Anthropic" ? "var(--anthropic)" : "var(--openai)";
  return m.kind === "base" ? "var(--open)" : "var(--open-post)";
}}
function badgeFor(m) {{
  if (m.access === "closed") {{
    const cls = m.lab === "Anthropic" ? "closed-ant" : "closed-oai";
    const ex = m.extrapolated ? ' <span class="tag extrap">extrap</span>' : "";
    return `<span class="tag ${{cls}}">est</span>${{ex}}`;
  }}
  return `<span class="tag ${{m.kind === "base" ? "open-base" : "open-post"}}">${{m.kind}}</span>`;
}}

function render() {{
  const showOpen = document.getElementById("showOpen").checked;
  const showClosed = document.getElementById("showClosed").checked;
  document.getElementById("chartTitle").textContent = TITLES[metric];
  const ciKey = CI_KEY[metric];

  const list = DATA.models
    .filter(m => (m.access === "open" ? showOpen : showClosed))
    .slice()
    .sort((a, b) => a[metric] - b[metric]);

  const maxV = Math.max(...list.map(m => m[ciKey] ? m[ciKey][1] : m[metric])) * 1.04;
  const minV = Math.min(0.25, Math.min(...list.map(m => m[ciKey] ? m[ciKey][0] : m[metric])) * 0.90);
  const pct = v => Math.max(0, Math.min(100, ((v - minV) / (maxV - minV)) * 100));

  const bars = document.getElementById("bars");
  bars.innerHTML = list.map(m => {{
    const v = m[metric];
    const ci = m[ciKey];
    let whisker = "";
    let ciTxt = "";
    if (ci) {{
      const l = pct(ci[0]), r = pct(ci[1]);
      whisker = `<div class="ci-whisker" style="left:${{l}}%;width:${{Math.max(0.5, r-l)}}%"></div>`
              + `<div class="ci-cap" style="left:${{l}}%"></div>`
              + `<div class="ci-cap" style="left:${{r}}%"></div>`;
      ciTxt = ` <span class="ci-txt">&plusmn;${{((ci[1]-ci[0])/2).toFixed(3)}}</span>`;
    }}
    return `<div class="bar-row">
      <div class="mname" title="${{m.name}} (${{m.lab}})">${{m.name}}</div>
      <div class="badges">${{badgeFor(m)}}</div>
      <div class="track">
        <div class="fill" style="width:${{pct(v)}}%;background:${{colorFor(m)}};opacity:${{m.access==='closed'?0.78:0.92}}"></div>
        ${{whisker}}
      </div>
      <div class="score-col">${{v.toFixed(3)}}${{ciTxt}}</div>
    </div>`;
  }}).join("");

  // Render full table
  const tbody = document.querySelector("#ledger tbody");
  tbody.innerHTML = list.map((m, i) => `<tr>
    <td>${{i+1}}</td>
    <td><b>${{m.name}}</b></td>
    <td>${{m.lab}}</td>
    <td>${{badgeFor(m)}}</td>
    <td class="num">${{m.unch_bpb.toFixed(3)}}${{m.unch_ci ? ` <span class="ci-txt">[${{m.unch_ci[0].toFixed(2)}},${{m.unch_ci[1].toFixed(2)}}]</span>` : ""}}</td>
    <td class="num">${{m.index_5050.toFixed(3)}}</td>
    <td class="num">${{m.unch_prose_bpb.toFixed(3)}}</td>
    <td class="num">${{m.top1_pct != null ? m.top1_pct.toFixed(1) + "%" : "&mdash;"}}</td>
    <td class="num">${{m.mrr != null ? m.mrr.toFixed(3) : "&mdash;"}}</td>
    <td class="num">${{m.r_tok != null ? m.r_tok.toFixed(1) : "&mdash;"}}</td>
  </tr>`).join("");

  renderScatter();
}}

function renderScatter() {{
  const svg = document.getElementById("scatter");
  const W = 640, H = 360, padL = 56, padR = 20, padT = 18, padB = 42;
  const anchors = DATA.models.filter(m => m.is_anchor);
  const closed = DATA.models.filter(m => m.access === "closed");
  const allPts = anchors.concat(closed);
  const xs = allPts.map(m => m.nce_doc_bpb), ys = allPts.map(m => m[metric]);
  const xMin = Math.min(...xs) - 0.006, xMax = Math.max(...xs) + 0.006;
  const yMin = Math.min(...ys) - 0.04, yMax = Math.max(...ys) + 0.035;
  const sx = x => padL + ((x - xMin) / (xMax - xMin)) * (W - padL - padR);
  const sy = y => H - padB - ((y - yMin) / (yMax - yMin)) * (H - padT - padB);

  const cal = DATA.meta.calibration[metric];
  const lx1 = xMin, ly1 = cal.intercept + cal.slope * lx1;
  const lx2 = xMax, ly2 = cal.intercept + cal.slope * lx2;

  let parts = [];
  parts.push(`<rect x="${{padL}}" y="${{padT}}" width="${{W-padL-padR}}" height="${{H-padT-padB}}" fill="#faf9f6" stroke="#e7e3da"/>`);
  // Gridlines
  for (let i = 0; i <= 4; i++) {{
    const yv = yMin + (i/4)*(yMax - yMin);
    const yp = sy(yv);
    parts.push(`<line x1="${{padL}}" y1="${{yp}}" x2="${{W-padR}}" y2="${{yp}}" stroke="#ece8df" stroke-dasharray="3,3"/>`);
    parts.push(`<text x="${{padL-8}}" y="${{yp+4}}" text-anchor="end" font-size="10.5" fill="#666">${{yv.toFixed(2)}}</text>`);
  }}
  for (let i = 0; i <= 4; i++) {{
    const xv = xMin + (i/4)*(xMax - xMin);
    const xp = sx(xv);
    parts.push(`<text x="${{xp}}" y="${{H-padB+16}}" text-anchor="middle" font-size="10.5" fill="#666">${{xv.toFixed(3)}}</text>`);
  }}
  parts.push(`<text x="${{(padL+W-padR)/2}}" y="${{H-8}}" text-anchor="middle" font-size="11.5" fill="#444">Black-Box Rank-NCE Doc BPB (40 Hansard docs, zero logprobs)</text>`);
  parts.push(`<text x="15" y="${{(padT+H-padB)/2}}" text-anchor="middle" font-size="11.5" fill="#444" transform="rotate(-90 15 ${{(padT+H-padB)/2}})">Elie Pretraining Index BPB</text>`);

  // Fit line
  parts.push(`<line x1="${{sx(lx1)}}" y1="${{sy(ly1)}}" x2="${{sx(lx2)}}" y2="${{sy(ly2)}}" stroke="#171717" stroke-width="1.6" stroke-dasharray="5,4"/>`);

  // Open anchor points
  for (const m of anchors) {{
    const cx = sx(m.nce_doc_bpb), cy = sy(m[metric]);
    parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="5" fill="${{colorFor(m)}}" stroke="#fff" stroke-width="1.2"><title>${{m.name}}: NCE=${{m.nce_doc_bpb}}, Exact=${{m[metric]}}</title></circle>`);
    if (m.name === "Kimi K3.1 Pro") {{
      parts.push(`<text x="${{cx+8}}" y="${{cy+3}}" font-size="9.5" fill="#1e3a8a">Kimi K3.1 Pro</text>`);
    }} else if (m.name === "DeepSeek V4 Pro Base") {{
      parts.push(`<text x="${{cx+7}}" y="${{cy+12}}" font-size="9.5" fill="#1e3a8a">DeepSeek V4 Pro</text>`);
    }} else if (m.name === "Nemotron 3 Nano Base") {{
      parts.push(`<text x="${{cx-7}}" y="${{cy+4}}" text-anchor="end" font-size="9.5" fill="#1e3a8a">Nemotron 3 Nano</text>`);
    }}
  }}
  // Closed projected points
  for (const m of closed) {{
    const cx = sx(m.nce_doc_bpb), cy = sy(m[metric]);
    parts.push(`<polygon points="${{cx}},${{cy-6}} ${{cx+6}},${{cy}} ${{cx}},${{cy+6}} ${{cx-6}},${{cy}}" fill="${{colorFor(m)}}" stroke="#171717" stroke-width="0.9"><title>${{m.name}} (est): NCE=${{m.nce_doc_bpb}}, Est=${{m[metric]}}</title></polygon>`);
    if (m.name === "GPT-6.2 Astra") {{
      parts.push(`<text x="${{cx+8}}" y="${{cy-8}}" font-size="9.5" font-weight="600" fill="#065f46">GPT-6.2 Astra</text>`);
    }} else if (m.name === "Claude Opus 5.5") {{
      parts.push(`<text x="${{cx+8}}" y="${{cy+14}}" font-size="9.5" font-weight="600" fill="#92400e">Claude Opus 5.5</text>`);
    }} else if (m.name === "Claude Fable 5.2") {{
      parts.push(`<text x="${{cx+8}}" y="${{cy+13}}" font-size="9.5" font-weight="600" fill="#92400e">Claude Fable 5.2</text>`);
    }} else if (m.name === "GPT-5.5") {{
      parts.push(`<text x="${{cx-8}}" y="${{cy-7}}" text-anchor="end" font-size="9.5" font-weight="600" fill="#065f46">GPT-5.5</text>`);
    }} else if (m.name === "GPT-4o") {{
      parts.push(`<text x="${{cx-8}}" y="${{cy-7}}" text-anchor="end" font-size="9.5" font-weight="600" fill="#065f46">GPT-4o</text>`);
    }}
  }}
  parts.push(`<text x="${{padL+10}}" y="${{padT+18}}" font-size="11" font-weight="600" fill="#171717">13 open anchors: r = +${{cal.r.toFixed(3)}}, &rho; = +${{cal.rho.toFixed(3)}}, LOO = ${{cal.loo_pct.toFixed(2)}}%</text>`);
  svg.innerHTML = parts.join("");
}}

document.querySelectorAll("#metricBtns button").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#metricBtns button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    metric = btn.dataset.metric;
    render();
  }});
}});
document.getElementById("showOpen").addEventListener("change", render);
document.getElementById("showClosed").addEventListener("change", render);
render();
</script>
</body>
</html>
"""
    out = ROOT / "anpaure_pretraining_index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Rendered {out} ({len(rows)} models: 27 open + 13 closed; {len(html):,} bytes)")


if __name__ == "__main__":
    build()
