"""Protocol A statistical estimators, control variates, and document-level bootstrap.

Implements all four complementary Protocol A estimators on matched Top-20
Direct OpenAI Responses observations:
  1. Certified Hard Lower Bound (BPB_LB):
       exact surprisal on hits; max(h_20, -log2(M_tail)) on misses.
  2. Zero-Proxy Geometric & Zipf(V) Tail Bracket (BPB_geom, BPB_zipf):
       closed-form tail expectations from (p_20, M_tail) alone, validated
       directly on live Sol outside-top-20 tail draws.
  3. Certified Operational Hybrid Compressor (BPB_comp, Upper Bound):
       exact closed-model Top-20 probabilities + M_tail * q(x_t | tail),
       yielding a valid normalized probability distribution (sum <= 1 ->
       Kraft-certified lossless compression rate, zero parametric extrapolation).
  4. Shifted Box-Cox Tobit Right-Censored MLE (BPB_tobit):
       exact Gaussian density in shifted Box-Cox space on uncensored hits +
       exact survival probability on right-censored misses, then imputes
       E[h_t | h_t >= censor_lb, h_t^(q)].

All aggregate metrics use slice-level ratio (sum bits / sum UTF-8 bytes) ->
macro-average across slices, with 95% confidence intervals bootstrapped over
whole documents (requests), plus optional proxy control-variate adjustment.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

LN2 = math.log(2.0)


@dataclass(frozen=True)
class PositionObservation:
    """Joined target observation + optional open-weight proxy features."""

    position_id: str
    slice_name: str
    doc_id: str
    target_utf8_bytes: int
    observed_exact: bool
    surprisal_bits: float | None
    censor_lb_bits: float
    h_20_bits: float
    p_20: float
    m_tail: float
    match_kind: str
    top_entries_logprobs: tuple[float, ...] = ()
    target_rank_in_top: int | None = None  # 1-indexed rank for WHOLE-target exact match only
    partial_prefix_rank_in_top: int | None = None  # 1-indexed rank for PARTIAL sub-prefix (strictly separate!)
    proxy_bits: float | None = None
    proxy_excluded_head_mass: float | None = None  # sum_{u in SolTop20} q(u), if available


def geom_tail_surprisal(p_20: float, m_tail: float) -> float:
    """Closed-form E[-log2(p_k) | k > 20] under geometric tail p_k = p_20 * r^(k-20)."""
    p20 = max(float(p_20), 1e-15)
    mt = max(float(m_tail), 1e-15)
    h20 = -math.log2(p20)
    # r = mt / (mt + p20); E[k - 20] = 1 / (1 - r) = (mt + p20) / p20
    # E[-log2(p_20 * r^(k-20))] = h20 + ((mt + p20) / p20) * log2(1 + p20 / mt)
    extra = ((mt + p20) / p20) * math.log2(1.0 + p20 / mt)
    return max(h20, -math.log2(mt), h20 + extra)


def zipf_tail_surprisal(p_20: float, m_tail: float, vocab_size: int = 200_000, k_head: int = 20) -> float:
    """Closed-form E[-log2(p_k) | k > k_head] under Zipf tail p_k = p_20 * (k/k_head)^(-alpha).

    Solves alpha uniquely so integral_{k_head+0.5}^{V+0.5} p_20*(x/k_head)^(-alpha) dx = m_tail.
    """
    p20 = max(float(p_20), 1e-15)
    mt = max(float(m_tail), 1e-15)
    h20 = -math.log2(p20)
    target = mt / p20
    u0 = (k_head + 0.5) / k_head
    u1 = (vocab_size + 0.5) / k_head

    def tail_sum(alpha: float) -> float:
        if abs(alpha - 1.0) < 1e-6:
            return k_head * math.log(u1 / u0)
        g = 1.0 - alpha
        return k_head * (u1**g - u0**g) / g

    if target >= tail_sum(0.01):
        # Nearly uniform over remaining vocabulary
        return max(h20, -math.log2(mt / max(1, vocab_size - k_head)))
    if target <= tail_sum(40.0):
        return max(h20, -math.log2(mt))

    lo, hi = 0.01, 40.0
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        if tail_sum(mid) > target:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)

    if abs(alpha - 1.0) < 1e-5:
        num = 0.5 * (math.log(u1) ** 2 - math.log(u0) ** 2)
        den = math.log(u1) - math.log(u0)
    else:
        g = 1.0 - alpha
        num = (u1**g * (g * math.log(u1) - 1.0) - u0**g * (g * math.log(u0) - 1.0)) / (g * g)
        den = (u1**g - u0**g) / g
    mean_ln_u = num / max(den, 1e-300)
    return max(h20, -math.log2(mt), h20 + (alpha / LN2) * mean_ln_u)


def _nelder_mead(f, x0: np.ndarray, step: float = 0.2, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Dependency-free deterministic Nelder-Mead optimizer."""
    n = len(x0)
    simplex = np.zeros((n + 1, n), dtype=float)
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += step
    vals = np.array([float(f(v)) for v in simplex], dtype=float)

    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if vals[-1] - vals[0] < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        vr = float(f(xr))
        if vals[0] <= vr < vals[-2]:
            simplex[-1], vals[-1] = xr, vr
        elif vr < vals[0]:
            xe = centroid + 2.0 * (xr - centroid)
            ve = float(f(xe))
            if ve < vr:
                simplex[-1], vals[-1] = xe, ve
            else:
                simplex[-1], vals[-1] = xr, vr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            vc = float(f(xc))
            if vc < vals[-1]:
                simplex[-1], vals[-1] = xc, vc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = float(f(simplex[i]))
    return simplex[0]


def _boxcox_shifted(h: np.ndarray, lam: float, shift: float) -> np.ndarray:
    u = np.maximum(h + shift, 1e-4)
    if abs(lam) < 1e-3:
        return np.log(u)
    return (np.power(u, lam) - 1.0) / lam


def _inv_boxcox_shifted(y: np.ndarray, lam: float, shift: float) -> np.ndarray:
    if abs(lam) < 1e-3:
        u = np.exp(np.clip(y, -10.0, 4.5))
    else:
        u = np.power(np.maximum(1e-4, 1.0 + lam * y), 1.0 / lam)
    return np.maximum(0.005, u - shift)


def fit_shifted_boxcox_tobit(
    obs: Sequence[PositionObservation],
    *,
    shift: float = 0.35,
    max_tail_bits: float = 50.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit Shifted Box-Cox Tobit right-censored MLE using proxy_bits as covariate.

    Returns `(per_position_bits, fit_diagnostics)`.
    Requires every observation in `obs` to have `proxy_bits is not None`.
    """
    M = len(obs)
    if M == 0:
        raise ValueError("Cannot fit Tobit on empty observations")

    proxy_bits = np.array([float(o.proxy_bits) for o in obs], dtype=float)
    obs_mask = np.array([o.observed_exact and o.surprisal_bits is not None for o in obs], dtype=bool)
    cens_mask = ~obs_mask

    h_obs = np.array(
        [float(o.surprisal_bits) if (o.observed_exact and o.surprisal_bits is not None) else 0.0 for o in obs],
        dtype=float,
    )
    h_cens = np.array([max(0.01, float(o.censor_lb_bits)) for o in obs], dtype=float)

    if int(np.sum(obs_mask)) < 10:
        raise ValueError("Too few uncensored observations (<10) to fit Tobit MLE")
    if int(np.sum(cens_mask)) == 0:
        return h_obs.copy(), {"a": 0.0, "b": 1.0, "sigma": 0.0, "lambda": 0.0, "censored_count": 0}

    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def norm_sf(z: np.ndarray) -> np.ndarray:
        return np.array([0.5 * math.erfc(float(v) * inv_sqrt2) for v in z], dtype=float)

    def neg_loglik(params: np.ndarray) -> float:
        a, b, log_sig, raw_lam = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        lam = 0.8 / (1.0 + math.exp(-max(-6.0, min(6.0, raw_lam))))
        sig = math.exp(max(-4.0, min(1.5, log_sig)))
        x = _boxcox_shifted(proxy_bits, lam, shift)
        mu = a + b * x

        yo = _boxcox_shifted(h_obs[obs_mask], lam, shift)
        ll_o = (
            -0.5 * math.log(2.0 * math.pi)
            - math.log(sig)
            - 0.5 * ((yo - mu[obs_mask]) / sig) ** 2
            + (lam - 1.0) * np.log(np.maximum(h_obs[obs_mask] + shift, 1e-4))
        )

        yc = _boxcox_shifted(h_cens[cens_mask], lam, shift)
        zc = (yc - mu[cens_mask]) / sig
        sf = np.maximum(norm_sf(zc), 1e-300)
        ll_c = np.log(sf)

        reg = 1e-3 * (raw_lam ** 2) + 1e-4 * (log_sig ** 2)
        return -float(np.sum(ll_o) + np.sum(ll_c)) + reg

    best_p, best_val = None, float("inf")
    for init_rl in (-2.0, -0.5, 0.8):
        p = _nelder_mead(neg_loglik, np.array([0.0, 1.0, -0.7, init_rl], dtype=float), max_iter=500)
        v = neg_loglik(p)
        if v < best_val:
            best_val, best_p = v, p

    assert best_p is not None
    a, b, log_sig, raw_lam = float(best_p[0]), float(best_p[1]), float(best_p[2]), float(best_p[3])
    lam = 0.8 / (1.0 + math.exp(-max(-6.0, min(6.0, raw_lam))))
    sig = math.exp(max(-4.0, min(1.5, log_sig)))
    x = _boxcox_shifted(proxy_bits, lam, shift)
    mu = a + b * x

    # Impute exact conditional expectation E[h | h >= h_cens, proxy_bits] on censored rows
    h_est = h_obs.copy()
    grid_steps = (np.arange(48, dtype=float) + 0.5) * (6.5 / 48.0)
    for idx in np.where(cens_mask)[0]:
        y_c = float(_boxcox_shifted(np.array([h_cens[idx]]), lam, shift)[0])
        z_lo = (y_c - float(mu[idx])) / sig
        zs = z_lo + grid_steps
        log_pdf = -0.5 * zs * zs
        log_pdf -= np.max(log_pdf)
        w = np.exp(log_pdf)
        w /= np.sum(w)
        hs = np.clip(_inv_boxcox_shifted(float(mu[idx]) + sig * zs, lam, shift), h_cens[idx], max_tail_bits)
        h_est[idx] = float(np.sum(w * hs))

    diag = {
        "a": a,
        "b": b,
        "sigma": sig,
        "lambda": lam,
        "shift": shift,
        "neg_loglik": best_val,
        "observed_count": int(np.sum(obs_mask)),
        "censored_count": int(np.sum(cens_mask)),
    }
    return h_est, diag


def compute_operational_hybrid_bits(obs: Sequence[PositionObservation]) -> np.ndarray:
    """Compute per-position bits for the Certified Operational Hybrid Compressor (BPB_comp).

    Constructs the normalized hybrid distribution:
      p_tilde(v) = p_closed(v)                                   for v in Top20(p_closed)
      p_tilde(v) = M_tail * q(v) / (1 - sum_{u in Top20} q(u))   for v not in Top20(p_closed).

    If `proxy_excluded_head_mass` is unavailable, falls back conservatively to
    unrenormalized `M_tail * q(x_t)` (which satisfies sum_v p_tilde(v) <= 1,
    still a strictly valid Kraft-inequality lossless compression upper bound!).
    """
    out = np.zeros(len(obs), dtype=float)
    for i, o in enumerate(obs):
        if o.observed_exact and o.surprisal_bits is not None:
            out[i] = float(o.surprisal_bits)
        else:
            if o.proxy_bits is None:
                raise ValueError("BPB_comp requires proxy_bits on censored positions")
            q_target = 2.0 ** (-float(o.proxy_bits))
            if o.proxy_excluded_head_mass is not None:
                q_tail_denom = max(q_target, 1.0 - float(o.proxy_excluded_head_mass), 1e-12)
            else:
                # Conservative Kraft upper bound: q(tail) <= 1.0
                q_tail_denom = 1.0
            p_tilde = max(1e-300, float(o.m_tail) * (q_target / q_tail_denom))
            # Enforce exact upper-bound consistency: -log2(p_tilde) >= censor_lb_bits?
            # Wait: for a valid lossless compressor, p_tilde is its OWN distribution
            # (not claiming p_closed = p_tilde), so -log2(p_tilde) IS the exact code length!
            out[i] = -math.log2(p_tilde)
    return out


def _macro_slice_bpb(
    slices: Sequence[str],
    doc_ids: Sequence[str],
    bits: np.ndarray,
    byte_lens: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Compute per-slice (sum bits / sum bytes) and unweighted macro-average across slices."""
    per_slice: dict[str, list[int]] = {}
    for i, s in enumerate(slices):
        per_slice.setdefault(s, []).append(i)
    slice_bpbs: dict[str, float] = {}
    for s, idxs in sorted(per_slice.items()):
        idx_arr = np.array(idxs, dtype=int)
        denom = float(np.sum(byte_lens[idx_arr]))
        slice_bpbs[s] = float(np.sum(bits[idx_arr]) / denom) if denom > 0 else float("nan")
    macro = float(np.mean(list(slice_bpbs.values()))) if slice_bpbs else float("nan")
    return macro, slice_bpbs


def document_bootstrap_bpb(
    obs: Sequence[PositionObservation],
    bits: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 2026,
    proxy_bits: np.ndarray | None = None,
    proxy_full_slice_bpb: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute macro BPB, per-slice BPB, and document-level bootstrap 95% CI.

    Optionally applies a slice-level proxy control variate when both `proxy_bits`
    and `proxy_full_slice_bpb` (exact full-corpus teacher-forced proxy BPB per slice)
    are supplied, resampling whole documents consistently for both target and proxy!
    """
    slices = [o.slice_name for o in obs]
    doc_ids = [o.doc_id for o in obs]
    bl = np.array([float(o.target_utf8_bytes) for o in obs], dtype=float)

    point_macro, point_slices = _macro_slice_bpb(slices, doc_ids, bits, bl)

    # Group indices by (slice -> doc_id -> list[pos_idx])
    slice_docs: dict[str, dict[str, list[int]]] = {}
    for i, o in enumerate(obs):
        slice_docs.setdefault(o.slice_name, {}).setdefault(o.doc_id, []).append(i)

    # Pre-aggregate per-document (sum_bits, sum_bytes, sum_proxy_bits) for fast bootstrap
    doc_tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for s, dmap in sorted(slice_docs.items()):
        d_bits = np.array([float(np.sum(bits[idxs])) for idxs in dmap.values()], dtype=float)
        d_bytes = np.array([float(np.sum(bl[idxs])) for idxs in dmap.values()], dtype=float)
        d_pbits = (
            np.array([float(np.sum(proxy_bits[idxs])) for idxs in dmap.values()], dtype=float)
            if proxy_bits is not None
            else None
        )
        doc_tables[s] = (d_bits, d_bytes, d_pbits)

    use_cv = (
        proxy_bits is not None
        and proxy_full_slice_bpb is not None
        and all(s in proxy_full_slice_bpb for s in doc_tables)
    )

    # Fit per-slice optimal OLS beta for control variate on document BPB ratios
    cv_betas: dict[str, float] = {}
    cv_point_slices: dict[str, float] = {}
    if use_cv:
        assert proxy_full_slice_bpb is not None
        for s, (d_bits, d_bytes, d_pbits) in doc_tables.items():
            assert d_pbits is not None
            y_doc = d_bits / np.maximum(d_bytes, 1.0)
            x_doc = d_pbits / np.maximum(d_bytes, 1.0)
            var_x = float(np.var(x_doc))
            beta = float(np.cov(y_doc, x_doc, ddof=0)[0, 1] / var_x) if (len(x_doc) > 2 and var_x > 1e-12) else 0.0
            beta = max(0.0, min(2.5, beta))
            cv_betas[s] = beta
            sub_q = float(np.sum(d_pbits) / np.sum(d_bytes))
            cv_point_slices[s] = point_slices[s] - beta * (sub_q - float(proxy_full_slice_bpb[s]))
        cv_point_macro = float(np.mean(list(cv_point_slices.values())))
    else:
        cv_point_macro = None

    rng = np.random.default_rng(seed)
    boot_raw = np.empty(n_boot, dtype=float)
    boot_cv = np.empty(n_boot, dtype=float) if use_cv else None

    slice_names = sorted(doc_tables.keys())
    for b in range(n_boot):
        s_raw_vals = []
        s_cv_vals = []
        for s in slice_names:
            d_bits, d_bytes, d_pbits = doc_tables[s]
            k = len(d_bits)
            draw = rng.integers(0, k, size=k)
            denom = float(np.sum(d_bytes[draw]))
            r_val = float(np.sum(d_bits[draw]) / denom) if denom > 0 else float("nan")
            s_raw_vals.append(r_val)
            if use_cv:
                assert d_pbits is not None and proxy_full_slice_bpb is not None
                q_val = float(np.sum(d_pbits[draw]) / denom) if denom > 0 else float("nan")
                s_cv_vals.append(r_val - cv_betas[s] * (q_val - float(proxy_full_slice_bpb[s])))
        boot_raw[b] = float(np.mean(s_raw_vals))
        if use_cv and boot_cv is not None:
            boot_cv[b] = float(np.mean(s_cv_vals))

    lo_raw, hi_raw = float(np.percentile(boot_raw, 2.5)), float(np.percentile(boot_raw, 97.5))
    result: dict[str, Any] = {
        "macro_bpb": point_macro,
        "ci95": [lo_raw, hi_raw],
        "half_width": 0.5 * (hi_raw - lo_raw),
        "per_slice_bpb": point_slices,
        "n_positions": len(obs),
        "n_documents": sum(len(dmap) for dmap in slice_docs.values()),
    }
    if use_cv and boot_cv is not None and cv_point_macro is not None:
        lo_cv, hi_cv = float(np.percentile(boot_cv, 2.5)), float(np.percentile(boot_cv, 97.5))
        result["control_variate"] = {
            "macro_bpb": cv_point_macro,
            "ci95": [lo_cv, hi_cv],
            "half_width": 0.5 * (hi_cv - lo_cv),
            "per_slice_bpb": cv_point_slices,
            "betas": cv_betas,
        }
    return result


def _temp_rescaled_bits(obs: Sequence[PositionObservation], temperature: float, vocab_size: int = 200_000) -> np.ndarray:
    """Compute per-position bits under post-hoc temperature rescaling p_T(v) ~ p(v)^(1/T).

    Uses the self-consistent tail normalizer Z_tail(T) = N_eff * (M_tail / N_eff)^(1/T)
    where N_eff = clamp(M_tail / p_20, 1, vocab_size), which equals M_tail exactly at T=1
    and preserves a valid normalized distribution for all T > 0.
    """
    out = np.zeros(len(obs), dtype=float)
    T = max(0.25, float(temperature))
    for i, o in enumerate(obs):
        if not o.top_entries_logprobs:
            out[i] = float(o.surprisal_bits) if (o.observed_exact and o.surprisal_bits is not None) else float(o.censor_lb_bits)
            continue
        lps = np.array(o.top_entries_logprobs, dtype=float)
        scaled = lps / T
        p20 = max(1e-15, float(o.p_20))
        mt = max(1e-15, float(o.m_tail))
        n_eff = max(1.0, min(float(vocab_size), mt / p20))
        log_z_tail = math.log(n_eff) + math.log(mt / n_eff) / T
        all_terms = np.append(scaled, log_z_tail)
        mx = float(np.max(all_terms))
        log_Z = mx + math.log(float(np.sum(np.exp(all_terms - mx))))

        raw_bits = float(o.surprisal_bits) if (o.observed_exact and o.surprisal_bits is not None) else float(o.censor_lb_bits)
        lp_T = (-raw_bits * LN2) / T - log_Z
        out[i] = max(0.0, -lp_T / LN2)
    return out


def cross_validated_temp_calibrated_bpb(
    obs: Sequence[PositionObservation],
    *,
    n_folds: int = 5,
    vocab_size: int = 200_000,
    n_boot: int = 2000,
    seed: int = 2026,
    proxy_bits: np.ndarray | None = None,
    proxy_full_slice_bpb: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """5-fold document-level CV post-hoc temperature calibration (undoing RL sharpening)."""
    docs = sorted({(o.slice_name, o.doc_id) for o in obs})
    k_folds = min(n_folds, max(2, len(docs)))
    doc_fold = {d: idx % k_folds for idx, d in enumerate(docs)}
    t_grid = np.linspace(0.7, 3.5, 57)

    bl_all = np.array([float(o.target_utf8_bytes) for o in obs], dtype=float)
    cv_bits = np.zeros(len(obs), dtype=float)
    fold_temps: list[float] = []

    for fold in range(k_folds):
        tr_idx = [i for i, o in enumerate(obs) if doc_fold[(o.slice_name, o.doc_id)] != fold]
        te_idx = [i for i, o in enumerate(obs) if doc_fold[(o.slice_name, o.doc_id)] == fold]
        if not tr_idx or not te_idx:
            continue
        tr_obs = [obs[i] for i in tr_idx]
        tr_bl = bl_all[tr_idx]
        best_T = float(min(t_grid, key=lambda t: float(np.sum(_temp_rescaled_bits(tr_obs, float(t), vocab_size)) / np.sum(tr_bl))))
        fold_temps.append(best_T)
        te_bits = _temp_rescaled_bits([obs[i] for i in te_idx], best_T, vocab_size)
        for pos_i, val in zip(te_idx, te_bits):
            cv_bits[pos_i] = float(val)

    res = document_bootstrap_bpb(
        obs, cv_bits, n_boot=n_boot, seed=seed, proxy_bits=proxy_bits, proxy_full_slice_bpb=proxy_full_slice_bpb
    )
    res["optimal_temperature_mean"] = float(np.mean(fold_temps)) if fold_temps else 1.0
    res["optimal_temperature_folds"] = [round(t, 3) for t in fold_temps]
    return res


def compute_rank_and_sharpening_diagnostics(obs: Sequence[PositionObservation]) -> dict[str, float]:
    """Temperature-invariant rank metrics (strictly whole-target vs partial-prefix) & sharpening."""
    whole_ranks = [o.target_rank_in_top if o.target_rank_in_top is not None else 25 for o in obs]
    r_arr = np.array(whole_ranks, dtype=float)
    partial_only = [
        1.0 if (o.target_rank_in_top is None and o.partial_prefix_rank_in_top is not None) else 0.0
        for o in obs
    ]
    head_ents = []
    top1_bits = []
    gap_12, gap_1_20 = [], []
    for o in obs:
        if o.top_entries_logprobs:
            ps = [math.exp(lp) for lp in o.top_entries_logprobs]
            head_ents.append(-sum(p * (lp / LN2) for p, lp in zip(ps, o.top_entries_logprobs)))
            top1_bits.append(-o.top_entries_logprobs[0] / LN2)
            if len(o.top_entries_logprobs) >= 2:
                gap_12.append(o.top_entries_logprobs[0] - o.top_entries_logprobs[1])
            if len(o.top_entries_logprobs) >= 20:
                gap_1_20.append(o.top_entries_logprobs[0] - o.top_entries_logprobs[19])
    mean_g12 = float(np.mean(gap_12)) if gap_12 else float("nan")
    mean_g120 = float(np.mean(gap_1_20)) if gap_1_20 else float("nan")
    return {
        "top1_accuracy": float(np.mean(r_arr == 1)),
        "top3_accuracy": float(np.mean(r_arr <= 3)),
        "top5_accuracy": float(np.mean(r_arr <= 5)),
        "top20_whole_target_accuracy": float(np.mean(r_arr <= 20)),
        "top20_accuracy": float(np.mean(r_arr <= 20)),  # backwards-compatible alias (strictly whole-target!)
        "top20_partial_prefix_only_rate": float(np.mean(partial_only)),
        "mrr": float(np.mean(1.0 / r_arr)),
        "mean_log2_rank": float(np.mean(np.log2(r_arr))),
        "median_rank": float(np.median(r_arr)),
        "mean_top20_entropy_bits": float(np.mean(head_ents)) if head_ents else float("nan"),
        "median_top1_surprisal_bits": float(np.median(top1_bits)) if top1_bits else float("nan"),
        "mean_gap_l1_l2_nats": mean_g12,
        "mean_gap_l1_l20_nats": mean_g120,
        "rl_winner_spike_ratio": float(mean_g120 / mean_g12) if (gap_12 and gap_1_20 and mean_g12 > 1e-6) else float("nan"),
    }


def bpb_to_perplexity_metrics(bpb: float, bytes_per_token: float = 5.36) -> dict[str, float]:
    """Convert tokenizer-invariant BPB into byte-PPL, nats/token (log-PPL), and token-PPL."""
    nat_tok = float(bpb) * float(bytes_per_token) * LN2
    return {
        "bpb": float(bpb),
        "byte_ppl": float(2.0 ** float(bpb)),
        "nats_per_token": nat_tok,
        "token_ppl": float(math.exp(min(nat_tok, 50.0))),
        "bytes_per_token": float(bytes_per_token),
    }


def head_calibrated_compressor_cv(
    obs: Sequence[PositionObservation],
    *,
    tail_cond_bits: float = 9.0,
    n_folds: int = 5,
    n_boot: int = 2000,
    seed: int = 2026,
    proxy_bits: np.ndarray | None = None,
    proxy_full_slice_bpb: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Zero-Z_tail(T)-assumption 5-fold doc-CV Head-Normalized + Empirical-Tail Calibrated Compressor.

    Decomposes the predictive distribution cleanly into:
      p_cal(v) = (1 - pi_miss) * softmax(ell_{1:K} / T)_v    for v in Top-K
               = pi_miss * q_tail(v)                          for v not in Top-K

    Unlike full-vocabulary temperature scaling, softmax normalization occurs strictly
    over the K OBSERVED head tokens (eliminating Z_tail(T) garbage-token blowup),
    while pi_miss is estimated out-of-fold from the empirical Top-K miss rate.
    """
    docs = sorted({(o.slice_name, o.doc_id) for o in obs})
    k_folds = min(n_folds, max(2, len(docs)))
    doc_fold = {d: idx % k_folds for idx, d in enumerate(docs)}
    t_grid = np.linspace(0.7, 4.2, 71)

    cv_total_bits = np.zeros(len(obs), dtype=float)
    cv_head_bits = np.zeros(len(obs), dtype=float)
    cv_head_mask = np.zeros(len(obs), dtype=bool)
    fold_temps: list[float] = []
    fold_pi_miss: list[float] = []

    for fold in range(k_folds):
        tr_idx = [i for i, o in enumerate(obs) if doc_fold[(o.slice_name, o.doc_id)] != fold]
        te_idx = [i for i, o in enumerate(obs) if doc_fold[(o.slice_name, o.doc_id)] == fold]
        if not tr_idx or not te_idx:
            continue
        tr_obs = [obs[i] for i in tr_idx]
        tr_hits = [o for o in tr_obs if o.observed_exact and o.match_kind == "exact_top20" and o.top_entries_logprobs and o.surprisal_bits is not None]
        n_miss = len(tr_obs) - len(tr_hits)
        pi_m = (n_miss + 0.5) / (len(tr_obs) + 1.0)
        fold_pi_miss.append(pi_m)

        def head_nll(T: float) -> float:
            s = 0.0
            for o in tr_hits:
                lps = np.array(o.top_entries_logprobs, dtype=float) / T
                mx = float(np.max(lps))
                lse = mx + math.log(float(np.sum(np.exp(lps - mx))))
                lp_t = (-float(o.surprisal_bits) * LN2) / T  # type: ignore[arg-type]
                s += -(lp_t - lse) / LN2
            return s

        best_T = float(min(t_grid, key=head_nll)) if tr_hits else 1.0
        fold_temps.append(best_T)

        for i in te_idx:
            o = obs[i]
            if o.observed_exact and o.match_kind == "exact_top20" and o.top_entries_logprobs and o.surprisal_bits is not None:
                lps = np.array(o.top_entries_logprobs, dtype=float) / best_T
                mx = float(np.max(lps))
                lse = mx + math.log(float(np.sum(np.exp(lps - mx))))
                lp_t = (-float(o.surprisal_bits) * LN2) / best_T
                cond_b = max(0.0, -(lp_t - lse) / LN2)
                cv_head_bits[i] = cond_b
                cv_head_mask[i] = True
                cv_total_bits[i] = -math.log2(max(1.0 - pi_m, 1e-12)) + cond_b
            else:
                # If proxy_bits is supplied, use proxy's conditional tail surprisal; else fallback constant
                if o.proxy_bits is not None:
                    q_t = 2.0 ** (-float(o.proxy_bits))
                    q_denom = max(q_t, 1.0 - float(o.proxy_excluded_head_mass or 0.0), 1e-12)
                    t_bits = -math.log2(max(q_t / q_denom, 1e-300))
                else:
                    t_bits = float(tail_cond_bits)
                cv_total_bits[i] = -math.log2(max(pi_m, 1e-12)) + t_bits

    res = document_bootstrap_bpb(
        obs, cv_total_bits, n_boot=n_boot, seed=seed, proxy_bits=proxy_bits, proxy_full_slice_bpb=proxy_full_slice_bpb
    )
    bl_all = np.array([float(o.target_utf8_bytes) for o in obs], dtype=float)
    head_denom = float(np.sum(bl_all[cv_head_mask]))
    res["head_cond_bpb"] = float(np.sum(cv_head_bits[cv_head_mask]) / head_denom) if head_denom > 0 else float("nan")
    res["optimal_head_temperature_mean"] = float(np.mean(fold_temps)) if fold_temps else 1.0
    res["empirical_miss_rate_mean"] = float(np.mean(fold_pi_miss)) if fold_pi_miss else 0.0
    res["tail_cond_bits_assumption"] = float(tail_cond_bits)
    return res


def shannon_rank_coded_bpb_cv(
    obs: Sequence[PositionObservation],
    *,
    tail_cond_bits: float = 9.0,
    k_head: int = 20,
    n_folds: int = 5,
    n_boot: int = 2000,
    seed: int = 2026,
    proxy_bits: np.ndarray | None = None,
    proxy_full_slice_bpb: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Pure Shannon (1951) Rank-Coded BPB via 5-fold doc-CV Zipf-Mandelbrot rank law.

    100% mathematically invariant to ANY arbitrary position-dependent strictly
    increasing RL sharpening function ell_t(v) -> f_t(ell_t(v)).

    Encodes each target atom losslessly via:
      - If target is at rank r in 1..k_head: code length = -log2((1 - pi_miss) * q_ZM(r))
        where q_ZM(r) proportional to (r + delta)^(-alpha) over r = 1..k_head.
      - If target misses Top-k_head:         code length = -log2(pi_miss) + tail_bits
        (using open proxy q conditional tail surprisal when supplied).
    """
    docs = sorted({(o.slice_name, o.doc_id) for o in obs})
    k_folds = min(n_folds, max(2, len(docs)))
    doc_fold = {d: idx % k_folds for idx, d in enumerate(docs)}
    r_idx = np.arange(1, k_head + 1, dtype=float)

    cv_bits = np.zeros(len(obs), dtype=float)
    alphas, deltas, pi_misses = [], [], []

    for fold in range(k_folds):
        tr_idx = [i for i, o in enumerate(obs) if doc_fold[(o.slice_name, o.doc_id)] != fold]
        te_idx = [i for i, o in enumerate(obs) if doc_fold[(o.slice_name, o.doc_id)] == fold]
        if not tr_idx or not te_idx:
            continue
        tr_obs = [obs[i] for i in tr_idx]
        tr_ranks = [
            int(o.target_rank_in_top)
            for o in tr_obs
            if o.observed_exact and o.match_kind == "exact_top20" and o.target_rank_in_top is not None and 1 <= o.target_rank_in_top <= k_head
        ]
        n_miss = len(tr_obs) - len(tr_ranks)
        pi_m = (n_miss + 0.5) / (len(tr_obs) + 1.0)
        pi_misses.append(pi_m)

        best_a, best_d, best_v = 1.5, 0.25, float("inf")
        if tr_ranks:
            for a in np.linspace(0.6, 3.2, 27):
                for d in (0.0, 0.25, 0.5, 1.0, 2.0):
                    w = np.power(r_idx + d, -a)
                    probs = w / np.sum(w)
                    val = -sum(math.log2(float(probs[r - 1])) for r in tr_ranks)
                    if val < best_v:
                        best_v, best_a, best_d = val, float(a), float(d)
        alphas.append(best_a)
        deltas.append(best_d)
        w = np.power(r_idx + best_d, -best_a)
        p_zm = w / np.sum(w)

        for i in te_idx:
            o = obs[i]
            if o.observed_exact and o.match_kind == "exact_top20" and o.target_rank_in_top is not None and 1 <= o.target_rank_in_top <= k_head:
                cv_bits[i] = -math.log2(max(1.0 - pi_m, 1e-12)) - math.log2(float(p_zm[o.target_rank_in_top - 1]))
            else:
                if o.proxy_bits is not None:
                    q_t = 2.0 ** (-float(o.proxy_bits))
                    q_denom = max(q_t, 1.0 - float(o.proxy_excluded_head_mass or 0.0), 1e-12)
                    t_bits = -math.log2(max(q_t / q_denom, 1e-300))
                else:
                    t_bits = float(tail_cond_bits)
                cv_bits[i] = -math.log2(max(pi_m, 1e-12)) + t_bits

    res = document_bootstrap_bpb(
        obs, cv_bits, n_boot=n_boot, seed=seed, proxy_bits=proxy_bits, proxy_full_slice_bpb=proxy_full_slice_bpb
    )
    res["zipf_mandelbrot_alpha_mean"] = float(np.mean(alphas)) if alphas else 1.5
    res["zipf_mandelbrot_delta_mean"] = float(np.mean(deltas)) if deltas else 0.25
    res["empirical_miss_rate_mean"] = float(np.mean(pi_misses)) if pi_misses else 0.0
    res["monotone_rl_invariant"] = True
    return res


def cross_model_outlier_and_winrate(
    model_doc_bpb: Mapping[str, Mapping[str, float]],
    *,
    r_threshold: float = -2.0,
) -> dict[str, Any]:
    """Elie Bakouch's Pre-training Index diagnostics across a panel of models:
      1. Multiplicative log2 residual: r(m, d) = log2(loss(m, d) / (difficulty(d) * skill(m)))
         where difficulty(d) = median_m loss(m, d), skill(m) = median_d (loss(m, d) / difficulty(d)).
         Documents with r < -2.0 (4x lower than expected) are flagged as memorization outliers.
      2. Pairwise document win-rate matrix: fraction of shared docs where model A < model B.
    """
    models = sorted(model_doc_bpb.keys())
    if len(models) < 2:
        raise ValueError("Need at least 2 models for cross-model diagnostics")
    shared_docs = sorted(set.intersection(*(set(model_doc_bpb[m].keys()) for m in models)))
    if not shared_docs:
        raise ValueError("No shared documents across models")

    # Matrix L[m_idx, d_idx]
    L = np.array([[max(1e-6, float(model_doc_bpb[m][d])) for d in shared_docs] for m in models], dtype=float)
    difficulty = np.median(L, axis=0)  # shape (D,)
    rel = L / np.maximum(difficulty[None, :], 1e-6)
    skill = np.median(rel, axis=1)  # shape (M,)
    expected = skill[:, None] * difficulty[None, :]
    r_mat = np.log2(L / np.maximum(expected, 1e-9))

    flagged_docs = []
    for di, d in enumerate(shared_docs):
        flagged_models = [models[mi] for mi in range(len(models)) if r_mat[mi, di] < r_threshold]
        if flagged_models:
            flagged_docs.append({"doc_id": d, "flagged_models": flagged_models, "min_r": float(np.min(r_mat[:, di]))})

    # Pairwise win rates (average win share across all other models)
    win_matrix: dict[str, dict[str, float]] = {m: {} for m in models}
    avg_win_rate: dict[str, float] = {}
    for i, mA in enumerate(models):
        wins_list = []
        for j, mB in enumerate(models):
            if i == j:
                continue
            wr = float(np.mean(L[i] < L[j]) + 0.5 * np.mean(L[i] == L[j]))
            win_matrix[mA][mB] = round(wr, 4)
            wins_list.append(wr)
        avg_win_rate[mA] = round(float(np.mean(wins_list)), 4)

    return {
        "n_models": len(models),
        "n_shared_docs": len(shared_docs),
        "r_threshold": r_threshold,
        "flagged_outlier_docs": flagged_docs,
        "flagged_fraction": len(flagged_docs) / len(shared_docs),
        "skill_multipliers": {models[i]: round(float(skill[i]), 4) for i in range(len(models))},
        "avg_pairwise_win_rate": avg_win_rate,
        "pairwise_win_matrix": win_matrix,
    }


def evaluate_protocol_a(
    obs: Sequence[PositionObservation],
    *,
    proxy_full_slice_bpb: Mapping[str, float] | None = None,
    vocab_size: int = 200_000,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run all Protocol A estimators (raw, tail-bracketed, temp-calibrated, Tobit) and rank metrics."""
    if not obs:
        raise ValueError("No valid observations to score")

    M = len(obs)
    has_proxy = all(o.proxy_bits is not None for o in obs)
    proxy_arr = np.array([float(o.proxy_bits) for o in obs], dtype=float) if has_proxy else None

    # 1. Hard Lower Bound
    bits_lb = np.array(
        [
            float(o.surprisal_bits) if (o.observed_exact and o.surprisal_bits is not None) else float(o.censor_lb_bits)
            for o in obs
        ],
        dtype=float,
    )
    # 2. Zero-Proxy Geometric & Zipf Tail Brackets (applied strictly to miss_censored;
    #    subtoken_censored retains its exact subtoken bound instead of assuming rank > 20!)
    bits_geom = np.array(
        [
            float(o.surprisal_bits)
            if (o.observed_exact and o.surprisal_bits is not None)
            else (
                max(float(o.censor_lb_bits), geom_tail_surprisal(o.p_20, o.m_tail))
                if o.match_kind == "miss_censored"
                else float(o.censor_lb_bits)
            )
            for o in obs
        ],
        dtype=float,
    )
    bits_zipf = np.array(
        [
            float(o.surprisal_bits)
            if (o.observed_exact and o.surprisal_bits is not None)
            else (
                max(float(o.censor_lb_bits), zipf_tail_surprisal(o.p_20, o.m_tail, vocab_size=vocab_size))
                if o.match_kind == "miss_censored"
                else float(o.censor_lb_bits)
            )
            for o in obs
        ],
        dtype=float,
    )

    out: dict[str, Any] = {
        "counts": {
            "total": M,
            "observed_exact": sum(1 for o in obs if o.observed_exact),
            "censored": sum(1 for o in obs if not o.observed_exact),
            "exact_rate": sum(1 for o in obs if o.observed_exact) / M,
            "by_match_kind": {
                k: sum(1 for o in obs if o.match_kind == k)
                for k in sorted({o.match_kind for o in obs})
            },
        },
        "rank_and_sharpening": compute_rank_and_sharpening_diagnostics(obs),
        "bpb_lower_bound": document_bootstrap_bpb(
            obs, bits_lb, n_boot=n_boot, seed=seed, proxy_bits=proxy_arr, proxy_full_slice_bpb=proxy_full_slice_bpb
        ),
        "bpb_zero_proxy_geom": document_bootstrap_bpb(
            obs, bits_geom, n_boot=n_boot, seed=seed, proxy_bits=proxy_arr, proxy_full_slice_bpb=proxy_full_slice_bpb
        ),
        "bpb_zero_proxy_zipf": document_bootstrap_bpb(
            obs, bits_zipf, n_boot=n_boot, seed=seed, proxy_bits=proxy_arr, proxy_full_slice_bpb=proxy_full_slice_bpb
        ),
    }

    if any(o.top_entries_logprobs for o in obs):
        out["bpb_shannon_rank_cv"] = shannon_rank_coded_bpb_cv(
            obs,
            tail_cond_bits=9.0,
            n_boot=n_boot,
            seed=seed,
            proxy_bits=proxy_arr,
            proxy_full_slice_bpb=proxy_full_slice_bpb,
        )
        out["bpb_head_calibrated_cv"] = head_calibrated_compressor_cv(
            obs,
            tail_cond_bits=9.0,
            n_boot=n_boot,
            seed=seed,
            proxy_bits=proxy_arr,
            proxy_full_slice_bpb=proxy_full_slice_bpb,
        )
        out["bpb_temp_calibrated_cv"] = cross_validated_temp_calibrated_bpb(
            obs,
            vocab_size=vocab_size,
            n_boot=n_boot,
            seed=seed,
            proxy_bits=proxy_arr,
            proxy_full_slice_bpb=proxy_full_slice_bpb,
        )

    if has_proxy and proxy_arr is not None:
        # 3. Certified Operational Hybrid Compressor (Upper Bound)
        bits_comp = compute_operational_hybrid_bits(obs)
        out["bpb_operational_hybrid_comp"] = document_bootstrap_bpb(
            obs, bits_comp, n_boot=n_boot, seed=seed, proxy_bits=proxy_arr, proxy_full_slice_bpb=proxy_full_slice_bpb
        )

        # 4. Shifted Box-Cox Tobit Right-Censored MLE
        bits_tobit, tobit_diag = fit_shifted_boxcox_tobit(obs)
        tobit_res = document_bootstrap_bpb(
            obs, bits_tobit, n_boot=n_boot, seed=seed, proxy_bits=proxy_arr, proxy_full_slice_bpb=proxy_full_slice_bpb
        )
        tobit_res["fit"] = tobit_diag
        out["bpb_shifted_boxcox_tobit"] = tobit_res

        # Reference proxy subsample BPB
        out["proxy_subsample"] = document_bootstrap_bpb(obs, proxy_arr, n_boot=n_boot, seed=seed)
        if proxy_full_slice_bpb is not None:
            out["proxy_full_corpus"] = {
                "macro_bpb": float(np.mean(list(proxy_full_slice_bpb.values()))),
                "per_slice_bpb": dict(proxy_full_slice_bpb),
            }

    return out
