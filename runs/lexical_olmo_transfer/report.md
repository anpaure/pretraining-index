# Held-Out OLMo-2-0425-1B Lexical Transfer Results (Smol + Qwen Training Only)

Training families: `smol`, `qwen` (family-balanced, 120 positions each). Holdout: `olmo` (120 positions, zero leakage).
Fixed escape cost: `9.0` bits | Shared fitted `pi_miss = 0.5413` (Smol `0.6901`, Qwen `0.3926`).

## Fitted Parameters (Smol + Qwen Only)
- **ShanZM:** `alpha = 2.00`, `delta = 2.00`
- **Scalar HeadCal:** `T = 1.45`
- **Kinked Winner-Spike HeadCal:** `w1 = 0.60`, `w_rest = 0.70` (`lambda = 0.002`)

## Held-Out OLMo-2-0425-1B Evaluation (Units: bits per UTF-8 byte)

| Method | Native Lexical BPB (OLMo) | Offset -> Raw 4B | Pred Base Raw 4B (Truth `0.97615`) | Signed Error Raw 4B [95% CI] | Pass `<=0.03` | Offset -> Full Doc | Pred Base Full Doc (Truth `0.74269`) | Signed Error Doc [95% CI] | Pass `<=0.03` |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| ShanZM | 1.44482 | -0.30631 | 1.13851 | +0.16236 [+0.06118, +0.26782] | FAIL | -0.56981 | 0.87501 | +0.13232 [+0.03850, +0.23018] | FAIL |
| Scalar HeadCal | 1.43017 | -0.30360 | 1.12657 | +0.15042 [+0.04860, +0.25548] | FAIL | -0.56709 | 0.86307 | +0.12038 [+0.02467, +0.22246] | FAIL |
| Kinked HeadCal | 1.43003 | -0.30323 | 1.12680 | +0.15065 [+0.04893, +0.25535] | FAIL | -0.56672 | 0.86331 | +0.12062 [+0.02535, +0.22202] | FAIL |

All integrity checks passed: `True` (7/7 checks).
