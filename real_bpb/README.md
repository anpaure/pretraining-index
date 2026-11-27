# Real BPB & White-Box Oracle Validation (`real_bpb` branch)

Evaluates exact, tokenizer-invariant teacher-forced **Bits-Per-Byte (BPB)** and **8-choice white-box oracle ranking** directly over the archived Nov-2026 ANPI corpus (`data/anpaure_v1_corpus_1000.jsonl`, `1,000` docs / `2.61 MB` UTF-8) and `954`-item ANPI testset (`runs/anpaure_v1_n1000.json`) via the Tinker API, plus strict zero-CoT black-box ANPI cross-provider reruns on the exact same weights.

See [`TINKER_VALIDATION_REPORT.md`](./TINKER_VALIDATION_REPORT.md) and `runs/bpb_tinker/SUMMARY.json` for full 14-model, 7-lab results (`r = +0.9916 / ρ = +0.9956` cross-provider black-box reproducibility; `ρ = +0.9341` white-box oracle vs. full-doc BPB; `ρ = +0.9385` black-box `-d'` vs. gold-span BPB).

## Quickstart

```bash
# White-box 954x8 ANPI testset + 1,000-doc corpus BPB
TINKER_API_KEY=... uvx --with tinker --with transformers --with sentencepiece --with tiktoken --with protobuf --with blobfile \
  python3 real_bpb/tinker_bpb.py --preset kimi-k3.1-pro --concurrency 96

# Strict zero-CoT black-box ANPI sanity rerun on Tinker
TINKER_API_KEY=... uvx --with tinker --with transformers --with jinja2 --with sentencepiece --with tiktoken --with protobuf --with blobfile \
  python3 real_bpb/tinker_blackbox_anpi.py --preset kimi-k3.1-pro --concurrency 96

# Regenerate 3-way comparison table & SUMMARY.json
python3 real_bpb/analyze_tinker_vs_anpi.py
```
