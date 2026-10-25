# Mali SPIR-V Pass Performance & Errata Benchmarks

## 1. Existing Pass Hardening Benchmark
Measurements taken on host x86_64 across representative game shader corpus (100 iterations per pass):

| Corpus Shader | Pass Configuration | Baseline (100 iters) | Hardened (100 iters) | Memory Impact | Def-Use Safety |
|---|---|---|---|---|---|
| `optbarrier.spv` | FixSpec + OptBarrier | 20.82s | 21.49s | Minimal stack overhead | Verified null guards & SSA def-use |
| `opselect.spv` | FixSpec + OptBarrier | 1.30s | 1.29s | Minimal stack overhead | Verified null guards & SSA def-use |
| `gta_frag.spv` | FixSpec + OptBarrier | 40.46s | 39.60s | Minimal stack overhead | Verified null guards & SSA def-use |
| `gta_vertex.spv` | FixSpec + OptBarrier | 3.46s | 3.45s | Minimal stack overhead | Verified null guards & SSA def-use |
| `failing_shader.spv` | FixSpec + OptBarrier | 16.65s | 16.31s | Minimal stack overhead | Verified null guards & SSA def-use |
| `clip_dist.spv` | FixSpec + OptBarrier | 40.34s | 39.80s | Minimal stack overhead | Verified null guards & SSA def-use |
