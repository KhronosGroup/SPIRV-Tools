# Mali SPIR-V Pass Performance Benchmarks

Measurements taken on host x86_64 across the same shader corpus (**10 iterations per shader**). Passes are enabled cumulatively in the order shown, so each column adds exactly one pass to the previous column:

- `FixSpec` = `--fix-mali-spec-constant-composite`
- `OptBarrier` = `--mali-optimization-barrier`
- `BitwiseScalar` = `--mali-bitwise-scalar`
- `ClampZero` = `--mali-clamp-zero`

| Corpus Shader | None | +FixSpec | +OptBarrier | +BitwiseScalar | +ClampZero |
|---|---|---|---|---|---|
| `optbarrier.spv` | 1.45s | 1.96s | 2.18s | 2.73s | 2.60s |
| `opselect.spv` | 0.09s | 0.13s | 0.12s | 0.13s | 0.13s |
| `gta_frag.spv` | 2.34s | 3.48s | 3.97s | 3.98s | 5.09s |
| `gta_vertex.spv` | 0.24s | 0.32s | 0.37s | 0.39s | 0.44s |
| `failing_shader.spv` | 1.25s | 1.66s | 1.69s | 2.11s | 2.31s |
| `clip_dist.spv` | 2.38s | 3.31s | 3.92s | 3.91s | 5.05s |
