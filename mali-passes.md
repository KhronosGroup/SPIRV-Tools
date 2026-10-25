# Mali SPIR-V Pass Performance Benchmarks

Measurements taken on host x86_64 across the same shader corpus (**10 iterations per shader**). Passes are enabled cumulatively in the order shown, so each column adds exactly one pass to the previous column:

- `FixSpec` = `--fix-mali-spec-constant-composite`
- `PerVertexBlock` = `--mali-pervertex-block`
- `OptBarrier` = `--mali-optimization-barrier`
- `BitwiseScalar` = `--mali-bitwise-scalar`
- `ClampZero` = `--mali-clamp-zero`

| Corpus Shader | None | +FixSpec | +PerVertexBlock | +OptBarrier | +BitwiseScalar | +ClampZero |
|---|---|---|---|---|---|---|
| `optbarrier.spv` | 1.44s | 2.04s | 0.03s | 0.03s | 0.03s | 0.03s |
| `opselect.spv` | 0.10s | 0.13s | 0.04s | 0.04s | 0.03s | 0.03s |
| `gta_frag.spv` | 2.26s | 3.22s | 0.04s | 0.03s | 0.03s | 0.03s |
| `gta_vertex.spv` | 0.22s | 0.30s | 0.03s | 0.03s | 0.03s | 0.03s |
| `failing_shader.spv` | 1.22s | 1.62s | 0.03s | 0.03s | 0.03s | 0.04s |
| `clip_dist.spv` | 2.28s | 3.40s | 0.03s | 0.03s | 0.03s | 0.04s |
