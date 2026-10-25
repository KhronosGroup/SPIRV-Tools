# Mali SPIR-V Pass Performance Benchmarks

Measurements taken on host x86_64 across the same shader corpus (**10 iterations per shader**). Passes are enabled cumulatively in the order shown, so each column adds exactly one pass to the previous column:

- `FixSpec` = `--fix-mali-spec-constant-composite`
- `OptBarrier` = `--mali-optimization-barrier`
- `BitwiseScalar` = `--mali-bitwise-scalar`
- `ClampZero` = `--mali-clamp-zero`
- `Lockstep` = `--mali-lockstep`
- `PerVertexBlock` = `--mali-pervertex-block`

| Corpus Shader | None | +FixSpec | +OptBarrier | +BitwiseScalar | +ClampZero | +Lockstep | +PerVertexBlock |
|---|---|---|---|---|---|---|---|
| `optbarrier.spv` | 1.15s | 1.55s | 1.69s | 2.17s | 2.07s | 0.03s | 0.03s |
| `opselect.spv` | 0.08s | 0.11s | 0.11s | 0.11s | 0.12s | 0.03s | 0.03s |
| `gta_frag.spv` | 1.79s | 2.61s | 3.48s | 3.24s | 4.03s | 0.03s | 0.03s |
| `gta_vertex.spv` | 0.19s | 0.23s | 0.28s | 0.30s | 0.33s | 0.03s | 0.03s |
| `failing_shader.spv` | 0.91s | 1.26s | 1.19s | 1.50s | 1.76s | 0.03s | 0.03s |
| `clip_dist.spv` | 1.69s | 2.65s | 3.08s | 2.94s | 3.77s | 0.03s | 0.03s |
