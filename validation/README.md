# Validation suite

The full-accuracy runs of the DESIGN.md §4 validation ladder, plus other long
runs. Each rung is compared against published reference values. The suite is
separate from `Pkg.test()` (the fast suite, ≤ 30 s per testset; owner
decision, issue #75). Each rung keeps a smoke test of the same code path in
`test/validation_smoke_test.jl`, or next to the unit tests of its module.

Run it from the repository root, using the test environment:

```bash
julia --project=test validation/runtests.jl
```

It must pass before merging any PR that changes numerics or physics
(AGENTS.md §3). Reference data and their provenance live in
`test/reference_data/`. AGENTS.md rule 2 applies: tolerances change only
through a dedicated issue and a human decision.

| File | Rung / check | Source |
|---|---|---|
| `v1_williamson.jl` | V1: Williamson cases 5 and 6, grid imprinting | Williamson et al. 1992; Thuburn & Cotter 2014 |
| `v3_lock_exchange.jl` | V3: walled lock-exchange front Froude number | Shin, Dalziel & Linden 2004; Benjamin 1968 |
| `x62_regrid_lock_channel.jl` | 3-day lock channel with z* regrid: bounds, buoyancy, z* target | #62 |
| `x65_zstar_pgf_stratified.jl` | 5-day stratified rest over a slope (DESIGN.md §7 z* PGF risk) | #65 |

Files are included in name order. `v*` files are validation-ladder rungs;
`x*` files are long dycore checks.
