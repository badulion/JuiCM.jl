# JuiCM.jl

**JuiCM** (Julia coupled Climate Model — *work in progress*) is a
from-scratch, fully-coupled ocean–atmosphere general circulation model,
written in pure Julia. It exists to explore a specific architecture:
icosahedral Voronoi horizontal grids with TRiSK (Thuburn–Ringler–Skamarock–Klemp)
operators, a z\*-coordinate ocean vertical grid, and an honest
coupled ocean–atmosphere design — built component by component, each one
validated against published benchmarks before the next is built on it.

> **Status: early development.** The dynamical cores, coupler, physics
> parameterizations, and mesh generator are all being written from
> scratch — this is a research project, not a production model. Nothing
> here should be used for anything real yet. Expect breaking changes
> without notice.

## Why from scratch?

Because the architecture *is* the research contribution. Existing
components (ClimaCore, Oceananigans, SpeedyWeather, MPAS meshes) embody
other groups' design decisions; reusing them would foreclose the design
space this project is meant to explore. Infrastructure dependencies
(NCDatasets, OffsetArrays, SeawaterPolynomials, KernelAbstractions,
Makie) are fine — they are tools, not dycores.

The full rationale — every major design decision, its alternatives, and
its risks — lives in [DESIGN.md](DESIGN.md), which is the source of
truth for the architecture. Contributions must not contradict it; if a
task requires deviating from it, open an issue first.

## Design highlights

| Decision | Choice |
|----------|--------|
| Grid family | Icosahedral Voronoi C-grid with TRiSK operators (D4/D9) |
| Coupling | Fully coupled ocean–atmosphere from day one, component-first milestones (D3) |
| Ocean vertical | z\* (rescaled height) coordinates (D7) |
| Purpose | Research / open source (D1) |
| First target | Single node; MPI-ready field types (D6) |
| v0.1 milestone | Validated shallow-water dynamical core (D8) |

## The validation ladder

Every component must pass published benchmarks before downstream
components may depend on it. No green test proves the *physics* is
right — numerical bugs look like physics — so each rung is a published
benchmark with a quantitative pass criterion:

| Rung | Benchmark | Component |
|------|-----------|-----------|
| V0 | 2D advection, convergence ≥ 1.5 | Mesh + TRiSK operators |
| V0.5 | Mesh quality suite | Mesh |
| V1 | Williamson shallow-water cases 2, 5, 6 | SW dycore |
| V2 | Held–Suarez 3D atmosphere | Atmosphere |
| V3 | Gravity current / lock exchange | Ocean |
| V4 | Overflow on slope | Ocean + z\* |
| V5 | Global spin-up toy forcing, budgets closed | Coupled ocean |
| V6 | Aqua-planet coupled spin-up | Everything |

Details and pass criteria: [DESIGN.md §4](DESIGN.md).

## Current status

- **Phase 0 (mesh):** icosahedron construction, 4:1 subdivision, spherical
  Voronoi dual, SCVT (Lloyd) relaxation, full TRiSK geometry tables, and
  the tangential reconstruction weights — verified to round-off against
  Thuburn et al. (2009)'s defining identities. Mesh visualization and the
  quality suite are in `docs/` and `test/`. ✅
- **Phase 1 (shallow water):** TRiSK operators, halo exchange, SSPRK
  time stepping, and the shallow-water dynamical core — the dycore runs
  partitioned (owner-computes, ghost exchange once per SSPRK stage) and
  is bitwise identical to the single-region run. Williamson case 2
  conserves mass to round-off and total energy to time-truncation level,
  with case-2 error norms in the published range for the resolution
  class. Validation suite (cases 3/5/6) in progress.

## Documentation

- [DESIGN.md](DESIGN.md) — architecture, decisions D1–D9, validation
  ladder, phase plan. **Read this first.**
- [AGENTS.md](AGENTS.md) — workflow rules for AI agents (and humans)
  contributing to this repo: issues → branches → PRs, docs + tests
  before merge, local green tests as the merge gate.

## Development

```bash
git clone https://github.com/badulion/JuiCM.jl
cd JuiCM.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Julia ≥ 1.10. The full test suite must be green before any PR is merged.

## License

MIT — see [LICENSE](LICENSE).
