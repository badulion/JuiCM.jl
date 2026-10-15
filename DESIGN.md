# JuiCM.jl — Design Document

A modular, fully-coupled General Circulation Model written from scratch in Julia.

**Status:** Planning (v0.0) — no code exists yet.
**Date:** 2026-10-15

---

## 1. Summary

JuiCM.jl is a from-scratch, research-grade coupled ocean-atmosphere GCM in pure
Julia. The architecture itself is part of the research contribution: a modular
component model where dynamical cores, physics parameterizations, and the
ocean-atmosphere coupler are independently swappable modules behind stable
interfaces.

The core thesis: **everything is a module behind a typed interface, validated
against published reference solutions before it is allowed to touch the coupled
system.**

---

## 2. Decision Log

| ID | Decision | Choice | Rationale |
|----|----------|--------|-----------|
| D1 | Project goal | Research / open source | Validation, docs, and extensibility are first-class requirements |
| D2 | Domain | Ocean / coupled | Coupled ocean-atmosphere system |
| D3 | Coupling strategy | Fully coupled from day one | Coupler interfaces designed against real coupled physics, not retrofitted |
| D4 | Horizontal grid | Unstructured (cubed-sphere) | No pole singularity; scales to eddying resolution; matches MPAS/ICON direction |
| D5 | Build vs reuse | From scratch | The architecture *is* the research contribution |
| D6 | Parallel target | Single node first | Fast debugging loop; field layout designed so MPI can slot in later |
| D7 | Vertical coordinate | z-star (rescaled depth) | Realistic bathymetry, free-surface motion, MOM6/MPAS-compatible validation |
| D8 | v0.1 milestone | Validated dycore benchmark | External ground truth is the only unambiguous bug detector |

### Resolved tension: D3 vs D8

"Coupled from day one" and "validate dycores first" conflict if read naively.
Resolution: **the architecture is coupled-first, the milestones are
component-first.**

- Coupler interfaces, state containers, and the driver exist from commit one.
- But each component is validated in isolation against published benchmarks
  before its output is trusted in a coupled run.
- The coupled system's first spin-up uses only components that have individually
  passed their benchmarks.

### Open sub-decision: cubed-sphere vs icosahedral (from D4)

These are different machines, not variants of one choice:

| | Cubed-sphere (finite-volume) | Icosahedral (Voronoi/Delaunay) |
|--|------------------------------|-------------------------------|
| Precedent | Oceananigans cubed sphere, GEOS, MITgim | MPAS-Ocean, ICON-O, DYNAMICO |
| Mesh generation | 6 structured faces, moderate | Fibonacci/recursive subdivision, then Voronoi dual — heavy |
| Operators | Structured stencils *within* faces + halo exchange at 8 face edges | TRiSK operator set on unstructured primal/dual meshes |
| Julia difficulty | Moderate — faces are ordinary arrays with periodic-style edge exchange | High — you own mesh generation, dual meshes, and flux reconstruction |

**Recommendation: cubed-sphere finite-volume.** Within each of the 6 faces the
grid is structured, so Julia arrays and stencil loops work directly; only the 8
face-edge boundaries need interpolation logic. This is the fastest path from
scratch to a validated 3D dycore, and Oceananigans.jl's cubed-sphere work
provides a published precedent for the face-edge exchange scheme. The TRiSK
icosahedral path is a research project in computational geometry before it is a
GCM.

This needs explicit sign-off before Phase 1 begins.

---

## 3. Architecture

### 3.1 Layer diagram

```
┌─────────────────────────────────────────────────────────┐
│                      Driver / Experiment                 │
│        (run scripts, checkpoint/restart, monitoring)      │
├──────────────┬──────────────────────────┬───────────────┤
│   Atmos      │        Coupler           │    Ocean      │
│  ┌─────────┐ │  ┌────────────────────┐  │ ┌───────────┐ │
│  │ Dycore  │ │  │  flux computation  │  │ │  Dycore   │ │
│  ├─────────┤ │  │  state remapping    │  │ ├───────────┤ │
│  │ Physics │◄┼──┤  (atmos ↔ ocean     ├──┼►│  Physics  │ │
│  │ modules │ │  │   grid interpolation)│  │ │  modules  │ │
│  └─────────┘ │  └────────────────────┘  │ └───────────┘ │
├──────────────┴──────────────────────────┴───────────────┤
│         Shared infrastructure (grids, fields, I/O)       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Package layout

One Julia package, submodules for each layer. Submodules — not separate
packages — so development stays atomic while boundaries stay clean. Any
submodule can be split into its own package later if the interface holds.

```
JuiCM.jl/
├── Project.toml
├── src/
│   ├── JuiCM.jl              # top module: using ./re-exports
│   ├── Grids/
│   │   ├── CubedSphere.jl    # face generation, coordinates, metric terms
│   │   ├── Connectivity.jl    # face-edge exchange tables, halos
│   │   └── VertGrids.jl      # z-star stretched vertical levels
│   ├── Fields/
│   │   ├── Field.jl         # Field{G, T, N}: data + grid + location
│   │   ├── Operators.jl      # grad, div, curl, interpolation (stencils)
│   │   └── Exchange.jl       # halo update across face edges
│   ├── Dycore/
│   │   ├── Atmos/
│   │   │   ├── Prognostics.jl    # state containers
│   │   │   ├── Dynamics.jl        # continuity, momentum, thermodynamics
│   │   │   └── TimeStepping.jl
│   │   └── Ocean/
│   │       ├── Prognostics.jl
│   │       ├── Dynamics.jl        # primitive eqs, z-star pressure gradient
│   │       └── TimeStepping.jl
│   ├── Physics/              # one file per parameterization, all same interface
│   │   ├── Interface.jl
│   │   ├── HeldSuarez.jl      # toy forcing (validation phase)
│   │   ├── VerticalMixing.jl
│   │   ├── Convection.jl
│   │   └── EOS.jl             # seawater equation of state (TEOS-10 via SeawaterPolynomials.jl)
│   ├── Coupler/
│   │   ├── Interface.jl
│   │   ├── Fluxes.jl          # bulk formulas → momentum, heat, freshwater
│   │   └── Remapping.jl      # atmos ↔ ocean grid interpolation
│   ├── Diagnostics/
│   │   ├── Budgets.jl        # mass/energy/tracer conservation monitors
│   │   └── Output.jl         # NetCDF writer
│   └── Benchmarks/
│       ├── ShallowWater/      # Williamson et al. 1992 test cases
│       ├── Atmos/             # Held-Suarez 1994
│       ├── Ocean/             # gravity current, overflow, double-gyre
│       └── Coupled/           # aqua-planet spin-up
├── test/
│   ├── runtests.jl
│   └── ...                    # per-module unit tests + reference comparisons
├── docs/
└── .github/workflows/ci.yml
```

### 3.3 Core abstractions

These five types are the architecture. Everything else is implementations of them.

```julia
# 1. Grid — immutable geometry + connectivity
abstract type AbstractGrid{FT} end
struct CubedSphereGrid{FT} <: AbstractGrid{FT}
    # 6 faces × Ni × Nj × Nk, coordinates, metric terms, edge tables
end

# 2. Field — data at a grid location. Parametric for future MPI/GPU.
struct Field{G<:AbstractGrid, T, N, AX}
    data::OffsetArray{T, N, AX}   # AX is abstract → Array today, MPIArray later
    grid::G
    location::FieldLocation        # Center / Face1 / Face2 / Face3 (C-grid staggering)
end

# 3. Prognostic state — what time integration advances
struct PrognosticState{FT}
    u::Field   # staggered velocities
    η::Field   # free surface (ocean) / surface pressure (atmos)
    θ::Field   # potential temperature
    S::Field   # salinity (ocean), tracers generally
end

# 4. Physics module — everything implements this
abstract type AbstractPhysicsModule end
"Compute tendencies into `tend` given state `state`."
function compute_tendency!(tend, state, mod::AbstractPhysicsModule, t)
    ...
end

# 5. Component — dycore + physics stack, what the coupler sees
abstract type AbstractComponent end
"Advance one macro time step; expose surface fluxes."
function step!(comp::AbstractComponent, t, Δt) ... end
function surface_fluxes(comp::AbstractComponent) ... end
```

**Why this works in Julia:** the physics interface is a single function
(`compute_tendency!`) dispatched on module type, so physics modules are pure
data + one function. Adding a parameterization = defining one struct + one
method. Swapping the ocean dycore = new `AbstractComponent` subtype; the
coupler never sees the difference. Function barriers (`step!`, the stencil
kernels inside `Operators.jl`) keep type-stability once grids and fields are
concrete.

**MPI escape hatch (D6):** `Field.data` is typed by an abstract array parameter.
Day one it's `Array`. Later, `MPIMultiArray`-style types can replace it without
touching physics code — provided physics code never indexes `data` directly,
only through operators. Enforce this in review and lint.

### 3.4 Coupler design

The coupler is a module, not a god-object:

1. **Pull** surface fluxes from each component (`surface_fluxes`).
2. Compute coupled fluxes from bulk formulas (momentum stress, latent +
   sensible heat, evaporation - precipitation).
3. **Remap** fluxes between the atmospheric and ocean grids (both cubed-sphere
   but potentially different resolutions — conservative remapping via
   overlap areas).
4. **Push** fluxes into each component's next tendency evaluation.

Conservation rule: the coupler never creates or destroys mass/energy; it moves
it. Every flux field crossing components is logged in `Budgets.jl` so a closed
budget is a runtime-checkable invariant, not a hope.

---

## 4. Validation ladder

Each rung is a published benchmark with a quantitative pass criterion. No
component advances to the next rung until it passes. This is the regression
suite for every future change.

| Rung | Test | Pass criterion | Component |
|------|------|----------------|-----------|
| V0 | 2D advection on cubed-sphere faces | convergence order ≥ 1.5 on smooth field | Operators.jl |
| V1 | Shallow-water Williamson cases 2, 5, 6 | Williamson error norms within published ranges | Grids + Exchange +Operators |
| V2 | 3D Held-Suarez (atmos) | zonal-mean T, u vs. published Held & Suarez 1994 figures | Atmos dycore |
| V3 | Gravity current / lock exchange (ocean) | front position vs. published laboratory data | Ocean dycore |
| V4 | Overflow / dense current on slope (ocean) | interface spreading rates vs. published results | Ocean dycore + z-star |
| V5 | Global ocean spin-up toy forcing | volume/tracer budgets closed to round-off | Ocean + physics |
| V6 | Aqua-planet coupled spin-up | coupled energy budget closed; no drift beyond tolerance | Coupler + everything |

Reference data: digitized from publications where necessary, stored in
`test/reference_data/` with provenance documented. Every reference dataset gets
a `README` stating source, digitization method, and expected tolerance.

---

## 5. Phased roadmap

Effort estimates assume one researcher working evenings + weekends with AI
pair-programming. Estimates are for working code; validation chasing bugs is
the variable.

### Phase 0 — Scaffolding (1 week)
- Package skeleton, CI (Julia 1.10+, `Pkg.test`, formatter, Aqua.jl)
- `Grids`: cubed-sphere face generation with metric terms
- `test`: face geometry invariants (areas sum to 4πr², coordinate round-trips)
- **Exit:** `] test` green, grid visualizations committed

### Phase 1 — Shallow water on the sphere (3–6 weeks)
- `Fields`, `Operators`, `Exchange` (face-edge halo interpolation)
- Shallow-water equations on the cubed-sphere
- Williamson test suite 2/5/6 → rungs V0, V1
- **Exit:** Williamson 5 (zonal flow over a mountain) within published error norms
- **This is the riskiest phase** — face-edge exchange correctness is where
  from-scratch cubed-sphere projects live or die. Budget extra time here.

### Phase 2 — 3D ocean dycore (6–10 weeks)
- Extend grid to 3D with z-star vertical
- Primitive equations: pressure gradient, continuity, vertical velocity
- Advection schemes (upwind → higher-order)
- Gravity current + overflow benchmarks → rungs V3, V4
- **Exit:** overflow benchmark matches published interface positions

### Phase 3 — 3D atmos dycore in parallel (4–8 weeks, overlaps Phase 2)
- Hydrostatic/anelastic atmosphere on the same grid machinery
- Held-Suarez → rung V2
- Shares Operators/Exchange entirely — this phase is mostly ocean-vs-atmos
  equation differences
- **Exit:** Held-Suarez zonal means match published figures

### Phase 4 — Coupler + toy physics (4–8 weeks)
- `Coupler`: flux computation, conservative remapping, coupling loop
- Ocean EOS (SeawaterPolynomials.jl), simple vertical mixing both components
- Conservation monitors (`Budgets.jl`) — coupled budgets close to round-off
- Aqua-planet spin-up → rung V6
- **Exit:** 100-day coupled spin-up with closed budgets and no numerical drift

### Phase 5 — Real physics modules (ongoing, the research)
- Each parameterization = one file + one interface method + its own tests
- Shortwave/longwave radiation, convection, sea ice, freshwater forcing
- This is where modular architecture pays off: every module validated against
  the already-trusted V6 system

---

## 6. Technology choices

| Choice | Decision | Why |
|--------|----------|-----|
| Julia version | ≥ 1.10 | stable performance features |
| Array types | `OffsetArray` for stencil halos | index-space locality at faces |
| Output | NetCDF via NCDatasets.jl | community-standard, cf-conventions |
| Equation of state | SeawaterPolynomials.jl | the one wheel not worth reinventing (a pure-Julia, citation-backed library — using it is reuse of *physics*, not architecture) |
| AD | Enzyme.jl later (parameter calibration) | defer; not needed for v0.x |
| GPU | KernelAbstractions.jl-compatible kernels from Phase 1 | kernel style costs little now, unlocks GPU later |
| Docs | Documenter.jl from day one | open-source requirement (D1) |

**The one dependency rule:** infrastructure dependencies (NCDatasets,
SeawaterPolynomials, OffsetArrays, KernelAbstractions) are fine — they're
tools. **No GCM code:** no ClimaCore, no Oceananigans, no SpeedyWeather. The
dycores, coupler, and physics are ours, from scratch (D5).

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cubed-sphere edge exchange bugs corrupt everything downstream | Critical | Phase 1 Williamson suite is dedicated to exactly this; do not skip |
| Project stalls in infrastructure before science (classic from-scratch death) | High | Validation ladder gives a shippable result at every rung; each rung is a demo |
| Single person can't sustain phases 2–4 | High | Phases 2 and 3 share all grid machinery; design doc is the recruiting doc |
| z-star pressure-gradient errors produce spurious flows | Medium | Known bug class — compare z-star runs against flat-bottom (simple-z) runs as debugging reference |
| Conservation leaks in the coupler | Medium | Budget monitors at runtime, fail-CI on any leak |
| Julia compile latency hurting iteration | Low | Precompile workloads; Revise.jl for dev loop |

---

## 8. Open questions

1. **Cubed-sphere vs icosahedral** (§2.1) — recommendation is cubed-sphere
   FV; needs sign-off.
2. **Atmosphere vertical coordinate** — this doc fixed the ocean's (z-star);
   the atmosphere needs its own decision (σ vs hybrid σ-p) by Phase 3.
3. **Time stepping** — start with explicit (RK4/SSPRK) for validation, implicit
   vertical later. Which implicit solver (native vs Krylov, matrix-free?) is a
   Phase 4 decision.
4. **Grid resolution strategy for coupled runs** — atmos typically runs 2–4×
   coarser than ocean; conservative remapping must handle this from V6 on.

---

*Generated from the office-hours planning session. Phase 0 cannot start until
§8.1 (grid family) is decided.*
