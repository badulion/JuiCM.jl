# JuiCM.jl — Design Document

A modular, fully-coupled General Circulation Model written from scratch in Julia.

**Status:** Planning (v0.1) — no code exists yet.
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
| D4 | Horizontal grid | Icosahedral Voronoi (TRiSK) | See D9 below |
| D5 | Build vs reuse | From scratch | The architecture *is* the research contribution |
| D6 | Parallel target | Single node first | Fast debugging loop; field layout designed so MPI can slot in later |
| D7 | Vertical coordinate | z-star (rescaled depth) | Realistic bathymetry, free-surface motion, MOM6/MPAS-compatible validation |
| D8 | v0.1 milestone | Validated dycore benchmark | External ground truth is the only unambiguous bug detector |
| D9 | Grid family (resolves #1) | Icosahedral TRiSK over cubed-sphere FV | Owner decision. TRiSK's mass/energy/potential-enstrophy conservation and the MPAS-O/ICON-O lineage outweigh cubed-sphere's faster time-to-first-result. Accepted costs: mesh generation is its own sub-project, every cell uses general unstructured machinery, debugging requires mesh visualization. Validation anchor is published benchmarks only (D9b), so no external-model grid-matching constraint applies |

### Resolved sub-decision: grid family (D9, was §8.1)

**Chosen: icosahedral C-grid (Voronoi) with TRiSK operators** — the MPAS-Ocean /
ICON-O / DYNAMICO lineage.

| | Icosahedral TRiSK (chosen) | Cubed-sphere FV (rejected) |
|--|----------------------------|----------------------------|
| Mesh generation | Icosahedron subdivision + Voronoi dual + Lloyd relaxation — a real sub-project | ~100 lines, gnomonic projection |
| Operators | TRiSK set (Ringler et al. 2010), uniform complexity everywhere | Structured stencils inside faces, seams special |
| Seams | None; 12 pentagon cells (dual of icosahedron vertices) are the only special cells | 12 edge pairs + 8 three-face corners |
| Conservation | Mass, energy, AND potential enstrophy by construction | Mass exact; enstrophy not natural |
| Precedent | MPAS-O, ICON-O, DYNAMICO | FV3, Oceananigans cubed sphere |

The rejected alternative remains a permitted future second dycore: the
`AbstractGrid`/`AbstractComponent` architecture allows a cubed-sphere component
as a sibling without touching the coupler. Revisit only if TRiSK mesh
generation proves fatal in Phase 0–1.

### Resolved tension: D3 vs D8

"Coupled from day one" and "validate dycores first" conflict if read naively.
Resolution: **the architecture is coupled-first, the milestones are
component-first.**

- Coupler interfaces, state containers, and the driver exist from commit one.
- But each component is validated in isolation against published benchmarks
  before its output is trusted in a coupled run.
- The coupled system's first spin-up uses only components that have individually
  passed their benchmarks.

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
│         Shared infrastructure (mesh, fields, I/O)        │
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
│   ├── Mesh/
│   │   ├── Icosahedron.jl    # icosahedron construction, recursive subdivision
│   │   ├── Voronoi.jl        # spherical Voronoi dual (C-grid), adjacency tables
│   │   ├── SCVT.jl           # Lloyd relaxation → smoothed centroidal Voronoi tessellation
│   │   └── MeshQuality.jl    # cell quality metrics, imprinting diagnostics
│   ├── Fields/
│   │   ├── Field.jl         # Field{G, T, N}: data + mesh + location
│   │   ├── TRiSK.jl          # TRiSK operators (Ringler et al. 2010):
│   │   │                     #   gradient, divergence, vorticity, curl,
│   │   │                     #   normal/tangential reconstruction, flux terms
│   │   └── Halo.jl           # halo update machinery (MPI-ready; local single-node first)
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
│   │   └── Remapping.jl      # atmos ↔ ocean mesh interpolation (conservative)
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
# 1. Mesh — immutable geometry + connectivity
abstract type AbstractMesh{FT} end
struct VoronoiSphereMesh{FT} <: AbstractMesh{FT}
    # primal (Voronoi cell) and dual (triangle) meshes:
    # xyz coordinates, cell/edge/vertex adjacency tables,
    # edge normal/tangential vectors, cell areas, edge lengths, weights
end

# 2. Field — data at a mesh location. Parametric for future MPI/GPU.
struct Field{M<:AbstractMesh, T, N, AX}
    data::AX                      # AX is abstract → Array today, MPIArray later
    mesh::M
    location::FieldLocation        # Cell / Edge / Vertex (C-grid staggering)
end

# 3. Prognostic state — what time integration advances
struct PrognosticState{FT}
    uₙ::Field   # normal velocities on edges (TRiSK prognostic)
    η::Field    # free surface (ocean) / surface pressure (atmos)
    θ::Field    # potential temperature
    S::Field    # salinity (ocean), tracers generally
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
coupler never sees the difference. Function barriers (`step!`, the TRiSK
operator kernels) keep type-stability once mesh and fields are concrete.

**TRiSK data layout note:** velocities live on edges, vorticity on vertices,
mass/tracers on cells. The adjacency tables in the mesh make every operator a
gather over precomputed neighbor lists — flat arrays, no indirect
decomposition needed for single-node runs.

**MPI escape hatch (D6):** `Field.data` is typed by an abstract array
parameter. Day one it's `Array`. Later, partitioned array types can replace it
without touching physics code — provided physics code never indexes `data`
directly, only through operators. Enforce this in review and lint.

### 3.4 Coupler design

The coupler is a module, not a god-object:

1. **Pull** surface fluxes from each component (`surface_fluxes`).
2. Compute coupled fluxes from bulk formulas (momentum stress, latent +
   sensible heat, evaporation - precipitation).
3. **Remap** fluxes between the atmospheric and ocean meshes (both icosahedral
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
| V0 | 2D advection on the icosahedral mesh | convergence order ≥ 1.5 on smooth field | Mesh + TRiSK operators |
| V0.5 | Mesh quality suite | 12 pentagons; total area 4πR²; cell quality metrics in bounds; imprinting below tolerance in SW tests | Mesh |
| V1 | Shallow-water Williamson cases 2, 5, 6 | Williamson error norms within published ranges | Mesh + TRiSK |
| V2 | 3D Held-Suarez (atmos) | zonal-mean T, u vs. published Held & Suarez 1994 figures | Atmos dycore |
| V3 | Gravity current / lock exchange (ocean) | front position vs. published laboratory data | Ocean dycore |
| V4 | Overflow / dense current on slope (ocean) | interface spreading rates vs. published results | Ocean dycore + z-star |
| V5 | Global ocean spin-up toy forcing | volume/tracer budgets closed to round-off | Ocean + physics |
| V6 | Aqua-planet coupled spin-up | coupled energy budget closed; no drift beyond tolerance | Coupler + everything |

Reference data: digitized from publications where necessary, stored in
`test/reference_data/` with provenance documented. Every reference dataset gets
a `README` stating source, digitization method, and expected tolerance.

**Benchmark-only validation anchor (D9b):** pass criteria are published values
from the benchmark literature, not intermodel comparisons. Cross-model
comparison (e.g. against MPAS-O test-case output) is permitted as extra
evidence but never a merge gate.

---

## 5. Phased roadmap

Effort estimates assume one researcher working evenings + weekends with AI
pair-programming. Estimates are for working code; validation chasing bugs is
the variable.

### Phase 0 — Scaffolding + mesh generation (2–4 weeks)
- Package skeleton, CI (Julia 1.10+, `Pkg.test`, formatter, Aqua.jl)
- `Mesh`: icosahedron → subdivision → spherical Voronoi dual → SCVT (Lloyd)
- `test`: mesh invariants (12 pentagons, areas sum to 4πr², adjacency
  consistency, quality metrics) → rung V0.5
- **Exit:** `] test` green, mesh quality report + visualization committed
- **Note:** for TRiSK the mesh generator is itself a deliverable. This is the
  largest single chunk of the project and where the D9 risk lands.

### Phase 1 — Shallow water on the sphere (3–6 weeks)
- `Fields` (cell/edge/vertex staggering), `TRiSK` operator set
- Shallow-water equations on the icosahedral C-grid, explicit SSPRK stepping
- Williamson test suite 2/5/6 → rungs V0, V1
- **Exit:** Williamson 5 (zonal flow over a mountain) within published error norms
- Grid imprinting and computational-mode checks are part of this phase's suite
  (icosahedral-specific failure modes; see §7).

### Phase 2 — 3D ocean dycore (6–10 weeks)
- Extend mesh to 3D with z-star vertical
- Primitive equations: pressure gradient, continuity, vertical velocity
- Advection schemes (upwind → higher-order, positive-definite on unstructured)
- Gravity current + overflow benchmarks → rungs V3, V4
- **Exit:** overflow benchmark matches published interface positions

### Phase 3 — 3D atmos dycore in parallel (4–8 weeks, overlaps Phase 2)
- Hydrostatic/anelastic atmosphere on the same mesh machinery
- Held-Suarez → rung V2
- Shares Mesh/TRiSK entirely — this phase is mostly ocean-vs-atmos
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
| Array types | flat arrays + precomputed adjacency tables | TRiSK gathers are neighbor-list loops; SIMD-friendly |
| Mesh generation | native Julia (Icosahedron → Voronoi → SCVT) | from-scratch rule (D5); JIGSAW/Gmsh forbidden as mesh code — the mesh is ours |
| Output | NetCDF via NCDatasets.jl | community-standard, cf-conventions |
| Equation of state | SeawaterPolynomials.jl | the one wheel not worth reinventing (a pure-Julia, citation-backed library — using it is reuse of *physics*, not architecture) |
| AD | Enzyme.jl later (parameter calibration) | defer; not needed for v0.x |
| GPU | KernelAbstractions.jl-compatible kernels from Phase 1 | kernel style costs little now, unlocks GPU later |
| Mesh visualization | Makie.jl (dev-time only dep) | debugging an unstructured mesh requires plotting it |

**The one dependency rule:** infrastructure dependencies (NCDatasets,
SeawaterPolynomials, KernelAbstractions, Makie) are fine — they're tools.
**No GCM code:** no ClimaCore, no Oceananigans, no SpeedyWeather, no
MPAS-derived meshes. The dycores, coupler, physics, and mesh generator are
ours, from scratch (D5).

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Mesh generation (Voronoi dual + SCVT convergence) stalls Phase 0 | Critical | Largest single estimate in the plan; icosahedral geodesic libraries' algorithms are published; subdivision-only (non-SCVT) mesh is an acceptable Phase 0 fallback with quality metrics recorded |
| Grid imprinting on icosahedron symmetry lines contaminates results | High | V0.5 imprinting suite + Williamson cases are designed to expose exactly this; document pentagon-cell error contribution |
| TRiSK computational modes / Hollingsworth-type instability | High | Follow Ringler et al. 2010 flux formulations exactly; Smagorinsky + instability checks in Phase 1 suite |
| Project stalls in infrastructure before science (classic from-scratch death) | High | Validation ladder gives a shippable result at every rung; each rung is a demo |
| Single person can't sustain phases 2–4 | High | Phases 2 and 3 share all mesh machinery; design doc is the recruiting doc |
| z-star pressure-gradient errors produce spurious flows | Medium | Known bug class — compare z-star runs against flat-bottom (simple-z) runs as debugging reference |
| Conservation leaks in the coupler | Medium | Budget monitors at runtime, fail-CI on any leak |
| Julia compile latency hurting iteration | Low | Precompile workloads; Revise.jl for dev loop |

---

## 8. Open questions

1. **Atmosphere vertical coordinate** — the atmosphere needs its own decision
   (σ vs hybrid σ-p) by Phase 3.
2. **Time stepping** — start with explicit (RK4/SSPRK) for validation, implicit
   vertical later. Which implicit solver (native vs Krylov, matrix-free?) is a
   Phase 4 decision.
3. **Mesh resolution strategy for coupled runs** — atmos typically runs 2–4×
   coarser than ocean; conservative remapping must handle this from V6 on.
4. **SCVT stopping criterion** — Lloyd relaxation convergence tolerance and
   iteration cap for Phase 0; acceptance thresholds in V0.5.
5. **Mesh persistence format** — regenerate per run vs cached JLD2/NetCDF
   mesh files (regeneration cost vs reproducibility tradeoff).

---

*Grid family decided (D9, closes #1): icosahedral TRiSK. Phase 0 may start.*
