# Shallow-water dycore on the icosahedral C-grid (issue #11).
#
# Implements the TRiSK shallow-water equations of Ringler et al. (2010)
# ("A unified approach to energy conservation and potential vorticity
# dynamics on grids and spheres", JCP 229, 3065–3090; R10 below):
#
#     ∂u/∂t = −q̂ F⊥ − ∇(g h + K)        (R10 Eq 20, momentum)
#     ∂h/∂t = −∇·(h u)                   (R10 Eq 16, continuity)
#
# with u the normal velocity on edges, h the thickness at cells, K the
# kinetic energy (R10 Eq 63), F⊥ the tangential flux (R10 Eq 24, T09
# weights), q the potential vorticity at vertices, and the
# energy-conserving PV-flux form of the Coriolis term (R10 Eq 49). The
# symmetric (q̃_e + q̃_{e'})/2 PV mean makes u·(q h u⊥) = 0 in the
# discrete global sum (R10 Eqs 37–47): mass conserves to round-off and
# total energy to time-truncation error — the case-2 acceptance test.
#
# Units (verified against the case-2 steady state in the #9 testset and
# the Williamson norms): the mesh geometry lives on the unit sphere
# (areas in steradians, lengths in radians). The state is physical
# (h [m], u [m/s]); the tendency is assembled in scaled variables
# h* = h/R (dimensionless), u* = u/R (rad/s) with dimensional time:
#
#     ∂h*/∂t = −∇·(ĥ* u*)                     [1/s]
#     ∂u*/∂t = −Q̂⊥ − ∇(g h*/R + K*)           [1/s²]
#
# q_v = (f_v + ζ_v)/h*_v with f_v = 2Ω sinφ, ζ = curl(u*), Q̂⊥ =
# pv_flux(q, ĥ*·u*) (R10 Eq 49), K* = kinetic_energy(u*) (R10 Eq 63).
# Physical tendencies: ∂h/∂t = R ∂h*/∂t [m/s], ∂u/∂t = R ∂u*/∂t [m/s²].
#
# Execution (DESIGN.md D6, ratified for #11): single-node, but the
# state is **partitioned** — one `HaloArray` payload per region in a
# `HaloWorld` (#10), the same arrays real MPI ranks would hold. Regions
# advance in lockstep; ghosts are exchanged once per SSPRK stage,
# before each L evaluation (owner-computes: every value a region reads
# was written by its owner in the same stage). Elementwise stage
# arithmetic propagates NaN-poisoned outer ghost slots, so each
# per-stage exchange is load-bearing — this is what makes the MPI
# transition a send-step swap (D6) rather than a rewrite. The
# partitioned result is bitwise identical to the single-region run
# (validated in the test suite).
#
# Physics hook (DESIGN.md §3.3, core abstraction 4): an
# `AbstractPhysicsModule` adds forcing via `compute_tendency!(tend,
# state, mod, t)`; the dycore ships `NullForcing` (zero tendency).
#
# AGENTS.md rule 3: no `Field.data` indexing outside `src/Fields/` —
# elementwise kernels (`Fields.lincomb!`, `Fields.axpy!`,
# `Fields.scale!`, `Fields.dot`, `Fields.valid_indices`) and the TRiSK
# operators are the only access paths.

module Atmos

using ..Voronoi: VoronoiSphereMesh
using ..Fields: Field, CellField, EdgeField, VertexField, CellLoc, EdgeLoc,
                location, mesh_of, similar_field, lincomb!, axpy!,
                similar_state, scale!, dot, valid_indices, copyto!
using ..TRiSK: divergence, divergence!, gradient, gradient!, curl,
               tangential_flux, edge_mean, kinetic_energy, pv_flux,
               potential_vorticity, integral
using ..TimeStepping: ssprk3_step!, ssprk3_integrate!
import ..TimeStepping: _stage, _lincomb!, _axpy!
using ..Halo: HaloWorld, HaloArray, halo_world, halo_update!,
              halo_exchange!, partition_mesh, build_halo_info
using ..ShallowWater: williamson_constants

export SWState, SWModel, sw_state, sw_model,
       AbstractPhysicsModule, compute_tendency!, NullForcing,
       sw_tendency!, sw_step!, sw_integrate!, sw_mass, sw_total_energy

# ---------------------------------------------------------------- state

"""
    SWState{H <: Field, U <: Field}

Shallow-water prognostic state (DESIGN.md §3.3, core abstraction 3,
single-layer case): thickness `h` at cells and normal velocity `u` on
edges, both as Fields carrying their mesh. Field payloads may be plain
vectors (single-region) or `HaloArray`s (#10, partitioned) — the
elementwise kernels are payload-agnostic.

Implements the TimeStepping state protocol: `TimeStepping._stage`
returns a fresh scratch state (`Fields.similar_state`), and the
`_lincomb!`/`_axpy!` kernels delegate to the stagger-checked
`Fields.lincomb!`/`Fields.axpy!`.
"""
struct SWState{H <: Field, U <: Field}
    h::H
    u::U
end

"""
    sw_state(h::Field, u::Field) -> SWState

Validate staggering and wrap a state: `h` must be a `CellField`, `u` an
`EdgeField`, both on the same mesh.
"""
function sw_state(h::Field, u::Field)
    location(h) == CellLoc() ||
        throw(ArgumentError("SWState h must be a CellField"))
    location(u) == EdgeLoc() ||
        throw(ArgumentError("SWState u must be an EdgeField"))
    mesh_of(h) === mesh_of(u) ||
        throw(ArgumentError("SWState fields must share a mesh"))
    return SWState(h, u)
end

_stage(s::SWState) = SWState(similar_state(s.h, s.u)...)
Base.copy(s::SWState) = SWState(copy(s.h), copy(s.u))
function _lincomb!(out::SWState, α, A::SWState, β, B::SWState)
    lincomb!(out.h, α, A.h, β, B.h)
    lincomb!(out.u, α, A.u, β, B.u)
    return out
end
function _axpy!(out::SWState, α, X::SWState)
    axpy!(out.h, α, X.h)
    axpy!(out.u, α, X.u)
    return out
end

# ------------------------------------------------- physics interface

"""
    AbstractPhysicsModule

A physics parameterization (DESIGN.md §3.3, core abstraction 4): pure
data plus one function, [`compute_tendency!`](@ref). Modules must not
mutate the state they read.
"""
abstract type AbstractPhysicsModule end

"""
    compute_tendency!(tend::SWState, state::SWState, mod::AbstractPhysicsModule, t)

Add the module's forcing tendencies into `tend` (which already holds
the dynamical tendency) — DESIGN.md §3.3's function-barrier seam. Every
concrete `AbstractPhysicsModule` must implement this; the generic
fallback errors loudly so a forgotten implementation cannot pass
silently.
"""
function compute_tendency!(tend::SWState, state::SWState,
                           ::AbstractPhysicsModule, t)
    error("compute_tendency! not implemented for $(typeof(mod)) — " *
          "every AbstractPhysicsModule must define the forcing tendency")
end

"""
    NullForcing

The no-physics module: zero forcing tendency. The dycore default.
"""
struct NullForcing <: AbstractPhysicsModule end

compute_tendency!(tend::SWState, state::SWState, ::NullForcing, t) = tend

# ------------------------------------------------------------ tendency

"""
    sw_tendency!(out::SWState, state::SWState; physics = NullForcing(), t = 0.0)
        -> SWState

Shallow-water dynamical tendency (R10 Eqs 16 and 20) plus the optional
physics hook, written into `out` (see the module-level units note for
the scaled-variable assembly):

    ∂h/∂t = −∇·(h u)                    continuity (R10 Eq 16; edge
                                         thickness ĥ from R10 Eq 52)
    ∂u/∂t = −Q̂⊥ − ∇(g h + K)            momentum (R10 Eq 20 in the
                                         energy-conserving PV-flux form,
                                         R10 Eq 49; K from R10 Eq 63)

`g` and `Ω` are the Williamson (Earth) constants. After this call
`out` holds the physical-unit tendencies; the physics hook
(`compute_tendency!`) then adds its forcing into the same buffers.

Sign note: `pv_flux` returns Q̂⊥ with the symmetric-q̃ form; R10 Eq 20's
momentum term is −q̂F⊥ − ∇(gh+K), so the u-tendency uses
`−Q̂⊥ − ∇(gh+K)`. Verified against the case-2 steady state (the #9
testset: residual vanishes at truncation level, 2nd order).

Requires all of `state`'s ghost slots to be current (call
`halo_update!` first in a partitioned run — the drivers do this).
"""
function sw_tendency!(out::SWState, state::SWState;
                      physics::AbstractPhysicsModule = NullForcing(),
                      t::Real = 0.0)
    mesh = mesh_of(state.h)
    (; R, Omega, g) = williamson_constants()

    # scaled variables h* = h/R, u* = u/R (module docstring)
    hs = similar_field(state.h)
    scale!(copyto!(hs, state.h), 1 / R)
    us = similar_field(state.u)
    scale!(copyto!(us, state.u), 1 / R)

    # continuity: ∂h*/∂t = −∇·(ĥ* u*)   →   ∂h/∂t = R · ∂h*/∂t  [m/s]
    Fu = edge_mean(hs) * us
    divergence!(out.h, Fu)
    scale!(out.h, -R)

    # momentum: ∂u*/∂t = −Q̂⊥ − ∇(g h*/R + K*)  →  ∂u/∂t = R · ∂u*/∂t
    q = potential_vorticity(us, hs,
                            VertexField(v -> 2 * Omega * mesh.vertex_xyz[3, v],
                                        mesh))
    Cor = pv_flux(q, Fu)                 # R10 Eq 49 (symmetric form)
    K = kinetic_energy(us)               # R10 Eq 63
    PhiK = lincomb!(similar_field(hs), g / R, hs, 1.0, K)   # [1/s²]
    du = gradient(PhiK)                  # R10 Eq 22
    lincomb!(out.u, -1.0, Cor, -1.0, du) # out.u = −Q̂⊥ − ∇(Φ+K)  (R10 Eq 20)
    scale!(out.u, R)                     # [m/s²]

    # physics hook: add forcing into the same buffers
    compute_tendency!(out, state, physics, t)
    return out
end

# -------------------------------------------------------------- model

"""
    SWModel

A partitioned shallow-water model: the mesh and one state per region
(each holding `HaloArray` payloads from a shared `HaloWorld` plan).
Construct with [`sw_model`](@ref); advance with [`sw_step!`](@ref).
"""
struct SWModel{FT, S <: SWState}
    mesh::VoronoiSphereMesh{FT}
    states::Vector{S}                    # per-region states
end

"""
    sw_model(mesh, h::CellField, u::EdgeField; n_regions = 4) -> SWModel

Partition `mesh` into `n_regions` regions and build one state per
region from the global initial conditions, with ghosts NaN-poisoned
(exchanged per stage by [`sw_step!`](@ref)).
"""
function sw_model(mesh::VoronoiSphereMesh{FT}, h::CellField, u::EdgeField;
                  n_regions::Int = 4) where {FT}
    p = partition_mesh(mesh, n_regions)
    ic = build_halo_info(p, mesh, CellLoc())
    ie = build_halo_info(p, mesh, EdgeLoc())
    wh = halo_world(ic; init = i -> h[i])
    wu = halo_world(ie; init = i -> u[i])
    states = [SWState(Field{typeof(mesh), FT, 1, typeof(wh[r]), CellLoc}(wh[r], mesh),
                     Field{typeof(mesh), FT, 1, typeof(wu[r]), EdgeLoc}(wu[r], mesh))
              for r in 1:n_regions]
    return SWModel(mesh, states)
end

# ------------------------------------------------------------ drivers

"""
    sw_step!(state::SWState, t, Δt; physics = NullForcing()) -> SWState

Advance one SSPRK(3,3) step (single-region state; no exchange).
"""
function sw_step!(state::SWState, t, Δt;
                  physics::AbstractPhysicsModule = NullForcing())
    L(out, s, τ) = sw_tendency!(out, s; physics = physics)
    return ssprk3_step!(L, state, t, Δt)
end

"""
    sw_integrate!(state::SWState, t0, t_final, n_steps; physics = NullForcing())
        -> SWState

Integrate a single-region state with SSPRK(3,3) over `n_steps` equal
steps.
"""
function sw_integrate!(state::SWState, t0, t_final, n_steps;
                       physics::AbstractPhysicsModule = NullForcing())
    L(out, s, τ) = sw_tendency!(out, s; physics = physics)
    return ssprk3_integrate!(L, state, t0, t_final, n_steps)
end

"""
    sw_step!(model::SWModel, t, Δt; physics = NullForcing()) -> SWModel

Advance all regions one SSPRK(3,3) step **in lockstep**, exchanging
ghosts once per stage — before each of the three L evaluations — so
every operator read sees owner-current values (owner-computes). In the
MPI transition only the exchange's send step changes; this driver is
untouched (DESIGN.md D6).

The stage loop mirrors `TimeStepping._ssprk3_core!` exactly (same
Butcher tableau, same stage times) with `halo_update!` inserted before
each L call.
"""
function sw_step!(model::SWModel, t, Δt;
                  physics::AbstractPhysicsModule = NullForcing())
    (; states) = model
    R = length(states)
    u1 = _stage.(states)
    u2 = _stage.(states)
    tmp = _stage.(states)
    # per-stage exchange: refresh ghosts of exactly the collection the
    # next L reads (owner-computes; see halo_exchange!)
    exchange(hs, us) = (halo_exchange!(hs); halo_exchange!(us); nothing)

    function L(out, s, τ)
        exchange([x.h for x in s], [x.u for x in s])
        for r in 1:R
            sw_tendency!(out[r], s[r]; physics = physics, t = τ)
        end
        return nothing
    end

    # stage 1: u1 = U + Δt L(U, t)
    L(u1, states, t)
    for r in 1:R
        _lincomb!(u1[r], 1.0, states[r], Δt, u1[r])
    end
    # stage 2: u2 = (3/4) U + (1/4)(u1 + Δt L(u1, t+Δt))
    L(tmp, u1, t + Δt)
    for r in 1:R
        _lincomb!(u2[r], 1.0, u1[r], Δt, tmp[r])
        _lincomb!(u2[r], 3 / 4, states[r], 1 / 4, u2[r])
    end
    # stage 3: Uⁿ⁺¹ = (1/3) U + (2/3)(u2 + Δt L(u2, t+Δt/2))
    L(tmp, u2, t + Δt / 2)
    for r in 1:R
        _lincomb!(tmp[r], 1.0, u2[r], Δt, tmp[r])
        _lincomb!(tmp[r], 1 / 3, states[r], 2 / 3, tmp[r])
    end
    # commit: the new states take tmp's buffers; the old state buffers
    # become the next call's tmp (allocation-free after warmup)
    for r in 1:R
        old = states[r]
        states[r] = tmp[r]
        tmp[r] = old
    end
    return model
end

# -------------------------------------------------------- diagnostics

"""
    sw_mass(state::SWState) -> Real

Global mass (volume): Σ A_i h_i (unit-sphere areas; multiply by R² for
physical units). Partitioned payloads contribute their owned slots only.
"""
function sw_mass(state::SWState)
    return integral(state.h)
end

"""
    sw_total_energy(state::SWState; g = 9.80616) -> Real

Global total energy: Σ A_i h_i (g h_i/2 + K_i) — thickness-weighted
potential plus kinetic (R10 Eq 63) energy, unit-sphere areas. Partitioned
payloads contribute their owned slots only.
"""
function sw_total_energy(state::SWState; g = 9.80616)
    K = kinetic_energy(state.u)
    # E = h·(g·h/2 + K) elementwise, then the area-weighted integral
    E = state.h * lincomb!(similar_field(state.h), g / 2, state.h, 1.0, K)
    return integral(E)
end

end # module Atmos
