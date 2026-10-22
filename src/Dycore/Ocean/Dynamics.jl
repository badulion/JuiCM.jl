# Layered primitive-equation ocean dycore (issue #30, DESIGN.md §3.2).
#
# The first fully-3D component: layered Boussinesq primitive equations on
# the icosahedral C-grid, horizontal operators via the #9 TRiSK set and
# the Phase-1 vector-invariant discretization (Ringler et al. 2010, R10;
# Ringler, Petersen, Jacobsen & Jones 2013 — the MPAS-Ocean framework —
# for the layered/mode-split aspects).
#
# Equation set (per layer k = 1 top, bottom at z = −H — flat H0 or the
# `bathymetry` field of issue #65 — z* or simple-z from the VerticalGrid
# mode; h_k = layer thickness, u_k = layer normal velocity, ρ_k = layer
# density from the pluggable EOS):
#
#   ∂h_k/∂t = −∇·(h_k u_k)                       (layer continuity)
#   ∂u_k/∂t = −q̂_k F⊥_k − ∇(p̄_k/ρ0 + K_k)
#              − b_k ∇z̄_k + g∇η̂                 (layer momentum)
#
# with F_k = ĥ_k u_k the per-layer edge thickness flux (R10 Eq 52),
# q_k = (f_v + ζ_k)/h*_v,k the layer potential vorticity at vertices,
# Q̂⊥_k = pv_flux(q_k, F_k) the energy-conserving PV flux (R10 Eq 49),
# K_k = kinetic_energy(u_k) (R10 Eq 63), and
#
#   p̄_k = g·Σ_{j<k} (ρ_j/ρ0)·h_j + (g/2)·(ρ_k/ρ0)·h_k   [m²/s²]
#
# the layer-mean hydrostatic pressure divided by the reference density
# ρ0 (interface pressures p_k = ρ0·g·Σ_{j<k}(ρ_j/ρ0)h_j, p̄_k their
# layer mean). b_k = g·ρ_k/ρ0 is the layer buoyancy and z̄_k the
# *actual* layer-center height from the state column: z_1 = η,
# z_{k+1} = z_k − h_k, z̄_k = (z_k + z_{k+1})/2 — NOT the coordinate
# map. Taking z̄ from the coordinate map instead would drop the
# thickness coupling and break the uniform-density limit (the layers
# would feel −(g/2)∇h_k instead of −g∇η); with actual heights the
# discrete PGF reduces exactly to the 2D shallow-water force for
# uniform density (every layer: −g∇η) and to the textbook two-layer
# internal form PGF_1 = −(ρ1 g/ρ0)∇η,
# PGF_2 = PGF_1 − (g/ρ0)(ρ2 − ρ1)∇ξ for K = 2.
#
# The +g∇η̂ term is the *baroclinic projection*: the depth-mean of the
# hydrostatic PGF over the column is exactly −g∇η (the barotropic
# force), so adding it back to every layer leaves the slow system
# force-free for the barotropic mode (uniform density: −∇(p̄/ρ0) −
# b∇z̄ = −g∇η cancels +g∇η̂ term by term). That removal is what makes
# the slow step internal-wave-limited rather than external-wave-
# limited; the barotropic mode itself lives in the fast subsystem
# (mode split below). The PGF pieces (−∇(p̄/ρ0) and −b_k∇z̄_k) are kept
# as separate gradient evaluations — their discrete cancellation on
# resting slopes is exactly the DESIGN.md §7 z-star PGF bug class, and
# merging them would mask it.
#
# Vertical momentum transport (3D Reynolds advection) and vertical
# viscosity/diffusion are Phase-4 physics: #30 carries the horizontal
# vector-invariant dynamics per layer, which is what the linear wave
# acceptance tests exercise.
#
# Mode split (user-ratified 2026-10-16; MPAS-O-style barotropic/
# baroclinic split with a state-carried fast mode):
#
#   fast (barotropic): the 2D shallow-water system for (H0 + η, ū) —
#     exactly the Phase-1 SW dycore (Coriolis included) — substepped
#     n_sub times per slow step with Δt_fast ≈ Δt_slow / n_sub set by
#     the barotropic CFL (√(g·H0)). The fast state (η, ū) is *carried
#     in OceanState* (fields `eta`, `ubar`) — it is prognostic, not a
#     per-step diagnostic.
#   slow (baroclinic): SSPRK(3,3) on the velocity *anomalies*
#     u′ = u − ū over the full Δt_slow, with the baroclinic projection
#     above removing the depth-mean force — the slow step is limited
#     by the internal-wave CFL (~100× weaker). The slow continuity
#     fluxes are PURE anomaly fluxes (∂h_k/∂t = −∇·(h_k u′_k)): the
#     column sum Σ_k h_k is then an exact invariant of the slow step —
#     no barotropic mass is pumped, which is what makes the frozen-η
#     projection exactly canceling and the split linearly stable.
#     (Using the full velocity in the slow fluxes would excite the
#     discrete barotropic mode at √(gH)·(1−O(1/K)) — verified by the
#     instability during #30 development; the column-mean transport is
#     carried entirely by the fast mode, and the baroclinic volume
#     redistribution reaches the layers through the re-stretch below.)
#
#   Complete-tendency projection and barotropic forcing (issue #40;
#   MPAS-O split-explicit class, Higdon 2005; Ringler et al. 2013 §3):
#   `ocean_step!` removes the thickness-weighted depth-mean of the
#   COMPLETE slow momentum tendency (pressure, Coriolis/Bernoulli,
#   viscosity, drag — `ocean_tendency!(...; projection = :full)`), so
#   u′ stays depth-mean free, and forces the fast mode with the part of
#   that depth-mean that is not the surface-pressure force:
#     G = ⟨∂u/∂t⟩(state) − ⟨∂u/∂t⟩(uniform density, u′ = 0)
#   (evaluated once at the step start, held over the substeps; the
#   reference evaluation is the slow operator's own response to η, so
#   G carries no surface-wave force). The fast state is carried as the
#   time-filtered pair (η̄, ū̄) — both averaged with the same trapezoid
#   weights — and the layers are re-projected after the step so that
#   their depth-mean velocity equals ū̄. Measured in the closed #40
#   channel: the pressure-only projection left 3 m/s of momentum-
#   without-mass depth-mean flow and a wedge-shaped dense current
#   (F = 0.28); the forced, filtered split has ⟨u⟩ = ū to round-off
#   and F = 0.48 (rung V3). Restarting from the filtered η̄ with the
#   unfiltered ū_end amplifies forced barotropic waves (measured
#   ×6 per 6 h) — hence the filtered pair.
#
#   Solid walls (`closed_edges`): normal velocity tendencies (layers
#   and fast mode) are zeroed on wall edges, relative vorticity is zero
#   at wall vertices (free slip; Adcroft & Marshall 1998 — a staircase
#   hexagon coastline is otherwise effectively no-slip), and wall cells
#   average the kinetic energy over their open edges.
#   coupling (step end): the depth-mean of the layers is REPLACED with
#     the fast solver's ū_end (u_k = u′_k + ū_end; the slow system's
#     barotropic velocity DOF is unphysical — its force was frozen —
#     so it is discarded, not added to), the column is re-stretched to
#     the coordinate map with the TIME-MEAN fast η over the substeps
#     (η̄; the end-of-step η of a barotropic wave is a random-phase
#     sample of a ~14 h oscillation — injecting it would alias the fast
#     wave into the slow state; the substep mean is the physical
#     volume change). Distribution of the column correction:
#     proportional stretch for :zstar (fractions preserved), whole
#     anomaly into the top layer for :simplez. (state.eta, state.ubar)
#     are set to (η̄, ū_end). After the re-stretch Σ_k h_k − H0 = η̄
#     holds exactly, and the total volume follows the fast solver's
#     flux-form continuity (conserved to round-off; Phase-1 SW mass
#     conservation).
#   tracers (T, S): two schemes, selected by `ocean_step!`'s
#     `tracer_advection` keyword.
#     :fct (default; issue #40 decision) — operator split: T/S are
#       frozen inside the slow SSPRK and advanced ONCE per mode-split
#       step by the FCT scheme (#36, `fct_advect_tracer!`) from the
#       start-of-step h to the re-stretched h. VOLUME COMPATIBILITY: the
#       tracer is transported by the time-integrated layer mass fluxes
#       that actually moved the thickness — the SSPRK-weighted
#       (1/6, 1/6, 2/3) stage fluxes ĥ_k u′_k of the slow step, with
#       their column sum replaced by the barotropic transport F̄_bt of
#       the fast substeps (the flux whose divergence is the η̄ change,
#       i.e. what the re-stretch imposes on the column), distributed
#       over layers by the start-of-step edge thickness. Then
#       Σ_k h_horiz,k = Σ_k h_new,k per column, the diagnosed surface
#       interface velocity w_1 vanishes to round-off (no tracer flux
#       through the sea surface), a constant tracer stays constant, and
#       Σ A h C is conserved to round-off. The re-stretch redistribution
#       between layers is the residual-diagnosed vertical transport.
#       (Requires a column-consistent input state, Σ_k h_k − H0 = η,
#       which every `ocean_step!` output satisfies.)
#     :upwind — the #32 reference path: flux-form upwind tendency
#       inside the slow SSPRK with the SAME layer fluxes the continuity
#       uses; the re-stretch keeps layer concentrations unchanged (the
#       fast barotropic mode carries no tracer in this path — the
#       content error is second order in the anomalies; see the
#       OceanTracers module header).
#
#   z* regrid (issue #62, DESIGN.md D7; opt-in `regrid = true`): the
#     proportional re-stretch above keeps the layer FRACTIONS of the
#     Lagrangian layers, so layers can collapse (#40: floor-pinned
#     sheets beside 30–110 km neighbours, whose Coriolis/Bernoulli
#     forcing drives a runaway). With regrid the step ends with a
#     vertical ALE remap to the z* target h_k = Δf_k·(H0 + η̄)
#     (`layer_thickness(grid, η̄)`), as in MPAS-O (Petersen et al.
#     2015) and MOM6 (Adcroft et al. 2019): u by a conservative,
#     MC-limited piecewise-linear remap of every edge column
#     (`_remap_edge_columns!`; Σ_k ĥ_k u_k exact, no new extrema), and
#     T/S by the FCT step with h_new = target, whose residual-diagnosed
#     vertical flux (`vertical_order = 2`, limited PLM) IS the remap.
#     Default `false`: the layered internal-wave tests exercise the
#     Lagrangian layers.
#     Refs: Petersen, M.R. et al. (2015), "Evaluation of the arbitrary
#     Lagrangian–Eulerian vertical coordinate method in the MPAS-Ocean
#     model", Ocean Modelling 86, 93–113; Adcroft, A. et al. (2019),
#     "The GFDL global ocean and sea ice model OM4.0", JAMES 11,
#     3167–3211 (MOM6 ALE); White & Adcroft (2008, JCP 227) and van
#     Leer (1977, JCP 23) for the limited remap.
#
# This is a Lie splitting: O(Δt_slow) splitting error per step. The
# depth-mean Coriolis of ū is advanced by the fast solver only (the
# slow system evolves anomalies), so no double-counting. The barotropic
# transport does not advect layer thicknesses within a slow step (the
# fast ū appears in the layers only through the re-stretch); this is
# the standard "anomaly-form" split and is adequate while the barotropic
# flow is weak relative to the internal flow (documented limitation;
# revisited with the partitioned driver in #45 if coupled tests demand
# it). Documented and tested by the #30 acceptance suite (resting
# state, internal wave, volume conservation).
#
# Units (the Phase-1 convention): mesh geometry on the unit sphere;
# state physical (h [m], u [m/s], T [K], S [psu]); the tendency is
# assembled in scaled variables h* = h/R, u* = u/R with dimensional
# time, then rescaled back (see Atmos/Dynamics.jl for the derivation).
# The pressure enters the momentum bracket scaled as
# ∇(p̄/ρ0)/R² + ∇K*_k  [1/(s²·rad)]; interface heights enter as
# (g/R²)·∇z̄ and the projection as (g/R²)·∇η̂.
#
# AGENTS.md rule 3: no `Field.data` indexing outside `src/Fields/` —
# elementwise kernels and the TRiSK operators are the only access
# paths. All layered kernels loop layers and delegate to the Field
# kernels, so HaloArray payloads carry through (MPI escape hatch, D6).

module Ocean

using ..Voronoi: VoronoiSphereMesh
using ..Fields: Field, CellField, EdgeField, VertexField, CellLoc, EdgeLoc,
                location, mesh_of, similar_field, lincomb!, axpy!,
                similar_state, scale!, copyto!, fill!, valid_indices
using ..TRiSK: divergence, divergence!, gradient, gradient!, curl,
               laplacian,
               tangential_flux, edge_mean, kinetic_energy, pv_flux,
               potential_vorticity, integral
using ..Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using ..Vertical: VerticalGrid
using ..OceanVertical: ssh_from_thickness, layer_center_z, layer_thickness
using ..OceanEOS: AbstractOceanEOS, ocean_density, ocean_density!
using ..OceanTracers: tracer_tendency!
using ..TracerFCT: fct_advect_tracer!, _mc
using ..ShallowWater: williamson_constants
using ..Atmos: SWState, sw_step!, sw_tendency!
using ..TimeStepping: ssprk3_step!, ssprk3_integrate!
import ..TimeStepping: _stage, _lincomb!, _axpy!

export OceanState, ocean_state, ocean_tendency!, ocean_step!,
       ocean_integrate!, ocean_volume, ocean_kinetic_energy,
       ocean_ssh, wall_vertices

# ---------------------------------------------------------------- state

"""
    OceanState{U, H, TC}

Layered-ocean prognostic state (issue #30; DESIGN.md §3.3 core
abstraction 3, layered case):

- `u` — per-layer normal velocity on edges [m/s] (`LayeredEdgeField`)
- `h` — per-layer thickness at cells [m] (`LayeredCellField`); the
  prognostic mass variable, evolved by the *baroclinic* continuity
  (pure anomaly flux divergence — the column sum Σ_k h_k is an exact
  invariant of the slow step). The column is re-stretched to the
  coordinate (z-star or simple-z) at the end of every
  [`ocean_step!`](@ref) using the fast mode's η, so the
  state-consistent surface Σ_k h_k − H0 agrees with `eta` exactly.
- `T` — per-layer temperature [K or °C] (`LayeredCellField`); advected
  by the FCT scheme (#36/#40, operator split, default) or the #32
  upwind scheme inside the slow step — either way with the mass fluxes
  that moved h (see `ocean_step!`'s `tracer_advection`)
- `S` — per-layer salinity [psu] (`LayeredCellField`); advected exactly
  like `T` (no sources/sinks yet — those are Phase-4 physics)
- `eta` — free-surface height [m] (`CellField`), the *fast barotropic*
  prognostic (advanced with ū by substepped 2D SW solves)
- `ubar` — barotropic (depth-mean) normal velocity [m/s] (`EdgeField`),
  the fast mode's velocity.

The slow system is purely baroclinic: it evolves the velocity
*anomalies* u′ = u − ū, its continuity fluxes carry no depth-mean
transport, and its pressure force is the full hydrostatic layer PGF
minus its own thickness-weighted depth-mean (the baroclinic
projection) — the barotropic mode would otherwise re-introduce the
external-wave CFL. The barotropic state reaches the layers at step end
through the fast solver's ū and the coordinate re-stretch (see
`ocean_step!`).

Field payloads may be plain vectors (single-region) or `HaloArray`s
(#10, partitioned) — the elementwise kernels are payload-agnostic.

Implements the TimeStepping state protocol (`_stage`, `_lincomb!`,
`_axpy!`) via the layered kernels.
"""
struct OceanState{U <: LayeredEdgeField, H <: LayeredCellField,
                  TC <: LayeredCellField}
    u::U
    h::H
    T::TC
    S::TC
    eta::CellField
    ubar::EdgeField
end

"""
    ocean_state(u, h, T, S[, eta, ubar]) -> OceanState

Validate staggering and wrap a layered state: `u` a `LayeredEdgeField`,
`h`/`T`/`S` `LayeredCellField`s, `eta` a surface `CellField` [m] and
`ubar` the barotropic `EdgeField` [m/s] (defaults: η = 0, ū = 0), all
on the same mesh with the same layer count.
"""
function ocean_state(u::LayeredEdgeField, h::LayeredCellField,
                     T::LayeredCellField, S::LayeredCellField,
                     eta = nothing, ubar = nothing)
    location(u.layers[1]) == EdgeLoc() ||
        throw(ArgumentError("OceanState u must be a LayeredEdgeField"))
    K = nlayers(u)
    for (nm, f) in (("h", h), ("T", T), ("S", S))
        nlayers(f) == K ||
            throw(ArgumentError("OceanState $nm must have $K layers"))
    end
    mesh_of(u) === mesh_of(h) === mesh_of(T) === mesh_of(S) ||
        throw(ArgumentError("OceanState fields must share a mesh"))
    et = eta === nothing ? CellField(i -> 0.0, mesh_of(h)) : eta
    ub = ubar === nothing ? EdgeField(e -> 0.0, mesh_of(h)) : ubar
    return OceanState(u, h, T, S, et, ub)
end

_stage(s::OceanState) = OceanState(similar(s.u), similar(s.h),
                                   similar(s.T), similar(s.S),
                                   similar_field(s.eta), similar_field(s.ubar))
Base.copy(s::OceanState) = OceanState(copy(s.u), copy(s.h), copy(s.T),
                                      copy(s.S), copy(s.eta), copy(s.ubar))
function _lincomb!(out::OceanState, α, A::OceanState, β, B::OceanState)
    lincomb!(out.u, α, A.u, β, B.u)
    lincomb!(out.h, α, A.h, β, B.h)
    lincomb!(out.T, α, A.T, β, B.T)
    lincomb!(out.S, α, A.S, β, B.S)
    lincomb!(out.eta, α, A.eta, β, B.eta)
    lincomb!(out.ubar, α, A.ubar, β, B.ubar)
    return out
end
function _axpy!(out::OceanState, α, X::OceanState)
    axpy!(out.u, α, X.u)
    axpy!(out.h, α, X.h)
    axpy!(out.T, α, X.T)
    axpy!(out.S, α, X.S)
    axpy!(out.eta, α, X.eta)
    axpy!(out.ubar, α, X.ubar)
    return out
end

"""
    ocean_ssh(grid, state) -> CellField

The state's free-surface height (the fast mode's η, carried in
`state.eta`; `ocean_step!` re-stretches the layer column with it at
step end, keeping Σ_k h_k − H0 equal to `eta` exactly).
"""
ocean_ssh(grid::VerticalGrid, s::OceanState) = s.eta

# ------------------------------------------------------------ tendency

"""
    ocean_tendency!(out::OceanState, state::OceanState; grid, eos,
                    eta = nothing, bathymetry = nothing, Omega = nothing,
                    nu_h = 0.0, c_drag = 0.0, advect_tracers = true,
                    free_slip_vertices = nothing, ke_closed_edges = nothing,
                    depth_mean_out = nothing, projection = :pressure)
        -> OceanState

Baroclinic (slow) tendency of the layered system — see the module
header for the equation set and scaled-variable assembly. `state.u`
holds the velocity *anomalies* u′ during a split step (the driver
projects before stepping). The continuity fluxes are pure anomaly
fluxes, so the column sum Σ_k h_k is an exact invariant of the slow
step, and the thickness-weighted depth-mean of the pressure force is
removed from the momentum (the baroclinic projection; see module
header). `eta` overrides the free surface used in the interface
heights; `nothing` = the state-consistent diagnostic (Σ h_k − H,
identical at every stage of a split step by the column invariant).
`bathymetry` is the depth H [m] used by that diagnostic: a `CellField`
(issue #65) or `nothing` for the flat `grid.H0`.

After this call `out` holds the physical-unit tendencies of
(u, h, T, S). With `advect_tracers = true` (default; the #32
behavior) T and S carry the flux-form upwind tendency
([`tracer_tendency!`](@ref)) with the same layer fluxes the continuity
uses; with `advect_tracers = false` they carry ZERO tendency (the
operator-split FCT path of [`ocean_step!`](@ref), which advances T/S
outside the SSPRK). `eta`/`ubar` have zero tendency (they are advanced
by the fast solver at the driver level, not inside the slow SSPRK).

Optional dissipation (issue #40; both 0 = off, the #30 dynamics):

- `nu_h` [m²/s] — CAP of the shock-sensor Laplacian viscosity
  `+ν_e ∇²u′` on every layer, with the per-edge sensor
  ν_e = min(ν_h, dx_e³ |∇²u′_e|), dx_e = R·d_e the physical cell-center
  distance (no tunable prefactor: the "C" of the residual-based
  artificial-dissipation class is fixed to 1). On smooth flow of scale
  ℓ the damping ratio is ~ (dx/ℓ)³; at an unresolved jump
  ν_e ~ |u| dx.
- `c_drag` [–] — quadratic-drag coefficient C_d of the collapsed-layer
  friction `−C_d u|u|/h_e`, applied only on edges whose thin flank
  h_e = min(h_i, h_j) is below 5% of `grid.H0`; h_e is clamped below
  at `1e-3·grid.H0` so a zero/negative stage thickness (possible inside
  the SSPRK, the thickness floor acts once per step) gives a finite,
  always-decelerating drag.

Projection (`projection`):

- `:pressure` (default; the #30 form for direct callers) — only the
  depth-mean of the pressure force is removed; the Coriolis/Bernoulli,
  viscosity and drag terms keep their depth-mean.
- `:full` (used by [`ocean_step!`](@ref)) — the depth-mean of the
  COMPLETE momentum tendency is removed (u′ stays depth-mean free) and
  written into `depth_mean_out` (an `EdgeField`, if given), from which
  the driver forms the barotropic forcing (module header).

Walls: `free_slip_vertices` (vertex indices; ζ = 0 there in the PV and
the viscous Laplacian) and `ke_closed_edges` (edge indices; wall cells
average the kinetic energy over open edges) — see
[`wall_vertices`](@ref) and `ocean_step!(...; closed_edges)`.

Requires all of `state`'s ghost slots to be current in a partitioned
run (single-region in #30; the partitioned driver lands with #45).
"""
function ocean_tendency!(out::OceanState, state::OceanState;
                         grid::VerticalGrid, eos::AbstractOceanEOS,
                         eta = nothing, bathymetry = nothing,
                         Omega = nothing, nu_h::Real = 0.0,
                         c_drag::Real = 0.0, advect_tracers::Bool = true,
                         free_slip_vertices = nothing, ke_closed_edges = nothing,
                         depth_mean_out = nothing, projection::Symbol = :pressure)
    mesh = mesh_of(state.h)
    projection in (:pressure, :full) ||
        throw(ArgumentError("ocean_tendency!: projection must be :pressure or :full"))
    (; R, g) = williamson_constants()
    Ω = Omega === nothing ? williamson_constants().Omega : Omega
    K = nlayers(state.h)

    # density from the pluggable EOS (per-layer cell fields)
    ρ = ocean_density(state.T, state.S, eos)

    # free surface: the state-consistent diagnostic. The slow continuity
    # fluxes use pure anomaly velocities, so Σ_k h_k (and hence η) is
    # EXACTLY invariant during the slow step — there is no barotropic
    # self-feedback (see the depth-mean projection below).
    η = eta !== nothing ? eta :
        bathymetry === nothing ? ssh_from_thickness(grid, state.h) :
        ssh_from_thickness(grid, state.h, bathymetry)

    # scaled per-layer velocities u* = u/R (anomalies during a split step)
    us = similar(state.u)
    for k in 1:K
        scale!(copyto!(us[k], state.u[k]), 1 / R)
    end

    # scaled thicknesses h* = h/R
    hs = similar(state.h)
    for k in 1:K
        scale!(copyto!(hs[k], state.h[k]), 1 / R)
    end

    # interface heights z_k [m] (cell fields): z_1 = η, z_{k+1} = z_k − h_k
    # — from the ACTUAL column, not the coordinate map (see module header)
    z = [similar_field(state.h[1]) for _ in 1:(K + 1)]
    copyto!(z[1], η)
    for k in 1:K
        lincomb!(z[k + 1], 1.0, z[k], -1.0, state.h[k])
    end
    # layer-center heights z̄_k = (z_k + z_{k+1})/2
    zbar = [lincomb!(similar_field(state.h[k]), 0.5, z[k], 0.5, z[k + 1])
            for k in 1:K]

    # per-layer Coriolis/Bernoulli scratch (edge fields, one per layer) —
    # assembled in the layer loop, re-attached after the depth-mean
    # projection (they are baroclinic dynamics, not pressure force)
    cors = [similar_field(state.u[1]) for _ in 1:K]

    # cumulative interface pressure p_k/ρ0 = Σ_{j<k} (ρ_j/ρ0) h_j  [m],
    # accumulated across layers (pbar = g·pacc + (g/2)(ρ_k/ρ0)h_k
    # assembles the pressure in [m²/s²])
    pacc = similar_field(state.h[1])
    fill!(pacc, 0.0)
    for k in 1:K
        rho_frac = similar_field(state.h[k])   # (ρ_k/ρ0) as a cell field
        copyto!(rho_frac, ρ[k])
        scale!(rho_frac, 1 / eos.ρ0)

        # layer-mean pressure p̄_k/ρ0 = p_k/ρ0 + (g/2)(ρ_k/ρ0) h_k
        pbar = lincomb!(similar_field(state.h[k]), g, pacc,
                        g / 2, rho_frac * state.h[k])     # [m²/s²]

        # --- continuity: ∂h_k/∂t = −∇·(h*_k u′*_k)·R   [m/s]
        # pure anomaly fluxes: Σ_k ∂h_k/∂t = −∇·(Σ_k ĥ_k u′_k)·R ≡ 0
        # (the depth-mean transport ū is the fast solver's business;
        # see module header)
        Fk = edge_mean(hs[k]) * us[k]
        divergence!(out.h[k], Fk)
        scale!(out.h[k], -R)

        # --- momentum bracket [m/s²], pressure part only:
        #   −∇(p̄_k/ρ0)/R² − (g ρ_k/ρ0)·∇z̄_k/R²  (NO Cor/KE yet)
        q = potential_vorticity(us[k], hs[k],
                                VertexField(v -> 2 * Ω *
                                                mesh.vertex_xyz[3, v], mesh);
                                free_slip_vertices = free_slip_vertices)
        Cor = pv_flux(q, Fk)
        du = gradient(pbar)                # [m²/s²/rad]
        bz = edge_mean(rho_frac) * gradient(zbar[k])   # (ρ_k/ρ0)∇z̄ [m/rad]
        Kk = kinetic_energy(us[k]; closed_edges = ke_closed_edges)
        lincomb!(out.u[k], -1.0 / R^2, du, 0.0, du)  # −∇(p̄/ρ0)/R²
        axpy!(out.u[k], -g / R^2, bz)      # − b ∇z̄_k
        scale!(out.u[k], R)                # → [m/s²]
        # Coriolis/Bernoulli bracket [m/s²]: −Q̂⊥_k − ∇K*_k·R
        # both terms are scaled-variable brackets [rad/s²] → ·R [m/s²]
        # (as in sw_tendency!; issue #66: the PV flux was once left
        # unscaled, i.e. 1/R too weak — no rotation, no ζ×u advection)
        copyto!(cors[k], Cor)
        scale!(cors[k], -R)
        axpy!(cors[k], -R, gradient(Kk))

        # accumulate for the next layer: pacc += (ρ_k/ρ0) h_k  [m]
        # (pbar = g·pacc + (g/2)(ρ_k/ρ0) h_k assembles the pressure)
        axpy!(pacc, 1.0, rho_frac * state.h[k])
    end

    # re-attach the Coriolis/Bernoulli terms: out.u = full slow
    # momentum tendency (pressure + Coriolis/Bernoulli; viscosity and
    # drag follow). Its depth-mean is removed at the END (baroclinic
    # projection, below) — ALL terms, not only the pressure force.
    projection === :pressure && _remove_depth_mean!(out.u, state.h, nothing)
    for k in 1:K
        axpy!(out.u[k], 1.0, cors[k])
    end

    # --- horizontal velocity dissipation (issue #40 decision): the
    # shock-localized dissipation path for the UNRESOLVED hydraulic
    # jump. The energy-conserving momentum discretization has no
    # dissipation path, and the measured #40 lock-exchange runaway
    # (|u| → ∞ at the stuck front cell; uniform-density bisection: NOT
    # density-driven) is exactly the jump-JKE imbalance. Form: a
    # per-edge Laplacian viscosity +∇·(ν_e ∇u) with the LAPLACIAN-
    # MAGNITUDE shock sensor ν_e = min(ν_h, dx_e³ |∇²u|) (residual-
    # based artificial dissipation class, Jameson). Why this sensor:
    # (a) a |u|-based cap CANNOT localize — |u| is the same at the
    # jump as along the smooth current, so the cap damps everything
    # (measured: pure |u|d_e cap → F = 0.23, the uniform-ν failure);
    # (b) a uniform ν_h can never balance the trail-step PGF (balance
    # needs u ≈ √(PGF·dx) = √(g′H) — measured runaway at the
    # floor-pinned trail step, layers pinned at h_min with a persistent
    # density step); the Laplacian sensor gives exactly that balance
    # while staying OFF on smooth flow (damping ratio ~ (dx/ℓ)³; the
    # class's prefactor C is fixed to 1 — no tunable coefficient).
    # ν_h is only a SAFETY CAP (effective ν set by the flow; the cap
    # binds only in extreme states). Wave-speed/stability cap
    # unnecessary: ν_e Δt / d_e² = |u| Δt / d_e ≪ 1 always (CFL).
    # Per-edge dissipation (q_e = ν_e (u_j − u_i)/d_e): one TRiSK
    # Laplacian per layer (each edge appears in exactly one cell pair
    # so the per-edge ν_e re-weights the flux divergence directly), on
    # the ANOMALY momentum (the fast mode carries the barotropic DOF
    # separately).
    if nu_h != 0.0
        mesh = mesh_of(us[1])
        for k in 1:K
            lapu = laplacian(us[k]; free_slip_vertices = free_slip_vertices)
            for e in 1:mesh.n_edges
                # Laplacian-magnitude shock sensor (residual-based
                # artificial dissipation): ν_e = min(ν_h, d_e³ R² |L_e|).
                # Units: L_e [1/(rad·s)] = R·∇²u_phys, so
                # d_e³ R² L_e = (d_e R)³ ∇²u = dx³ ∇²u [m²/s]. At a
                # jump ∇²u ~ |u|/dx² → ν_e = |u|dx (the Re ≈ 1 capture
                # value); on a smooth scale ℓ ≫ dx, ν_e ~ dx³ |u|/ℓ²
                # → damping ratio (ν_e/|u|ℓ) ~ (dx/ℓ)³ — cubic in the
                # resolution ratio, so smooth flow is untouched.
                νe = mesh.d_e[e]^3 * R^2 * abs(lapu[e])
                νe = νe < nu_h ? νe : nu_h
                lapu[e] *= νe                  # ν_e-weighted [u·m/rad²·1/rad]
            end
            axpy!(out.u[k], 1.0 / R, lapu)     # [m/s²] (∇²_phys u = L_rad(u)/R²)
        end
    end

    # --- quadratic drag on COLLAPSED layers (issue #40): C_d·|u|u/h_k
    # — boundary-layer friction where a layer has collapsed to the
    # thickness floor. The measured #40 trail runaway: a pinned sheet
    # (h = h_min, ANY layer — measured in layer 1, the return flow, not
    # just the bottom) under a persistent interface step accelerates
    # without bound (u → 300 m/s): the sheet is smooth (the shock
    # sensor cannot see it) while the step's PGF pumps it forever, and
    # the FCT limiter refuses the flux that would erode the pinned
    # step (monotonicity keeps the step). Drag balances that PGF at
    # u* = √(PGF·h_min/C_d) (≈ 12 m/s at the #40 scaling) and is
    # strictly zero for full-depth layers. Real physics, not a patch:
    # a floor-collapsed sheet IS boundary flow; the laboratory currents
    # have exactly this friction, and every production ocean model
    # carries quadratic drag (C_d ≈ 2.5e-3, the canonical coefficient).
    if c_drag != 0.0
        # stage-thickness guard: inside the SSPRK an h_k can be ≤ 0 (the
        # thickness floor acts once per step, after the tracer step);
        # h_e ≤ 0 would give Inf/NaN (h_e = 0) or ANTI-drag (h_e < 0).
        # Clamp to a positive drag thickness (1e-3·H0, a fiftieth of
        # the 5%-H0 activation threshold) — finite and decelerating.
        h_drag_min = 1.0e-3 * grid.H0 / R
        h_drag_on = 0.05 * grid.H0 / R
        for k in 1:K
            hK = hs[k]                     # scaled thickness [rad]
            ue = us[k]                     # scaled velocity [rad/s]
            for e in 1:mesh.n_edges
                i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
                # MIN of the flanking thicknesses, not the mean: at a
                # pinned-sheet edge (h = h_min beside a deep layer) the
                # mean dilutes the drag ~20× and the sheet accelerates
                # without bound (measured: the #40 day-7 runaway, u →
                # 400 m/s). Boundary friction acts on the SHEET — the
                # thin cell sets the drag scale.
                h_e = max(min(hK[i], hK[j]), h_drag_min)
                h_e < h_drag_on || continue
                out.u[k][e] -= c_drag * ue[e] * abs(ue[e]) *
                               R / h_e            # u|u|/h_phys [m/s²]
            end
        end
    end

    # ------------------------------------------------ baroclinic
    # projection (exact, discrete; MPAS-O split-explicit class,
    # Higdon 2005 / Ringler et al. 2013 §3): subtract the thickness-
    # weighted depth-mean of the COMPLETE slow momentum tendency
    # (pressure, Coriolis/Bernoulli, viscosity, drag), so the anomaly
    # u′ stays depth-mean free and the barotropic DOF lives only in the
    # fast solver. The removed mean is returned in `depth_mean_out`
    # (when given): ocean_step! forces the fast mode with it. Removing
    # only the pressure-force mean (the pre-#40 form) left the depth-
    # mean of −Q̂⊥ − ∇K in u′ — a spurious barotropic velocity that
    # carries momentum but no mass, opposed by no pressure (measured:
    # |⟨u⟩ − ū| = 3 m/s in the closed lock-exchange channel, the
    # dense current dragged into a slow wedge).
    if projection === :full
        _remove_depth_mean!(out.u, state.h, depth_mean_out)
    elseif depth_mean_out !== nothing
        fill!(depth_mean_out, 0.0)
    end

    # T and S. advect_tracers = true: the #32 flux-form upwind
    # tendency with the SAME layer fluxes the continuity used
    # (compatibility by construction — Tracers.jl module header; the
    # continuity tendency out.h is passed so the product-rule closure
    # matches exactly). advect_tracers = false: zero tendency — the
    # operator-split FCT path of ocean_step! (issue #40 decision; the
    # Zalesak limiter is defined for complete content steps, not
    # partial SSPRK-stage increments) advances T/S once per step with
    # the stage-integrated mass fluxes (module header, "tracers").
    if advect_tracers
        tracer_tendency!(out.T, state.T, state.h, state.u; dhdt = out.h)
        tracer_tendency!(out.S, state.S, state.h, state.u; dhdt = out.h)
    else
        fill!(out.T, 0.0)
        fill!(out.S, 0.0)
    end
    # eta/ubar: zero slow tendency (advanced by the fast solver)
    fill!(out.eta, 0.0)
    fill!(out.ubar, 0.0)
    return out
end

# ------------------------------------------------------ z* regrid

"""
    _remap_edge_columns!(u::LayeredEdgeField, hsrc::LayeredCellField,
                         hdst::LayeredCellField) -> u

Conservative vertical ALE remap of the layer velocities on every edge
column (issue #62, the momentum half of the z* regrid in
[`ocean_step!`](@ref)) from the source edge thicknesses
ĥ^s_k = (hsrc_k,i + hsrc_k,j)/2 (the post-step Lagrangian layers) to the
target ones ĥ^d_k (edge means of `hdst`, the z* layers). The two
columns must have the same total (it is enforced to round-off: the
bottom target interface is set to the source one).

Exact overlap remap ([`_overlap_remap!`](@ref)), with interface depths
z_k measured downward from the surface (z_1 = 0, z_{k+1} = z_k + ĥ_k):
each source layer m carries the MC-limited piecewise-linear profile

    u(ξ) = u_m + s_m (ξ − ½),  ξ ∈ [0, 1] across the layer,
    s_m  = MC(u_m − u_{m−1}, u_{m+1} − u_m)   (s = 0 in the top and
                                               bottom layers)

(MC = monotonized-central limiter, `TracerFCT._mc`; van Leer 1977), and
each target layer receives the exact integral of that profile over its
overlap with every source layer. The source integrals partition the
column, so Σ_k ĥ_k u_k is conserved exactly (round-off), and every
reconstructed value lies between neighbouring layer means, so the
remap creates no new extrema — for ANY interface displacement. (The
#62 flux form, one swept-volume flux per interface, was valid only for
|d_k| ≤ ĥ_donor; with thin layers (60 z* levels, 3.3 m) the lock-
exchange interface swept more than a layer per step and the momentum
remap went unstable: measured NaN at day 6.) This is the piecewise-
linear member of the remap family of MOM6's ALE (White & Adcroft 2008).

Per-edge scalar loop through `Field[i]` getindex (AGENTS.md rule 3).
"""
function _remap_edge_columns!(u::LayeredEdgeField, hsrc::LayeredCellField,
                              hdst::LayeredCellField)
    K = nlayers(u)
    nlayers(hsrc) == nlayers(hdst) == K ||
        throw(ArgumentError("_remap_edge_columns!: layer counts differ"))
    mesh = mesh_of(u)
    zs = zeros(K + 1); zd = zeros(K + 1); uc = zeros(K); q = zeros(K)
    sl = zeros(K)
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        for k in 1:K
            zs[k + 1] = zs[k] + 0.5 * (hsrc[k][i] + hsrc[k][j])
            zd[k + 1] = zd[k] + 0.5 * (hdst[k][i] + hdst[k][j])
            uc[k] = u[k][e]
            q[k] = (zs[k + 1] - zs[k]) * uc[k]
        end
        zd[K + 1] = zs[K + 1]                 # identical totals (round-off)
        _overlap_remap!(q, zs, uc, zd, sl)
        for k in 1:K
            hk = zd[k + 1] - zd[k]
            u[k][e] = hk > 0 ? q[k] / hk : uc[k]
        end
    end
    return u
end

"""
    _overlap_remap!(q, zs, c, zd, sl) -> q

Conservative overlap remap of one column: source layer means `c` on the
interfaces `zs` (length K+1, non-decreasing, downward) to the target
interfaces `zd` (same end points). Writes the target layer CONTENTS
(∫ c dz) into `q`; `sl` is a length-K slope scratch. Piecewise-linear
MC-limited reconstruction, zero slope in the end layers (see
[`_remap_edge_columns!`](@ref)).
"""
function _overlap_remap!(q, zs, c, zd, sl)
    K = length(c)
    for m in 1:K
        sl[m] = (1 < m < K) ? _mc(c[m] - c[m - 1], c[m + 1] - c[m]) : 0.0
    end
    fill!(q, 0.0)
    m = 1
    for k in 1:K
        a, b = zd[k], zd[k + 1]
        b > a || continue
        while m < K && zs[m + 1] <= a
            m += 1
        end
        mm = m
        while mm <= K && zs[mm] < b
            lo = max(a, zs[mm]); hi = min(b, zs[mm + 1])
            hm = zs[mm + 1] - zs[mm]
            if hi > lo && hm > 0
                ξ = ((lo + hi) / 2 - zs[mm]) / hm   # overlap midpoint
                q[k] += (hi - lo) * (c[mm] + sl[mm] * (ξ - 0.5))
            end
            mm += 1
        end
    end
    return q
end

# -------------------------------------------------------------- drivers

"""
    _remove_depth_mean!(u, h, mean_out) -> u

Subtract from every layer of the edge tendency `u` its thickness-
weighted depth-mean Σ_k ĥ_k u_k / Σ_k ĥ_k (ĥ = edge_mean(h)); write the
removed mean into `mean_out` unless it is `nothing`.
"""
function _remove_depth_mean!(u, h, mean_out)
    K = nlayers(u)
    htot = similar_field(h[1])
    fill!(htot, 0.0)
    for k in 1:K
        axpy!(htot, 1.0, h[k])
    end
    dent = edge_mean(htot)
    fmean = similar_field(u[1])
    fill!(fmean, 0.0)
    for k in 1:K
        axpy!(fmean, 1.0, edge_mean(h[k]) * u[k])
    end
    for i in eachindex(fmean)
        fmean[i] = dent[i] != 0 ? fmean[i] / dent[i] : 0.0
    end
    for k in 1:K
        axpy!(u[k], -1.0, fmean)
    end
    mean_out === nothing || copyto!(mean_out, fmean)
    return u
end

"""
    wall_vertices(mesh, closed_edges) -> Union{Nothing, Vector{Int}}

The vertices on solid walls: every end vertex of a closed (no-flux)
edge, sorted and unique; `nothing` for `closed_edges = nothing`.
[`ocean_step!`](@ref) applies the free-slip condition ζ_v = 0 there
(Adcroft & Marshall 1998; see `TRiSK.potential_vorticity!`).
"""
wall_vertices(mesh, ::Nothing) = nothing
wall_vertices(mesh, closed) =
    sort!(unique!([v for e in closed
                   for v in (mesh.edge_vertex_pos[e], mesh.edge_vertex_neg[e])]))

"Zero the edge field `f` on the wall edges `closed` (no-op for `nothing`)."
_close_edges!(f, ::Nothing) = f
function _close_edges!(f, closed)
    for e in closed
        f[e] = 0.0
    end
    return f
end

"""
    ocean_step!(state::OceanState, grid, eos, t, Δt;
                n_sub = nothing, cfl = 0.3, Omega = nothing,
                nu_h = 0.0, c_drag = 0.0, min_h_frac = 0.0,
                tracer_advection = :fct, floor_stats = nothing,
                regrid = false, closed_edges = nothing,
                bathymetry = nothing)
        -> OceanState

Advance one mode-split step of Δt [s] (single-region state):

1. **fast**: the 2D SW system (H + η, ū) — from the carried
   `state.eta`/`state.ubar` — substepped `n_sub` times (SSPRK(3,3) on
   the Phase-1 `sw_tendency!`, identical to `sw_step!`, with bottom
   topography b = −H when `bathymetry` is given; `n_sub = nothing`
   picks the smallest count with Δt/n_sub within the barotropic CFL
   limit `cfl·dx/√(g·max H)`, minimum 1),
2. **slow**: SSPRK(3,3) baroclinic step on the velocity anomalies
   u′ = u − ū ([`ocean_tendency!`](@ref), anomaly continuity fluxes)
   over Δt,
3. **coupling**: add the fast solver's ū_end to the layers
   (`u_k = u′_k + ū_end`), re-stretch the column to the coordinate map
   with the time-mean fast η, and store (η̄, ū_end) back into the state
   (see module header),
4. **tracers** (`tracer_advection`): `:fct` (default, issue #40) —
   operator-split FCT step ([`fct_advect_tracer!`](@ref)) driven by the
   time-integrated layer mass fluxes that actually moved h (slow
   SSPRK stages + fast substeps; module header "tracers") — volume
   compatible: constant tracers stay constant and Σ A h C is conserved
   to round-off; `:upwind` — the #32 upwind tendency inside the slow
   SSPRK,
5. **z* regrid** (only if `regrid = true`): see below,
6. **floor** (only if `min_h_frac > 0`): see
   [`_apply_thickness_floor!`](@ref).

Keywords:

- `nu_h` [m²/s] — cap of the shock-sensor viscosity
  ν_e = min(ν_h, dx_e³|∇²u′_e|) (see [`ocean_tendency!`](@ref); 0 = off).
- `c_drag` [–] — collapsed-layer quadratic drag coefficient
  (see [`ocean_tendency!`](@ref); 0 = off).
- `min_h_frac` [–] — minimum layer thickness as a fraction of the
  column depth (h_min = min_h_frac·H, per column with `bathymetry`,
  min_h_frac·`grid.H0` without), applied once per step after the
  tracer step. **Opt-in** (default 0 = no floor). Must satisfy
  `min_h_frac·K < 1` (otherwise a resting column is entirely below the
  floor) — an `ArgumentError` is thrown.
- `floor_stats` — `nothing` or a `Dict{Symbol, Float64}` that
  accumulates the floor's non-standard interventions (see
  [`_apply_thickness_floor!`](@ref)).
- `regrid` — z* vertical coordinate (issue #62, DESIGN.md D7). `true`:
  after the re-stretch the layers are restored to the z* target
  `layer_thickness(grid, η̄, H)` (h_k = Δf_k·(H + η̄)) every step — the
  vertical ALE remap of MPAS-O/MOM6. u is remapped per edge column by
  [`_remap_edge_columns!`](@ref) (Σ_k ĥ_k u_k exact, MC-limited, no new
  extrema), T/S by the FCT step with h_new = target and the limited
  second-order vertical flux (`vertical_order = 2`; content exact).
  Layers then cannot collapse, so the floor and the collapsed-layer
  drag stay inactive in practice. Requires `tracer_advection = :fct`
  (`ArgumentError` otherwise). Default `false`: Lagrangian layers
  re-stretched with fractions preserved (the #30 behaviour the layered
  internal-wave tests exercise).

- `closed_edges` — `nothing` (default: every edge open, the closed
  sphere) or a collection of edge indices that are solid walls
  (no-flux / no-normal-flow lateral boundaries: channels, basins,
  coastlines). The normal-velocity tendency of every layer AND of the
  fast barotropic mode is zeroed on these edges, so an input state with
  u = ū = 0 there keeps u ≡ 0: the mass flux ĥu, the tracer flux and
  the vertical remap of u vanish across the wall exactly (volume and
  tracer content of each enclosed region are conserved separately).
  Tangential (vorticity) terms see u = 0 on the wall — the TRiSK
  free-slip wall of Ringler et al. (2010) §6 class. The input state
  must already have u = ū = 0 on closed edges.
- `bathymetry` — `nothing` (default: flat bottom at `grid.H0`) or a
  `CellField` of positive depths H [m] (issue #65, DESIGN.md D7; e.g.
  the #49 `bathymetry_*` profiles). The bottom sits at z = −H(x, y):
  the fast mode carries H + η over topography b = −H, and the
  re-stretch, the z* regrid target and the thickness floor use the
  per-column H (z* layers h_k = Δf_k·(H + η) follow the terrain). The
  slow PGF needs no change: its interface heights come from the actual
  column (z_1 = η, z_{k+1} = z_k − h_k), so the bottom interface is −H
  automatically and a resting uniform-density ocean over any H feels
  exactly −g∇η = 0. Requires the z* coordinate (`:simplez` levels are
  fixed fractions of a flat H0 — `ArgumentError` otherwise). The input
  state must be column-consistent with it (Σ_k h_k − H = η, e.g.
  `layer_thickness(grid, η, H)`).

Requires a column-consistent input state (Σ_k h_k − H = η, which
every `ocean_step!` output satisfies) for the `:fct` budget to be
exact. Split scheme documented in the module header (Lie splitting,
MPAS-O-flavored barotropic/baroclinic split).
"""
function ocean_step!(state::OceanState, grid::VerticalGrid,
                     eos::AbstractOceanEOS, t::Real, Δt::Real;
                     n_sub = nothing, cfl::Real = 0.3, Omega = nothing,
                     nu_h::Real = 0.0, min_h_frac::Real = 0.0,
                     c_drag::Real = 0.0, tracer_advection::Symbol = :fct,
                     floor_stats = nothing, regrid::Bool = false,
                     closed_edges = nothing, bathymetry = nothing)
    mesh = mesh_of(state.h)
    (; R, g) = williamson_constants()
    H0 = grid.H0
    K = nlayers(state.h)
    (bathymetry !== nothing && grid.mode === :simplez) &&
        throw(ArgumentError("ocean_step!: variable bathymetry requires the " *
                            "z* coordinate (:simplez levels are fixed " *
                            "fractions of a flat H0)"))
    # column depth H [m] as a cell field (flat H0 without bathymetry) and
    # the fast-mode bottom topography b = −H (nothing for the flat
    # bottom: the Phase-1 code path, ∇b ≡ 0)
    Hcol = bathymetry === nothing ? fill!(similar_field(state.eta), H0) :
           bathymetry
    topo = bathymetry === nothing ? nothing :
           scale!(copyto!(similar_field(state.eta), bathymetry), -1.0)
    Hmax = bathymetry === nothing ? H0 : maximum(bathymetry[i] for i in
                                                 eachindex(bathymetry))
    tracer_advection in (:fct, :upwind) ||
        throw(ArgumentError("ocean_step!: tracer_advection must be " *
                            ":fct or :upwind, got $tracer_advection"))
    0 <= min_h_frac * K < 1 ||
        throw(ArgumentError("ocean_step!: min_h_frac = $min_h_frac with " *
                            "$K layers puts the resting column below the " *
                            "floor (need 0 ≤ min_h_frac·K < 1)"))
    use_fct = tracer_advection === :fct
    wallv = wall_vertices(mesh, closed_edges)
    (regrid && !use_fct) &&
        throw(ArgumentError("ocean_step!: regrid = true requires " *
                            "tracer_advection = :fct (the tracer remap is " *
                            "the FCT vertical flux)"))

    # SSPRK(3,3) Butcher weights b = (1/6, 1/6, 2/3) of the three L
    # evaluations (TimeStepping._ssprk3_core! calls L in stage order):
    # the step's h increment is exactly −(Δt/R)∇·(Σ_s b_s F_s), so the
    # b-weighted stage fluxes are the time-integrated mass flux.
    ssprk_b = (1 / 6, 1 / 6, 2 / 3)

    # --- fast mode: 2D SW system (H + η, ū) from the carried state ---
    eta0 = state.eta
    ubar0 = state.ubar
    h2 = lincomb!(similar_field(eta0), 1.0, Hcol, 1.0, eta0)   # H + η
    # substep count from the barotropic CFL: wave speed √(g·max H) [m/s],
    # dx = min cell-center distance [rad] → physical dx = R·dx
    if n_sub === nothing
        dx = minimum(mesh.d_e)             # cell-center distances [rad]
        c = sqrt(g * Hmax)                 # [m/s]
        dt_max = cfl * R * dx / c          # [s]
        n_sub = max(1, ceil(Int, Δt / dt_max))
    end
    Δtf = Δt / n_sub
    fast = SWState(h2, ubar0)
    # time-mean fast η over the substeps (trapezoid across substep
    # ends): the end-of-step η of a barotropic wave is a random-phase
    # sample of a 2π/ω_bt ≈ 14 h oscillation — injecting it into the
    # layers via the re-stretch aliases the fast wave into the slow
    # state. The substep mean converges to the step-mean η (the
    # physically meaningful volume change); the oscillation averages
    # out. Includes both endpoints: η̄ = (η_0/2 + Σ_{n=1..n_sub−1} η_n +
    # η_end/2)/n_sub.
    #
    # Barotropic transport consistent with η̄ (tracer budget, :fct):
    # η_n − η_0 = −(Δtf/R) ∇·Σ_{m≤n} F_m with F_m the b-weighted SSPRK
    # flux ĥū of substep m, so
    #   η̄ − η_0 = −(Δt/R) ∇·F̄_bt,  F̄_bt = Σ_m (n_sub − m + ½)/n_sub² F_m
    # — exactly the column volume change the re-stretch imposes.
    Fbt = fill!(similar_field(ubar0), 0.0)        # [m²/s]
    wsub = Ref(0.0)
    fstage = Ref(0)

    # --- barotropic forcing (split-explicit, MPAS-O class): the depth-
    # mean of the slow momentum tendency at the step start that is NOT
    # the surface-pressure force −g∇η (the fast solver's own term):
    #   G = ⟨∂u/∂t⟩(state) − ⟨∂u/∂t⟩(state with ρ ≡ ρ0, u′ = 0),
    # held fixed over the substeps. The reference evaluation has uniform
    # density and no anomaly flow, so its depth-mean is exactly the
    # slow operator's discrete response to η alone; subtracting it
    # removes the η-dependence with the SAME discretization. (Subtracting
    # the fast solver's −g∇η instead leaves the slow/fast discretization
    # mismatch — the layer-weighted depth-mean PGF is only approximately
    # −g∇η — in G as a lagged, step-frozen surface-wave force: measured
    # exponential barotropic blow-up, |ū| → 900 m/s in 2 days, faster
    # with more substeps.) G carries the baroclinic pressure, Coriolis/
    # Bernoulli, viscosity and drag depth-means; without it the fast mode
    # never feels them and, in a closed basin, the barotropic response
    # that cancels them is missing.
    G = fill!(similar_field(ubar0), 0.0)
    begin
        start = copy(state)
        for k in 1:K
            axpy!(start.u[k], -1.0, ubar0)     # anomaly u′ at the start
        end
        tend0 = copy(start)
        ocean_tendency!(tend0, start; grid = grid, eos = eos, eta = eta0,
                        Omega = Omega, nu_h = nu_h, c_drag = c_drag,
                        advect_tracers = false, free_slip_vertices = wallv,
                        ke_closed_edges = closed_edges, depth_mean_out = G,
                        projection = :full)
        # uniform reference water mass (ρ ≡ ρ0 for the linear EOS; any
        # uniform T/S for a general EOS — only its uniformity matters)
        Tr = hasproperty(eos, :T_ref) ? eos.T_ref : state.T[1][1]
        Sr = state.S[1][1]
        ref = copy(start)
        for k in 1:K
            fill!(ref.u[k], 0.0)
            fill!(ref.T[k], Tr)
            fill!(ref.S[k], Sr)
        end
        Gref = similar_field(ubar0)
        ocean_tendency!(copy(ref), ref; grid = grid, eos = eos, eta = eta0,
                        Omega = Omega, nu_h = nu_h, c_drag = c_drag,
                        advect_tracers = false, free_slip_vertices = wallv,
                        ke_closed_edges = closed_edges, depth_mean_out = Gref,
                        projection = :full)
        axpy!(G, -1.0, Gref)
        _close_edges!(G, closed_edges)
    end

    function Lfast(out, s, τ)
        sw_tendency!(out, s; topography = topo)   # == sw_step!'s L
        axpy!(out.u, 1.0, G)               # baroclinic → barotropic forcing
        _close_edges!(out.u, closed_edges)
        if use_fct
            fstage[] = fstage[] % 3 + 1
            axpy!(Fbt, wsub[] * ssprk_b[fstage[]], edge_mean(s.h) * s.u)
        end
        return out
    end
    eta_mean = lincomb!(similar_field(eta0), 0.5, Hcol, 0.5, eta0)
    ubar_mean = lincomb!(similar_field(ubar0), 0.5, ubar0, 0.0, ubar0)
    for n in 1:n_sub
        wsub[] = (n_sub - n + 0.5) / n_sub^2
        fast = ssprk3_step!(Lfast, fast, t + (n - 1) * Δtf, Δtf)
        if n < n_sub
            axpy!(eta_mean, 1.0, fast.h)   # accumulate H + η_n
            axpy!(ubar_mean, 1.0, fast.u)
        end
    end
    axpy!(eta_mean, 0.5, fast.h)           # + η_end/2 (trapezoid close)
    scale!(eta_mean, 1.0 / n_sub)          # [m]: time-mean (H + η)
    axpy!(ubar_mean, 0.5, fast.u)
    scale!(ubar_mean, 1.0 / n_sub)         # time-mean ū (same weights)
    eta1 = lincomb!(similar_field(eta0), 1.0, eta_mean,
                    -1.0, Hcol)                            # η̄ = mean − H
    # time-mean ū with the SAME trapezoid weights as η̄: the fast state
    # is carried as the filtered pair (η̄, ū̄) — MPAS-O split-explicit
    # (Ringler et al. 2013 §3c) restarts the barotropic subcycle from the
    # time-averaged state. Pairing the filtered η̄ with the instantaneous
    # end-of-step ū (the pre-#40 form) is an amplifying restart for
    # barotropic waves once they are forced (measured with the
    # barotropic forcing G: exponential blow-up ×6 per 6 h).
    ubar1 = ubar_mean

    # --- slow mode: baroclinic SSPRK(3,3) on the anomalies u′ = u − ū ---
    # ū_start removed before the step, ū_end attached after
    work = copy(state)
    for k in 1:K
        axpy!(work.u[k], -1.0, ubar0)      # project: u′ = u − ū_start
    end
    # time-integrated slow layer fluxes Σ_s b_s ĥ_k,s u′_k,s [m²/s]
    Fsl = [fill!(similar_field(state.u[k]), 0.0) for k in 1:K]
    sstage = Ref(0)
    function L(out, s, τ)
        ocean_tendency!(out, s; grid = grid, eos = eos, eta = eta0,
                        Omega = Omega, nu_h = nu_h, c_drag = c_drag,
                        advect_tracers = !use_fct,
                        free_slip_vertices = wallv,
                        ke_closed_edges = closed_edges, projection = :full)
        for k in 1:K
            _close_edges!(out.u[k], closed_edges)
        end
        if use_fct
            sstage[] = sstage[] % 3 + 1
            for k in 1:K
                axpy!(Fsl[k], ssprk_b[sstage[]], edge_mean(s.h[k]) * s.u[k])
            end
        end
        return out
    end
    # ssprk3_step! returns a NEW state (the TimeStepping contract)
    work = ssprk3_step!(L, work, t, Δt)

    # --- coupling: attach the fast solver's ū_end to the layers
    for k in 1:K
        axpy!(work.u[k], 1.0, ubar1)       # u = u′ + ū_end
    end
    copyto!(work.eta, eta1)                # time-mean η (see above)
    copyto!(work.ubar, ubar1)

    # re-stretch the column to the coordinate map with the TIME-MEAN
    # fast η (anti-aliasing; see the accumulation above). The column
    # correction Δcol = (H + η̄) − Σh preserves the baroclinic
    # structure; its distribution is coordinate-dependent:
    colsum = similar_field(work.h[1])
    fill!(colsum, 0.0)
    for k in 1:K
        axpy!(colsum, 1.0, work.h[k])
    end
    #   :zstar   — proportional stretch h_k ← h_k·(H+η̄)/Σh (the column
    #              stretches with the surface; fractions preserved)
    #   :simplez — the whole anomaly goes into the top layer (the
    #              lower levels sit at fixed depths)
    dcol = lincomb!(similar_field(eta1), 1.0, Hcol, 1.0, eta1)  # H + η̄
    axpy!(dcol, -1.0, colsum)                              # Δcol [m]
    if grid.mode === :simplez
        axpy!(work.h[1], 1.0, dcol)
    else
        ratio = lincomb!(similar_field(eta1), 1.0, Hcol, 1.0, eta1)
        for i in eachindex(ratio)
            ratio[i] = ratio[i] / colsum[i]
        end
        for k in 1:K
            copyto!(work.h[k], work.h[k] * ratio)   # payload-preserving
        end
    end

    # --- tracer advance (:fct; operator split, issue #40 decision):
    # T/S with the production FCT scheme over the FULL step, from the
    # pre-step thicknesses (state.h) to the re-stretched pre-floor
    # thicknesses, transported by the mass fluxes that ACTUALLY moved h
    # (volume compatibility; module header "tracers"):
    #   F_k = F_sl,k + (ĥ_k/Σ_j ĥ_j)·(F̄_bt − Σ_j F_sl,j),  ĥ = edge_mean(h_old)
    # Σ_k F_k = F̄_bt, so Σ_k h_horiz,k = Σ_k h_old,k − (Δt/R)∇·F̄_bt
    # = H + η̄ = Σ_k h_new,k: the surface interface velocity diagnosed
    # by the FCT vanishes and no tracer crosses the sea surface. The
    # slow column transport Σ F_sl (nonzero whenever u′ has a column
    # mean) is replaced by F̄_bt exactly as the re-stretch replaces the
    # slow column volume by H + η̄. `fct_advect_tracer!` takes a
    # VELOCITY and forms F = ĥ_old·u itself, so the flux is passed as
    # u_tr = F/ĥ_old (exact up to one rounding). Edges with ĥ_old,k = 0
    # (layer empty on both flanks at step start) cannot carry flux in
    # this interface (u_tr = 0); their mismatch would reach w_1 — only
    # possible without the floor.
    if use_fct
        hm = [edge_mean(state.h[k]) for k in 1:K]
        u_tr = similar(work.u)
        for e in eachindex(Fbt)
            htot = 0.0
            Fcol = 0.0
            for k in 1:K
                htot += hm[k][e]
                Fcol += Fsl[k][e]
            end
            corr = htot != 0 ? (Fbt[e] - Fcol) / htot : 0.0
            for k in 1:K
                u_tr[k][e] = hm[k][e] != 0 ? Fsl[k][e] / hm[k][e] + corr : 0.0
            end
        end
        if regrid
            # --- z* ALE regrid (issue #62, DESIGN.md D7; module
            # header): restore the layers to the z* target
            # h_k = Δf_k·(H + η̄). Its column total equals the
            # re-stretched Σ_k h_k (both H + η̄), so the regrid is a
            # pure vertical redistribution: u by the conservative
            # limited edge-column remap from the Lagrangian layers to
            # the target; T/S by the same FCT step with h_new = target,
            # whose continuity-residual vertical flux carries the
            # Lagrangian → target transport (limited PLM, one flux per
            # interface: content exact).
            target = bathymetry === nothing ? layer_thickness(grid, eta1) :
                     layer_thickness(grid, eta1, bathymetry)
            _remap_edge_columns!(work.u, work.h, target)
            fct_advect_tracer!(work.T, state.h, target, u_tr, Δt;
                               vertical_order = 2)
            fct_advect_tracer!(work.S, state.h, target, u_tr, Δt;
                               vertical_order = 2)
            for k in 1:K
                copyto!(work.h[k], target[k])
            end
        else
            fct_advect_tracer!(work.T, state.h, work.h, u_tr, Δt)
            fct_advect_tracer!(work.S, state.h, work.h, u_tr, Δt)
        end
    end

    # --- re-projection: the layers' thickness-weighted depth-mean
    # velocity must equal the fast solver's ū (u′ depth-mean free).
    # The projected tendencies keep it so up to the stage/thickness
    # mismatch of the SSPRK stages and the re-stretch/remap; remove the
    # residual so no momentum-without-mass barotropic flow accumulates.
    let hm = [edge_mean(work.h[k]) for k in 1:K]
        for e in eachindex(work.ubar)
            num = 0.0
            den = 0.0
            for k in 1:K
                num += hm[k][e] * work.u[k][e]
                den += hm[k][e]
            end
            den > 0 || continue
            d = num / den - work.ubar[e]
            for k in 1:K
                work.u[k][e] -= d
            end
        end
    end

    # --- minimum-thickness floor (opt-in) ---
    if min_h_frac > 0
        if bathymetry === nothing
            _apply_thickness_floor!(work.h, (work.T, work.S), work.eta,
                                    min_h_frac * H0, H0; stats = floor_stats)
        else
            _apply_thickness_floor!(work.h, (work.T, work.S), work.eta,
                                    scale!(copyto!(similar_field(bathymetry), bathymetry),
                                           min_h_frac),
                                    bathymetry; stats = floor_stats)
        end
    end
    return work
end

"Per-column value of a scalar (flat) or `CellField` (per-cell) quantity."
_column_value(x::Real, i) = x
_column_value(x::CellField, i) = x[i]

"""
    _apply_thickness_floor!(h, tracers, eta, h_min, H0; stats = nothing)

Minimum-thickness floor of [`ocean_step!`](@ref) (issue #40,
user-ratified; the MPAS-O min-thickness treatment class, Petersen et
al. 2018 §2.6), per column i:

1. **Ordinary column** (headroom Σ_k max(h_k − h_min, 0) > deficit
   Σ_k max(h_min − h_k, 0)): layers below h_min are raised to h_min and
   the deficit is charged to the layers with headroom by scaling their
   excess above h_min with f = 1 − deficit/headroom. Column volume
   Σ_k h_k and tracer content Σ_k h_k C_k are conserved to round-off:
   headroom layers keep their concentration, a clipped layer receives
   the excess-weighted mixture of the donor layers.
2. **Starved column with positive volume** (0 < Σ_k h_k ≤ K·h_min,
   headroom ≤ deficit): the floor cannot be met conservatively. The
   column is HOMOGENIZED: every layer gets h_k = Σ_j h_j/K (> 0, but
   below h_min) and every tracer its column-mean concentration
   Σ h C/Σ h — volume and content conserved, the column's vertical
   structure is lost. Counted in `stats[:homogenized_columns]`.
3. **Non-positive column volume** (Σ_k h_k ≤ 0): no conservative fix
   exists. Every layer is set to h_min (tracer concentrations kept;
   layers with h ≤ 0 take the column's thickness-positive-weighted mean,
   or keep their own value if none), η is raised accordingly so the
   column stays consistent (Σ h − H = η), and the added volume
   K·h_min − Σ_k h_k [m] is NOT conserved: counted in
   `stats[:added_volume_columns]` and summed (per-column metres, not
   area-weighted) in `stats[:added_volume_m]`.

`h` is a `LayeredCellField`, `tracers` a tuple of `LayeredCellField`s
remapped with h, `eta` the surface `CellField`; `stats` is `nothing` or
a `Dict{Symbol, Float64}` that is accumulated into. `h_min` and the
column depth `H0` are numbers (flat bottom) or `CellField`s (per-column
values under variable bathymetry, issue #65).
"""
function _apply_thickness_floor!(h::LayeredCellField, tracers, eta::CellField,
                                 h_min_col, H_col; stats = nothing)
    K = nlayers(h)
    taken = zeros(Float64, K)
    gained = zeros(Float64, K)
    hpre = zeros(Float64, K)
    cpre = zeros(Float64, K)
    bump!(key, v) = stats === nothing ||
                    (stats[key] = get(stats, key, 0.0) + v; nothing)
    for i in eachindex(h[1])
        h_min = _column_value(h_min_col, i)
        deficit = 0.0
        headroom = 0.0
        col = 0.0
        for k in 1:K
            hk = h[k][i]
            col += hk
            hk < h_min ? (deficit += h_min - hk) : (headroom += hk - h_min)
        end
        deficit > 0 || continue
        if headroom > deficit
            # (1) ordinary column: conservative scaled-excess transfer
            f = 1.0 - deficit / headroom
            for k in 1:K
                hk = h[k][i]
                hpre[k] = hk
                taken[k] = 0.0
                gained[k] = 0.0
                if hk < h_min
                    h[k][i] = h_min
                    gained[k] = h_min - hk
                else
                    h[k][i] = h_min + (hk - h_min) * f
                    taken[k] = (hk - h_min) * (1.0 - f)
                end
            end
            # tracer remap: headroom layers keep their concentration
            # (their h and content scale together); clipped layers
            # receive the excess-weighted donor mixture.
            for C in tracers
                for k in 1:K
                    cpre[k] = C[k][i]
                end
                for k in 1:K
                    gained[k] == 0.0 && continue
                    m = hpre[k] * cpre[k]
                    for j in 1:K
                        taken[j] == 0.0 || (m += gained[k] * taken[j] /
                                            deficit * cpre[j])
                    end
                    C[k][i] = m / h_min
                end
            end
        elseif col > 0
            # (2) starved, positive volume: homogenize (conservative)
            hbar = col / K
            for C in tracers
                q = 0.0
                for k in 1:K
                    q += h[k][i] * C[k][i]
                end
                for k in 1:K
                    C[k][i] = q / col
                end
            end
            for k in 1:K
                h[k][i] = hbar
            end
            bump!(:homogenized_columns, 1.0)
        else
            # (3) non-positive volume: refill to h_min (NOT conservative)
            for C in tracers
                w = 0.0
                q = 0.0
                for k in 1:K
                    if h[k][i] > 0
                        w += h[k][i]
                        q += h[k][i] * C[k][i]
                    end
                end
                for k in 1:K
                    (h[k][i] <= 0 && w > 0) && (C[k][i] = q / w)
                end
            end
            added = K * h_min - col
            for k in 1:K
                h[k][i] = h_min
            end
            eta[i] = K * h_min - _column_value(H_col, i)
            bump!(:added_volume_columns, 1.0)
            bump!(:added_volume_m, added)
        end
    end
    return h
end

"""
    ocean_integrate!(state, grid, eos, t0, t_final, n_steps;
                     n_sub = nothing, Omega = nothing, nu_h = 0.0,
                     min_h_frac = 0.0, c_drag = 0.0,
                     tracer_advection = :fct, floor_stats = nothing,
                     regrid = false, closed_edges = nothing,
                     bathymetry = nothing)
        -> OceanState

Integrate `n_steps` equal mode-split steps (see [`ocean_step!`](@ref)
for the keywords).
"""
function ocean_integrate!(state::OceanState, grid::VerticalGrid,
                          eos::AbstractOceanEOS, t0::Real, t_final::Real,
                          n_steps::Integer; n_sub = nothing, Omega = nothing,
                          nu_h::Real = 0.0, min_h_frac::Real = 0.0,
                          c_drag::Real = 0.0, tracer_advection::Symbol = :fct,
                          floor_stats = nothing, regrid::Bool = false,
                          closed_edges = nothing, bathymetry = nothing)
    Δt = (t_final - t0) / n_steps
    for n in 1:n_steps
        state = ocean_step!(state, grid, eos, t0 + (n - 1) * Δt, Δt;
                            n_sub = n_sub, Omega = Omega, nu_h = nu_h,
                            min_h_frac = min_h_frac, c_drag = c_drag,
                            tracer_advection = tracer_advection,
                            floor_stats = floor_stats, regrid = regrid,
                            closed_edges = closed_edges,
                            bathymetry = bathymetry)
    end
    return state
end

# ---------------------------------------------------------- diagnostics

"""
    ocean_volume(grid, state) -> Real

Global ocean volume Σ A_i (H0 + η_i) = Σ A_i Σ_k h_k (unit-sphere
areas). Partitioned payloads contribute owned slots only. The #30 mass
conservation rung: conserved to round-off (via the fast solver's
flux-form continuity and the column re-stretch).
"""
function ocean_volume(grid::VerticalGrid, state::OceanState)
    (; H0) = grid
    m = mesh_of(state.h)
    s = 0.0
    for i in valid_indices(state.h[1])
        col = 0.0
        for k in 1:nlayers(state.h)
            col += state.h[k][i]
        end
        s += m.cell_area[i] * col
    end
    return s
end

"""
    ocean_kinetic_energy(state) -> Real

Global kinetic energy Σ A_i Σ_k (h_k · K_k) with K_k the R10 Eq 63
layer kinetic energy (unit-sphere areas).
"""
function ocean_kinetic_energy(state::OceanState)
    m = mesh_of(state.h)
    K = nlayers(state.h)
    s = 0.0
    for k in 1:K
        Kk = kinetic_energy(state.u[k])
        for i in valid_indices(state.h[k])
            s += m.cell_area[i] * state.h[k][i] * Kk[i]
        end
    end
    return s
end

end # module Ocean
