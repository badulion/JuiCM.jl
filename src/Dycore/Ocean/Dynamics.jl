# Layered primitive-equation ocean dycore (issue #30, DESIGN.md §3.2).
#
# The first fully-3D component: layered Boussinesq primitive equations on
# the icosahedral C-grid, horizontal operators via the #9 TRiSK set and
# the Phase-1 vector-invariant discretization (Ringler et al. 2010, R10;
# Ringler, Petersen, Jacobsen & Jones 2013 — the MPAS-Ocean framework —
# for the layered/mode-split aspects).
#
# Equation set (per layer k = 1 top, flat bottom, z* or simple-z from
# the VerticalGrid mode; h_k = layer thickness, u_k = layer normal
# velocity, ρ_k = layer density from the pluggable EOS):
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
               tangential_flux, edge_mean, kinetic_energy, pv_flux,
               potential_vorticity, integral
using ..Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using ..Vertical: VerticalGrid
using ..OceanVertical: ssh_from_thickness, layer_center_z
using ..OceanEOS: AbstractOceanEOS, ocean_density, ocean_density!
using ..ShallowWater: williamson_constants
using ..Atmos: SWState, sw_step!
using ..TimeStepping: ssprk3_step!, ssprk3_integrate!
import ..TimeStepping: _stage, _lincomb!, _axpy!

export OceanState, ocean_state, ocean_tendency!, ocean_step!,
       ocean_integrate!, ocean_volume, ocean_kinetic_energy,
       ocean_ssh

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
- `T` — per-layer temperature [K or °C] (`LayeredCellField`)
- `S` — per-layer salinity [psu] (`LayeredCellField`); **prognostic
  slot with no dynamics in #30** (user-ratified): carried unchanged so
  the Phase-4 EOS/physics plugs in without touching the state layout.
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
                    eta = nothing, Omega = nothing) -> OceanState

Baroclinic (slow) tendency of the layered system — see the module
header for the equation set and scaled-variable assembly. `state.u`
holds the velocity *anomalies* u′ during a split step (the driver
projects before stepping). The continuity fluxes are pure anomaly
fluxes, so the column sum Σ_k h_k is an exact invariant of the slow
step, and the thickness-weighted depth-mean of the pressure force is
removed from the momentum (the baroclinic projection; see module
header). `eta` overrides the free surface used in the interface
heights; `nothing` = the state-consistent diagnostic (Σ h_k − H0,
identical at every stage of a split step by the column invariant).

After this call `out` holds the physical-unit tendencies of
(u, h, T, S); T and S have zero dynamical tendency in #30 (no tracer
advection until #32 — their tendencies here are exactly zero), and
`eta`/`ubar` have zero tendency (they are advanced by the fast solver
at the driver level, not inside the slow SSPRK).

Requires all of `state`'s ghost slots to be current in a partitioned
run (single-region in #30; the partitioned driver lands with #45).
"""
function ocean_tendency!(out::OceanState, state::OceanState;
                         grid::VerticalGrid, eos::AbstractOceanEOS,
                         eta = nothing, Omega = nothing)
    mesh = mesh_of(state.h)
    (; R, g) = williamson_constants()
    Ω = Omega === nothing ? williamson_constants().Omega : Omega
    K = nlayers(state.h)

    # density from the pluggable EOS (per-layer cell fields)
    ρ = ocean_density(state.T, state.S, eos)

    # free surface: the state-consistent diagnostic. The slow continuity
    # fluxes use pure anomaly velocities, so Σ_k h_k (and hence η) is
    # EXACTLY invariant during the slow step — there is no barotropic
    # self-feedback (see the depth-mean projection below).
    η = eta === nothing ? ssh_from_thickness(grid, state.h) : eta

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
                                                mesh.vertex_xyz[3, v], mesh))
        Cor = pv_flux(q, Fk)
        du = gradient(pbar)                # [m²/s²/rad]
        bz = edge_mean(rho_frac) * gradient(zbar[k])   # (ρ_k/ρ0)∇z̄ [m/rad]
        Kk = kinetic_energy(us[k])
        lincomb!(out.u[k], -1.0 / R^2, du, 0.0, du)  # −∇(p̄/ρ0)/R²
        axpy!(out.u[k], -g / R^2, bz)      # − b ∇z̄_k
        scale!(out.u[k], R)                # → [m/s²]
        # Coriolis/Bernoulli bracket [m/s²]: −Q̂⊥_k − ∇K*_k·R
        copyto!(cors[k], Cor)
        scale!(cors[k], -1.0)
        axpy!(cors[k], -R, gradient(Kk))  # (gradient in bracket·R units)

        # accumulate for the next layer: pacc += (ρ_k/ρ0) h_k  [m]
        # (pbar = g·pacc + (g/2)(ρ_k/ρ0) h_k assembles the pressure)
        axpy!(pacc, 1.0, rho_frac * state.h[k])
    end

    # ------------------------------------------------ baroclinic
    # projection (exact, discrete): subtract the thickness-weighted
    # depth-mean of the pressure force. In the continuum the depth-mean
    # of the hydrostatic PGF is exactly −g∇η̂; discretely the layer
    # weighting (ρ_k, z̄_k, edge means) makes it only approximately
    # −g∇η̂, and the RESIDUAL (layer-dependent, half-strength barotropic
    # force) is what destabilizes the split. Subtracting the actual
    # discrete depth-mean removes the barotropic mode of the DISCRETE
    # system exactly (uniform-density resting state still exact: the
    # force is zero before the projection).
    htot = similar_field(state.h[1])
    fill!(htot, 0.0)
    for k in 1:K
        axpy!(htot, 1.0, state.h[k])               # [m]
    end
    dent = edge_mean(htot)
    fmean = similar_field(out.u[1])
    fill!(fmean, 0.0)
    for k in 1:K
        axpy!(fmean, 1.0, edge_mean(state.h[k]) * out.u[k])   # [m³/s⁴]·R
    end
    for i in eachindex(fmean.data)
        fmean.data[i] = fmean.data[i] / dent.data[i]
    end                                            # depth-mean [m/s²]
    for k in 1:K
        axpy!(out.u[k], -1.0, fmean)
        # re-attach the Coriolis/Bernoulli terms (baroclinic dynamics —
        # these are the anomaly momentum, not the pressure force)
        axpy!(out.u[k], 1.0, cors[k])
    end

    # T and S: zero dynamical tendency in #30 (tracers arrive with #32);
    # eta/ubar: zero slow tendency (advanced by the fast solver)
    fill!(out.T, 0.0)
    fill!(out.S, 0.0)
    fill!(out.eta, 0.0)
    fill!(out.ubar, 0.0)
    return out
end

# -------------------------------------------------------------- drivers

"""
    ocean_step!(state::OceanState, grid, eos, t, Δt;
                n_sub = nothing, cfl = 0.3) -> OceanState

Advance one mode-split step of Δt [s] (single-region state):

1. **fast**: the 2D SW system (H0 + η, ū) — from the carried
   `state.eta`/`state.ubar` — substepped `n_sub` times (SSPRK(3,3) via
   the Phase-1 `sw_step!`; `n_sub = nothing` picks the smallest count
   with Δt/n_sub within the barotropic CFL limit `cfl·dx/√(g·H0)`,
   minimum 1),
2. **slow**: SSPRK(3,3) baroclinic step on the velocity anomalies
   u′ = u − ū ([`ocean_tendency!`](@ref), pure anomaly continuity
   fluxes — the column sum is invariant) over Δt,
3. **coupling**: replace the layers' depth-mean with the fast
   solver's ū_end (`u_k = u′_k + ū_end`), re-stretch the column to the
   coordinate map with the time-mean fast η, and store (η̄, ū_end)
   back into the state (see module header).

Split scheme documented in the module header (Lie splitting,
MPAS-O-flavored barotropic/baroclinic split).
"""
function ocean_step!(state::OceanState, grid::VerticalGrid,
                     eos::AbstractOceanEOS, t::Real, Δt::Real;
                     n_sub = nothing, cfl::Real = 0.3, Omega = nothing)
    mesh = mesh_of(state.h)
    (; R, g) = williamson_constants()
    H0 = grid.H0
    K = nlayers(state.h)

    # --- fast mode: 2D SW system (H0 + η, ū) from the carried state ---
    eta0 = state.eta
    ubar0 = state.ubar
    h2 = lincomb!(similar_field(eta0), 1.0,
                  fill!(similar_field(eta0), H0), 1.0, eta0)   # H0 + η
    # substep count from the barotropic CFL: wave speed √(g·H0) [m/s],
    # dx = min cell-center distance [rad] → physical dx = R·dx
    if n_sub === nothing
        dx = minimum(mesh.d_e)             # cell-center distances [rad]
        c = sqrt(g * H0)                   # [m/s]
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
    eta_mean = lincomb!(similar_field(eta0), 1.0,
                        fill!(similar_field(eta0), H0 / 2), 0.5, eta0)
    for n in 1:n_sub
        fast = sw_step!(fast, t + (n - 1) * Δtf, Δtf)
        if n < n_sub
            axpy!(eta_mean, 1.0, fast.h)   # accumulate H0 + η_n
        end
    end
    axpy!(eta_mean, 0.5, fast.h)           # + η_end/2 (trapezoid close)
    scale!(eta_mean, 1.0 / n_sub)          # [m]: time-mean (H0 + η)
    eta1 = lincomb!(similar_field(eta0), 1.0, eta_mean,
                    -1.0, fill!(similar_field(eta0), H0))  # η̄ = mean − H0
    ubar1 = fast.u                         # end-of-step fast ū

    # --- slow mode: baroclinic SSPRK(3,3) on the anomalies u′ = u − ū ---
    # pure anomaly fluxes in the slow continuity (exact column
    # invariant); ū_start removed before the step, ū_end attached after
    work = copy(state)
    for k in 1:K
        axpy!(work.u[k], -1.0, ubar0)      # project: u′ = u − ū_start
    end
    L(out, s, τ) = ocean_tendency!(out, s; grid = grid, eos = eos,
                                   eta = eta0, Omega = Omega)
    # ssprk3_step! returns a NEW state (the TimeStepping contract)
    work = ssprk3_step!(L, work, t, Δt)

    # --- coupling: REPLACE the layers' depth-mean with the fast
    # solver's ū_end (the slow system's barotropic velocity DOF is
    # unphysical — its force was frozen — so it is discarded)
    for k in 1:K
        axpy!(work.u[k], 1.0, ubar1)       # u = u′ + ū_end
    end
    copyto!(work.eta, eta1)                # time-mean η (see above)
    copyto!(work.ubar, ubar1)

    # re-stretch the column to the coordinate map with the TIME-MEAN
    # fast η (anti-aliasing; see the accumulation above). The column
    # correction Δcol = (H0 + η̄) − Σh preserves the baroclinic
    # structure; its distribution is coordinate-dependent:
    colsum = similar_field(work.h[1])
    fill!(colsum, 0.0)
    for k in 1:K
        axpy!(colsum, 1.0, work.h[k])
    end
    #   :zstar   — proportional stretch h_k ← h_k·(H0+η̄)/Σh (the column
    #              stretches with the surface; fractions preserved)
    #   :simplez — the whole anomaly goes into the top layer (the
    #              lower levels sit at fixed depths)
    dcol = lincomb!(similar_field(eta1), 1.0,
                    fill!(similar_field(eta1), H0), 1.0, eta1)  # H0 + η̄
    axpy!(dcol, -1.0, colsum)                              # Δcol [m]
    if grid.mode === :simplez
        axpy!(work.h[1], 1.0, dcol)
    else
        ratio = lincomb!(similar_field(eta1), 1.0,
                         fill!(similar_field(eta1), H0), 1.0, eta1)
        for i in eachindex(ratio.data)
            ratio.data[i] = ratio.data[i] / colsum.data[i]
        end
        for k in 1:K
            copyto!(work.h[k], work.h[k] * ratio)   # payload-preserving
        end
    end
    return work
end

"""
    ocean_integrate!(state, grid, eos, t0, t_final, n_steps;
                     n_sub = nothing) -> OceanState

Integrate `n_steps` equal mode-split steps (see [`ocean_step!`](@ref)).
"""
function ocean_integrate!(state::OceanState, grid::VerticalGrid,
                          eos::AbstractOceanEOS, t0::Real, t_final::Real,
                          n_steps::Integer; n_sub = nothing, Omega = nothing)
    Δt = (t_final - t0) / n_steps
    for n in 1:n_steps
        state = ocean_step!(state, grid, eos, t0 + (n - 1) * Δt, Δt;
                            n_sub = n_sub, Omega = Omega)
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
