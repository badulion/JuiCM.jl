# Gravity current / lock-exchange benchmark (issue #40, DESIGN.md §4
# rung V3 — "front position vs. published laboratory data").
#
# Configuration (documented per the issue; the locally-Cartesian patch
# convention):
#
#   The benchmark lives on an equatorial band of the icosahedral
#   Voronoi mesh — the same "locally-Cartesian patch" convention the
#   #49 bathymetry profiles use (`bathymetry_channel`). Longitude λ
#   plays the role of the channel coordinate x [rad], and the channel
#   depth is H0 (flat bottom — lock exchange is a flat-bed experiment;
#   Huppert & Simpson 1980, Shin, Dalziel & Linden 2004 are both
#   flat-bed). No bathymetry call is needed: H = H0 flat.
#
#   TWO configurations (documented operating points):
#
#   1. **Patch** (physics demonstration): a finite dense patch
#      `|λ| < λ_lock, |φ| < φ_lock` releases a radially-spreading
#      current. On the sphere every longitude band WRAPS at the
#      antimeridian — a λ < 0 / λ > 0 half-domain split creates a
#      SECOND interface at λ ≈ ±π (24 wrap edges at nsub = 3) and the
#      run collapses at both ends (measured; see the smoke notes). The
#      finite patch is also the actual laboratory geometry (SDL04).
#
#   2. **Channel** (Froude measurement): dense fluid in
#      `−λ_lock < λ < λ_lock` full-width in latitude over a bounded
#      equatorial band `|φ| < φ_wide`, ambient beyond — the interface
#      at BOTH patch edges propagates currents; tracking the EAST edge
#      along the equator gives the 1-D lock-exchange front. The
#      published slumping Froude number F = U/√(g'H) ≈ 0.45 (SDL04
#      full-depth lock; Benjamin's energy-conserving front: 0.5) is
#      measured on this front.
#
#   Boussinesq T anomaly: with LinearEOS ρ = ρ0(1 − α(T − T_ref)), a
#   temperature difference ΔT produces g' = g·α·ΔT. The diagnostics
#   are non-dimensional:
#
#     F = (d x_f/dt) / √(g' H)   vs the published 0.45
#
#   so any (g', H) with the same dynamic regime reproduces the
#   laboratory scaling; the absolute values pick a resolved, stable
#   operating point (documented at the test).
#
# Diagnostics (the rung):
#   - dense-fraction profile f(λ): inside the channel band |φ| < band,
#     the area-weighted zonal-bin mean of the column dense fraction
#     f_i = Σ_k h_k (T_amb − T_k)/ΔT / Σ_k h_k (1 = full-depth dense,
#     0 = ambient); empty λ-bins are filled from the nearest non-empty
#     bin ([`dense_fraction_profile`](@ref));
#   - front position x_f(t): the easternmost λ-bin east of the lock
#     centre with f > frac (threshold fixed by the IC's T_amb and ΔT,
#     not re-measured from the evolving field);
#   - front speed during slumping: least-squares fit of x_f(t) over the
#     slumping window (before the returning bore meets the front —
#     SDL04 §3);
#   - buoyancy conservation: Σ A_i Σ_k h_k (T_amb − T) invariant
#     (closed channel, no-flux boundaries);
#   - monotonicity: f(λ) is non-increasing away from the lock centre
#     along the channel axis (no numerical reflection off the front).
#
# Dycore treatment documented in this PR (issue #40 decision,
# user-ratified): tracers advance by operator split (FCT once per
# mode-split step; the slow SSPRK solves (u, h) with T/S frozen), and
# `ocean_step!` carries a column-conserving minimum-thickness floor
# (the MPAS-O min-thickness class, Petersen et al. 2018 §2.6) — the
# collapse drains the head cell below the layer stability floor and the
# velocity update at h → 0 amplifies the PGF without bound (measured:
# h 200 → 0 m, NaN at day 3 without the floor).
#
# Reference:
# - Shin, J.O., Dalziel, S.B., Linden, P.F. (2004): "Gravity currents
#   produced by lock exchange", J. Fluid Mech. 521, 1–34.
#   Full-depth lock (D/H = 1): F = U/√(g'H) ≈ 0.45 (slumping phase);
#   Benjamin (1968) energy-conserving front: 0.5.
# - Huppert, H.E., Simpson, J.E. (1980): "The slumping of gravity
#   currents", J. Fluid Mech. 99, 785–799.
# - Petersen, M.R. et al. (2018): MPAS-Ocean Model User's Guide 6.0
#   (min-thickness treatment class).
#
# AGENTS.md rule 3: per-cell scalar loops go through `Field[i]`
# getindex; elementwise kernels through the Fields module API.

module GravityCurrent

using ..Fields: Field, CellField, EdgeField, mesh_of
using ..Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using ..Vertical: VerticalGrid
using ..OceanVertical: layer_thickness
using ..OceanEOS: LinearEOS
using ..ShallowWater: williamson_constants
using ..Ocean: ocean_state, ocean_step!, OceanState

export lock_exchange_state, lock_channel_state, lock_channel_T,
       dense_fraction_profile, front_position, front_speed,
       buoyancy_content, interface_monotone, gravity_current_benchmark!

# ------------------------------------------------------------- helpers

_lonlat(mesh, i) = (atan(mesh.cell_xyz[2, i], mesh.cell_xyz[1, i]),
                    asin(clamp(mesh.cell_xyz[3, i], -1.0, 1.0)))

"Mean dual-edge length d_e [rad] of `mesh` (the cell-centre spacing)."
_mean_de(mesh) = sum(mesh.d_e) / length(mesh.d_e)

"The lock predicate shared by every IC builder: `|λ| < λ_lock && |φ| < φ_lock`."
function _in_lock(mesh, i, λ_lock, φ_lock)
    λ, φ = _lonlat(mesh, i)
    return abs(λ) < λ_lock && abs(φ) < φ_lock
end

"Temperature step ΔT = g′/(g·α) realizing reduced gravity `g_prime`."
_ΔT(g_prime, eos::LinearEOS) = g_prime / (williamson_constants().g * eos.alpha)

function _lock_state(mesh, grid, eos, g_prime, λ_lock, φ_lock, T_amb)
    ΔT = _ΔT(g_prime, eos)
    K = nlayers(grid)
    h = layer_thickness(grid, CellField(i -> 0.0, mesh))
    T = LayeredCellField(mesh, K)
    S = LayeredCellField(mesh, K)
    for k in 1:K, i in 1:mesh.n_cells
        T[k][i] = _in_lock(mesh, i, λ_lock, φ_lock) ? T_amb - ΔT : T_amb
        S[k][i] = 35.0
    end
    u = LayeredEdgeField(mesh, K)
    return ocean_state(u, h, T, S), g_prime
end

# ------------------------------------------------------- configuration

"""
    lock_exchange_state(mesh, grid, eos; g_prime, L = nothing,
                        λ_lock = nothing, φ_lock = π/8,
                        T_amb = 20.0) -> (state, g′)

Lock-exchange initial state at rest: a full-depth dense-water LOCK
PATCH `|λ| < λ_lock, |φ| < φ_lock` (temperature `T_amb − ΔT` in every
layer; `T_amb` elsewhere, S = 35) on a flat bottom of depth `grid.H0`.
A patch, not a half-sphere: on the sphere every longitude band WRAPS at
the antimeridian, so a λ < 0 / λ > 0 split creates a SECOND interface
at λ ≈ ±π and the run collapses at both ends (measured). The finite
patch is also the laboratory geometry (Shin, Dalziel & Linden 2004;
Huppert & Simpson 1980 finite-volume slumping).

The lock half-length is set by exactly one of:
- `λ_lock` [rad] — the longitude half-width directly;
- `L` [m] — the physical half-length, `λ_lock = L / R` (R = Earth
  radius from `williamson_constants`);
- neither — the configuration of record, `λ_lock = 2.5·d̄_e` (2.5 mean
  cell spacings of `mesh`).
Passing both throws `ArgumentError`.

`g_prime` [m/s²] is the target reduced gravity; the temperature step
ΔT = g′/(g·α) realizes it with the `LinearEOS` (Boussinesq
ρ = ρ0(1 − α(T − T_ref))). Returns the state and the realized g′
(the discrete ΔT is exact, so realized = target).
"""
function lock_exchange_state(mesh, grid::VerticalGrid, eos::LinearEOS;
                             g_prime::Real, L = nothing, λ_lock = nothing,
                             φ_lock::Real = π / 8, T_amb::Real = 20.0)
    if λ_lock !== nothing && L !== nothing
        throw(ArgumentError("lock_exchange_state: pass L or λ_lock, not both"))
    end
    λl = λ_lock !== nothing ? Float64(λ_lock) :
         L !== nothing ? Float64(L) / williamson_constants().R :
         2.5 * _mean_de(mesh)
    λl > 0 || throw(ArgumentError("lock_exchange_state: lock half-width must be > 0"))
    return _lock_state(mesh, grid, eos, g_prime, λl, φ_lock, T_amb)
end

"""
    lock_channel_state(mesh, grid, eos; g_prime, λ_lock = 2.5·d̄_e,
                       φ_wide = 2·d̄_e, T_amb = 20.0) -> (state, g′)

Channel lock-exchange IC (the front-speed measurement configuration):
full-depth dense fluid in `|λ| < λ_lock` AND `|φ| < φ_wide` (a bounded
equatorial band), ambient elsewhere, at rest. Both λ-edges are
interfaces (the lock-exchange symmetry); the EAST edge is the tracked
front. The antimeridian stays ambient (the band is bounded), so no
wrap interface exists. Defaults are the configuration of record
(d̄_e = mean cell spacing of `mesh` [rad]). Same predicate as
[`lock_exchange_state`](@ref) and [`lock_channel_T`](@ref).
"""
function lock_channel_state(mesh, grid::VerticalGrid, eos::LinearEOS;
                            g_prime::Real,
                            λ_lock::Real = 2.5 * _mean_de(mesh),
                            φ_wide::Real = 2.0 * _mean_de(mesh),
                            T_amb::Real = 20.0)
    return _lock_state(mesh, grid, eos, g_prime, λ_lock, φ_wide, T_amb)
end

"""
    lock_channel_T(mesh, λ_lock, φ_wide, ΔT, T_amb = 20.0) -> CellField

The lock-exchange temperature of any single layer of the IC as a
`CellField`: `T_amb − ΔT` where `|λ| < λ_lock && |φ| < φ_wide`,
`T_amb` elsewhere — identical to every layer of
[`lock_channel_state`](@ref) with the same arguments (the IC is
full-depth). Used for IC verification.
"""
function lock_channel_T(mesh, λ_lock::Real, φ_wide::Real, ΔT::Real,
                        T_amb::Real = 20.0)
    return CellField(i -> _in_lock(mesh, i, λ_lock, φ_wide) ?
                          T_amb - ΔT : T_amb, mesh)
end

# ---------------------------------------------------------- diagnostics

"""
    dense_fraction_profile(T, h; T_amb, ΔT, band = 0.1, nbins = 200)
        -> (λc, f)

Along-channel profile of the column dense-water fraction. Per cell,

    f_i = Σ_k h_k (T_amb − T_k) / (ΔT · Σ_k h_k)

(1 = full-depth dense fluid, 0 = ambient). Cells with `|φ| < band`
[rad] are binned into `nbins` equal λ-bins on [−π, π); each bin holds
the area-weighted mean of f, and empty bins take the value of the
nearest non-empty bin (ties to the west). `T_amb` and `ΔT` are the IC's
values (fixed thresholds). Returns the bin centres `λc` [rad] and `f`.
`band` should not exceed the lock's latitude half-width, otherwise
bins inside the lock mix dense and ambient cells.
"""
function dense_fraction_profile(T::LayeredCellField, h::LayeredCellField;
                                T_amb::Real, ΔT::Real, band::Real = 0.1,
                                nbins::Integer = 200)
    m = mesh_of(T)
    K = nlayers(T)
    w = zeros(nbins); fa = zeros(nbins)
    for i in 1:m.n_cells
        λ, φ = _lonlat(m, i)
        abs(φ) < band || continue
        num = 0.0; col = 0.0
        for k in 1:K
            num += h[k][i] * (T_amb - T[k][i])
            col += h[k][i]
        end
        b = clamp(floor(Int, (λ + π) / (2π) * nbins) + 1, 1, nbins)
        w[b] += m.cell_area[i]
        fa[b] += m.cell_area[i] * num / (ΔT * col)
    end
    filled = findall(>(0), w)
    isempty(filled) && throw(ArgumentError("dense_fraction_profile: no cells with |φ| < band"))
    f = zeros(nbins)
    for b in 1:nbins
        if w[b] > 0
            f[b] = fa[b] / w[b]
        else
            j = filled[argmin(abs.(filled .- b))]
            f[b] = fa[j] / w[j]
        end
    end
    λc = [-π + (b - 0.5) * 2π / nbins for b in 1:nbins]
    return λc, f
end

"""
    front_position(T, h; T_amb, ΔT, frac = 0.5, band = 0.1,
                   nbins = 200) -> Real

Front longitude [rad] of the EAST-going dense current: the centre of
the easternmost λ-bin in [0, π) whose dense fraction
([`dense_fraction_profile`](@ref)) exceeds `frac`. The threshold is
fixed by the IC's `T_amb` and `ΔT` (passed explicitly, so it does not
drift as mixing erodes the extrema). Returns `0.0` (the lock centre)
if no east bin exceeds `frac`. Resolution: one bin, 2π/nbins.
"""
function front_position(T::LayeredCellField, h::LayeredCellField;
                        T_amb::Real, ΔT::Real, frac::Real = 0.5,
                        band::Real = 0.1, nbins::Integer = 200)
    λc, f = dense_fraction_profile(T, h; T_amb = T_amb, ΔT = ΔT,
                                   band = band, nbins = nbins)
    xf = 0.0
    for b in eachindex(λc)
        λc[b] >= 0 && f[b] > frac && (xf = max(xf, λc[b]))
    end
    return xf
end

"""
    front_speed(xf_t, t_t; window = nothing) -> Real

Slumping-phase front speed [rad/s if `xf_t` in rad]: least-squares
slope of x_f(t) at times `t_t` [s], over all points or over
`window = (t_lo, t_hi)` [s]. Throws if fewer than 2 points fall in the
window or the lengths differ.
"""
function front_speed(xf_t::Vector{<:Real}, t_t::Vector{<:Real};
                     window = nothing)
    length(xf_t) == length(t_t) ||
        throw(ArgumentError("front_speed: trajectory length mismatch"))
    idx = collect(eachindex(t_t))
    if window !== nothing
        idx = filter(i -> window[1] <= t_t[i] <= window[2], idx)
    end
    length(idx) >= 2 ||
        throw(ArgumentError("front_speed: fewer than 2 points in window"))
    tm = sum(t_t[i] for i in idx) / length(idx)
    xm = sum(xf_t[i] for i in idx) / length(idx)
    num = sum((t_t[i] - tm) * (xf_t[i] - xm) for i in idx)
    den = sum((t_t[i] - tm)^2 for i in idx)
    return num / den
end

"""
    buoyancy_content(T, h; T_amb = 20.0) -> Real

Temperature-deficit content Σ_i A_i Σ_k h_k (T_amb − T_{k,i}) over all
cells and layers, with `A_i` the cell area on the UNIT sphere [rad²]
and `h` in [m] (result in [K·m·rad²]; multiply by R²·g·α for buoyancy
content in [m⁴/s²]). With a linear EOS this is proportional to the
buoyancy content, the conserved quantity of the closed lock exchange
(FCT is conservative and the boundaries are no-flux); invariance is
what the rung checks. For the at-rest lock IC it equals
ΔT · H0 · Σ_{i ∈ lock} A_i.
"""
function buoyancy_content(T::LayeredCellField, h::LayeredCellField;
                          T_amb::Real = 20.0)
    m = mesh_of(T)
    s = 0.0
    for k in 1:nlayers(T), i in 1:m.n_cells
        s += m.cell_area[i] * h[k][i] * (T_amb - T[k][i])
    end
    return s
end

"""
    interface_monotone(T, h; T_amb, ΔT, band = 0.1, nbins = 200,
                       tol = 1e-6) -> Bool

Non-oscillatory interface check (issue #40): along the channel axis
(cells with `|φ| < band`), the dense fraction profile
([`dense_fraction_profile`](@ref)) must be non-increasing with
distance from the lock centre on BOTH sides — east of λ = 0 for
increasing λ, west of λ = 0 for decreasing λ — up to `tol`. A
numerical reflection off the front (a fold in the dense layer) shows
up as an increase. `T_amb`, `ΔT` are the IC values.
"""
function interface_monotone(T::LayeredCellField, h::LayeredCellField;
                            T_amb::Real, ΔT::Real, band::Real = 0.1,
                            nbins::Integer = 200, tol::Real = 1e-6)
    λc, f = dense_fraction_profile(T, h; T_amb = T_amb, ΔT = ΔT,
                                   band = band, nbins = nbins)
    east = findall(>=(0), λc)                 # increasing λ
    west = reverse(findall(<(0), λc))         # decreasing λ
    for side in (east, west)
        for j in 2:length(side)
            f[side[j]] > f[side[j - 1]] + tol && return false
        end
    end
    return true
end

# ------------------------------------------------------------- driver

"""
    gravity_current_benchmark!(state, grid, eos, T_amb, ΔT; n_steps,
                               save_every = 1, Δt = 450.0, n_sub = 4,
                               nu_h = 1e5, c_drag = 2.5e-3,
                               min_h_frac = 0.01, Omega = nothing,
                               frac = 0.5, band = 0.1, nbins = 200,
                               step_kwargs = (;))
        -> (; state, t, xf, buoyancy)

Run the lock-exchange release from `state` (an at-rest
[`lock_channel_state`](@ref)/[`lock_exchange_state`](@ref) result):
`n_steps` mode-split [`ocean_step!`](@ref) calls of `Δt` [s], recording
the front position ([`front_position`](@ref), thresholds from the IC's
`T_amb`, `ΔT`) and the buoyancy content ([`buoyancy_content`](@ref))
at t = 0 and every `save_every` steps.

The stepping defaults are the configuration of record (issue #40):
Δt = 450 s, n_sub = 4 barotropic substeps, shock-sensor viscosity cap
ν_h = 1e5 m²/s, collapsed-layer quadratic drag c_drag = 2.5e-3,
minimum-thickness floor min_h_frac = 0.01 (with H0 = 2e5 m, K = 10,
g′ = 4e-4 m/s² and an n_subdivisions = 4 mesh, set by the caller via
`grid`/`state`). `Omega = nothing` uses `ocean_step!`'s default
rotation. `step_kwargs` is forwarded verbatim to `ocean_step!` (and
overrides the named keywords) for dycore options not listed here.

`ocean_step!` returns a new state; the input `state` binding is not
updated, so use the returned `state` (the final state). Also returns
the save times `t` [s], front positions `xf` [rad] and buoyancy
content series `buoyancy` (units of [`buoyancy_content`](@ref)).
"""
function gravity_current_benchmark!(state::OceanState, grid::VerticalGrid,
                                    eos::LinearEOS, T_amb::Real, ΔT::Real;
                                    n_steps::Integer, save_every::Integer = 1,
                                    Δt::Real = 450.0, n_sub = 4,
                                    nu_h::Real = 1e5, c_drag::Real = 2.5e-3,
                                    min_h_frac::Real = 0.01, Omega = nothing,
                                    frac::Real = 0.5, band::Real = 0.1,
                                    nbins::Integer = 200,
                                    step_kwargs::NamedTuple = (;))
    diag(st) = (front_position(st.T, st.h; T_amb = T_amb, ΔT = ΔT,
                               frac = frac, band = band, nbins = nbins),
                buoyancy_content(st.T, st.h; T_amb = T_amb))
    kw = merge((; n_sub = n_sub, nu_h = nu_h, c_drag = c_drag,
                  min_h_frac = min_h_frac, Omega = Omega), step_kwargs)
    t = Float64[0.0]
    x0, b0 = diag(state)
    xf = Float64[x0]; buoy = Float64[b0]
    for n in 1:n_steps
        state = ocean_step!(state, grid, eos, (n - 1) * Δt, Δt; kw...)
        if n % save_every == 0
            x, b = diag(state)
            push!(t, n * Δt); push!(xf, x); push!(buoy, b)
        end
    end
    return (; state, t, xf, buoyancy = buoy)
end

end # module GravityCurrent
