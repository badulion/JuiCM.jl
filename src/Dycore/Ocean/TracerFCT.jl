# Higher-order positive-definite tracer advection (issue #36, DESIGN.md
# §5 Phase 2 "Advection schemes (upwind → higher-order,
# positive-definite on unstructured)", second rung: the production
# scheme).
#
# Flux-corrected transport (FCT, Zalesak 1979) on the layered C-grid:
#
#   1. **High-order flux** — second-order donor-gradient (flex-PG
#      style, Skamarock 2008 §2c): the edge tracer value is
#      reconstructed from the DONOR (upwind) cell with the Green–Gauss
#      cell gradient (`TRiSK.cell_gradient` (least-squares over edge means), exact for linear
#      fields on the hexagonal Voronoi cells) evaluated at the edge
#      midpoint:
#
#         C^Ho_e = C_d + ∇C_d · (x_e − x_d),   d = donor cell
#
#      clipped into [min(C_i, C_j), max(C_i, C_j)] of the flanking
#      cells (monotonicity prescreen — keeps a pentagon-biased
#      gradient from seeding an extreme reconstruction, see below).
#      The flux is F_e · C^Ho_e with the SAME F_e = ĥ_e u_e as the
#      volume budget (compatibility preserved; only the tracer factor
#      differs from #32's upwind value).
#   2. **Low-order flux** — the #32 first-order upwind flux (positive,
#      diffusive).
#   3. **Antidiffusive flux** — A_e = F_e·(C^Ho_e − C^Lo_e) (zero
#      wherever the reconstruction equals the upwind value).
#   4. **Zalesak limiter** (eqs. (12)–(25) in the flux form): per cell
#      i, with the neighborhood extrema Cmax_i/Cmin_i (cell + edge
#      neighbors) and content headrooms
#
#         Q^↑_i = (Cmax_i − C_i)·A_i ,  Q^↓_i = (C_i − Cmin_i)·A_i ,
#
#      the antidiffusive fluxes entering/leaving i are summed (P^↑_i,
#      P^↓_i in content units), the cell factor is
#
#         β_i = min(1, Q^↑_i/P^↑_i, Q^↓_i/P^↓_i)
#
#      (P = 0 ⇒ no constraint from that side), and the edge correction
#      is scaled by
#
#         β_e = min(β_i, β_j)
#
#      (Zalesak eq. (25) in the classic single-factor-per-cell form:
#      both constraints couple into one β_i, so a cell at its local
#      extremum freezes ALL its antidiffusive fluxes, in and out).
#      Scaling a shared edge flux keeps the flux form EXACT: the
#      corrected edge flux is still one number seen by both flanks,
#      so the global content Σ A h C is conserved to round-off
#      regardless of the limiter.
#
# The result: second-order (horizontally) in smooth regions (β = 1,
# centered reconstruction), degrading to the upwind flux near extrema —
# monotone and positive while conservative (see the column bound below
# for the one, reported, exception).
#
# Pentagon note (DESIGN.md §7 imprinting risk): the Green–Gauss
# gradient on the 12 pentagonal cells carries a first-order
# midpoint-rule bias (asymmetric face weights). Two safeguards handle
# it without special-casing: the reconstruction prescreen clips the
# donor value into the flanking-cell range, and the Zalesak limiter
# clips any remaining overshoot. Pentagon cells therefore cannot
# produce negative concentrations; they may show slightly elevated
# local diffusion (documented behavior; the acceptance tests measure
# global errors, where the 12/642 pentagon contribution is negligible).
#
# Vertical (#40 revision): FIRST-ORDER upwind interface fluxes from
# the post-horizontal concentration C_hor = Q/h^horiz, signed by the
# interface velocity diagnosed from the continuity residual. Each
# interior interface carries ONE flux, subtracted from the layer it
# leaves and added to the one it enters, so the column content is
# conserved to round-off. (The #36 version used a centered value on
# the inflow side and the upwind value on the outflow side of the same
# interface — two different numbers, a content source/sink ∝ the
# vertical curvature of C; the "clip" of a centered mean into its two
# flanks was also a no-op.) The #36 acceptance tests have no vertical
# tracer gradient under vertical motion, so they do not distinguish
# upwind from a higher-order vertical flux.
# Surface closure: flow through the top of layer 1 carries layer 1's
# own concentration (#32); the bottom is closed.
#
# Vertical, second order (#62, `vertical_order = 2`): the interface
# value is the donor layer's MC-limited piecewise-linear
# reconstruction evaluated with the van Leer Courant correction
# (layer index increasing downward, d = donor layer, c = |w|Δt/h^hor_d):
#
#   s_d    = MC(C_d − C_{d−1}, C_{d+1} − C_d)
#   C_face = C_d ± ½ s_d (1 − c)    (+ for downward flow, − for upward)
#
# with the monotonized-central limiter (van Leer 1977)
#   MC(a, b) = 0 if ab ≤ 0, else sign(a)·min(2|a|, 2|b|, |a + b|/2).
# The face value stays between the donor's and the receiving layer's
# concentrations, and it is still ONE number per interface, so column
# content stays exact. This is the vertical remap step of the z* ALE
# regrid (`ocean_step!(…; regrid = true)`: the MPAS-O/MOM6 z*
# coordinate with a limited piecewise-linear remap, White & Adcroft
# 2008 class), where the residual-diagnosed vertical transport is a
# full layer-to-target remap rather than a small correction.
#
# Column-conservative bound: when the residual-driven outflow volume
# Δt·w exceeds the thickness h^horiz of a thin layer (measured #40: a
# ≈ 13 km/step residual against a 2-km floor-pinned sheet), upwind
# over-drains it and Q/h leaves the physical range (→ spurious EOS
# density step → the #40 day-7 momentum runaway). After the vertical
# step every cell is clipped into [Bmin, Bmax]·h_new, where B are the
# extrema of the PRE-STEP concentration over the cell's 2-ring (the
# horizontal limiter's C_lo bounds reach the 2-ring) in layers
# k−1:k+1 (the vertical donors) — a superset of every value the
# monotone parts of the scheme can produce. The clipped content is NOT
# discarded: it is redistributed within the same column in proportion
# to each layer's headroom (footroom), which keeps every layer inside
# its bounds. Only if the whole column lacks headroom is the remainder
# dropped; that loss is reported through the `clip_residual` keyword
# (measured 0 in the guard tests; it is a non-conservation, not a
# transport).
#
# Drained cells: a cell with h_new ≤ h_neglect = 1e-6·max h_new
# (including h_new ≤ 0) is "vanished": it keeps its pre-step
# concentration and its content h_new·C_old; the mismatch with the
# transported content enters the same column redistribution, so it is
# conserved too (and Σ h·C counts such cells with their actual,
# possibly non-positive, h_new). The horizontal limiter treats a cell
# with h^horiz ≤ h_neglect the same way (C_lo = C_old, β = 0).
# h_neglect is relative to the GLOBAL max thickness (MOM6
# set_h_neglect class) — a global reduction, to revisit under MPI.
#
# References:
# - Zalesak, S.T. (1979): "Fully multidimensional flux-corrected
#   transport algorithms for fluids", J. Comput. Phys. 31, 335–362
#   (limiter eqs. (12)–(25)).
# - Skamarock, W.C. (2008): "A linear analysis of FCT-limited
#   advection on the sphere in flux form", Mon. Wea. Rev. 136, 372–384
#   (donor-gradient / flex-PG flux form on C-grids).
# - van Leer, B. (1977): "Towards the ultimate conservative difference
#   scheme IV", J. Comput. Phys. 23, 276–299 (MC limiter, PLM with
#   Courant correction).
# - White, L., Adcroft, A. (2008): "A high-order finite volume
#   remapping scheme for nonuniform grids", J. Comput. Phys. 227,
#   7394–7422 (limited piecewise-polynomial vertical remap, MOM6 ALE).
# - Thuburn, J., Ringler, T.D., Klemp, J.B., Skamarock, W.C. (2014):
#   "A numerical framework for the solution of the primitive equations
#   on an unstructured mesh", J. Comput. Phys. 262, 1–24.
#
# AGENTS.md rule 3: per-cell scalar loops go through `Field[i]`
# getindex; elementwise kernels through the Fields module API.

module TracerFCT

using ..Fields: Field, CellField, EdgeField, mesh_of, similar_field,
                lincomb!, axpy!, scale!, copyto!, fill!
using ..TRiSK: divergence!, edge_mean, cell_gradient
using ..Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using ..Vertical: VerticalGrid

export fct_advect_tracer!

_default_R() = 6.37122e6

"""
    _mc(a, b) -> Float64

Monotonized-central (MC) limited slope from the two one-sided
differences `a` (upper) and `b` (lower) (van Leer 1977):
`0` if `a·b ≤ 0`, else `sign(a)·min(2|a|, 2|b|, |a + b|/2)`. Shared by
the `vertical_order = 2` FCT vertical flux and the z* momentum remap
of `ocean_step!`.
"""
_mc(a, b) = a * b <= 0 ? 0.0 :
            sign(a) * min(2 * abs(a), 2 * abs(b), abs(a + b) / 2)

# ------------------------------------------------------- flux assembly

"""
    _fct_fluxes(F_lo, F_hi, C, hs, us, mesh) -> nothing

Low-order (upwind) and high-order (donor-gradient, Green–Gauss
reconstruction at the edge midpoint, clipped into the flanking-cell
range) edge tracer fluxes for one layer, sharing the volume flux
`F_e = ĥ_e u_e` (already in scaled units h* = h/R, u* = u/R). Writes
into caller-provided `F_lo`/`F_hi` [concentration·flux units].
"""
function _fct_fluxes(F_lo::EdgeField, F_hi::EdgeField, C::CellField,
                     hs::CellField, us::EdgeField)
    mesh = C.mesh
    grad = cell_gradient(C)
    gx, gy, gz = grad[1], grad[2], grad[3]
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        Fe = us[e] * (hs[i] + hs[j]) / 2
        if us[e] > 0
            d, r = i, j
        else
            d, r = j, i
        end
        # donor reconstruction at the edge midpoint, clipped into the
        # flanking-cell range (pentagon prescreen; module header)
        val = C[d] + gx[d] * (mesh.edge_x[1, e] - mesh.cell_xyz[1, d]) +
              gy[d] * (mesh.edge_x[2, e] - mesh.cell_xyz[2, d]) +
              gz[d] * (mesh.edge_x[3, e] - mesh.cell_xyz[3, d])
        lo = min(C[i], C[j])
        hi = max(C[i], C[j])
        val = clamp(val, lo, hi)
        F_lo[e] = Fe * C[d]
        F_hi[e] = Fe * val
    end
    return nothing
end

# ----------------------------------------------------- Zalesak limiter

"""
    _apply_fct!(Q_out, Q_old, C, hhor, F_lo, F_hi, Abeta, ΔtR;
                h_neglect = 0.0) -> nothing

One horizontal FCT content update (classical Zalesak 1979 form):

1. low-order content solution `Q_lo = Q_old − ΔtR·∇·F_lo` (monotone
   by the upwind construction, module header);
2. low-order concentration `C_lo = Q_lo / hhor` and its neighborhood
   extrema `Cmax`/`Cmin` (cell + edge neighbors) — the limiter's
   bounds (Zalesak eq. (12) applied to the low-order state);
3. per-cell antidiffusive in/out sums `P_up`/`P_dn` (content units)
   and headrooms `Q_up_i = (Cmax_i − C_lo_i)·hhor_i`,
   `Q_dn_i = (C_lo_i − Cmin_i)·hhor_i` (Zalesak eqs. (23)–(24));
4. cell factor `β_i = min(1, Q_up_i/P_up_i, Q_dn_i/P_dn_i)` (empty
   side ⇒ no constraint from it; Zalesak eq. (25), classic form —
   both constraints couple into one β_i so a cell at its local
   extremum freezes ALL its antidiffusive fluxes), and the per-edge
   scaling

      β_e = min(β_i, β_j)

5. corrected content `Q_out = Q_old − ΔtR·∇·(F_lo + β A)`
   (= `Q_lo − ΔtR·∇·(β A)`, the low-order divergence taken once).

`Abeta` is scratch holding `F_hi − F_lo`; it is overwritten with the
limited version. Cells with `hhor ≤ h_neglect` (keyword, default 0)
have no meaningful low-order concentration: they use the pre-step `C`
in the bounds and get `β_i = 0`. The symmetric scaling keeps each corrected edge flux
a single number seen by both flanks — flux-form conservation is exact
regardless of the limiter.
"""
function _apply_fct!(Q_out::CellField, Q_old::CellField, C::CellField,
                     hhor::CellField, F_lo::EdgeField, F_hi::EdgeField,
                     Abeta::Vector{Float64}, ΔtR::Real; h_neglect::Real = 0.0)
    mesh = C.mesh
    n = mesh.n_cells
    # antidiffusive edge flux
    for e in 1:mesh.n_edges
        Abeta[e] = F_hi[e] - F_lo[e]
    end
    # low-order content solution
    Q_lo = similar_field(C)
    divergence!(Q_lo, F_lo)
    scale!(Q_lo, -ΔtR)
    axpy!(Q_lo, 1.0, Q_old)            # Q_lo = Q_old − ΔtR ∇·F_lo
    # low-order concentration
    C_lo = similar_field(C)
    # (a horizontally drained cell, hhor ≤ h_neglect, has no meaningful
    # low-order concentration: it contributes its pre-step value to the
    # bounds and takes no antidiffusive flux, β_i = 0 below)
    for i in 1:n
        C_lo[i] = hhor[i] > h_neglect ? Q_lo[i] / hhor[i] : C[i]
    end
    # neighborhood extrema of the low-order state (Zalesak eq. 12)
    Cmax = similar_field(C); fill!(Cmax, -Inf)
    Cmin = similar_field(C); fill!(Cmin, Inf)
    for i in 1:n
        for e in mesh.cell_edges[i]
            j = mesh.edge_cells[1, e] == i ? mesh.edge_cells[2, e] :
                mesh.edge_cells[1, e]
            Cmax[i] = max(Cmax[i], C_lo[j])
            Cmin[i] = min(Cmin[i], C_lo[j])
        end
        Cmax[i] = max(Cmax[i], C_lo[i])
        Cmin[i] = min(Cmin[i], C_lo[i])
    end
    # per-cell in/out sums of the antidiffusive flux (content units)
    P_up = zeros(n); P_dn = zeros(n)   # entering (fills), leaving (drains)
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        A = Abeta[e] * mesh.l_e[e]
        if A > 0        # leaves i, enters j
            P_dn[i] += A; P_up[j] += A
        elseif A < 0    # leaves j, enters i
            P_dn[j] -= A; P_up[i] -= A
        end
    end
    β = ones(n)                        # single factor per cell (both constraints)
    for i in 1:n
        if hhor[i] ≤ h_neglect
            β[i] = 0.0
            continue
        end
        # headroom in content units vs the entering/leaving antidiffusive
        # transport over the step: ΔtR·P/A_i is the content-per-area a
        # fully-kept antidiffusive flux would move (Zalesak eqs. 23–25
        # with the Δt and area factors of this discretization)
        q_up = max(Cmax[i] - C_lo[i], 0.0) * hhor[i] * mesh.cell_area[i]
        q_dn = max(C_lo[i] - Cmin[i], 0.0) * hhor[i] * mesh.cell_area[i]
        P_up[i] > 0 && (β[i] = min(β[i], q_up / (ΔtR * P_up[i])))
        P_dn[i] > 0 && (β[i] = min(β[i], q_dn / (ΔtR * P_dn[i])))
    end
    # per-edge factor: symmetric min of the two flanking cells (Zalesak
    # eq. 25 — a shared edge flux is ONE number seen by both flanks)
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        Abeta[e] *= min(β[i], β[j])
    end
    # corrected content increment: Q_out = Q_old − ΔtR ∇·(F_lo + β A)
    # (= Q_lo − ΔtR ∇·(β A): the low-order divergence is already inside
    # Q_lo, so accumulating onto Q_lo would apply it TWICE)
    Ftot = similar_field(F_lo)
    for e in 1:mesh.n_edges
        Ftot[e] = F_lo[e] + Abeta[e]
    end
    divergence!(Q_out, Ftot)
    scale!(Q_out, -ΔtR)
    axpy!(Q_out, 1.0, Q_old)
    return nothing
end

# -------------------------------------------------------- entry point

"""
    fct_advect_tracer!(C::LayeredCellField, h_old::LayeredCellField,
                       h_new::LayeredCellField, u::LayeredEdgeField,
                       Δt::Real; R = 6.37122e6,
                       clip_residual = nothing,
                       vertical_order = 1) -> LayeredCellField

Advance a layered tracer one FCT step of Δt [s] in place — the
production scheme of #36 (Zalesak 1979 FCT with donor-gradient
high-order fluxes; module header for the scheme). Signature and
semantics match [`OceanTracers.advect_tracer!`](@ref): content form
with `h_old → h_new` given by the caller (the volume evolution is the
dycore's), horizontal FCT fluxes, vertical interface velocities
diagnosed from the continuity residual with upwind interface
concentrations (one flux per interface, seen by both layers), a
column-conservative bound, back to concentration. Returns `C`.

Guards (module header, "Column-conservative bound" and "Drained cells"):

- cells with `h_new ≤ h_neglect = 1e-6·max(h_new)` (including
  `h_new ≤ 0`) keep their pre-step concentration; their content
  mismatch is moved to the rest of the column;
- the post-vertical bound clips each cell into the min/max of the
  pre-step concentration over its 2-ring in layers k−1:k+1 and
  redistributes the clipped content within the column.

`vertical_order` selects the interior interface value of the vertical
flux: `1` (default) — upwind; `2` — MC-limited piecewise-linear donor
reconstruction with the Courant correction (module header; used by
the z* regrid of `ocean_step!`). Both are single-valued per interface.

Column content is conserved exactly unless a column cannot absorb its
clipped content within its bounds; that dropped content
(Σ_i A_i·|residual_i|, A_i = `mesh.cell_area`) is written to
`clip_residual[]` when a `Ref{Float64}` is passed (0 otherwise).
"""
function fct_advect_tracer!(C::LayeredCellField, h_old::LayeredCellField,
                            h_new::LayeredCellField, u::LayeredEdgeField,
                            Δt::Real; R::Real = _default_R(),
                            clip_residual::Union{Nothing, Ref{Float64}} = nothing,
                            vertical_order::Integer = 1)
    K = nlayers(C)
    vertical_order in (1, 2) ||
        throw(ArgumentError("fct_advect_tracer!: vertical_order must be " *
                            "1 or 2, got $vertical_order"))
    nlayers(h_old) == nlayers(h_new) == K == nlayers(u) ||
        throw(ArgumentError("fct_advect_tracer!: layer counts differ"))
    Δt > 0 || throw(ArgumentError("fct_advect_tracer!: Δt must be positive"))
    mesh = mesh_of(C)
    n = mesh.n_cells

    # h_neglect: tracer-division floor, machine-tiny relative to the
    # thickest layer (MOM6 set_h_neglect class). Below it a cell is
    # "vanished": dividing its content by a TINY-but-nonzero h explodes
    # (measured #40: T = −222 K → |ΔT| = 4.8 K across an edge → fresh
    # EOS density step → runaway at day 7).
    hmax = 0.0
    for k in 1:K, i in eachindex(h_new[k])
        hmax = max(hmax, h_new[k][i])
    end
    h_neglect = 1.0e-6 * hmax

    C_old = [copyto!(similar_field(C[k]), C[k]) for k in 1:K]
    Q = [similar_field(C[k]) for k in 1:K]          # content h·C
    hhor = [similar_field(h_old[k]) for k in 1:K]
    Abeta = zeros(mesh.n_edges)                     # shared scratch
    for k in 1:K
        hs = scale!(copyto!(similar_field(h_old[k]), h_old[k]), 1 / R)
        us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
        F = edge_mean(hs) * us                      # shared volume flux
        # horizontal thickness update from the SAME volume flux
        d = divergence!(similar_field(C[k]), F)
        scale!(d, -R * Δt)
        copyto!(hhor[k], h_old[k])
        axpy!(hhor[k], 1.0, d)                      # h^horiz
        # FCT content update Q = Q_old − ΔtR ∇·(F_lo + β A)
        F_lo = similar_field(u[k])
        F_hi = similar_field(u[k])
        _fct_fluxes(F_lo, F_hi, C[k], hs, us)
        Q_old = similar_field(C[k])
        copyto!(Q_old, h_old[k] * C[k])
        _apply_fct!(Q[k], Q_old, C[k], hhor[k], F_lo, F_hi, Abeta, Δt * R;
                    h_neglect = h_neglect)
    end

    # post-horizontal concentration: the donor values of the vertical
    # step (a horizontally drained cell falls back to its pre-step value)
    Chor = [similar_field(C[k]) for k in 1:K]
    for k in 1:K, i in eachindex(Chor[k])
        Chor[k][i] = hhor[k][i] > h_neglect ? Q[k][i] / hhor[k][i] :
                     C_old[k][i]
    end

    # --- vertical: interface velocities from the continuity residual ---
    # w[k] is the velocity through the TOP of layer k, positive downward
    # (h_new_k = hhor_k + Δt·(w[k] − w[k+1]) by construction)
    w = [similar_field(C[1]) for _ in 1:(K + 1)]
    fill!(w[K + 1], 0.0)                            # bottom: closed
    for k in K:-1:1
        res = h_new[k] - hhor[k]
        lincomb!(w[k], 1.0, w[k + 1], 1 / Δt, res)
    end
    # surface closure: whatever enters/leaves through the top of layer 1
    # carries layer 1's own concentration (matches #32)
    for i in 1:n
        Q[1][i] += Δt * w[1][i] * Chor[1][i]
    end
    # interior interfaces: ONE flux per interface (upwind, or the
    # limited PLM donor value for vertical_order = 2), subtracted from
    # the layer it leaves and added to the one it enters — column
    # content is conserved to round-off
    for k in 2:K, i in 1:n
        wk = w[k][i]
        d = wk > 0 ? k - 1 : k                      # donor layer
        Cf = Chor[d][i]
        if vertical_order == 2 && wk != 0
            up = d > 1 ? Chor[d][i] - Chor[d - 1][i] : 0.0
            dn = d < K ? Chor[d + 1][i] - Chor[d][i] : 0.0
            c = min(1.0, abs(wk) * Δt / max(hhor[d][i], h_neglect, eps()))
            Cf += (wk > 0 ? 0.5 : -0.5) * _mc(up, dn) * (1 - c)
        end
        G = Δt * wk * Cf
        Q[k - 1][i] -= G
        Q[k][i] += G
    end

    # --- column-conservative bound (module header) ---
    # bounds: extrema of the PRE-STEP concentration over the 2-ring of
    # each cell (the horizontal limiter's C_lo bounds reach the 2-ring)
    # in layers k−1:k+1 (the vertical donors)
    B1max = [similar_field(C[k]) for k in 1:K]
    B1min = [similar_field(C[k]) for k in 1:K]
    B2max = [similar_field(C[k]) for k in 1:K]
    B2min = [similar_field(C[k]) for k in 1:K]
    _ring_extrema!(B1max, B1min, C_old, C_old, mesh)
    _ring_extrema!(B2max, B2min, B1max, B1min, mesh)
    lo = zeros(K); hi = zeros(K); live = falses(K)
    resid = 0.0
    for i in 1:n
        pool = 0.0
        for k in 1:K
            hn = h_new[k][i]
            if hn > h_neglect
                bmax = B2max[k][i]; bmin = B2min[k][i]
                k > 1 && (bmax = max(bmax, B2max[k - 1][i]);
                          bmin = min(bmin, B2min[k - 1][i]))
                k < K && (bmax = max(bmax, B2max[k + 1][i]);
                          bmin = min(bmin, B2min[k + 1][i]))
                lo[k] = bmin * hn; hi[k] = bmax * hn; live[k] = true
                q = clamp(Q[k][i], lo[k], hi[k])
            else
                # vanished cell: hold the pre-step concentration
                live[k] = false
                q = hn * C_old[k][i]
            end
            pool += Q[k][i] - q
            Q[k][i] = q
        end
        if pool > 0
            room = 0.0
            for k in 1:K
                live[k] && (room += hi[k] - Q[k][i])
            end
            f = room > 0 ? min(1.0, pool / room) : 0.0
            for k in 1:K
                live[k] && (Q[k][i] += f * (hi[k] - Q[k][i]))
            end
            resid += mesh.cell_area[i] * max(pool - room, 0.0)
        elseif pool < 0
            room = 0.0
            for k in 1:K
                live[k] && (room += Q[k][i] - lo[k])
            end
            f = room > 0 ? min(1.0, -pool / room) : 0.0
            for k in 1:K
                live[k] && (Q[k][i] -= f * (Q[k][i] - lo[k]))
            end
            resid += mesh.cell_area[i] * max(-pool - room, 0.0)
        end
        for k in 1:K
            hn = h_new[k][i]
            C[k][i] = hn > h_neglect ? Q[k][i] / hn : C_old[k][i]
        end
    end
    clip_residual === nothing || (clip_residual[] = resid)
    return C
end

"""
    _ring_extrema!(Mx, Mn, Ahi, Alo, mesh) -> nothing

Per layer, `Mx[k][i] = max(Ahi[k][i], max_j Ahi[k][j])` and
`Mn[k][i] = min(Alo[k][i], min_j Alo[k][j])` over the edge neighbours
j of i (one ring). Applied twice it gives the 2-ring extrema.
"""
function _ring_extrema!(Mx, Mn, Ahi, Alo, mesh)
    for k in eachindex(Mx), i in 1:mesh.n_cells
        mx = Ahi[k][i]; mn = Alo[k][i]
        for e in mesh.cell_edges[i]
            j = mesh.edge_cells[1, e] == i ? mesh.edge_cells[2, e] :
                mesh.edge_cells[1, e]
            mx = max(mx, Ahi[k][j]); mn = min(mn, Alo[k][j])
        end
        Mx[k][i] = mx; Mn[k][i] = mn
    end
    return nothing
end

end # module TracerFCT
