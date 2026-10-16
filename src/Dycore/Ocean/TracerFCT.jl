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
# The result: second-order in smooth regions (β = 1, centered
# reconstruction), degrading to the upwind flux near extrema —
# monotone and positive while exactly conservative.
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
# Vertical: second-order limited interface fluxes — centered
# (arithmetic-mean) interface concentration clipped into the two
# flanking layer values, signed by the diagnosed interface velocity
# (upwind fallback at the surface/bottom closures, matching #32).
#
# References:
# - Zalesak, S.T. (1979): "Fully multidimensional flux-corrected
#   transport algorithms for fluids", J. Comput. Phys. 31, 335–362
#   (limiter eqs. (12)–(25)).
# - Skamarock, W.C. (2008): "A linear analysis of FCT-limited
#   advection on the sphere in flux form", Mon. Wea. Rev. 136, 372–384
#   (donor-gradient / flex-PG flux form on C-grids).
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
    _apply_fct!(Q_out, Q_old, C, hhor, F_lo, F_hi, Abeta, ΔtR) -> nothing

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
limited version. The symmetric scaling keeps each corrected edge flux
a single number seen by both flanks — flux-form conservation is exact
regardless of the limiter.
"""
function _apply_fct!(Q_out::CellField, Q_old::CellField, C::CellField,
                     hhor::CellField, F_lo::EdgeField, F_hi::EdgeField,
                     Abeta::Vector{Float64}, ΔtR::Real)
    mesh = C.mesh
    n = mesh.n_cells
    # antidiffusive edge flux
    for e in 1:mesh.n_edges
        Abeta[e] = F_hi.data[e] - F_lo.data[e]
    end
    # low-order content solution
    Q_lo = similar_field(C)
    divergence!(Q_lo, F_lo)
    scale!(Q_lo, -ΔtR)
    axpy!(Q_lo, 1.0, Q_old)            # Q_lo = Q_old − ΔtR ∇·F_lo
    # low-order concentration
    C_lo = similar_field(C)
    for i in 1:n
        C_lo[i] = Q_lo[i] / hhor[i]
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
        Ftot.data[e] = F_lo.data[e] + Abeta[e]
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
                       Δt::Real; R = 6.37122e6) -> LayeredCellField

Advance a layered tracer one FCT step of Δt [s] in place — the
production scheme of #36 (Zalesak 1979 FCT with donor-gradient
high-order fluxes; module header for the scheme). Signature and
semantics match [`OceanTracers.advect_tracer!`](@ref): content form
with `h_old → h_new` given by the caller (the volume evolution is the
dycore's), horizontal FCT fluxes, vertical interface velocities
diagnosed from the continuity residual with limited centered
interface concentrations, back to concentration. Returns `C`.
"""
function fct_advect_tracer!(C::LayeredCellField, h_old::LayeredCellField,
                            h_new::LayeredCellField, u::LayeredEdgeField,
                            Δt::Real; R::Real = _default_R())
    K = nlayers(C)
    nlayers(h_old) == nlayers(h_new) == K == nlayers(u) ||
        throw(ArgumentError("fct_advect_tracer!: layer counts differ"))
    Δt > 0 || throw(ArgumentError("fct_advect_tracer!: Δt must be positive"))
    mesh = mesh_of(C)

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
        # FCT content increment dQ = Q_lo − ΔtR ∇·(β A)
        F_lo = similar_field(u[k])
        F_hi = similar_field(u[k])
        _fct_fluxes(F_lo, F_hi, C[k], hs, us)
        Q_old = similar_field(C[k])
        copyto!(Q_old, h_old[k] * C[k])
        dQ = similar_field(C[k])
        _apply_fct!(dQ, Q_old, C[k], hhor[k], F_lo, F_hi, Abeta, Δt * R)
        copyto!(Q[k], dQ)                           # Q = Q_lo + limited A
    end

    # --- vertical: interface velocities from the continuity residual ---
    w = [similar_field(C[1]) for _ in 1:(K + 1)]
    fill!(w[K + 1], 0.0)                            # bottom: closed
    for k in K:-1:1
        res = h_new[k] - hhor[k]
        lincomb!(w[k], 1.0, w[k + 1], 1 / Δt, res)
    end
    for k in 1:K
        Ck = C[k]
        Qi = Q[k]
        wtop = w[k]
        wbot = w[k + 1]
        for i in eachindex(Qi)
            # second-order vertical: centered interface concentration
            # clipped into the flanking layer values (limited; the
            # surface k = 1 and bottom k = K closures fall back to the
            # local layer value, matching the #32 closures)
            Cup = k == 1 ? Ck[i] : C[k - 1][i]
            Cdn = k == K ? Ck[i] : C[k + 1][i]
            Ctop = clamp((Ck[i] + Cup) / 2, min(Ck[i], Cup),
                         max(Ck[i], Cup))
            Cbot = clamp((Ck[i] + Cdn) / 2, min(Ck[i], Cdn),
                         max(Ck[i], Cdn))
            Gtop = wtop[i] * (wtop[i] > 0 ? Ctop : Ck[i])
            Gbot = wbot[i] * (wbot[i] > 0 ? Ck[i] : Cbot)
            Qi[i] -= Δt * (Gbot - Gtop)
        end
    end

    for k in 1:K
        copyto!(C[k], Q[k])
        for i in eachindex(C[k])
            C[k][i] = Q[k][i] / h_new[k][i]
        end
    end
    return C
end

end # module TracerFCT
