# Layered tracer advection (issue #32, DESIGN.md §5 Phase 2 "Advection
# schemes (upwind → higher-order, positive-definite on unstructured)",
# first rung: first-order upwind).
#
# Flux-form upwind advection of cell-centered tracers on the layered
# C-grid, driven by the SAME edge transports the continuity uses —
# compatibility of tracer and volume fluxes is by construction, not by
# tuning (Ringler et al. 2010 §2.4 flux form; MPAS-O tracer transport,
# Ringler, Petersen, Jacobsen & Jones 2013 §2e):
#
#   ∂(h_k C_k)/∂t = −∇·(F_k C_k^up)                (content, per layer)
#
# with F_k = ĥ_k u_k the layer thickness flux (R10 Eq 52) and C^up the
# UPWIND concentration at each edge (first order). The concentration
# tendency used by the SSPRK stages follows by the product rule with
# the layer continuity ∂h/∂t = −∇·F:
#
#   ∂C_k/∂t = [−∇·(F_k C_k^up) − C_k ∂h_k/∂t] / h_k
#
# Consequences used by the acceptance tests (all exact properties of
# this discrete form):
#   • Compatibility: a horizontally-uniform tracer under ANY flow has
#     zero tendency — ∇·(F·C) = C·∇·F exactly (divergence is linear in
#     the edge flux and C multiplies each edge value), so the two terms
#     cancel to round-off. No "non-divergent flow" caveat is needed.
#   • Conservation: Σ_i A_i (h_i C_i) is an exact invariant of the
#     content form on the closed sphere — the flux divergence sums to
#     the edge-pair cancellation Σ_e l_e F_e (C^up_{e,1} − C^up_{e,2})
#     = 0 bit-exactly (each edge contributes +flux to one flank and
#     −flux to the other). With concentration as the prognostic and a
#     multi-stage integrator the invariant Σ A h C is quadratic in the
#     state; it is then conserved to the integrator's quadratic-
#     invariant error (measured, small), not bit-exactly — documented
#     at the test.
#   • Positivity: the update is a convex combination of the cell value
#     and its upwind neighbors (coefficients Δt·l_e·F_e/h_new ≥ 0 under
#     the CFL), so a nonnegative tracer stays nonnegative.
#
# Vertical advection: interface tracer fluxes with the interface
# velocities IMPLIED BY THE CONTINUITY ("interface fluxes from
# continuity", issue #32). With the layer budgets
#
#   ∂h_k/∂t = −∇·F_k − (w_{k+1/2} − w_{k−1/2})
#
# (w positive downward, w_{bottom} = 0), the interface velocities are
# diagnosed bottom-up from the thickness evolution:
#
#   w_{k−1/2} = w_{k+1/2} + [(h_k^{new} − h_k^{horiz})/Δt]
#
# where h^{horiz} is the horizontal-flux-only update. The tracer flux
# through an interface is first-order upwind in the vertical: downward
# flow (w > 0) carries the concentration of the layer ABOVE, upward
# flow carries the layer BELOW. The surface (k = 1/2) carries the
# top-layer concentration in both directions (the inflow-through-the-
# surface boundary closure: water entering the column carries the local
# surface concentration). Summed over the column the vertical fluxes
# telescope, so the column tracer budget is closed exactly like the
# column volume budget. In the #30 slow dynamics the continuity is pure
# anomaly horizontal fluxes, so the diagnosed w is identically zero and
# the vertical flux vanishes — consistency with volume conservation by
# construction (the issue's requirement).
#
# Coordinate re-stretch closure (the piece #30's split adds): at step
# end `ocean_step!` re-stretches the layer column to the coordinate map
# with the fast mode's η. The re-stretch is a REMAP, not a flux: the
# fast (barotropic) mode carries no tracer in the anomaly-form split
# (the same documented approximation as #30's velocity-anomaly split —
# the barotropic transport does not advect layer properties). The
# closure adopted here: layer concentrations are UNCHANGED by the
# re-stretch (each layer's content scales with its volume; the
# exchanged water carries the local layer concentration). This keeps a
# horizontally-uniform T(z) profile exact through any fast-mode
# activity, and its conservation error is O(η′·C′) — second order in
# the anomalies, vanishing for uniform tracers and uniform η.
#
# Units: the Phase-1 convention — mesh geometry on the unit sphere,
# state physical (h [m], u [m/s], C arbitrary), tendencies in C/s; the
# assembly scales h* = h/R, u* = u/R, assembles flux divergences in
# scaled coordinates and multiplies by R back to physical units (see
# Atmos/Dynamics.jl for the derivation).
#
# AGENTS.md rule 3: no `Field.data` indexing outside `src/Fields/` —
# elementwise kernels (`lincomb!`, `axpy!`, `scale!`, `rdivide!`) and
# the TRiSK operators are the only access paths; per-cell scalar loops
# go through `Field[i]` getindex.

module OceanTracers

using ..Fields: Field, CellField, EdgeField, mesh_of, similar_field,
                lincomb!, axpy!, scale!, rdivide!, copyto!, fill!,
                valid_indices
using ..TRiSK: divergence!, edge_mean, upwind_flux
using ..Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using ..Vertical: VerticalGrid

export tracer_tendency!, advect_tracer!, tracer_total

_default_R() = 6.37122e6

# ------------------------------------------------------- dycore kernel

"""
    tracer_tendency!(out::LayeredCellField, C::LayeredCellField,
                     h::LayeredCellField, u::LayeredEdgeField;
                     dhdt = nothing, R = 6.37122e6) -> LayeredCellField

Flux-form upwind advection tendency of a layered tracer (concentration
form, per layer):

    ∂C_k/∂t = [−∇·(F_k C_k^up) − C_k·(∂h_k/∂t)] / h_k        [C/s]

with `F_k = ĥ_k u_k` the layer thickness flux (R10 Eq 52 — the SAME
flux the volume continuity uses, making the tracer and volume budgets
compatible by construction) and `C^up` the upwind concentration. The
layer thickness tendency `∂h_k/∂t` [m/s] must be the one the volume
continuity prescribes: pass `dhdt` to reuse a precomputed tendency
(the dycore passes the continuity tendency it already assembled);
`nothing` computes it from the same fluxes.

The content form ∂(hC)/∂t = −∇·(F C^up) — of which this is the product
rule — conserves Σ A_i h_i C_i exactly on the closed sphere (module
header). `out` is overwritten; all fields must share mesh and layer
count. R is the planet radius [m] (unit-sphere geometry rescaling,
Phase-1 convention).
"""
function tracer_tendency!(out::LayeredCellField, C::LayeredCellField,
                          h::LayeredCellField, u::LayeredEdgeField;
                          dhdt = nothing, R::Real = _default_R())
    K = nlayers(C)
    nlayers(h) == K == nlayers(u) ||
        throw(ArgumentError("tracer_tendency!: layer counts differ"))
    dhdt === nothing || nlayers(dhdt) == K ||
        throw(ArgumentError("tracer_tendency!: dhdt layer count differs"))
    mesh = mesh_of(C)
    for k in 1:K
        # scaled fluxes (Phase-1 convention): F* = ĥ* u*  [m²/(s·rad)]
        hs = scale!(copyto!(similar_field(h[k]), h[k]), 1 / R)
        us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
        F = edge_mean(hs) * us                       # volume flux
        # content tendency −∇·(F C^up)·R  [C·m/s]
        FT = upwind_flux(us, hs, C[k])
        content = divergence!(similar_field(C[k]), FT)
        scale!(content, -R)
        # thickness tendency: precomputed or from the same flux
        dh = dhdt === nothing ?
             begin
             d = divergence!(similar_field(C[k]), F)
             scale!(d, -R)
             end : dhdt[k]                            # [m/s]
        # ∂C/∂t = (content − C·∂h/∂t)/h
        lincomb!(out[k], 1.0, content, -1.0, C[k] * dh)
        rdivide!(out[k], out[k], h[k])
    end
    return out
end

# -------------------------------------------------------- entry points

"""
    advect_tracer!(C::LayeredCellField, h_old::LayeredCellField,
                   h_new::LayeredCellField, u::LayeredEdgeField, Δt::Real;
                   R = 6.37122e6) -> LayeredCellField

Advance a layered tracer one explicit upwind step of Δt [s] in place,
from thickness `h_old` [m] to `h_new` [m] (both given — the volume
evolution is the caller's; e.g. the dycore's continuity + re-stretch)
and layer normal velocity `u` [m/s]:

1. horizontal: content update with the upwind flux
   `(h C)^horiz = h_old C − Δt·R·∇·(F C^up)`,
2. vertical: interface velocities diagnosed from the continuity
   residual `w_{k−1/2} = w_{k+1/2} + (h_new − h^horiz)_k/Δt`
   (bottom-up, `w_bottom = 0`; module header), vertical content fluxes
   applied with the first-order upwind concentration,
3. back to concentration: `C_new = (h C)^{new} / h_new`.

The content form makes Σ A h C an exact invariant of the step up to
the final division round-off, for ANY (h_old → h_new) volume evolution
— the vertical fluxes are the continuity's own, so tracer and volume
budgets close identically (issue #32's consistency requirement).
`h_new` entries must be positive. Returns `C`.
"""
function advect_tracer!(C::LayeredCellField, h_old::LayeredCellField,
                        h_new::LayeredCellField, u::LayeredEdgeField,
                        Δt::Real; R::Real = _default_R())
    K = nlayers(C)
    nlayers(h_old) == nlayers(h_new) == K == nlayers(u) ||
        throw(ArgumentError("advect_tracer!: layer counts differ"))
    Δt > 0 || throw(ArgumentError("advect_tracer!: Δt must be positive"))
    mesh = mesh_of(C)

    # --- horizontal content + thickness updates (same flux F) ---
    Q = [similar_field(C[k]) for k in 1:K]     # content h·C
    hhor = [similar_field(h_old[k]) for k in 1:K]   # horizontal-only h
    for k in 1:K
        hs = scale!(copyto!(similar_field(h_old[k]), h_old[k]), 1 / R)
        us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
        F = edge_mean(hs) * us
        d = divergence!(similar_field(C[k]), F)
        scale!(d, -R * Δt)                     # Δh from horizontal fluxes
        copyto!(hhor[k], h_old[k])
        axpy!(hhor[k], 1.0, d)                 # h^horiz
        FT = upwind_flux(us, hs, C[k])
        content = divergence!(similar_field(C[k]), FT)
        scale!(content, -R * Δt)               # Δ(hC) horizontal
        copyto!(Q[k], h_old[k] * C[k])
        axpy!(Q[k], 1.0, content)
    end

    # --- vertical: interface velocities from the continuity residual ---
    # w[k] = velocity at interface k (k = 1 surface, k = K+1 bottom),
    # positive downward; w_k = w_{k+1} + (h_new − h^horiz)_k/Δt
    w = [similar_field(C[1]) for _ in 1:(K + 1)]
    fill!(w[K + 1], 0.0)                       # bottom: closed
    for k in K:-1:1
        res = h_new[k] - hhor[k]               # elementwise Field −
        lincomb!(w[k], 1.0, w[k + 1], 1 / Δt, res)
    end
    # vertical content fluxes: Q_k −= Δt (G_{k+1} − G_k),
    # G_k = w_k · C_up, upwind in the vertical; the surface (k = 1)
    # carries the top-layer concentration in both directions (closure:
    # inflow through the surface has the local surface concentration)
    for k in 1:K
        Ck = C[k]
        Qi = Q[k]
        wtop = w[k]
        wbot = w[k + 1]
        Cup_top = k == 1 ? Ck : C[k - 1]       # upwind above interface k
        for i in eachindex(Qi)
            Gtop = wtop[i] * (wtop[i] > 0 ? (k == 1 ? Ck[i] : Cup_top[i])
                              : Ck[i])
            Gbot = wbot[i] * (wbot[i] > 0 ? Ck[i] :
                              (k == K ? zero(Ck[i]) : C[k + 1][i]))
            Qi[i] -= Δt * (Gbot - Gtop)
        end
    end

    # --- back to concentration ---
    for k in 1:K
        copyto!(C[k], Q[k])
        rdivide!(C[k], C[k], h_new[k])
    end
    return C
end

"""
    advect_tracer!(C::LayeredCellField, h::LayeredCellField,
                   u::LayeredEdgeField, Δt::Real; R = 6.37122e6)
        -> LayeredCellField

Convenience form with the volume evolution computed internally from
the same fluxes (pure horizontal advection: `h_new = h − Δt·∇·(F)·R`,
no vertical fluxes — the diagnosed interface velocities vanish
identically). See the full method for the general form.
"""
function advect_tracer!(C::LayeredCellField, h::LayeredCellField,
                        u::LayeredEdgeField, Δt::Real;
                        R::Real = _default_R())
    K = nlayers(C)
    hnew = [similar_field(h[k]) for k in 1:K]
    for k in 1:K
        hs = scale!(copyto!(similar_field(h[k]), h[k]), 1 / R)
        us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
        d = divergence!(similar_field(C[k]), edge_mean(hs) * us)
        scale!(d, -R * Δt)
        copyto!(hnew[k], h[k])
        axpy!(hnew[k], 1.0, d)
    end
    return advect_tracer!(C, h, LayeredField(hnew), u, Δt; R = R)
end

"""
    tracer_total(C::LayeredCellField, h::LayeredCellField) -> Real

Global tracer content Σ_i A_i Σ_k h_k,i C_k,i (unit-sphere areas) —
the conserved quantity of the flux-form scheme (issue #32's closed-
domain budget rung). Partitioned payloads contribute owned slots only
(see `valid_indices`).
"""
function tracer_total(C::LayeredCellField, h::LayeredCellField)
    nlayers(C) == nlayers(h) ||
        throw(ArgumentError("tracer_total: layer counts differ"))
    m = mesh_of(C)
    s = 0.0
    for k in 1:nlayers(C), i in valid_indices(C[k])
        s += m.cell_area[i] * h[k][i] * C[k][i]
    end
    return s
end

end # module OceanTracers
