# Horizontal pressure-gradient force (PGF) schemes for the layered ocean
# on z* (terrain-following) layers (issue #73, DESIGN.md §7 risk
# "z-star pressure-gradient errors produce spurious flows").
#
# All schemes return, per layer k and edge e (normal n̂_e from cell
# i = edge_cells[1, e] to j = edge_cells[2, e]), the pressure-gradient
# acceleration −(1/ρ0)∂p/∂n at constant height, averaged over the layer
# [m/s²]. Inputs are the actual column state (`ocean_tendency!`):
# interface heights z_1 = η, z_{k+1} = z_k − h_k (k = 1 top), layer
# thicknesses h_k and layer densities ρ_k (layer-constant, from the EOS).
# Hydrostatic Boussinesq pressure, p/ρ0 = g(η − z) + ∫_z^η g ρ′/ρ0 dz′
# with ρ′ = ρ − ρ0. Every scheme reproduces −g∇η exactly for uniform
# density (ρ′ ≡ 0).
#
# Schemes (`pgf` keyword):
#
# :twoterm — the #30 form −∇p̄_k − b_k∇z̄_k (layer-mean pressure and
#   layer-centre height, full density) — the classic σ-coordinate
#   two-term discretization. Computed in `ocean_tendency!` itself (not
#   here); kept as the default reference.
#
# :adcroft2008 — finite-volume PGF with analytic integration, Adcroft,
#   Hallberg & Harrison (2008), Ocean Modelling 22, 106–113. The force on
#   the layer control volume between the two columns is the contour
#   integral of the pressure anomaly pa = p + ρ0 g z around it (the
#   −ρ0 g z reference integrates to zero on a closed contour): the
#   column integrals ∫pa dz at both cell centres and the integrals of pa
#   along the sloping top and bottom interfaces. With layer-constant
#   density, the layer properties interpolated linearly along the
#   interfaces between the columns, and a linear EOS, all integrals are
#   exact (products of linear functions). In MOM6's notation (the
#   Boussinesq branch of MOM_PressureForce_FV.F90):
#     PFu = 2/(ρ0 dx (h_i + h_j)) · [ (pa_i h_i + intz_i) − (pa_j h_j + intz_j)
#             + (h_j − h_i)·intx_pa_top − (z_bot,j − z_bot,i)·intx_dpa ]
#   with pa at the layer top, intz = ∫(pa − pa_top) dz over the layer,
#   intx_pa_top the along-interface mean of pa on the top interface and
#   intx_dpa the along-layer mean of the change of pa through the layer.
#
# :mom6 — MOM6's production configuration: the same FV integral with a
#   piecewise-linear (PLM) reconstruction of density within each layer
#   (MOM6 RECONSTRUCT_FOR_PRESSURE = True, PRESSURE_RECONSTRUCTION_SCHEME
#   = 1, BOUNDARY_EXTRAPOLATION_PRESSURE = True — the defaults with ALE
#   regridding), ported from MOM6 src/ALE/PLM_functions.F90 and
#   MOM_ALE.F90 (ALE_PLM_edge_values): thickness-weighted limited slopes
#   (PLM_slope_wa), the edge-consistency pass (PLM_monotonized_slope),
#   and extrapolated slopes in the top and bottom layers
#   (PLM_extrapolate_slope). MOM6 reconstructs T and S; with a linear EOS
#   reconstructing ρ is identical. The reconstruction preserves the layer
#   mean, so for a linear EOS only the column integral changes:
#   intz = (g/ρ0) h² [(ρ_t − ρ0)/2 + (ρ_b − ρ_t)/6] (ρ_t, ρ_b the
#   reconstructed top/bottom values). Exact (round-off) for a linear
#   profile ρ(z) in every layer.
#
# :roms — the density-Jacobian scheme of Shchepetkin & McWilliams (2003),
#   JGR 108(C3), 3090, as implemented in ROMS prsgrd32.h: pressure at the
#   layer centres from a vertical integration with monotonized cubic
#   (harmonic-mean "elementary differences") corrections, and the
#   along-layer force
#     PF = [P_i − P_j − (g/2ρ0)( (ρ′_j + ρ′_i)(z_j − z_i)
#            − ⅕[(dRx_j − dRx_i)(z_j − z_i − (dZx_j + dZx_i)/12)
#                − (dZx_j − dZx_i)(ρ′_j − ρ′_i − (dRx_j + dRx_i)/12)] )] / dx
#   (z = layer-centre heights). On the structured ROMS grid dRx, dZx at
#   a cell are harmonic means of the differences across its two faces in
#   the direction of the force; on the Voronoi mesh the second face is
#   the edge of the cell OPPOSITE to e (the neighbour whose centre is
#   most anti-parallel to the edge direction — exact for hexagons), its
#   difference rescaled to the spacing d_e of e. A wall edge contributes
#   a zero difference (ROMS masking).
#
# AGENTS.md rule 3: per-cell scalar loops through `Field[i]` getindex.

module OceanPGF

using ..Voronoi: VoronoiSphereMesh
using ..Fields: Field, CellField, EdgeField, mesh_of
using ..Layered: LayeredCellField, nlayers
using ..ShallowWater: williamson_constants
using LinearAlgebra: dot

export PGF_SCHEMES, pressure_gradient

"Selectable pressure-gradient schemes of the ocean dycore (`pgf` keyword)."
const PGF_SCHEMES = (:twoterm, :adcroft2008, :mom6, :roms)

# ∫₀¹ (a_i(1−t) + a_j t)(b_i(1−t) + b_j t) dt for linear a, b
_lin2(ai, aj, bi, bj) = (ai * bi + aj * bj) / 3 + (ai * bj + aj * bi) / 6

"""
    pressure_gradient(scheme, h, z, ρ, ρ0; closed_edges = nothing)
        -> Vector{EdgeField}

Per-layer pressure-gradient acceleration [m/s²] along every edge normal
for `scheme ∈ (:adcroft2008, :mom6, :roms)` (see the module header):
`h` the layer thicknesses (`LayeredCellField`, k = 1 top), `z` the
K + 1 interface heights (`CellField`s, z[1] = η), `ρ` the layer
densities and `ρ0` the Boussinesq reference density. `closed_edges`
(wall edge indices) masks the faces of the `:roms` horizontal stencil.
"""
function pressure_gradient(scheme::Symbol, h::LayeredCellField, z, ρ, ρ0::Real;
                           closed_edges = nothing)
    scheme in (:adcroft2008, :mom6) && return _pgf_fv(h, z, ρ, ρ0, scheme === :mom6)
    scheme === :roms && return _pgf_roms(h, z, ρ, ρ0; closed = closed_edges)
    throw(ArgumentError("pressure_gradient: unknown scheme $scheme " *
                        "(one of $(PGF_SCHEMES[2:end]); :twoterm lives in ocean_tendency!)"))
end

# ---------------------------------------------------- finite volume

# MOM6 PLM (src/ALE/PLM_functions.F90); slopes are the change across the
# cell from top (k − 1 side) to bottom (k + 1 side), k = 1 top
function _plm_slope_wa(hl, hc, hr, ul, uc, ur)
    σr = ur - uc; σl = uc - ul
    σc = 2.0 * (ur - ul) * (hc / (hl + 2.0 * hc + hr))
    umin = min(ul, uc, ur); umax = max(ul, uc, ur)
    s = σl * σr > 0.0 ? sign(σc) * min(abs(σc), 2.0 * min(uc - umin, umax - uc)) : 0.0
    if uc - 0.5 * abs(s) < umin || uc + 0.5 * abs(s) > umax
        s *= (1.0 - eps(s))
    end
    return abs(s) < 1.0e-140 ? 0.0 : s
end
function _plm_monotonized_slope(ul, uc, ur, sl, sc, sr)
    almost_two = 2.0 * (1.0 - eps(sc))
    er = ul + 0.5 * sl; el = ur - 0.5 * sr
    slp = abs(sc)
    edge = uc - 0.5 * sc
    if (edge - er) * (uc - edge) < 0.0
        edge = 0.5 * (edge + er); slp = min(slp, abs(edge - uc) * almost_two)
    end
    edge = uc + 0.5 * sc
    if (edge - uc) * (el - edge) < 0.0
        edge = 0.5 * (edge + el); slp = min(slp, abs(edge - uc) * almost_two)
    end
    return sign(sc) * slp
end
_plm_extrapolate_slope(hl, hc, ul, uc) = 2.0 * (uc - (ul * hc + uc * hl) / (hl + hc))

"Top/bottom edge values (ρ_t, ρ_b) of the MOM6 PLM reconstruction of one column."
function _plm_edges!(ρt, ρb, slp, hcol, ρcol)
    K = length(ρcol)
    if K < 3
        ρt .= ρcol; ρb .= ρcol
        return
    end
    slp[1] = 0.0; slp[K] = 0.0
    for k in 2:(K - 1)
        slp[k] = _plm_slope_wa(hcol[k - 1], hcol[k], hcol[k + 1],
                               ρcol[k - 1], ρcol[k], ρcol[k + 1])
    end
    for k in 2:(K - 1)
        ms = _plm_monotonized_slope(ρcol[k - 1], ρcol[k], ρcol[k + 1],
                                    slp[k - 1], slp[k], slp[k + 1])
        ρt[k] = ρcol[k] - 0.5 * ms; ρb[k] = ρcol[k] + 0.5 * ms
    end
    ms = -_plm_extrapolate_slope(hcol[2], hcol[1], ρcol[2], ρcol[1])
    ρt[1] = ρcol[1] - 0.5 * ms; ρb[1] = ρcol[1] + 0.5 * ms
    ms = _plm_extrapolate_slope(hcol[K - 1], hcol[K], ρcol[K - 1], ρcol[K])
    ρt[K] = ρcol[K] - 0.5 * ms; ρb[K] = ρcol[K] + 0.5 * ms
    return
end

function _pgf_fv(h, z, ρ, ρ0, recon::Bool)
    m = mesh_of(h)
    K = nlayers(h)
    (; g, R) = williamson_constants()
    nc = m.n_cells
    # per column: anomaly pressure at layer tops pa_k/ρ0 [m²/s²], the
    # in-layer column integral intz_k [m³/s²] and the layer buoyancy
    # anomaly b_k = g(ρ_k − ρ0)/ρ0 [m/s²]
    pa = zeros(K, nc); intz = zeros(K, nc); b = zeros(K, nc); hh = zeros(K, nc)
    zb = zeros(K, nc); η = zeros(nc)
    hcol = zeros(K); ρcol = zeros(K); ρt = zeros(K); ρb = zeros(K); slp = zeros(K)
    for i in 1:nc
        η[i] = z[1][i]
        if recon
            for k in 1:K
                hcol[k] = h[k][i]; ρcol[k] = ρ[k][i]
            end
            _plm_edges!(ρt, ρb, slp, hcol, ρcol)
        end
        p = g * η[i]                        # pa_1/ρ0 = g η (ρ_ref = ρ0)
        for k in 1:K
            hk = h[k][i]
            bk = g * (ρ[k][i] - ρ0) / ρ0
            pa[k, i] = p; b[k, i] = bk; hh[k, i] = hk; zb[k, i] = z[k + 1][i]
            if recon
                bt = g * (ρt[k] - ρ0) / ρ0          # reconstructed top value
                bb = g * (ρb[k] - ρ0) / ρ0          # … bottom value
                intz[k, i] = hk^2 * (bt / 2 + (bb - bt) / 6)
            else
                intz[k, i] = 0.5 * bk * hk^2
            end
            p += bk * hk
        end
    end
    out = [EdgeField(m) for _ in 1:K]
    for e in 1:m.n_edges
        i, j = m.edge_cells[1, e], m.edge_cells[2, e]
        dx = R * m.d_e[e]
        intx_pa = g * (η[i] + η[j]) / 2     # along-surface mean of pa/ρ0
        for k in 1:K
            hi, hj = hh[k, i], hh[k, j]
            intx_dpa = _lin2(b[k, i], b[k, j], hi, hj)
            out[k][e] = ((pa[k, i] * hi + intz[k, i]) - (pa[k, j] * hj + intz[k, j]) +
                         (hj - hi) * intx_pa - (zb[k, j] - zb[k, i]) * intx_dpa) *
                        2 / (dx * (hi + hj))
            intx_pa += intx_dpa
        end
    end
    return out
end

# --------------------------------------------- ROMS (SM03, prsgrd32)

# per mesh: for every edge e, the cell "behind" i and "ahead of" j along
# the line i → j, and the edge crossed to reach it
# (cached per mesh object; the mesh is immutable, keyed by its objectid)
const _STENCIL = Dict{UInt, NTuple{4, Vector{Int}}}()

function _opposite(m, c, e)
    # neighbour of cell c across the edge of c most anti-parallel to
    # the direction from c across e
    other(e2) = m.edge_cells[1, e2] == c ? m.edge_cells[2, e2] : m.edge_cells[1, e2]
    xc = m.cell_xyz[:, c]
    d = m.cell_xyz[:, other(e)] .- xc
    best = 0; beste = 0; bestv = Inf
    for e2 in m.cell_edges[c]
        e2 == e && continue
        v = dot(m.cell_xyz[:, other(e2)] .- xc, d)
        v < bestv && (bestv = v; best = other(e2); beste = e2)
    end
    return best, beste
end

function _stencil(m)
    get!(_STENCIL, objectid(m)) do
        ne = m.n_edges
        ib = zeros(Int, ne); eb = zeros(Int, ne); jf = zeros(Int, ne); ef = zeros(Int, ne)
        for e in 1:ne
            i, j = m.edge_cells[1, e], m.edge_cells[2, e]
            ib[e], eb[e] = _opposite(m, i, e)
            jf[e], ef[e] = _opposite(m, j, e)
        end
        (ib, eb, jf, ef)
    end
end

# ROMS harmonic "elementary difference": 2ab/(a+b) if ab > eps, else 0
_harm(a, b) = (c = 2a * b; c > 1.0e-10 ? c / (a + b) : 0.0)

function _pgf_roms(h, z, ρ, ρ0; closed = nothing)
    m = mesh_of(h)
    K = nlayers(h)
    (; g, R) = williamson_constants()
    nc = m.n_cells
    GR = g / ρ0
    # ROMS index r = 1 bottom … K top ↔ layer k = K − r + 1
    zr = zeros(K, nc); rr = zeros(K, nc); P = zeros(K, nc)
    dR = zeros(K + 1); dZ = zeros(K + 1)          # dR[r+1] ↔ ROMS dR(r), r = 0..K
    for i in 1:nc
        for r in 1:K
            k = K - r + 1
            zr[r, i] = 0.5 * (z[k][i] + z[k + 1][i])
            rr[r, i] = ρ[k][i] - ρ0
        end
        zw = z[1][i]                               # ROMS z_w(N) = η
        if K == 1
            P[1, i] = g * zw + GR * rr[1, i] * (zw - zr[1, i])
            continue
        end
        for r in 1:(K - 1)
            dR[r + 1] = rr[r + 1, i] - rr[r, i]
            dZ[r + 1] = zr[r + 1, i] - zr[r, i]
        end
        dR[K + 1] = dR[K]; dZ[K + 1] = dZ[K]
        dR[1] = dR[2]; dZ[1] = dZ[2]
        for r in K:-1:1                            # in place, downward
            dR[r + 1] = _harm(dR[r + 1], dR[r])
            dZ[r + 1] = 2 * dZ[r + 1] * dZ[r] / (dZ[r + 1] + dZ[r])
        end
        cff2 = 0.5 * (rr[K, i] - rr[K - 1, i]) * (zw - zr[K, i]) / (zr[K, i] - zr[K - 1, i])
        P[K, i] = g * zw + GR * (rr[K, i] + cff2) * (zw - zr[K, i])
        for r in (K - 1):-1:1
            P[r, i] = P[r + 1, i] + 0.5 * GR *
                ((rr[r + 1, i] + rr[r, i]) * (zr[r + 1, i] - zr[r, i]) -
                 0.2 * ((dR[r + 2] - dR[r + 1]) *
                        (zr[r + 1, i] - zr[r, i] - (dZ[r + 2] + dZ[r + 1]) / 12) -
                        (dZ[r + 2] - dZ[r + 1]) *
                        (rr[r + 1, i] - rr[r, i] - (dR[r + 2] + dR[r + 1]) / 12)))
        end
    end
    ib, eb, jf, ef = _stencil(m)
    cl = closed === nothing ? Set{Int}() : Set(closed)
    out = [EdgeField(m) for _ in 1:K]
    for e in 1:m.n_edges
        i, j = m.edge_cells[1, e], m.edge_cells[2, e]
        dx = R * m.d_e[e]
        sb = m.d_e[e] / m.d_e[eb[e]]               # rescale to the spacing of e
        sf = m.d_e[e] / m.d_e[ef[e]]
        wb = eb[e] in cl ? 0.0 : 1.0
        wf = ef[e] in cl ? 0.0 : 1.0
        for r in 1:K
            k = K - r + 1
            dz_e = zr[r, j] - zr[r, i];  dr_e = rr[r, j] - rr[r, i]
            dz_b = wb * sb * (zr[r, i] - zr[r, ib[e]]); dr_b = wb * sb * (rr[r, i] - rr[r, ib[e]])
            dz_f = wf * sf * (zr[r, jf[e]] - zr[r, j]); dr_f = wf * sf * (rr[r, jf[e]] - rr[r, j])
            dZx_i = _harm(dz_b, dz_e); dZx_j = _harm(dz_e, dz_f)
            dRx_i = _harm(dr_b, dr_e); dRx_j = _harm(dr_e, dr_f)
            out[k][e] = (P[r, i] - P[r, j] - 0.5 * GR *
                         ((rr[r, j] + rr[r, i]) * dz_e -
                          0.2 * ((dRx_j - dRx_i) * (dz_e - (dZx_j + dZx_i) / 12) -
                                 (dZx_j - dZx_i) * (dr_e - (dRx_j + dRx_i) / 12)))) / dx
        end
    end
    return out
end

end # module OceanPGF
