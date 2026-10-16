# TRiSK discrete operators (Ringler, Thuburn, Klemp & Skamarock 2010,
# cited as R10 with equation numbers).
#
# Sign conventions (established in Mesh/Voronoi.jl, verified there against
# solid-body circulation and T09's defining identities):
#   n_{e,i} = +1 iff i == edge_cells[1,e]   (n̂_e points cell 1 → cell 2,
#             so n̂_e is the outward normal of cell 1)
#   t_{e,v} = +1 iff v == edge_vertex_pos[e] (t̂_e points toward the
#             `edge_vertex_pos` end of the edge)
#
# This module lives in src/Fields/ — it is the ONE place allowed to
# access Field.data directly (AGENTS.md hard rule 3): physics and dycore
# code must go through these operators.

module TRiSK

using LinearAlgebra

using ..Fields: Field, CellField, EdgeField, VertexField,
                CellLoc, EdgeLoc, VertexLoc,
                similar_field, valid_indices
using ..Voronoi: VoronoiSphereMesh

export divergence, divergence!, gradient, gradient!,
       curl, curl!, tangential_flux, tangential_flux!,
       edge_mean, edge_mean!, kinetic_energy, kinetic_energy!,
       pv_flux, pv_flux!, potential_vorticity, potential_vorticity!,
       integral, relative_vorticity, relative_vorticity!

# ------------------------------------------------------------- divergence

"""
    divergence!(out::CellField, F::EdgeField) -> CellField

Discrete divergence of an edge flux field at cell centers (R10 Eq 21):

    (∇·F)_i = (1/A_i) Σ_{e∈EC(i)} n_{e,i} F_e l_e

`F_e > 0` transports quantity in the `n̂_e` direction (from
`edge_cells[1,e]` toward `edge_cells[2,e]`). `out` must be a `CellField`
on the same mesh; its prior contents are overwritten.
"""
function divergence!(out::CellField, F::EdgeField)
    mesh = F.mesh
    fill!(out, 0)
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        flux = mesh.l_e[e] * F.data[e]
        out.data[i] += flux
        out.data[j] -= flux
    end
    out.data ./= mesh.cell_area
    return out
end

"""
    divergence(F::EdgeField) -> CellField

Allocating version of [`divergence!`](@ref).
"""
divergence(F::EdgeField) = divergence!(similar_field(F, eltype(F), CellLoc()), F)

# --------------------------------------------------------------- gradient

"""
    gradient!(out::EdgeField, h::CellField) -> EdgeField

Discrete normal gradient of a cell-centered scalar at edge points
(R10 Eq 22):

    (∇h)_e = (1/d_e) Σ_{i∈CE(e)} (−n_{e,i}) h_i

which for the two flanking cells is simply `(h_j − h_i)/d_e` with the
result oriented along `n̂_e` (cell 1 → cell 2).
"""
function gradient!(out::EdgeField, h::CellField)
    mesh = h.mesh
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        out.data[e] = (h.data[j] - h.data[i]) / mesh.d_e[e]
    end
    return out
end

"""
    gradient(h::CellField) -> EdgeField

Allocating version of [`gradient!`](@ref).
"""
gradient(h::CellField) = gradient!(similar_field(h, eltype(h), EdgeLoc()), h)

# ------------------------------------------------------------------- curl

"""
    curl!(out::VertexField, F::EdgeField) -> VertexField

Discrete curl of an edge field at dual cells / vertices (R10 Eq 23):

    k·(∇×F)_v = (1/A_v) Σ_{e∈EV(v)} t_{e,v} F_e d_e

`t_{e,v}` orients each edge contribution for counterclockwise traversal
of the dual-cell boundary (positive circulation = right-hand rule about
`x̂_v`).
"""
function curl!(out::VertexField, F::EdgeField)
    mesh = F.mesh
    fill!(out, 0)
    for v in 1:mesh.n_vertices
        acc = 0.0
        for e in mesh.vertex_edges[v]
            t = mesh.edge_vertex_pos[e] == v ? 1 : -1
            acc += t * mesh.d_e[e] * F.data[e]
        end
        out.data[v] = acc / mesh.vertex_area[v]
    end
    return out
end

"""
    curl(F::EdgeField) -> VertexField

Allocating version of [`curl!`](@ref).
"""
curl(F::EdgeField) = curl!(VertexField(F.mesh, eltype(F)), F)

"""
    relative_vorticity(u::EdgeField) -> VertexField

Relative vorticity at vertices: the curl of the normal velocity field
(R10 §3; T09 Eq 9). See [`curl`](@ref).
"""
relative_vorticity(u::EdgeField) = curl(u)

# ------------------------------------------------- tangential flux (M)

"""
    tangential_flux!(out::EdgeField, F::EdgeField) -> EdgeField

Tangential flux at edge points from surrounding normal fluxes (R10 Eq 24,
the flux mapping `M`; weights from T09 Eq 33, precomputed on the mesh):

    F⊥_e = Σ_{e'∈ECP(e)} w_{e,e'} F_{e'}

with `w_{e,e'}` already including the `l_{e'}/d_e` factors. `out` may
alias `F` only if `scratch` is provided; otherwise a fresh field is
allocated.
"""
function tangential_flux!(out::EdgeField, F::EdgeField,
                          scratch::Union{Nothing, EdgeField} = nothing)
    mesh = F.mesh
    src = F.data
    if out.data === src
        scratch === nothing &&
            error("tangential_flux!: out aliases F — pass a scratch EdgeField")
        src = scratch.data
        copyto!(src, F.data)
    end
    for e in 1:mesh.n_edges
        acc = 0.0
        wts = mesh.ecp_weights[e]
        eds = mesh.ecp_edges[e]
        for k in eachindex(eds)
            acc += wts[k] * src[eds[k]]
        end
        out.data[e] = acc
    end
    return out
end

"""
    tangential_flux(F::EdgeField) -> EdgeField

Allocating version of [`tangential_flux!`](@ref).
"""
tangential_flux(F::EdgeField) = tangential_flux!(similar_field(F), F)

# -------------------------------------------------------------- edge mean

"""
    edge_mean!(out::EdgeField, h::CellField) -> EdgeField

Cell-centered field averaged to edge points (R10 Eq 52, from the
averaging identity R10 Appendix A.2 Eq A.9):

    ĥ_e = (h_i + h_j) / 2   for the two flanking cells i, j

(MPAS: `layerThicknessEdge = 0.5·(cell1 + cell2)`.)
"""
function edge_mean!(out::EdgeField, h::CellField)
    mesh = h.mesh
    for e in 1:mesh.n_edges
        out.data[e] = (h.data[mesh.edge_cells[1, e]] +
                       h.data[mesh.edge_cells[2, e]]) / 2
    end
    return out
end

"""
    edge_mean(h::CellField) -> EdgeField

Allocating version of [`edge_mean!`](@ref).
"""
edge_mean(h::CellField) = edge_mean!(similar_field(h, eltype(h), EdgeLoc()), h)

"""
    edge_mean!(out::EdgeField, q::VertexField) -> EdgeField

Vertex-centered field averaged to edge points (R10 Eq 50):

    q̃_e = (q_{v1} + q_{v2}) / 2   for the two end vertices of e
"""
function edge_mean!(out::EdgeField, q::VertexField)
    mesh = q.mesh
    for e in 1:mesh.n_edges
        out.data[e] = (q.data[mesh.edge_vertex_pos[e]] +
                       q.data[mesh.edge_vertex_neg[e]]) / 2
    end
    return out
end

"""
    edge_mean(q::VertexField) -> EdgeField

Allocating version of the vertex→edge mean (R10 Eq 50).
"""
edge_mean(q::VertexField) = edge_mean!(similar_field(q, eltype(q), EdgeLoc()), q)

# -------------------------------------------------------- kinetic energy

"""
    kinetic_energy!(out::CellField, u::EdgeField) -> CellField

Cell-centered kinetic energy per unit mass (R10 Eq 63):

    K_i = (1/A_i) Σ_{e∈EC(i)} (A_e/4) u_e²,   A_e = l_e d_e   (R10 Eq 48)

The weights A_e/4 sum to A_i over the edges of i (R10 Eq 64), making K
a convex combination of the surrounding u² — no spurious extrema.
"""
function kinetic_energy!(out::CellField, u::EdgeField)
    mesh = u.mesh
    fill!(out, 0)
    for e in 1:mesh.n_edges
        w = mesh.l_e[e] * mesh.d_e[e] / 4 * u.data[e]^2
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        out.data[i] += w
        out.data[j] += w
    end
    out.data ./= mesh.cell_area
    return out
end

"""
    kinetic_energy(u::EdgeField) -> CellField

Allocating version of [`kinetic_energy!`](@ref).
"""
kinetic_energy(u::EdgeField) = kinetic_energy!(similar_field(u, eltype(u), CellLoc()), u)

# --------------------------------------------------------------- PV flux

"""
    pv_flux!(out::EdgeField, F::EdgeField, q::VertexField,
             scratch::Union{Nothing, EdgeField} = nothing) -> EdgeField

Energy-conserving potential-vorticity flux at edge points (R10 Eqs 49–50):

    Q⊥_e = (1/d_e) Σ_{e'∈ECP(e)} w_{e,e'} l_{e'} F_{e'} (q̃_e + q̃_{e'})/2

with `q̃_e` the arithmetic mean of the vertex PV at the two ends of `e`
(R10 Eq 50). The symmetric `(q̃_e + q̃_{e'})/2` makes the Coriolis term
energetically neutral in the discrete global sum (R10 Eqs 37–47).

The stored `ecp_weights` already include the `l_{e'}/d_e` factors, so
`Q⊥_e = Σ w_{e,e'} F_{e'} (q̃_e + q̃_{e'})/2` where `F` is the *normal*
flux field — Eq 49 is Eq 24 (the flux mapping) with the symmetric PV
mean inserted into the same sum, applied once. `out` may alias `F` only
if `scratch` is provided.
"""
function pv_flux!(out::EdgeField, q::VertexField, F::EdgeField,
                  scratch::Union{Nothing, EdgeField} = nothing)
    mesh = F.mesh
    q̃ = edge_mean(q)                       # q at edge points (R10 Eq 50)
    src = F.data
    if out.data === src
        scratch === nothing &&
            error("pv_flux!: out aliases F — pass a scratch EdgeField")
        copyto!(scratch.data, F.data)
        src = scratch.data
    end
    for e in 1:mesh.n_edges
        acc = 0.0
        wts = mesh.ecp_weights[e]
        eds = mesh.ecp_edges[e]
        for k in eachindex(eds)
            ep = eds[k]
            # w includes l_{e'}/d_e; the (q̃_e+q̃_{e'})/2 symmetric mean:
            acc += wts[k] * src[ep] * (q̃.data[e] + q̃.data[ep]) / 2
        end
        out.data[e] = acc
    end
    return out
end

"""
    pv_flux(q::VertexField, F::EdgeField) -> EdgeField

Allocating version of [`pv_flux!`](@ref).
"""
pv_flux(q::VertexField, F::EdgeField) =
    pv_flux!(similar_field(F), q, F, similar_field(F))

# --------------------------------------------------- potential vorticity

"""
    potential_vorticity!(out::VertexField, u::EdgeField, h::CellField,
                         f::Union{Nothing, VertexField} = nothing) -> VertexField

Absolute potential vorticity at vertices (R10 §2.4):

    q_v = (f_v + ζ_v) / h_v,     ζ = ∇×u   (R10 Eq 23)

with `h_v` the equal-distribution (normalized inverse-corner-count)
cell-thickness interpolation — the weights `R_{i,v}` normalized to sum
to one over the cells surrounding each vertex:

    h_v = Σ_i (h_i / n_i) / Σ_i (1 / n_i)

(`n_i` = corner count of cell `i`; pentagons weight 1/5 vs 1/6 for
hexagons.) `f = nothing` gives relative PV `ζ / h_v`. `out` must be a
`VertexField` on the same mesh.
"""
function potential_vorticity!(out::VertexField, u::EdgeField, h::CellField,
                              f::Union{Nothing, VertexField} = nothing)
    mesh = h.mesh
    ζ = curl(u)
    for v in 1:mesh.n_vertices
        num = 0.0
        den = 0.0
        for c in mesh.vertex_cells[v]
            nck = 1 / length(mesh.cell_corners[c])
            num += nck * h.data[c]
            den += nck
        end
        fv = f === nothing ? 0.0 : f.data[v]
        out.data[v] = (fv + ζ.data[v]) / (num / den)
    end
    return out
end

"""
    potential_vorticity(u::EdgeField, h::CellField[, f]) -> VertexField

Allocating version of [`potential_vorticity!`](@ref).
"""
potential_vorticity(u::EdgeField, h::CellField,
                    f::Union{Nothing, VertexField} = nothing) =
    potential_vorticity!(similar_field(h, eltype(h), VertexLoc()), u, h, f)

"""
    integral(f::Union{CellField, EdgeField}) -> Real

Global area-weighted integral of a field over the unit sphere: for a
`CellField` the weights are the cell areas `A_i`; for an `EdgeField`
the edge areas `A_e = l_e d_e` (R10 Eq 48). On a partitioned payload
only the region's owned slots enter (see
[`Fields.valid_indices`](@ref)) — ghosts are copies and deeper slots
are NaN-poisoned, so the result equals the single-region integral in
both cases. (The mesh geometry lives on the unit sphere; multiply by
`R²` for physical units.)
"""
function integral(f::CellField)
    mesh = f.mesh
    s = zero(eltype(f))
    for i in valid_indices(f)
        s += f.data[i] * mesh.cell_area[i]
    end
    return s
end

function integral(f::EdgeField)
    mesh = f.mesh
    s = zero(eltype(f))
    for e in valid_indices(f)
        s += f.data[e] * mesh.l_e[e] * mesh.d_e[e]
    end
    return s
end

end # module TRiSK
