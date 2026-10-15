"""Generation of base icosahedra and their recursive subdivision.

The icosahedron is the root of every icosahedral (Voronoi/Delaunay) mesh:
its 12 vertices seed the Voronoi cell centers (including the 12 pentagon
cells that persist at every subdivision level), and recursive subdivision
of its 20 triangular faces generates the Delaunay triangulations whose
duals are the target meshes (Ringler et al. 2010, §2).
"""
module Icosahedron

export IcosphereMesh, subdivide, nvertices, nedges, nfaces, vertices, edges, faces

using LinearAlgebra: norm, normalize, cross, dot
using Base: @propagate_inbounds

"""A triangulated unit sphere: vertex coordinates plus edge and face
connectivity tables.

Faces are oriented consistently (counterclockwise seen from outside the
sphere), which the Voronoi dual construction relies on for sign
conventions.

Fields (all public, read-only by convention):
- `xyz::Matrix{FT}` — 3×N vertex coordinates, unit vectors
- `edge_vertices::Matrix{Int}` — 2×E edge endpoint indices (into xyz columns)
- `face_vertices::Matrix{Int}` — 3×F face vertex indices, CCW oriented
"""
struct IcosphereMesh{FT}
    xyz::Matrix{FT}
    edge_vertices::Matrix{Int}
    face_vertices::Matrix{Int}
    edge_midpoint_cache::Vector{Tuple{Int,Int}}   # per edge: (parent a, parent b)
end

vertices(m::IcosphereMesh) = m.xyz
edges(m::IcosphereMesh) = m.edge_vertices
faces(m::IcosphereMesh) = m.face_vertices
nvertices(m::IcosphereMesh) = size(m.xyz, 2)
nedges(m::IcosphereMesh) = size(m.edge_vertices, 2)
nfaces(m::IcosphereMesh) = size(m.face_vertices, 2)

"""    icosahedron(FT=Float64) -> IcosphereMesh{FT}

The unit icosahedron: 12 vertices, 30 edges, 20 faces.

Vertex coordinates from the standard construction: two poles at
`(0,0,±1)` and two rings of five vertices at latitude `±atan(1/2)`,
chosen so every edge has equal length
(Ronchi, Iacono, Paolucci 1996, §2; any rigid rotation of the result is
equally valid).
"""
function icosahedron(::Type{FT} = Float64) where {FT}
    φ = convert(FT, MathConstants.golden)   # (1+√5)/2
    # 12 vertices: cyclic permutations of (±1, ±φ, 0), normalized to the
    # unit sphere (standard golden construction; every edge has equal length)
    raw = [
        (-1,  φ, 0), (1,  φ, 0), (-1, -φ, 0), (1, -φ, 0),
        (0, -1,  φ), (0,  1,  φ), (0, -1, -φ), (0,  1, -φ),
        ( φ, 0, -1), (φ, 0,  1), (-φ, 0, -1), (-φ, 0,  1),
    ]
    xyz = reduce(hcat, [normalize(FT.(collect(p))) for p in raw])

    # Edges of the icosahedron: all vertex pairs at the icosahedral edge
    # distance. Edge length of the unit-sphere icosahedron:
    ℓ = 2 / sqrt(φ^2 + 1)
    ev = Tuple{Int,Int}[]
    for i in 1:12, j in i+1:12
        d = norm(xyz[:, i] .- xyz[:, j])
        if abs(d - ℓ) < 100 * eps(FT)
            push!(ev, (i, j))
        end
    end
    length(ev) == 30 || error("icosahedron construction: found $(length(ev)) edges, expected 30")

    # Faces: all vertex triples that are pairwise adjacent. Orientation is
    # fixed by requiring outward normal (right-hand rule) — enforce below.
    adjacent(i, j) = any(e -> (e == (i, j) || e == (j, i)), ev)
    fv = Tuple{Int,Int,Int}[]
    for i in 1:12, j in i+1:12, k in j+1:12
        (adjacent(i, j) && adjacent(j, k) && adjacent(i, k)) || continue
        # orient counterclockwise seen from outside:
        n = cross(xyz[:, j] .- xyz[:, i], xyz[:, k] .- xyz[:, i])
        if dot(n, xyz[:, i]) > 0
            push!(fv, (i, j, k))
        else
            push!(fv, (i, k, j))
        end
    end
    length(fv) == 20 || error("icosahedron construction: found $(length(fv)) faces, expected 20")

    edge_vertices = hcat([collect(e) for e in ev]...)
    face_vertices = hcat([collect(f) for f in fv]...)
    cache = [(0, 0) for _ in ev]
    return IcosphereMesh{FT}(xyz, edge_vertices, face_vertices, cache)
end

"""    subdivide(m::IcosphereMesh) -> IcosphereMesh

One 4:1 subdivision step: every triangle is split into four by edge
midpoints (projected to the unit sphere). Vertex count goes
V → V + E; faces F → 4F; edges E → 2E + 3F.

Edge midpoints are shared by construction (a dictionary keys them by the
sorted endpoint pair), so the result is a valid triangulation of the
sphere with no duplicate vertices (Loop-style 1-to-4 scheme, as used for
geodesic grids; cf. Ringler et al. 2010, Fig. 1).
"""
function subdivide(m::IcosphereMesh{FT}) where {FT}
    old_v = m.xyz
    old_e = m.edge_vertices
    old_f = m.face_vertices
    nv_old = size(old_v, 2)
    ne_old = size(old_e, 2)

    # midpoint index lookup: sorted vertex pair -> new vertex index
    mid = Dict{Tuple{Int,Int},Int}()
    new_xyz = [old_v[:, i] for i in 1:nv_old]   # Vector{Vector{FT}}
    for e in 1:ne_old
        a, b = old_e[1, e], old_e[2, e]
        key = a < b ? (a, b) : (b, a)
        p = normalize((old_v[:, a] .+ old_v[:, b]) ./ 2)
        push!(new_xyz, p)
        mid[key] = length(new_xyz)
    end

    nv = length(new_xyz)
    xyz = hcat(new_xyz...)

    # New edges: for each old face (i,j,k) with midpoints (p,q,r):
    #   corners: (i,p), (p,j), (j,q), (q,k), (k,r), (r,i)
    #   middle:  (p,q), (q,r), (r,p)
    # Corner edges are shared by two adjacent faces — deduplicate with a
    # sorted-pair set so every edge appears exactly once.
    edge_set = Set{Tuple{Int,Int}}()
    new_ev = Tuple{Int,Int}[]
    new_fv = Tuple{Int,Int,Int}[]
    for f in 1:size(old_f, 2)
        i, j, k = old_f[1, f], old_f[2, f], old_f[3, f]
        p = mid[i < j ? (i, j) : (j, i)]
        q = mid[j < k ? (j, k) : (k, j)]
        r = mid[k < i ? (k, i) : (i, k)]
        for (a, b) in ((i, p), (p, j), (j, q), (q, k), (k, r), (r, i), (p, q), (q, r), (r, p))
            key = a < b ? (a, b) : (b, a)
            if !(key in edge_set)
                push!(edge_set, key)
                push!(new_ev, (a, b))
            end
        end
        # orientation inherited from the parent face
        push!(new_fv, (i, p, r))
        push!(new_fv, (p, j, q))
        push!(new_fv, (r, q, k))
        push!(new_fv, (p, q, r))
    end

    edge_vertices = hcat([collect(e) for e in new_ev]...)
    face_vertices = hcat([collect(f) for f in new_fv]...)
    cache = [(0, 0) for _ in new_ev]
    return IcosphereMesh{FT}(xyz, edge_vertices, face_vertices, cache)
end

end # module Icosahedron
