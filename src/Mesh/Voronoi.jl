"""Spherical Voronoi dual meshes (C-grids) from icosphere triangulations.

Given a Delaunay triangulation of the unit sphere, the Voronoi cell of
each triangulation vertex is the spherical polygon of points closer to
that vertex than to any other. Cell centers = triangulation vertices,
cell corners = triangle circumcenters, cell edges connect the
circumcenters of adjacent triangles (crossing the shared triangulation
edge at right angles).

This module implements the full TRiSK mesh geometry (Ringler et al.
2010, §2, Fig. 1–3; Thuburn et al. 2009 §2):

- primal mesh: Voronoi cells P_i with centers x_i, areas A_i
- dual mesh: triangles D_v with centers at circumcenters v, areas A_v
- per primal edge e: flanking cells CE(e), end vertices VE(e),
  crossing point x_e, lengths l_e (primal edge, circumcenter to
  circumcenter) and d_e (dual edge, cell-center to cell-center),
  unit vectors n_e (cell i → j,
  tangent to sphere) and t_e = k̂ × n_e
- per cell corner (i, v): kite area (P_i ∩ D_v) and corner weight
  R_{i,v} = area(P_i ∩ D_v) / A_i  (T09 Eq 12–13; Σ_v R_{i,v} = 1)
- per ordered edge pair: tangential-velocity reconstruction weights
  w_{e,e'} (T09 Eq 33, with a = 1/2):
  w_{e,e'} = n_{e',i} · t_{e,v₂} · (Σ_{v between e' and e} R_{i,v} − 1/2)
  where cell i contains both edges, and v₂ is the last vertex between
  them traversing CCW around cell i.

Sign conventions (T09 Eq 9, Ringler Eq 21–23):
- n_{e,i} = +1 if n_e is an outward normal of cell i, else −1
- t_{e,v} = +1 if vertex v is at the left end of edge e looking along
  n_e (i.e. v = edge_vertex_pos[e]), else −1
"""
module Voronoi

export VoronoiSphereMesh, generate_icosahedral_mesh,
       mesh_summary, adjacency_consistent, kite_uniformity

using LinearAlgebra: norm, normalize, cross, dot
using ..Icosahedron: icosahedron, subdivide

"""A spherical Voronoi (cell) mesh with its Delaunay (triangle) dual —
the C-grid for TRiSK discretizations.

TRiSK staggering (Ringler et al. 2010):
- mass and tracers live at **cells** (Voronoi cell centers)
- normal velocities live on **edges** (between adjacent cells, at x_e)
- vorticity lives at **vertices** (triangle circumcenters)

Fields (public, read-only by convention) — see the module docstring
for the geometry and sign conventions:

Geometry: `cell_xyz`, `cell_area`, `vertex_xyz`, `vertex_area`,
`edge_cells`, `edge_vertex_pos`, `edge_vertex_neg`, `edge_x`,
`edge_normal`, `edge_tangent`, `l_e`, `d_e`

Connectivity (all CCW seen from outside): `cell_edges` (edges around
each cell), `cell_corners` (vertices around each cell),
`vertex_cells`, `vertex_edges`

TRiSK tables: `corner_area` (kite areas, parallel to `cell_corners`),
`corner_weight` (R_{i,v}, parallel to `cell_corners`),
`ecp_edges` / `ecp_weights` (parallel arrays per edge: e' ∈ ECP(e) and
w_{e,e'}), `pentagon_cells`

Diagnostics: `cell_quality` (mean kite-area uniformity per cell)
"""
struct VoronoiSphereMesh{FT}
    n_cells::Int
    n_edges::Int
    n_vertices::Int
    cell_xyz::Matrix{FT}
    cell_area::Vector{FT}
    vertex_xyz::Matrix{FT}
    vertex_area::Vector{FT}
    edge_cells::Matrix{Int}          # 2×Ne: (i, j); n_e points i → j
    edge_vertex_pos::Vector{Int}    # v at the left end of e looking along n_e
    edge_vertex_neg::Vector{Int}
    edge_x::Matrix{FT}               # 3×Ne: crossing point x_e (unit vector)
    edge_normal::Matrix{FT}         # 3×Ne: n_e
    edge_tangent::Matrix{FT}        # 3×Ne: t_e = k̂ × n_e
    l_e::Vector{FT}
    d_e::Vector{FT}
    cell_edges::Vector{Vector{Int}}       # CCW around each cell
    cell_corners::Vector{Vector{Int}}      # CCW around each cell
    vertex_cells::Vector{Vector{Int}}     # CCW around each vertex
    vertex_edges::Vector{Vector{Int}}      # CCW around each vertex
    corner_area::Vector{Vector{FT}}       # kite areas, parallel to cell_corners
    corner_weight::Vector{Vector{FT}}     # R_{i,v}, parallel to cell_corners
    ecp_edges::Vector{Vector{Int}}        # e' ∈ ECP(e) per edge
    ecp_weights::Vector{Vector{FT}}       # w_{e,e'}, parallel to ecp_edges
    pentagon_cells::Vector{Int}
    cell_quality::Vector{FT}
end

include("Quality.jl")

# ---------------------------------------------------------------- geometry

central_angle(a, b) = atan(norm(cross(a, b)), dot(a, b))

"""    spherical_circumcenter(a, b, c) -> unit vector

Exact spherical circumcenter of the spherical triangle with corners
`a`, `b`, `c` (unit vectors), for triangles within a hemisphere.

A point p is equidistant (angular) from a and b iff p·a = p·b, i.e.
p ⊥ (a−b). The circumcenter lies along the intersection line of the
planes with normals (a−b) and (b−c), both through the origin:
p ∝ (a−b) × (b−c).
"""
function spherical_circumcenter(a, b, c)
    p = cross(a .- b, b .- c)
    np = norm(p)
    np > 1e-12 || error("degenerate triangle in circumcenter computation")
    p = p ./ np
    dot(p, a .+ b .+ c) < 0 && (p = .-p)
    return p
end

"""    spherical_polygon_area(corners) -> FT

Signed area of the spherical polygon given by corner unit vectors, via
Girard's theorem in turning-angle form: A = 2π − Σ τᵢ, where τᵢ is the
signed turning (deflection) angle at corner i, positive for left turns.

CCW order (seen from outside the sphere) gives the polygon's own
(positive) area; clockwise order gives its negative.
"""
function spherical_polygon_area(corners::Vector{Vector{FT}}) where {FT}
    n = length(corners)
    n >= 3 || error("polygon needs ≥ 3 corners")
    total = zero(FT)
    for i in 1:n
        a = corners[mod1(i - 1, n)]
        b = corners[i]
        c = corners[mod1(i + 1, n)]
        # arrival tangent at b on arc a→b: p(s) = cos s a + sin s u,
        # u = normalize(b − (a·b)a) ⇒ p'(s_b) ∝ (a·b) b − a
        t_in = normalize(dot(a, b) .* b .- a)
        # departure tangent at b on arc b→c
        t_out = normalize(c .- dot(c, b) .* b)
        total += atan(dot(cross(t_in, t_out), b), dot(t_in, t_out))
    end
    A = 2 * π - total
    # turning-angle form returns the area of the region to the LEFT of
    # the traversal; CW input yields the complement. Normalize winding:
    total < 0 && (A -= 4 * π)
    return A
end

"""    spherical_triangle_centroid_integral(a, b, c) -> (vector, area)

Exact ∫∫ x dA over the spherical triangle with unit-vector corners:
the integral is A · (a+b+c)/|a+b+c| (the centroid of a spherical
triangle lies along a+b+c).
"""
function spherical_triangle_centroid_integral(a, b, c)
    A = spherical_polygon_area([a, b, c])
    s = a .+ b .+ c
    ns = norm(s)
    ns > 1e-12 || error("degenerate triangle in centroid integral")
    return (A ./ ns) .* s, A
end

"""    arc_intersection(p1, p2, q1, q2) -> unit vector

Intersection point of the great-circle arc p1→p2 with the great-circle
arc q1→q2 (both minor arcs). Returns the point lying on both segments.
"""
function arc_intersection(p1, p2, q1, q2)
    n1 = cross(p1, p2)
    n2 = cross(q1, q2)
    c = cross(n1, n2)
    nc = norm(c)
    nc > 1e-12 || error("arcs are parallel — no unique intersection")
    c = c ./ nc
    dot(c, p1 .+ p2) < 0 && (c = .-c)
    dot(c, q1 .+ q2) < 0 && (c = .-c)
    # c is on both great circles; verify it is within both minor arcs
    on_segment(c, a, b) = dot(c, a) > dot(c, .-a) && dot(c, b) > dot(c, .-b) &&
                          dot(cross(cross(a, c), cross(c, b)), c) >= -1e-10
    on_segment(c, p1, p2) || on_segment(.-c, p1, p2) || error("arc intersection outside segment p")
    on_segment(c, q1, q2) || on_segment(.-c, q1, q2) || error("arc intersection outside segment q")
    return c
end

# ------------------------------------------------------------ construction

"""    generate_icosahedral_mesh(FT=Float64; n_subdivisions=0,
                                lloyd_iterations=0) -> VoronoiSphereMesh

Build the spherical Voronoi C-grid (TRiSK geometry) from the icosphere
triangulation with `n_subdivisions` subdivision steps, then optionally
relax toward a Spherical Centroidal Voronoi Tessellation with
`lloyd_iterations` Lloyd steps (each step moves every cell center to
the exact centroid of its Voronoi cell, keeping the connectivity
fixed, then rebuilds the geometry; cf. Ringler et al. 2010 §2, Ju et
al. 2002).

Cell counts: 0 → 12 cells, 1 → 42, 2 → 162, 3 → 642, ...
"""
function generate_icosahedral_mesh(::Type{FT} = Float64;
                                   n_subdivisions::Int = 0,
                                   lloyd_iterations::Int = 0) where {FT}
    n_subdivisions >= 0 || throw(ArgumentError("n_subdivisions must be ≥ 0"))
    lloyd_iterations >= 0 || throw(ArgumentError("lloyd_iterations must be ≥ 0"))

    tri = icosahedron(FT)
    for _ in 1:n_subdivisions
        tri = subdivide(tri)
    end

    mesh = build_from_triangulation(tri)
    for _ in 1:lloyd_iterations
        new_centers = similar(mesh.cell_xyz)
        for i in 1:mesh.n_cells
            new_centers[:, i] = cell_centroid(mesh, i)
        end
        # rebuild the triangulation with moved centers, connectivity fixed
        tri2 = rebuild_triangulation(tri, new_centers)
        mesh = build_from_triangulation(tri2)
    end
    return mesh
end

function rebuild_triangulation(tri, new_centers)
    # connectivity unchanged; only vertex coordinates move (still unit
    # vectors by construction of the centroid)
    return typeof(tri)(new_centers, tri.edge_vertices, tri.face_vertices, tri.edge_midpoint_cache)
end

"""Exact centroid of Voronoi cell i (unit vector), computed by
splitting the cell into spherical triangles (x_i, v_k, v_{k+1}) over
its CCW corners and using the exact spherical-triangle centroid
integral."""
function cell_centroid(mesh, i)
    corners = mesh.cell_corners[i]
    acc = zeros(eltype(mesh.cell_xyz), 3)
    for k in 1:length(corners)
        a = mesh.vertex_xyz[:, corners[k]]
        b = mesh.vertex_xyz[:, corners[mod1(k + 1, end)]]
        m, _ = spherical_triangle_centroid_integral(mesh.cell_xyz[:, i], a, b)
        acc .+= m
    end
    return normalize(acc)
end

function build_from_triangulation(tri)
    FT = eltype(tri.xyz)
    xyz = tri.xyz                       # cell centers (triangulation vertices)
    tri_edges = tri.edge_vertices
    tri_faces = tri.face_vertices
    nv_tri = size(tri_faces, 2)         # triangles = Voronoi corners (vertices)
    nc = size(xyz, 2)
    ne = size(tri_edges, 2)

    # circumcenters of all triangles
    circum = [spherical_circumcenter(xyz[:, tri_faces[1, f]],
                                     xyz[:, tri_faces[2, f]],
                                     xyz[:, tri_faces[3, f]]) for f in 1:nv_tri]

    # triangle → global triangulation-edge lookup, CCW pairs (i,j),(j,k),(k,i)
    edge_index = Dict{Tuple{Int,Int},Int}()
    for e in 1:ne
        a, b = tri_edges[1, e], tri_edges[2, e]
        edge_index[a < b ? (a, b) : (b, a)] = e
    end
    face_edges(f) = begin
        i, j, k = tri_faces[1, f], tri_faces[2, f], tri_faces[3, f]
        (edge_index[i < j ? (i, j) : (j, i)],
         edge_index[j < k ? (j, k) : (k, j)],
         edge_index[k < i ? (k, i) : (i, k)])
    end

    # faces adjacent to each triangulation edge (exactly 2 on a sphere)
    faces_of_edge = [Int[] for _ in 1:ne]
    for f in 1:nv_tri
        for e in face_edges(f)
            push!(faces_of_edge[e], f)
        end
    end
    all(fs -> length(fs) == 2, faces_of_edge) ||
        error("triangulation edge not shared by exactly 2 faces")

    # --- per primal edge -------------------------------------------------
    edge_cells = zeros(Int, 2, ne)
    edge_normal = zeros(FT, 3, ne)
    edge_tangent = zeros(FT, 3, ne)
    l_e = zeros(FT, ne)
    d_e = zeros(FT, ne)
    edge_vertex_pos = zeros(Int, ne)
    edge_vertex_neg = zeros(Int, ne)
    edge_x = zeros(FT, 3, ne)

    for e in 1:ne
        i, j = tri_edges[1, e], tri_edges[2, e]
        edge_cells[1, e] = i
        edge_cells[2, e] = j
        # crossing point first: primal arc (x_i, x_j) ∩ dual arc (circumcenters)
        f1, f2 = faces_of_edge[e]
        xe = arc_intersection(xyz[:, i], xyz[:, j], circum[f1], circum[f2])
        edge_x[:, e] = xe
        # n_e: tangent to the sphere AT x_e, pointing from cell i to cell j
        v = xyz[:, j] .- xyz[:, i]
        n = normalize(v .- dot(v, xe) .* xe)
        edge_normal[:, e] = n
        # t_e = k̂ × n_e with k̂ the local vertical at x_e (Ringler Eq. §3,
        # T09 §2): right-handed frame (n_e, t_e, x̂_e) at the edge
        edge_tangent[:, e] = cross(xe, n)
        l_e[e] = central_angle(circum[f1], circum[f2])
        d_e[e] = central_angle(xyz[:, i], xyz[:, j])
        # pos/neg end vertices: t_e points from neg to pos
        if dot(circum[f1] .- circum[f2], edge_tangent[:, e]) > 0
            edge_vertex_pos[e], edge_vertex_neg[e] = f1, f2
        else
            edge_vertex_pos[e], edge_vertex_neg[e] = f2, f1
        end
    end

    # --- cells around each vertex, ordered CCW ---------------------------
    # (walk the triangle: faces sharing vertex v, CCW seen from outside)
    vertex_cells = [Int[] for _ in 1:nv_tri]
    for f in 1:nv_tri
        for c in tri_faces[:, f]
            push!(vertex_cells[f], c)
        end
    end
    for v in 1:nv_tri
        vertex_cells[v] = order_cells_ccw(vertex_cells[v], xyz, circum[v])
    end

    # --- cell corners and edges, ordered CCW ----------------------------
    # corners of cell c: circumcenters of the triangles containing c.
    # CCW order: sort by signed angle around the cell center direction.
    corners_of_cell = [Int[] for _ in 1:nc]
    for f in 1:nv_tri
        for c in tri_faces[:, f]
            push!(corners_of_cell[c], f)
        end
    end
    cell_corners = Vector{Vector{Int}}(undef, nc)
    for c in 1:nc
        fs = corners_of_cell[c]
        ref = circum[fs[1]]
        angs = map(f -> signed_angle(ref, circum[f], xyz[:, c]), fs)
        cell_corners[c] = fs[sortperm(angs)]
    end

    # edges of cell c in CCW order: edge between consecutive corners
    cell_edges = [Int[] for _ in 1:nc]
    for c in 1:nc
        cs = cell_corners[c]
        n = length(cs)
        es = Int[]
        for k in 1:n
            v1, v2 = cs[k], cs[mod1(k + 1, n)]
            # find the triangulation edge of cell c whose two adjacent
            # triangles are v1 and v2
            e = edge_between(c, v1, v2, faces_of_edge, edge_cells)
            push!(es, e)
        end
        cell_edges[c] = es
    end

    # --- cell areas (Girard over corners) --------------------------------
    cell_area = zeros(FT, nc)
    for c in 1:nc
        cs = cell_corners[c]
        cell_area[c] = spherical_polygon_area([circum[f] for f in cs])
        cell_area[c] > 0 || error("cell $c has non-positive area — mesh not a valid Voronoi diagram (non-Delaunay triangulation?)")
    end

    # --- kites and corner weights R_{i,v} ---------------------------------
    # kite at corner v of cell i: (v, x_{e_next}, x_i, x_{e_prev}) CCW,
    # where e_prev/e_next are the cell edges before/after corner v.
    corner_area = [Vector{FT}(undef, length(cell_corners[c])) for c in 1:nc]
    corner_weight = [Vector{FT}(undef, length(cell_corners[c])) for c in 1:nc]
    for c in 1:nc
        cs = cell_corners[c]
        es = cell_edges[c]
        n = length(cs)
        for k in 1:n
            v = cs[k]
            e_prev = es[mod1(k - 1, n)]   # edge INTO corner v (CCW)
            e_next = es[k]                # edge OUT of corner v (CCW)
            kite = [circum[v], edge_x[:, e_next], xyz[:, c], edge_x[:, e_prev]]
            a = spherical_polygon_area(kite)
            a > 0 || error("negative kite at cell $c corner $v — non-Delaunay triangulation")
            corner_area[c][k] = a
            corner_weight[c][k] = a / cell_area[c]
        end
    end

    # --- dual (vertex) areas: Girard over the adjacent cell centers -------
    vertex_area = zeros(FT, nv_tri)
    vertex_xyz = hcat(circum...)
    for v in 1:nv_tri
        cs = vertex_cells[v]
        vertex_area[v] = spherical_polygon_area([xyz[:, c] for c in cs])
        vertex_area[v] > 0 || error("non-positive dual cell area at vertex $v")
    end

    # --- edges around each vertex, CCW --------------------------------------
    vertex_edges = [Int[] for _ in 1:nv_tri]
    for v in 1:nv_tri
        es = [e for e in 1:ne if v in (edge_vertex_pos[e], edge_vertex_neg[e])]
        # order by angle around the vertex direction
        ref = edge_x[:, es[1]]
        angs = map(e -> signed_angle(ref, edge_x[:, e], circum[v]), es)
        vertex_edges[v] = es[sortperm(angs)]
    end

    # --- tangential reconstruction weights w_{e,e'} (T09 Eq 33) ------------
    # For each e and each cell i flanking e, and each e' ∈ EC(i), e' ≠ e:
    # walk CCW around i from e' to e, summing the corner weights R_{i,v}
    # of every corner crossed (v₁ = first corner after e', v₂ = arrival
    # corner at e). Then
    #   w_{e,e'} = n_i^{e'} · t^e_{v₂} · (Σ R_{i,v} − 1/2) · l_{e'} / d_e
    # with n_i^{e'} = +1 iff i == cellsOnEdge(e',1) (n̂_{e'} outward from i)
    # and t^e_{v₂} = +1 iff v₂ == edge_vertex_pos[e] (the +t̂_e end of e).
    # Contributions from both flanking cells add. The reconstruction is
    #   u⊥_e = Σ_{e'} w_{e,e'} u_{e'}
    # with u_{e'} positive along n̂_{e'}. Verified properties (tests):
    # T09 Eq 17/20 (local identities, round-off), Eq 14 (dual-divergence
    # interpolation identity, round-off), Eq 39 (bare antisymmetry).
    ecp_edges = [Int[] for _ in 1:ne]
    ecp_weights = [Vector{FT}(undef, 0) for _ in 1:ne]
    for e in 1:ne
        i, j = edge_cells[1, e], edge_cells[2, e]
        i == j && error("edge $e has both sides in cell $i")
        acc = Dict{Int,FT}()
        for c in (i, j)
            ces = cell_edges[c]
            n_c = length(ces)
            p_e = findfirst(==(e), ces)
            p_e === nothing && error("edge $e not found in cell $c")
            for eprime in ces
                eprime == e && continue
                p_ep = findfirst(==(eprime), ces)
                # CCW walk from e' to e: corner positions p_ep+1, ..., p_e
                S = zero(FT)
                q = p_ep
                while true
                    q = mod1(q + 1, n_c)
                    S += corner_weight[c][q]
                    q == p_e && break
                end
                v2 = cell_corners[c][p_e]
                n_out = edge_cells[1, eprime] == c ? 1 : -1
                # t̂_e points toward the `edge_vertex_pos` end of e (the
                # convention verified against solid-body circulation in
                # the curl/vorticity operator). The CCW walk around cell c
                # arrives at the corner where e ENTERS the walk — but the
                # walk starts at e' and moves CCW, so the arrival corner
                # is the CCW-EXIT corner of e, which is the +t̂ end only
                # when t̂ aligns with the CCW traversal of this cell.
                # Rather than re-deriving per cell, evaluate both ends of
                # e geometrically: t^e_{v2} = +1 iff v2 is the
                # edge_vertex_pos end.
                t_e_v2 = edge_vertex_pos[e] == v2 ? 1 : -1
                acc[eprime] = get(acc, eprime, zero(FT)) +
                              n_out * t_e_v2 * (S - FT(1 / 2)) * l_e[eprime] / d_e[e]
            end
        end
        # deterministic order
        ecp_edges[e] = sort!(collect(keys(acc)))
        ecp_weights[e] = [acc[k] for k in ecp_edges[e]]
    end

    # --- pentagons and quality ----------------------------------------------
    pentagon_cells = Int[c for c in 1:nc if length(cell_corners[c]) == 5]
    cell_quality = [minimum(corner_area[c]) / (cell_area[c] / length(corner_area[c]))
                    for c in 1:nc]

    return VoronoiSphereMesh{FT}(
        nc, ne, nv_tri, xyz, cell_area, vertex_xyz, vertex_area,
        edge_cells, edge_vertex_pos, edge_vertex_neg, edge_x,
        edge_normal, edge_tangent, l_e, d_e,
        cell_edges, cell_corners, vertex_cells, vertex_edges,
        corner_area, corner_weight, ecp_edges, ecp_weights,
        pentagon_cells, cell_quality,
    )
end

# --- helpers -----------------------------------------------------------

"""Signed angle from a to b around axis n (right-handed), in (−π, π]."""
function signed_angle(a, b, n)
    c = cross(a, b)
    s = dot(c, n) >= 0 ? 1.0 : -1.0
    return s * atan(norm(c), dot(a, b))
end

"""Order cell indices around vertex direction `vdir` CCW (seen from
outside the sphere): sort by signed angle of the cell-center vectors
around vdir."""
function order_cells_ccw(cells, xyz, vdir)
    ref = xyz[:, cells[1]]
    angs = map(c -> signed_angle(ref, xyz[:, c], vdir), cells)
    return cells[sortperm(angs)]
end

"""The triangulation edge of cell c whose two adjacent triangles are
exactly v1 and v2."""
function edge_between(c, v1, v2, faces_of_edge, edge_cells)
    for e in 1:length(faces_of_edge)
        fs = faces_of_edge[e]
        (v1 in fs && v2 in fs) || continue
        (edge_cells[1, e] == c || edge_cells[2, e] == c) && return e
    end
    error("no edge of cell $c between vertices $v1 and $v2")
end

end # module Voronoi
