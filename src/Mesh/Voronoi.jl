"""Spherical Voronoi dual meshes (C-grids) from icosphere triangulations.

Given a Delaunay triangulation of the unit sphere, the Voronoi cell of
each triangulation vertex is the spherical polygon of points closer to
that vertex than to any other. Cell centers = triangulation vertices,
cell corners = triangle circumcenters, cell edges connect the
circumcenters of adjacent triangles (crossing the shared triangulation
edge at right angles).

On a unit sphere every triangle circumcenter is the normalized sum of
the vertex vectors (spherical circumcenter; on the unit sphere the
barycentric-like construction with equal vertex weights is exact for the
spherical case only if the triangle is small — for geodesic icosahedral
meshes the exact spherical circumcenter formula is required. We use the
exact spherical formula via the planar normal-projection method, valid
for triangles spanning at most a hemisphere.)

This module implements the construction and the connectivity tables the
TRiSK operators need (Ringler et al. 2010, §2 and Fig. 2).
"""
module Voronoi

export VoronoiSphereMesh, generate_icosahedral_mesh

using LinearAlgebra: norm, normalize, cross, dot
using ..Icosahedron: icosahedron, subdivide

"""A spherical Voronoi (cell) mesh with its Delaunay (triangle) dual —
the C-grid for TRiSK discretizations.

TRiSK staggering (Ringler et al. 2010):
- mass and tracers live at **cells** (Voronoi cell centers = icosphere vertices)
- normal velocities live on **edges** (primal mesh edges, between adjacent cells)
- vorticity lives at **vertices** (triangle circumcenters = Voronoi cell corners)

Fields (public, read-only by convention):
- `cell_xyz` — 3×Nc unit vectors, cell centers
- `cell_area` — Nc cell areas (sum = 4π)
- `n_cells`, `n_edges`, `n_vertices` — counts
- `edge_cells` — 2×Ne indices of the cells flanking each edge
- `edge_vertex_pos` / `edge_vertex_neg` — Ne indices of the vertices (circumcenters) at the edge ends, pos = left of `edge_normal` (going from cell 1 to cell 2)
- `edge_normal` — 3×Ne unit vectors, tangent to the sphere, pointing from `edge_cells[1,e]` to `edge_cells[2,e]`
- `edge_tangent` — 3×Ne unit vectors, `edge_tangent = edge_normal × edge_normal_from_center`, i.e. 90° CCW rotation of the normal in the tangent plane
- `l_e` — Ne primal edge lengths (great-circle distance between flanking cell centers)
- `d_e` — Ne dual edge lengths (great-circle distance between the edge's two end vertices)
- `vertex_cells` — Vector{Vector{Int}}: cells around each vertex, ordered CCW (seen from outside)
- `vertex_area` — Nv vertex (dual-cell) areas, K_v weights for vorticity (sum = 4π)
- `edge_sign_on_vertex` — Vector{Vector{Int}}: sign of each vertex-adjacent edge's circulation contribution (+1 if edge normal points CCW around the vertex, −1 otherwise)
- `pentagon_cells` — indices of the 12 cells with 5 edges (icosahedron vertices)
"""
struct VoronoiSphereMesh{FT}
    n_cells::Int
    n_edges::Int
    n_vertices::Int
    cell_xyz::Matrix{FT}
    cell_area::Vector{FT}
    edge_cells::Matrix{Int}
    edge_vertex_pos::Vector{Int}
    edge_vertex_neg::Vector{Int}
    edge_normal::Matrix{FT}
    edge_tangent::Matrix{FT}
    l_e::Vector{FT}
    d_e::Vector{FT}
    vertex_cells::Vector{Vector{Int}}
    vertex_area::Vector{FT}
    edge_sign_on_vertex::Vector{Vector{Int}}
    pentagon_cells::Vector{Int}
end

"""    spherical_circumcenter(a, b, c) -> unit vector

Exact spherical circumcenter of the spherical triangle with corners
`a`, `b`, `c` (unit vectors), for triangles within a hemisphere.

Method: intersect the three planes through the origin with normals
`a`, `b`, `c`... — no. The circumcenter is equidistant from a, b, c, so
it lies on the intersection of the perpendicular bisector planes of ab
and bc, i.e. on the line through the origin crossed with (b−a) and
(c−b): p ∝ (b−a)×(c−b)... — the standard robust form: p ∝
(a+b+c)-normalized is only valid for small triangles; we use the exact
plane-intersection: p ∝ n1 × n2 where n1 = normalize(a+b) × (b−a) (normal
of plane 1 through origin and midpoint direction of ab), n2 likewise
for bc, choosing the solution direction with positive dot to the
triangle's inward side.
"""
function spherical_circumcenter(a::AbstractVector{FT}, b::AbstractVector{FT}, c::AbstractVector{FT}) where {FT}
    # A point p is equidistant (angular) from a and b iff p·a = p·b, i.e.
    # p ⊥ (a − b). The circumcenter therefore lies along the intersection
    # line of the planes with normals (a−b) and (b−c), both through the
    # origin: p ∝ (a−b) × (b−c).
    p = cross(a .- b, b .- c)
    np = norm(p)
    np > 100 * eps(FT) || error("degenerate triangle in circumcenter computation")
    p = p ./ np
    # Two antipodal candidates; pick the one on the same side as the triangle
    # (closer to the triangle interior: positive dot with a+b+c).
    if dot(p, a .+ b .+ c) < 0
        p = .-p
    end
    return p
end

"""    spherical_polygon_area(corners) -> FT

Signed area of the spherical polygon given by corner unit vectors, via
Girard's theorem in turning-angle form: A = 2π − Σ τᵢ, where τᵢ is the
signed turning (deflection) angle at corner i, positive for left turns.

Convention: corners ordered counterclockwise seen from outside the
sphere give the polygon's own (positive) area; clockwise order gives the
negative of the polygon area (NOT the complement). Handles convex and
non-convex polygons.
"""
function spherical_polygon_area(corners::Vector{Vector{FT}}) where {FT}
    n = length(corners)
    n >= 3 || error("polygon needs ≥ 3 corners")
    total = zero(FT)
    for i in 1:n
        a = corners[mod1(i - 1, n)]
        b = corners[i]
        c = corners[mod1(i + 1, n)]
        # arrival tangent at b (direction of travel along arc a→b):
        # p(s) = cos(s) a + sin(s) u with u = normalize(b − (a·b) a),
        # so p'(s_b) ∝ (a·b) b − a.
        t_in = normalize((dot(a, b)) .* b .- a)
        # departure tangent at b along arc b→c
        t_out = normalize(c .- (dot(c, b)) .* b)
        total += atan(dot(cross(t_in, t_out), b), dot(t_in, t_out))
    end
    A = 2 * π - total
    # Normalize the winding: the turning-angle form above returns the area of
    # the region to the LEFT of the traversal. For CW input that is the
    # complement (4π − A); detect orientation via the polygon centroid and
    # return the signed area of the small region consistently.
    centroid = normalize(sum(corners))
    # centroid of the polygon's own region lies on the same side as the
    # corner centroid only if the polygon is the smaller region — check with a
    # point known to be inside: the normalized centroid of the corners is
    # inside any convex polygon; for the complement region it is not.
    # Robust test: sum of turning angles. CCW (small region) → Στ ∈ (0, 2π).
    if total < 0
        A = A - 4 * π   # CW input: shift complement to negative signed area
    end
    return A
end

"""    generate_icosahedral_mesh(FT=Float64; n_subdivisions=0) -> VoronoiSphereMesh

Build the spherical Voronoi C-grid from the icosphere triangulation
with `n_subdivisions` subdivision steps applied to the base icosahedron
(0 → 12 cells, 1 → 42, 2 → 162, ...; standard MPAS-style levels).

The construction (Ringler et al. 2010, §2):

1. Icosahedron vertices are cell centers; subdivide for refinement.
2. Triangle circumcenters are Voronoi cell corners (mesh vertices).
3. For each triangulation edge (i, j): the Voronoi cell edge between
   cells i and j runs between the circumcenters of the (one or two)
   triangles containing edge (i, j). The dual edge length d_e is the
   corner-to-corner arc; the primal length l_e is the center-to-center
   arc.
4. Cell areas by Girard's theorem over the corner polygon (ordered CCW).
5. Vertex (dual-cell) areas: K_v = Σ over the three adjacent cells of the
   area of the sub-triangle (cell center, corner1, corner2) — computed
   here as the sum of corner-to-center-to-corner spherical triangle areas.

Sign conventions follow the paper: for edge e between cells i (pos
side) and j (neg side), `edge_normal` points i → j, and the tangential
velocity reconstruction uses vertices ordered CCW when viewed from
outside the sphere.
"""
function generate_icosahedral_mesh(::Type{FT} = Float64; n_subdivisions::Int = 0) where {FT}
    n_subdivisions >= 0 || throw(ArgumentError("n_subdivisions must be ≥ 0"))

    # 1. Triangulation
    tri = icosahedron(FT)
    for _ in 1:n_subdivisions
        tri = subdivide(tri)
    end
    xyz = tri.xyz                      # cell centers (triangulation vertices)
    tri_edges = tri.edge_vertices
    tri_faces = tri.face_vertices
    nv_tri = size(tri_faces, 2)        # number of triangles = Voronoi corners

    # 2. Circumcenters of all triangles = Voronoi corners
    circum = [spherical_circumcenter(xyz[:, tri_faces[1, f]],
                                    xyz[:, tri_faces[2, f]],
                                    xyz[:, tri_faces[3, f]]) for f in 1:nv_tri]

    # 3. Per-triangulation-edge Voronoi edge data.
    # Triangle → edge lookup: for each triangle, which of its 3 vertex pairs
    # is which global triangulation edge.
    edge_index = Dict{Tuple{Int,Int},Int}()
    for e in 1:size(tri_edges, 2)
        a, b = tri_edges[1, e], tri_edges[2, e]
        edge_index[a < b ? (a, b) : (b, a)] = e
    end
    # for each triangle, the three edge pairs in CCW order (i,j), (j,k), (k,i)
    tri_edge_of_face(f) = begin
        i, j, k = tri_faces[1, f], tri_faces[2, f], tri_faces[3, f]
        (edge_index[i < j ? (i, j) : (j, i)],
         edge_index[j < k ? (j, k) : (k, j)],
         edge_index[k < i ? (k, i) : (i, k)])
    end

    # Faces adjacent to each triangulation edge (1 or 2; on the sphere always 2
    # for a full triangulation, but boundary-free construction allows the check)
    faces_of_edge = [Int[] for _ in 1:size(tri_edges, 2)]
    for f in 1:nv_tri
        for e in tri_edge_of_face(f)
            push!(faces_of_edge[e], f)
        end
    end
    all(fs -> length(fs) == 2, faces_of_edge) ||
        error("triangulation edge not shared by exactly 2 faces — mesh corrupt")

    ne = size(tri_edges, 2)
    nc = size(xyz, 2)

    edge_cells = zeros(Int, 2, ne)
    edge_normal = zeros(FT, 3, ne)
    edge_tangent = zeros(FT, 3, ne)
    l_e = zeros(FT, ne)
    d_e = zeros(FT, ne)
    edge_vertex_pos = zeros(Int, ne)
    edge_vertex_neg = zeros(Int, ne)

    for e in 1:ne
        i, j = tri_edges[1, e], tri_edges[2, e]
        edge_cells[1, e] = i
        edge_cells[2, e] = j
        # normal: tangent to sphere, from i to j
        nvec = xyz[:, j] .- xyz[:, i]
        n = normalize(nvec .- dot(nvec, xyz[:, i]) .* xyz[:, i])
        edge_normal[:, e] = n
        # tangent: 90° CCW rotation of normal in tangent plane (n × r̂)
        r̂ = normalize(xyz[:, i] .+ xyz[:, j])
        edge_tangent[:, e] = normalize(cross(n, r̂))
        # primal edge length: great-circle distance i to j
        l_e[e] = central_angle(xyz[:, i], xyz[:, j])
        # dual edge length: distance between the two adjacent circumcenters
        f1, f2 = faces_of_edge[e]
        d_e[e] = central_angle(circum[f1], circum[f2])
        # end vertices of the Voronoi edge: choose pos/neg so that
        # (pos − neg) direction agrees with tangent.
        v1, v2 = circum[f1], circum[f2]
        if dot(v1 .- v2, edge_tangent[:, e]) > 0
            edge_vertex_pos[e], edge_vertex_neg[e] = f1, f2
        else
            edge_vertex_pos[e], edge_vertex_neg[e] = f2, f1
        end
    end

    # 4. Cell corner polygons (CCW order) and areas (Girard).
    # Corners of cell c = circumcenters of triangles containing c, ordered CCW
    # around c (seen from outside the sphere).
    vertex_cells = [Int[] for _ in 1:nv_tri]      # cells around each corner
    corners_of_cell = [Int[] for _ in 1:nc]
    for f in 1:nv_tri
        for c in tri_faces[:, f]
            push!(corners_of_cell[c], f)
            push!(vertex_cells[f], c)
        end
    end

    cell_area = zeros(FT, nc)
    for c in 1:nc
        fs = corners_of_cell[c]
        # order corners CCW around the cell center:
        ordered = order_ccw(circum, fs, xyz[:, c])
        cell_area[c] = spherical_polygon_area([circum[f] for f in ordered])
    end

    # 5. Vertex (dual cell) areas: K_v = sum over the 3 adjacent cells of
    # spherical area of sub-triangle (center, cornerA, cornerB) where
    # cornerA, cornerB are the two Voronoi corners adjacent within that cell.
    vertex_area = zeros(FT, nv_tri)
    for v in 1:nv_tri
        cs = vertex_cells[v]     # the 3 cells around this corner, any order
        total = zero(FT)
        for c in cs
            fs = corners_of_cell[c]
            # the two corners of cell c's polygon that are adjacent to v...
            # corner v itself is a vertex of the polygon; its neighbors in the
            # polygon are found via the CCW-ordered polygon. Instead of the
            # polygon, use the two triangle edges of face v that touch c.
            # Simpler and equivalent: vertex area contribution of cell c at
            # corner v = spherical area of triangle (c, x, y) where x, y are
            # the two other corners of the triangles adjacent to edges (c, *)
            # of face v.
            total += dual_subtriangle_area(c, v, tri, circum, corners_of_cell, vertex_cells)
        end
        vertex_area[v] = total
    end

    # 6. Pentagon cells (degree-5): icosahedron vertices
    pentagon_cells = Int[c for c in 1:nc if length(corners_of_cell[c]) == 5]

    # 7. edge signs for circulation at vertices: for vertex v, edge e is
    # adjacent iff one of its end cells... we store per-vertex the adjacent
    # edges and the sign of their contribution to circulation around v.
    edge_sign_on_vertex = [Int[] for _ in 1:nv_tri]
    edges_of_vertex = [Int[] for _ in 1:nv_tri]
    for e in 1:ne
        for v in (edge_vertex_pos[e], edge_vertex_neg[e])
            push!(edges_of_vertex[v], e)
        end
    end
    for v in 1:nv_tri
        cs = order_ccw_cells_around(vertex_cells[v], xyz, v, tri_faces)
        signs = Int[]
        for e in edges_of_vertex[v]
            i, j = edge_cells[1, e], edge_cells[2, e]
            # positive if edge normal points CCW around v (i.e. cell i is the
            # CCW-preceding neighbor): consistent with circulation ∮ u·t ds
            # where t is CCW around v.
            s = circulation_sign(e, v, edge_cells, edge_vertex_pos, edge_vertex_neg, edge_tangent, circum)
            push!(signs, s)
        end
        edge_sign_on_vertex[v] = signs
    end

    return VoronoiSphereMesh{FT}(
        nc, ne, nv_tri, xyz, cell_area,
        edge_cells, edge_vertex_pos, edge_vertex_neg,
        edge_normal, edge_tangent, l_e, d_e,
        vertex_cells, vertex_area, edge_sign_on_vertex, pentagon_cells,
    )
end

# --- helpers -------------------------------------------------------------

central_angle(a, b) = atan(norm(cross(a, b)), dot(a, b))

"""Order circumcenter indices `fs` counterclockwise around direction `c`."""
function order_ccw(circum, fs, c)
    vs = [circum[f] for f in fs]
    ref = vs[1]
    angles = map(v -> signed_angle(ref, v, c), vs)
    perm = sortperm(angles)
    return fs[perm]
end

function signed_angle(a, b, n)
    c = normalize(cross(a, b))
    s = dot(c, n) >= 0 ? 1.0 : -1.0
    return s * atan(norm(cross(a, b)), dot(a, b))
end

function dual_subtriangle_area(c, v, tri, circum, corners_of_cell, vertex_cells)
    # area of spherical triangle (cell center c, corner v, corner w) summed
    # over the two Voronoi corners w flanking v in cell c's polygon —
    # implemented via the two triangles of face v that contain vertex c:
    # each such triangle contributes the sub-triangle (c, v, other-circumcenter).
    error("dual subtriangle area: replaced during SCVT milestone (issue #3 phase 2)")
end

function order_ccw_cells_around(cs, xyz, v, tri_faces)
    return cs  # ordering not needed for signs; placeholder
end

function circulation_sign(e, v, edge_cells, ev_pos, ev_neg, edge_tangent, circum)
    # sign of edge e in the circulation around vertex v:
    # the circulation integrand is u_n * l... the standard TRiSK convention
    # (Ringler et al. 2010, Eq. 27): vorticity at vertex v uses the tangential
    # velocities reconstructed at the edges of EV(v) with signs s = ±1.
    # s = +1 if the edge's tangent direction (pos → neg vertex) runs CCW
    # around v, i.e. if rotating from (v → edge midpoint) by +90° aligns with
    # the tangent.
    i, j = edge_cells[1, e], edge_cells[2, e]
    p_pos = circum[ev_pos[e]]
    p_neg = circum[ev_neg[e]]
    t = edge_tangent[:, e]
    # radial direction from v to edge midpoint (on the sphere):
    m = normalize(p_pos .+ p_neg)
    r = normalize(m .- dot(m, circum[v]) .* circum[v])
    # CCW around v (outside view) means t ≈ cross(r, circum[v]):
    return dot(t, cross(r, circum[v])) > 0 ? 1 : -1
end

end # module Voronoi
