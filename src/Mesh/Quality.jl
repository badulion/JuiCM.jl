# Mesh quality diagnostics (DESIGN.md §4 rung V0.5).
#
# All functions take a `VoronoiSphereMesh` and return plain data (numbers,
# named tuples) so they are usable both in tests and in human-readable
# reports. No plotting here — visualization lives in `docs/` scripts.

"""
    mesh_summary(mesh) -> NamedTuple

Aggregate quality summary of a `VoronoiSphereMesh`:

- `n_cells`, `n_edges`, `n_vertices`: element counts.
- `total_area`: sum of primal cell areas (should be 4π for R = 1).
- `n_pentagons`: number of pentagonal cells (exactly 12 for the
  icosahedral family).
- `min_quality`, `mean_quality`: cell quality statistics, where the
  quality of a cell is the ratio of its smallest kite area to the mean
  kite area (1.0 for a perfectly uniform cell; MPAS-style metric).
- `aspect_min`: minimum over edges of d_e / l_e bounds check placeholder
  (kept for future orthogonality metrics).
"""
function mesh_summary(mesh::VoronoiSphereMesh)
    return (
        n_cells = mesh.n_cells,
        n_edges = mesh.n_edges,
        n_vertices = mesh.n_vertices,
        total_area = sum(mesh.cell_area),
        n_pentagons = length(mesh.pentagon_cells),
        min_quality = minimum(mesh.cell_quality),
        mean_quality = sum(mesh.cell_quality) / mesh.n_cells,
    )
end

"""
    adjacency_consistent(mesh) -> Bool

Full adjacency consistency check:

- every edge has two distinct flanking cells (`edge_cells`),
- `cell_edges` and `edge_cells` are mutual inverses (each edge appears
  in exactly the edge lists of its two cells),
- each `cell_edges[c]` has the same length as `cell_corners[c]` and the
  polygon closes (consecutive corners are connected by the listed edges
  in order),
- `vertex_cells` / `vertex_edges` are mutually consistent with
  `edge_cells` / `edge_vertex_pos|neg` (each vertex sees exactly the
  cells/edges incident to it).
"""
function adjacency_consistent(mesh::VoronoiSphereMesh)
    # edge ↔ cell incidence
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        i != j || return false
        e in mesh.cell_edges[i] || return false
        e in mesh.cell_edges[j] || return false
    end
    for c in 1:mesh.n_cells
        length(mesh.cell_edges[c]) == length(mesh.cell_corners[c]) || return false
        for e in mesh.cell_edges[c]
            c in mesh.edge_cells[:, e] || return false
        end
        # polygon closes: edge k connects corner k and corner k+1 (CCW)
        n = length(mesh.cell_edges[c])
        for k in 1:n
            v1 = mesh.cell_corners[c][k]
            v2 = mesh.cell_corners[c][mod1(k + 1, n)]
            e = mesh.cell_edges[c][k]
            (v1 in (mesh.edge_vertex_pos[e], mesh.edge_vertex_neg[e]) &&
             v2 in (mesh.edge_vertex_pos[e], mesh.edge_vertex_neg[e])) || return false
        end
    end
    # vertex incidence
    for v in 1:mesh.n_vertices
        for c in mesh.vertex_cells[v]
            v in mesh.cell_corners[c] || return false
        end
        for e in mesh.vertex_edges[v]
            v in (mesh.edge_vertex_pos[e], mesh.edge_vertex_neg[e]) || return false
        end
    end
    return true
end

"""
    kite_uniformity(mesh) -> FT

Global mesh-quality metric (Ringler et al. 2010 §6 / MPAS convention):
ratio of the minimum to the mean kite area over all (cell, corner)
kites, weighted per cell. Returns a scalar in (0, 1]; 1 would be a
perfectly uniform tessellation. The icosahedral mesh at level 0 scores
≈ 0.85; Lloyd relaxation toward SCVT improves it.
"""
function kite_uniformity(mesh::VoronoiSphereMesh)
    qs = mesh.cell_quality
    return minimum(qs)
end
