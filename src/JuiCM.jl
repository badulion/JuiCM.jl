module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")
include("Diagnostics/ErrorNorms.jl")

using .Voronoi: VoronoiSphereMesh, generate_icosahedral_mesh,
                mesh_summary, adjacency_consistent, kite_uniformity
using .Fields: Field, CellField, EdgeField, VertexField

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       CellField, EdgeField, VertexField,
       mesh_summary, adjacency_consistent, kite_uniformity,
       l1_norm, l2_norm, linf_norm, error_norms

end # module JuiCM
