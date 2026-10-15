module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")
include("Diagnostics/ErrorNorms.jl")

using .Voronoi: VoronoiSphereMesh, mesh_summary, adjacency_consistent,
                kite_uniformity

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       mesh_summary, adjacency_consistent, kite_uniformity,
       l1_norm, l2_norm, linf_norm, error_norms

end # module JuiCM
