module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")

using .Voronoi: VoronoiSphereMesh, mesh_summary, adjacency_consistent,
                kite_uniformity

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       mesh_summary, adjacency_consistent, kite_uniformity

end # module JuiCM
