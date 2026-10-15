module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")

using .Voronoi: VoronoiSphereMesh, generate_icosahedral_mesh,
                mesh_summary, adjacency_consistent, kite_uniformity
using .Fields: Field, CellField, EdgeField, VertexField

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       CellField, EdgeField, VertexField,
       mesh_summary, adjacency_consistent, kite_uniformity

end # module JuiCM
