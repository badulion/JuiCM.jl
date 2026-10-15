module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")

export VoronoiSphereMesh, generate_icosahedral_mesh, Field

end # module JuiCM
