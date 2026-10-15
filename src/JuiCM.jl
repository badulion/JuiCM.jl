module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")
include("Diagnostics/ErrorNorms.jl")

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       l1_norm, l2_norm, linf_norm, error_norms

end # module JuiCM
