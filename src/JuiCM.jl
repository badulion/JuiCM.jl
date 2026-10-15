module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")
include("Fields/TRiSK.jl")

using .Voronoi: VoronoiSphereMesh, generate_icosahedral_mesh,
                mesh_summary, adjacency_consistent, kite_uniformity
using .Fields: Field, CellField, EdgeField, VertexField
using .TRiSK: divergence, gradient, curl, tangential_flux, edge_mean,
              kinetic_energy, pv_flux, relative_vorticity

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       CellField, EdgeField, VertexField,
       mesh_summary, adjacency_consistent, kite_uniformity,
       divergence, gradient, curl, tangential_flux, edge_mean,
       kinetic_energy, pv_flux, relative_vorticity

end # module JuiCM
