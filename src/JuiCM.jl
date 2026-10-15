module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")
include("Diagnostics/ErrorNorms.jl")
include("Benchmarks/ShallowWater/ShallowWater.jl")
include("TimeStepping/SSPRK.jl")

using .Voronoi: VoronoiSphereMesh, generate_icosahedral_mesh,
                mesh_summary, adjacency_consistent, kite_uniformity
using .Fields: Field, CellField, EdgeField, VertexField
using .ShallowWater: williamson_constants, case2_velocity, case2_height,
                     case5_velocity, case5_height, case5_mountain,
                     case6_velocity, case6_height, case6_phase_shift
using .TimeStepping: ssprk3_step!, ssprk3_integrate!, rk4_step!, rk4_integrate!

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       CellField, EdgeField, VertexField,
       mesh_summary, adjacency_consistent, kite_uniformity,
       l1_norm, l2_norm, linf_norm, error_norms,
       williamson_constants, case2_velocity, case2_height,
       case5_velocity, case5_height, case5_mountain,
       case6_velocity, case6_height, case6_phase_shift,
       ssprk3_step!, ssprk3_integrate!, rk4_step!, rk4_integrate!

end # module JuiCM
