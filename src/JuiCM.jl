module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")
include("Fields/TRiSK.jl")
include("Fields/Halo.jl")
include("Diagnostics/ErrorNorms.jl")
include("Benchmarks/ShallowWater/ShallowWater.jl")
include("TimeStepping/SSPRK.jl")
include("Dycore/Atmos/Dynamics.jl")

using .Voronoi: VoronoiSphereMesh, generate_icosahedral_mesh,
                mesh_summary, adjacency_consistent, kite_uniformity
using .Fields: Field, CellField, EdgeField, VertexField
using .TRiSK: divergence, gradient, curl, tangential_flux, edge_mean,
              kinetic_energy, pv_flux, relative_vorticity
using .Halo: MeshPartition, partition_mesh, n_regions, region_of,
             owned_indices, HaloInfo, build_halo_info, HaloArray,
             halo_array, HaloWorld, halo_world, halo_update!,
             halo_exchange!, partitioned_fields, owned_view
using .Fields: lincomb!, axpy!, similar_field, scale!, valid_indices
using .ShallowWater: williamson_constants, case2_velocity, case2_height,
                     case5_velocity, case5_height, case5_mountain,
                     case6_velocity, case6_height, case6_phase_shift
using .TimeStepping: ssprk3_step!, ssprk3_integrate!, rk4_step!, rk4_integrate!
using .Atmos: SWState, SWModel, sw_state, sw_model, AbstractPhysicsModule,
              compute_tendency!, NullForcing, sw_tendency!, sw_step!,
              sw_integrate!, sw_mass, sw_total_energy

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       CellField, EdgeField, VertexField,
       mesh_summary, adjacency_consistent, kite_uniformity,
       divergence, gradient, curl, tangential_flux, edge_mean,
       kinetic_energy, pv_flux, relative_vorticity,
       partition_mesh, halo_world, halo_update!, halo_exchange!,
       partitioned_fields, owned_view,
       SWState, SWModel, sw_state, sw_model, AbstractPhysicsModule,
       compute_tendency!, NullForcing, sw_tendency!, sw_step!,
       sw_integrate!, sw_mass, sw_total_energy,
       l1_norm, l2_norm, linf_norm, error_norms,
       williamson_constants, case2_velocity, case2_height,
       case5_velocity, case5_height, case5_mountain,
       case6_velocity, case6_height, case6_phase_shift,
       ssprk3_step!, ssprk3_integrate!, rk4_step!, rk4_integrate!,
       SWState, sw_state, AbstractPhysicsModule, compute_tendency!,
       NullForcing, sw_tendency!, sw_step!, sw_integrate!,
       sw_mass, sw_total_energy

end # module JuiCM
