module JuiCM

using OffsetArrays

include("Mesh/Icosahedron.jl")
include("Mesh/Voronoi.jl")
include("Fields/Field.jl")
include("Fields/TRiSK.jl")
include("Fields/Halo.jl")
include("Fields/Layered.jl")
include("Mesh/Vertical.jl")
include("Diagnostics/ErrorNorms.jl")
include("Benchmarks/ShallowWater/ShallowWater.jl")
include("TimeStepping/SSPRK.jl")
include("Dycore/Ocean/Vertical.jl")
include("Dycore/Ocean/EOS.jl")
include("Dycore/Ocean/Tracers.jl")
include("Dycore/Ocean/TracerFCT.jl")
include("Dycore/Atmos/Dynamics.jl")
include("Dycore/Ocean/Dynamics.jl")
include("Benchmarks/Ocean/GravityCurrent.jl")

using .Voronoi: VoronoiSphereMesh, generate_icosahedral_mesh,
                mesh_summary, adjacency_consistent, kite_uniformity
using .Fields: Field, CellField, EdgeField, VertexField
using .Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using .Vertical: VerticalGrid, zstar_thicknesses
using .TRiSK: divergence, gradient, curl, tangential_flux, edge_mean,
              laplacian, kinetic_energy, pv_flux, relative_vorticity,
              divergence!, cell_gradient
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
using .OceanVertical: zstar_stretch_factor, interface_z, layer_thickness,
                      layer_center_z, dz_interfaces, interp_to_interfaces,
                      ssh_from_thickness, simplez_center_depths,
                      bathymetry_flat,
                      bathymetry_channel, bathymetry_slope,
                      bathymetry_shelf_break, regrid_zstar,
                      effective_layer_counts, thickness_from_ssh,
                      vertical_neighbors
using .OceanEOS: AbstractOceanEOS, LinearEOS, ocean_density,
                 ocean_density!
using .OceanTracers: tracer_tendency!, advect_tracer!, tracer_total
using .TracerFCT: fct_advect_tracer!
using .Atmos: SWState, sw_step!
using .Ocean: OceanState, ocean_state, ocean_tendency!, ocean_step!,
              ocean_integrate!, ocean_volume, ocean_kinetic_energy,
              ocean_ssh, wall_vertices, coriolis_parameter
using .GravityCurrent: lock_exchange_state, lock_channel_state,
                       dense_fraction_profile, front_position, front_speed,
                       buoyancy_content, interface_monotone, lock_channel_T,
                       gravity_current_benchmark!, walled_channel,
                       walled_lock_state, walled_front_position,
                       walled_lock_exchange!

export VoronoiSphereMesh, generate_icosahedral_mesh, Field,
       CellField, EdgeField, VertexField,
       LayeredField, LayeredCellField, LayeredEdgeField,
       VerticalGrid, zstar_thicknesses,
       mesh_summary, adjacency_consistent, kite_uniformity,
       divergence, gradient, curl, tangential_flux, edge_mean,
       kinetic_energy, pv_flux, relative_vorticity, divergence!,
       cell_gradient,
       partition_mesh, halo_world, halo_update!, halo_exchange!,
       partitioned_fields, owned_view,
       zstar_stretch_factor, interface_z, layer_thickness,
       layer_center_z, dz_interfaces, interp_to_interfaces,
       ssh_from_thickness, nlayers, simplez_center_depths,
       bathymetry_flat, bathymetry_channel, bathymetry_slope,
       bathymetry_shelf_break, regrid_zstar, effective_layer_counts,
       thickness_from_ssh, vertical_neighbors,
       AbstractOceanEOS, LinearEOS, ocean_density, ocean_density!,
       tracer_tendency!, advect_tracer!, tracer_total,
       fct_advect_tracer!,
       OceanState, ocean_state, ocean_tendency!, ocean_step!,
       ocean_integrate!, ocean_volume, ocean_kinetic_energy, ocean_ssh,
       wall_vertices, coriolis_parameter, walled_channel, walled_lock_state,
       walled_front_position, walled_lock_exchange!,
       lock_exchange_state, lock_channel_state, front_position, front_speed,
       buoyancy_content, interface_monotone, lock_channel_T,
       dense_fraction_profile, gravity_current_benchmark!, laplacian,
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
