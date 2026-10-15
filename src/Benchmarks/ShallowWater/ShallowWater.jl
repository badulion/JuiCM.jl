"""Williamson et al. (1992) shallow-water test-case initial conditions.

Grid-independent analytic reference states for validation rung V1
(DESIGN.md §4): cases 2, 5, and 6 as functions of (lon, lat).
"""
module ShallowWater

include("WilliamsonICs.jl")

using .WilliamsonICs: williamson_constants, case2_velocity, case2_height,
                      case5_velocity, case5_height, case5_mountain,
                      case6_velocity, case6_height, case6_phase_shift

export williamson_constants, case2_velocity, case2_height,
       case5_velocity, case5_height, case5_mountain,
       case6_velocity, case6_height, case6_phase_shift

end # module ShallowWater
