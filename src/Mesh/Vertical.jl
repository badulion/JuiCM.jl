# Vertical grid tables (#28, DESIGN.md D7 z-star, §5 Phase 2).
#
# (header comment: see bottom of file for the module boundaries)
#
# The z*-space interface positions are fixed fractions of the reference
# depth, measured downward from the free surface (Adcroft & Campin 2004;
# MOM6/MPAS-O convention): an interface at fraction f of the reference
# depth H sits, in the moving column, at depth-below-surface
# ζ = f·(H + η), i.e. position z = η − f·(H + η). The bottom (f = 1)
# therefore always lands at z = −H — fixed bathymetry, as required.
# In z-coordinate form the map is z = η + s·(1 + η/H) with
# s = −f·H ∈ [−H, 0]; see Dycore/Ocean/Vertical.jl.
#
# This file holds only the *tables* (a `VerticalGrid`); the z-star
# mapping and the vertical metric operators on layered fields live in
# `Dycore/Ocean/Vertical.jl`. Spatially varying bathymetry H(x, y),
# regridding helpers, and minimum-thickness treatment are issue #49 —
# this file's H is the flat-bottom reference depth H0 (strict #28/#49
# split, user-ratified).

module Vertical

export VerticalGrid, zstar_thicknesses

import ..Layered: nlayers

"""
    VerticalGrid{FT}

Vertical discretization for the layered ocean: `nlayers` layers whose
interfaces sit at fixed fractions of the reference (flat-bottom) depth
`H0`, in z-star space. `zstar_fractions[k]` is the fraction of `H0` at
interface `k`, measured downward from the free surface: entry 1 is 0
(surface), entry `nlayers + 1` is 1 (bottom). Actual interface depths
follow from the z-star map in `Dycore/Ocean/Vertical.jl`.

Reference: Adcroft & Campin (2004), "Revised stretched grids for ocean
general circulation models" (the z-star coordinate); MOM6/MPAS-O use the
same convention (DESIGN.md D7).

Construct with [`VerticalGrid(H0, nlayers; stretching)`](@ref).
"""
struct VerticalGrid{FT}
    """Reference (flat-bottom) depth [m], positive; bathymetry −H0."""
    H0::FT
    """Interface fractions in [0, 1], length nlayers + 1, monotone,
    measured downward from the free surface (0 = surface, 1 = bottom)."""
    zstar_fractions::Vector{FT}
end

"""
    VerticalGrid(H0, nlayers; stretching = identity) -> VerticalGrid

Uniform (or stretched) vertical grid with `nlayers` layers over
reference depth `H0` [m]. `stretching` maps the uniform fraction
u ∈ [0, 1] to the interface fraction: `f = stretching(u)`. Use e.g.
`u -> u^2` for finer layers near the surface. The result is validated:
monotone, f(0) = 0, f(1) = 1.
"""
function VerticalGrid(H0::Real, nlayers::Integer; stretching = identity)
    H0 > 0 || throw(ArgumentError("H0 must be positive"))
    nlayers >= 1 || throw(ArgumentError("nlayers must be ≥ 1"))
    f = [stretching((k - 1) / nlayers) for k in 1:(nlayers + 1)]
    (f[1] == 0 && f[end] == 1) ||
        throw(ArgumentError("stretching must map 0 → 0 and 1 → 1"))
    all(>(0), diff(f)) ||
        throw(ArgumentError("interface fractions must be strictly increasing"))
    return VerticalGrid(Float64(H0), Float64.(f))
end

"""Number of layers of a vertical grid."""
nlayers(g::VerticalGrid) = length(g.zstar_fractions) - 1

"""Reference layer thicknesses Δz*_k = (f_{k+1} − f_k)·H0 [m]."""
function zstar_thicknesses(g::VerticalGrid)
    return [g.zstar_fractions[k + 1] - g.zstar_fractions[k]
            for k in 1:nlayers(g)] .* g.H0
end

end # module Vertical
