# Vertical grid tables (#28 z-star, #30 coordinate modes, DESIGN.md D7,
# §5 Phase 2).
#
# (header comment: see bottom of file for the module boundaries)
#
# Two coordinate modes (user-ratified for #30):
#
#   :zstar   — interfaces at fixed fractions f_k of the reference depth
#              H0, measured downward from the free surface (Adcroft &
#              Campin 2004; MOM6/MPAS-O convention): an interface at
#              fraction f of the reference depth H sits, in the moving
#              column, at depth-below-surface ζ = f·(H + η), i.e.
#              position z = η − f·(H + η). The bottom (f = 1) therefore
#              always lands at z = −H — fixed bathymetry, as required.
#              In z-coordinate form the map is z = η + s·(1 + η/H) with
#              s = −f·H ∈ [−H, 0]; see Dycore/Ocean/Vertical.jl.
#
#   :simplez — flat-bottom fixed-depth levels: interface k sits at the
#              fixed depth −f_k·H0 for every η (no surface stretching);
#              the free-surface displacement η is carried entirely by
#              the top layer, whose thickness is h_1 = Δz*_1 + η. The
#              column sum Σ h_k = H0 + η still holds exactly. This is
#              the cross-reference configuration for the DESIGN.md §7
#              z-star PGF-on-slope bug class (#30 resting-state test,
#              #45 overflow cross-reference).
#
# This file holds only the *tables* (a `VerticalGrid`); the coordinate
# maps (z-star AND simple-z) and the vertical metric operators on
# layered fields live in `Dycore/Ocean/Vertical.jl`. Spatially varying
# bathymetry H(x, y) is issue #49 — the :simplez mode is flat-bottom
# only by construction.

module Vertical

export VerticalGrid, zstar_thicknesses

import ..Layered: nlayers

"""
    VerticalGrid{FT}

Vertical discretization for the layered ocean: `nlayers` layers whose
interfaces sit at fixed fractions of the reference (flat-bottom) depth
`H0`. `mode` selects the coordinate:

- `:zstar` (default) — interfaces follow the free surface: depth-below-
  surface ζ_k = f_k·(H0 + η) (Adcroft & Campin 2004; MOM6/MPAS-O,
  DESIGN.md D7). Works with any bathymetry (#49).
- `:simplez` — fixed-depth levels z_k = −f_k·H0, independent of η
  (flat bottom only); η moves the top layer alone.

`zstar_fractions[k]` is the fraction of `H0` at interface `k`, measured
downward from the surface: entry 1 is 0, entry `nlayers + 1` is 1
(bottom in both modes). Actual interface positions and layer thicknesses
follow from the mode-dependent maps in `Dycore/Ocean/Vertical.jl`.

Construct with [`VerticalGrid(H0, nlayers; stretching, mode)`](@ref).
"""
struct VerticalGrid{FT}
    """Reference (flat-bottom) depth [m], positive; bathymetry −H0."""
    H0::FT
    """Interface fractions in [0, 1], length nlayers + 1, monotone,
    measured downward from the surface (0 = surface, 1 = bottom)."""
    zstar_fractions::Vector{FT}
    """Coordinate mode: `:zstar` or `:simplez`."""
    mode::Symbol
end

"""
    VerticalGrid(H0, nlayers; stretching = identity, mode = :zstar)
        -> VerticalGrid

Uniform (or stretched) vertical grid with `nlayers` layers over
reference depth `H0` [m]. `stretching` maps the uniform fraction
u ∈ [0, 1] to the interface fraction: `f = stretching(u)`. Use e.g.
`u -> u^2` for finer layers near the surface. `mode` is `:zstar`
(free-surface-following interfaces, the DESIGN.md D7 default) or
`:simplez` (fixed-depth levels, flat bottom only — the #30/#45
cross-reference configuration). The result is validated: monotone,
f(0) = 0, f(1) = 1.
"""
function VerticalGrid(H0::Real, nlayers::Integer; stretching = identity,
                      mode::Symbol = :zstar)
    H0 > 0 || throw(ArgumentError("H0 must be positive"))
    nlayers >= 1 || throw(ArgumentError("nlayers must be ≥ 1"))
    mode in (:zstar, :simplez) ||
        throw(ArgumentError("mode must be :zstar or :simplez, got $mode"))
    f = [stretching((k - 1) / nlayers) for k in 1:(nlayers + 1)]
    (f[1] == 0 && f[end] == 1) ||
        throw(ArgumentError("stretching must map 0 → 0 and 1 → 1"))
    all(>(0), diff(f)) ||
        throw(ArgumentError("interface fractions must be strictly increasing"))
    return VerticalGrid(Float64(H0), Float64.(f), mode)
end

"""Number of layers of a vertical grid."""
nlayers(g::VerticalGrid) = length(g.zstar_fractions) - 1

"""Reference layer thicknesses Δz*_k = (f_{k+1} − f_k)·H0 [m]."""
function zstar_thicknesses(g::VerticalGrid)
    return [g.zstar_fractions[k + 1] - g.zstar_fractions[k]
            for k in 1:nlayers(g)] .* g.H0
end

end # module Vertical
