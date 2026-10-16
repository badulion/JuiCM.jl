# z-star vertical coordinate machinery (#28, DESIGN.md D7, §5 Phase 2).
#
# Coordinate definition (Adcroft & Campin 2004; MOM6/MPAS-O convention,
# DESIGN.md D7). Interfaces sit at fixed fractions f_k of the reference
# depth H0, measured downward from the free surface. In the moving
# column (free surface at z = η, bathymetric bottom fixed at z = −H0):
#
#   depth below surface of interface k:  ζ_k = f_k·(H0 + η)
#   position of interface k:             z_k = η − ζ_k
#                                       = η − f_k·(H0 + η)
#   layer thickness:                     h_k = (f_{k+1} − f_k)·(H0 + η)
#                                        = Δz*_k · (1 + η/H0)
#
# so the column sum Σ_k h_k = H0 + η exactly and the bottom interface
# (f = 1) sits at z = −H0 for every η. In the equivalent z-form
# (s = −f·H0 ∈ [−H0, 0], surface s = 0, bottom s = −H0):
# z = η + s·(1 + η/H0).
#
# All functions here are exact per column — they are algebraic maps, no
# time stepping. Spatially varying bathymetry H(x, y) and the regridding
# helpers are issue #49; everything here takes the grid's flat H0
# (strict #28/#49 split, user-ratified).

module OceanVertical

using ..Fields: Field, CellField, mesh_of
using ..Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using ..Vertical: VerticalGrid
using ..Voronoi: VoronoiSphereMesh

export zstar_stretch_factor, interface_z, layer_thickness, layer_center_z,
       dz_interfaces, interp_to_interfaces, ssh_from_thickness

"""
    zstar_stretch_factor(eta, H0) -> (H0 + eta)/H0

The z-star stretch factor σ = 1 + η/H0 (Adcroft & Campin 2004): layer
thicknesses scale by σ, interface depths below the surface scale by σ.
`eta` may be a number or a `CellField` (returns matching).
"""
zstar_stretch_factor(eta::Real, H0::Real) = (H0 + eta) / H0
zstar_stretch_factor(eta::CellField, H0::Real) =
    CellField(i -> (H0 + eta[i]) / H0, mesh_of(eta))

"""
    interface_z(grid::VerticalGrid, eta) -> Vector{CellField}
    interface_z(grid::VerticalGrid, eta::Real, mesh) -> Vector{CellField}

Positions z_k [m] of the `nlayers + 1` interfaces (entry 1 = free
surface z = η, entry K+1 = bottom z = −H0):

    z_k = η − f_k·(H0 + η)      (f_k = grid.zstar_fractions[k])

`eta` may be a `CellField`, or a scalar together with the target `mesh`
(uniform-elevation convenience form).
"""
function interface_z(grid::VerticalGrid, eta::CellField)
    (; H0, zstar_fractions) = grid
    m = mesh_of(eta)
    return [CellField(i -> eta[i] - f * (H0 + eta[i]), m)
            for f in zstar_fractions]
end
function interface_z(grid::VerticalGrid, eta::Real,
                     mesh::VoronoiSphereMesh)
    (; H0, zstar_fractions) = grid
    return [CellField(i -> eta - f * (H0 + eta), mesh)
            for f in zstar_fractions]
end

"""
    layer_thickness(grid::VerticalGrid, eta) -> LayeredCellField
    layer_thickness(grid::VerticalGrid, eta::Real, mesh) -> LayeredCellField

Layer thicknesses h_k [m] (k = 1 top):

    h_k = (f_{k+1} − f_k)·(H0 + η) = Δz*_k·(1 + η/H0)

Column invariants (exact, see test/ocean_vertical_test.jl):
Σ_k h_k = H0 + η and η = Σ_k h_k − H0. `eta` may be a `CellField`, or a
scalar together with the target `mesh`.
"""
function layer_thickness(grid::VerticalGrid, eta::CellField)
    (; H0, zstar_fractions) = grid
    m = mesh_of(eta)
    K = nlayers(grid)
    return LayeredCellField((i, k) ->
        (zstar_fractions[k + 1] - zstar_fractions[k]) * (H0 + eta[i]),
        m, K)
end
function layer_thickness(grid::VerticalGrid, eta::Real,
                         mesh::VoronoiSphereMesh)
    (; H0, zstar_fractions) = grid
    K = nlayers(grid)
    return LayeredCellField((i, k) ->
        (zstar_fractions[k + 1] - zstar_fractions[k]) * (H0 + eta),
        mesh, K)
end

"""
    layer_center_z(grid::VerticalGrid, eta) -> LayeredCellField

Positions z [m] of the layer centers (arithmetic mean of the
neighboring interfaces in z space):

    z_c,k = (z_k + z_{k+1}) / 2

with `interface_z`'s indexing (layer k lies between interfaces k and
k + 1).
"""
function layer_center_z(grid::VerticalGrid, eta::CellField)
    zi = interface_z(grid, eta)
    m = mesh_of(eta)
    K = nlayers(grid)
    return LayeredCellField((i, k) -> (zi[k][i] + zi[k + 1][i]) / 2, m, K)
end
function layer_center_z(grid::VerticalGrid, eta::Real,
                        mesh::VoronoiSphereMesh)
    zi = interface_z(grid, eta, mesh)
    K = nlayers(grid)
    return LayeredCellField((i, k) -> (zi[k][i] + zi[k + 1][i]) / 2, mesh, K)
end

"""
    dz_interfaces(h::LayeredCellField) -> Vector{CellField}

Vertical metric: grid spacing between the centers of adjacent layers at
each *interior* interface (K − 1 entries for K layers, entry k sits
between layers k and k+1):

    Δz_c,k = (h_k + h_{k+1}) / 2

This is the denominator for centered vertical differences of layered
quantities (e.g. vertical diffusion in #30+).
"""
function dz_interfaces(h::LayeredCellField)
    K = nlayers(h)
    K >= 2 || throw(ArgumentError("need ≥ 2 layers"))
    m = mesh_of(h)
    return [CellField(i -> (h[k][i] + h[k + 1][i]) / 2, m) for k in 1:(K - 1)]
end

"""
    interp_to_interfaces(q::LayeredCellField, h::LayeredCellField)
        -> Vector{CellField}

Interpolate a layer-centered quantity to the K + 1 interfaces,
thickness-weighted (linear in z for a piecewise-linear profile; exact
on linear stratification):

    surface (k = 1):        q_1        (top layer value; a surface BC
                             replaces this in #30+)
    interior (k = 2..K):    (h_{k−1}·q_k + h_k·q_{k−1}) /
                            (h_{k−1} + h_k)
    bottom (k = K+1):       q_K        (bottom layer value)

The thickness weighting is the standard layered-model interface
interpolation (MPAS-O `vert_interp`; MOM6 `FKAvgE`-style weighted mean):
closer (thinner) layers get more weight.
"""
function interp_to_interfaces(q::LayeredCellField, h::LayeredCellField)
    K = nlayers(q)
    K >= 1 || throw(ArgumentError("need ≥ 1 layer"))
    nlayers(h) == K || throw(ArgumentError("q and h layer counts differ"))
    m = mesh_of(q)
    out = Vector{CellField}(undef, K + 1)
    out[1] = CellField(i -> q[1][i], m)
    for k in 2:K   # interior interface k lies between layers k−1 (above) and k (below)
        out[k] = CellField(i -> (h[k - 1][i] * q[k][i] +
                                 h[k][i] * q[k - 1][i]) /
                                (h[k - 1][i] + h[k][i]), m)
    end
    out[K + 1] = CellField(i -> q[K][i], m)
    return out
end

"""
    ssh_from_thickness(grid::VerticalGrid, h::LayeredCellField) -> CellField

Diagnostic free-surface height from the layer thicknesses (the inverse
of [`layer_thickness`](@ref)):

    η = Σ_k h_k − H0

Exact by construction (column sum invariant); this is how the ocean
dycore recovers η after updating thicknesses.
"""
function ssh_from_thickness(grid::VerticalGrid, h::LayeredCellField)
    (; H0) = grid
    m = mesh_of(h)
    return CellField(nlayers(grid) >= 1 ? i ->
        sum(h[k][i] for k in 1:nlayers(h)) - H0 : i -> -H0, m)
end

end # module OceanVertical
