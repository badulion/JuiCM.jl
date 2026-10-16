# z-star vertical coordinate machinery (#28, bathymetry + regridding
# #49, DESIGN.md D7, §5 Phase 2).
#
# Coordinate definition (Adcroft & Campin 2004; MOM6/MPAS-O convention,
# DESIGN.md D7). Interfaces sit at fixed fractions f_k of the reference
# depth, measured downward from the free surface. With bathymetric depth
# H (flat H0 from #28, or a per-cell CellField from #49) and free
# surface at z = η (bottom fixed at z = −H):
#
#   depth below surface of interface k:  ζ_k = f_k·(H + η)
#   position of interface k:             z_k = η − ζ_k
#                                        = η − f_k·(H + η)
#   layer thickness:                     h_k = (f_{k+1} − f_k)·(H + η)
#                                        = Δz*_k · (1 + η/H)  [per cell]
#
# so the column sum Σ_k h_k = H + η exactly per cell and the bottom
# interface (f = 1) sits at z = −H for every η. In the equivalent z-form
# (s = −f·H ∈ [−H, 0], surface s = 0, bottom s = −H):
# z = η + s·(1 + η/H) per cell.
#
# All functions here are exact per column — algebraic maps, no time
# stepping. Dispatch on `H`: a `Real` (the grid's flat H0, #28) or a
# `CellField` (spatially varying bathymetry, #49).
#
# Bathymetry (#49): analytic profiles in `Bathymetry`, minimum-thickness
# treatment via bottom-level reduction (`regrid_zstar` /
# `effective_layer_counts`), vertical neighbor tables
# (`vertical_neighbors`), and the regridding round trip
# (`thickness_from_ssh`) — all below.

module OceanVertical

using ..Fields: Field, CellField, mesh_of
using ..Layered: LayeredField, LayeredCellField, LayeredEdgeField, nlayers
using ..Vertical: VerticalGrid
using ..Voronoi: VoronoiSphereMesh

export zstar_stretch_factor, interface_z, layer_thickness, layer_center_z,
       dz_interfaces, interp_to_interfaces, ssh_from_thickness,
       bathymetry_flat, bathymetry_channel, bathymetry_slope,
       bathymetry_shelf_break, regrid_zstar, effective_layer_counts,
       thickness_from_ssh, vertical_neighbors

"""
    zstar_stretch_factor(eta, H) -> (H + eta)/H

The z-star stretch factor σ = 1 + η/H (Adcroft & Campin 2004): layer
thicknesses scale by σ, interface depths below the surface scale by σ.
`eta` and `H` may each be a number or a `CellField` (at least one a
field; a field result is returned when either input is a field).
"""
zstar_stretch_factor(eta::Real, H::Real) = (H + eta) / H
function zstar_stretch_factor(eta::CellField, H::Real)
    return CellField(i -> (H + eta[i]) / H, mesh_of(eta))
end
function zstar_stretch_factor(eta::Real, H::CellField)
    return CellField(i -> (H[i] + eta) / H[i], mesh_of(H))
end
function zstar_stretch_factor(eta::CellField, H::CellField)
    mesh_of(eta) == mesh_of(H) ||
        throw(ArgumentError("eta and H must live on the same mesh"))
    return CellField(i -> (H[i] + eta[i]) / H[i], mesh_of(eta))
end

"""
    interface_z(grid::VerticalGrid, eta[, H]) -> Vector{CellField}
    interface_z(grid::VerticalGrid, eta::Real, mesh) -> Vector{CellField}

Positions z_k [m] of the `nlayers + 1` interfaces (entry 1 = free
surface z = η, entry K+1 = bottom z = −H):

    z_k = η − f_k·(H + η)      (f_k = grid.zstar_fractions[k])

`H` is the bathymetric depth: omitted/`Real` uses the grid's flat H0;
a `CellField` gives per-cell bathymetry (#49). `eta` may be a
`CellField`, or a scalar together with the target `mesh` (the
uniform-elevation form always uses flat H0).
"""
function interface_z(grid::VerticalGrid, eta::CellField)
    return interface_z(grid, eta, CellField(i -> grid.H0, mesh_of(eta)))
end
function interface_z(grid::VerticalGrid, eta::CellField, H::Real)
    (; zstar_fractions) = grid
    m = mesh_of(eta)
    return [CellField(i -> eta[i] - f * (H + eta[i]), m)
            for f in zstar_fractions]
end
function interface_z(grid::VerticalGrid, eta::CellField, H::CellField)
    (; zstar_fractions) = grid
    m = mesh_of(eta)
    mesh_of(H) == m ||
        throw(ArgumentError("eta and H must live on the same mesh"))
    return [CellField(i -> eta[i] - f * (H[i] + eta[i]), m)
            for f in zstar_fractions]
end
function interface_z(grid::VerticalGrid, eta::Real,
                     mesh::VoronoiSphereMesh)
    (; H0, zstar_fractions) = grid
    return [CellField(i -> eta - f * (H0 + eta), mesh)
            for f in zstar_fractions]
end

"""
    layer_thickness(grid::VerticalGrid, eta[, H]) -> LayeredCellField
    layer_thickness(grid::VerticalGrid, eta::Real, mesh) -> LayeredCellField

Layer thicknesses h_k [m] (k = 1 top):

    h_k = (f_{k+1} − f_k)·(H + η) = Δz*_k·(1 + η/H)

Column invariants (exact, see test/ocean_vertical_test.jl):
Σ_k h_k = H + η per cell and η = Σ_k h_k − H. `H` is the bathymetric
depth: omitted/`Real` uses the grid's flat H0; a `CellField` gives
per-cell bathymetry (#49). `eta` may be a `CellField`, or a scalar
together with the target `mesh` (the uniform-elevation form always uses
flat H0).
"""
function layer_thickness(grid::VerticalGrid, eta::CellField)
    return layer_thickness(grid, eta,
                           CellField(i -> grid.H0, mesh_of(eta)))
end
function layer_thickness(grid::VerticalGrid, eta::CellField, H::Real)
    (; zstar_fractions) = grid
    m = mesh_of(eta)
    K = nlayers(grid)
    return LayeredCellField((i, k) ->
        (zstar_fractions[k + 1] - zstar_fractions[k]) * (H + eta[i]),
        m, K)
end
function layer_thickness(grid::VerticalGrid, eta::CellField, H::CellField)
    (; zstar_fractions) = grid
    m = mesh_of(eta)
    mesh_of(H) == m ||
        throw(ArgumentError("eta and H must live on the same mesh"))
    K = nlayers(grid)
    return LayeredCellField((i, k) ->
        (zstar_fractions[k + 1] - zstar_fractions[k]) * (H[i] + eta[i]),
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
function layer_center_z(grid::VerticalGrid, eta::CellField, H::Real)
    zi = interface_z(grid, eta, H)
    m = mesh_of(eta)
    K = nlayers(grid)
    return LayeredCellField((i, k) -> (zi[k][i] + zi[k + 1][i]) / 2, m, K)
end
function layer_center_z(grid::VerticalGrid, eta::CellField, H::CellField)
    zi = interface_z(grid, eta, H)
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
    ssh_from_thickness(grid::VerticalGrid, h::LayeredCellField, H) -> CellField

Diagnostic free-surface height from the layer thicknesses (the inverse
of [`layer_thickness`](@ref)):

    η = Σ_k h_k − H

`H` is the bathymetric depth: omitted/`Real` uses the grid's flat H0;
a `CellField` gives per-cell bathymetry (#49). Exact by construction
(column sum invariant); this is how the ocean dycore recovers η after
updating thicknesses.
"""
function ssh_from_thickness(grid::VerticalGrid, h::LayeredCellField)
    (; H0) = grid
    m = mesh_of(h)
    return CellField(i -> sum(h[k][i] for k in 1:nlayers(h)) - H0, m)
end
function ssh_from_thickness(grid::VerticalGrid, h::LayeredCellField,
                            H::Real)
    m = mesh_of(h)
    return CellField(i -> sum(h[k][i] for k in 1:nlayers(h)) - H, m)
end
function ssh_from_thickness(grid::VerticalGrid, h::LayeredCellField,
                            H::CellField)
    m = mesh_of(h)
    mesh_of(H) == m ||
        throw(ArgumentError("h and H must live on the same mesh"))
    return CellField(i -> sum(h[k][i] for k in 1:nlayers(h)) - H[i], m)
end

# ------------------------------------------------------ bathymetry (#49)

# Analytic bathymetry profiles, per-cell depth H(x, y) > 0 [m] as
# CellFields. All take (mesh, params...) and are smooth enough for the
# V3/V4 benchmark configurations (channel, linear slope, shelf break).

"""    bathymetry_flat(mesh, H0) -> CellField

Flat bottom: H(x, y) = H0 everywhere."""
bathymetry_flat(mesh::VoronoiSphereMesh, H0::Real) =
    CellField(i -> H0, mesh)

"""
    bathymetry_channel(mesh, H0, H_deep; x_min = -π/2, x_max = π/2,
                       transition = π/8) -> CellField

Closed-channel profile: deep region `H_deep` for λ ∈ (x_min + transition,
x_max − transition), shallow shelf `H0` beyond, smoothly blended with a
cosine taper over `transition` radians of longitude (a locally-Cartesian
patch interpretation: the channel runs along the equator between the
longitude bounds; at |lat| > π/4 the profile is the deep value — the
benchmark domain is the equatorial band).

Reference: the lock-exchange/overflow channel convention (STSWM
channel configurations; Huppert & Simpson 1980 use a closed box).
"""
function bathymetry_channel(mesh::VoronoiSphereMesh, H0::Real, H_deep::Real;
                            x_min = -π / 2, x_max = π / 2,
                            transition = π / 8)
    H0 < H_deep || throw(ArgumentError("H0 (shelf) must be < H_deep"))
    return CellField(mesh) do i
        λ = atan(mesh.cell_xyz[2, i], mesh.cell_xyz[1, i])
        t = transition
        # smooth step up from H0 at the ends to H_deep in the interior;
        # λ outside [x_min, x_max] (longitude wrap) maps to the nearest
        # ramp end so the profile stays in [H0, H_deep]
        λw = λ < (x_min + x_max) / 2 ? max(λ, x_min) : min(λ, x_max)
        west = λw < x_min + t ? H0 + (H_deep - H0) *
               (1 - cos(π * (λw - x_min) / t)) / 2 : H_deep
        east = λw > x_max - t ? H0 + (H_deep - H0) *
               (1 - cos(π * (x_max - λw) / t)) / 2 : H_deep
        return min(west, east)
    end
end

"""
    bathymetry_slope(mesh, H_shallow, H_deep, θ0 = π/6) -> CellField

Linear downslope profile along longitude: `H_shallow` at λ = −π/2
rising linearly to `H_deep` at λ = π/2 (clamped beyond), the
locally-Cartesian slope of the V4 overflow configuration (Legg et al.
2006 class). On the sphere the profile is a function of longitude only;
latitude dependence is intentionally absent (periodic in the cross-slope
direction for the benchmark patch).
"""
function bathymetry_slope(mesh::VoronoiSphereMesh, H_shallow::Real,
                          H_deep::Real, λ_min = -π / 2, λ_max = π / 2)
    H_shallow < H_deep ||
        throw(ArgumentError("H_shallow must be < H_deep"))
    return CellField(mesh) do i
        λ = atan(mesh.cell_xyz[2, i], mesh.cell_xyz[1, i])
        λc = clamp(λ, λ_min, λ_max)
        return H_shallow +
               (H_deep - H_shallow) * (λc - λ_min) / (λ_max - λ_min)
    end
end

"""
    bathymetry_shelf_break(mesh, H_shelf, H_deep; λ_break = 0.0,
                           width = π/16) -> CellField

Continental-shelf profile: `H_shelf` for λ < λ_break, dropping
smoothly (tanh) to `H_deep` over `width` radians of longitude —
the shelf-break/slope configuration used in dense-water overflow
studies (e.g. Wåhlin & Cenedese class experiments).
"""
function bathymetry_shelf_break(mesh::VoronoiSphereMesh, H_shelf::Real,
                                H_deep::Real; λ_break = 0.0,
                                width = π / 16)
    width > 0 || throw(ArgumentError("width must be positive"))
    return CellField(mesh) do i
        λ = atan(mesh.cell_xyz[2, i], mesh.cell_xyz[1, i])
        return H_shelf + (H_deep - H_shelf) *
               (1 + tanh((λ - λ_break) / width)) / 2
    end
end

# ------------------------------------------- min thickness + regridding

"""
    effective_layer_counts(grid, H, h_min_frac = 0.01) -> Vector{Int}

Bottom-level reduction (MPAS-style, user-ratified convention for #49):
at each cell, bottom layers are merged upward so the *merged bottom
layer* keeps at least `h_min_frac·H(i)` of reference thickness; layers
above it keep their fixed z-star-space thicknesses Δz*_k·H(i). Returns
the per-cell active layer count n_k(i) ∈ [1, nlayers(grid)] — layer n
is the merged remainder.

The reduction is cumulative from the bottom (the deepest layers are
dropped first), and the reference column (η = 0) drives it, so the
layer structure is time-independent — it changes only if H or
h_min_frac changes, and the z-star regridding never re-merges layers
mid-run.
"""
function effective_layer_counts(grid::VerticalGrid, H::CellField,
                                h_min_frac::Real = 0.01)
    (; zstar_fractions) = grid
    K = nlayers(grid)
    m = mesh_of(H)
    h_min_frac > 0 && h_min_frac < 1 ||
        throw(ArgumentError("h_min_frac must be in (0, 1)"))
    counts = Vector{Int}(undef, m.n_cells)
    for i in 1:m.n_cells
        Hi = H[i]
        floor_hi = h_min_frac * Hi
        # walk from the top (k = 1) down, accumulating reference
        # thickness; stop before the accumulated remainder would leave
        # the bottom layers below the floor. Layers 1..n stay active.
        n = K
        acc = 0.0
        for k in 1:K
            dz = (zstar_fractions[k + 1] - zstar_fractions[k]) * Hi
            if k < K && acc + dz > Hi - floor_hi
                n = k - 1 >= 1 ? k - 1 : 1
                break
            end
            acc += dz
            n = k
        end
        counts[i] = max(n, 1)
    end
    return counts
end

"""
    regrid_zstar(grid, eta, H; h_min_frac = 0.01)
        -> (thickness::LayeredCellField, counts::Vector{Int})

Recompute the z-star layer structure under arbitrary bathymetry and
surface elevation — the piece that keeps the coordinate well-defined on
a slope (#49). With the bottom-level-reduction convention: layers below
the per-cell active count (see [`effective_layer_counts`](@ref)) are
merged into the layer above (their thicknesses added, upward), so

    Σ_{k≤n(i)} h_k = H(i) + η(i)   still holds exactly per cell,

and every active layer keeps positive thickness. The returned counts
let the dycore skip inactive bottom layers.
"""
function regrid_zstar(grid::VerticalGrid, eta::CellField, H::CellField;
                      h_min_frac::Real = 0.01)
    mesh_of(eta) == mesh_of(H) ||
        throw(ArgumentError("eta and H must live on the same mesh"))
    (; zstar_fractions) = grid
    m = mesh_of(eta)
    K = nlayers(grid)
    counts = effective_layer_counts(grid, H, h_min_frac)

    # reference thicknesses Δz*_k (fixed fractions), then the z-star
    # scale per cell, then merge from the bottom
    h = layer_thickness(grid, eta, H)
    out = LayeredCellField(m, K)
    for i in 1:m.n_cells
        n = counts[i]
        # active layers 1..n−1 keep their z-star thickness; layer n
        # absorbs everything below it (the merged remainder)
        upper = sum(h[k][i] for k in 1:(n - 1); init = 0.0)
        for k in 1:(n - 1)
            out[k][i] = h[k][i]
        end
        out[n][i] = (H[i] + eta[i]) - upper
        for k in (n + 1):K
            out[k][i] = 0.0     # inactive slots
        end
    end
    return (out, counts)
end

"""
    thickness_from_ssh(grid, eta, H; h_min_frac = 0.01) -> LayeredCellField

Convenience wrapper around [`regrid_zstar`](@ref) returning only the
thicknesses (the regridding round trip used in the acceptance test:
apply prescribed η(t) forcing, recompute, recover the analytic profile).
"""
function thickness_from_ssh(grid::VerticalGrid, eta::CellField, H::CellField;
                            h_min_frac::Real = 0.01)
    return first(regrid_zstar(grid, eta, H; h_min_frac = h_min_frac))
end

"""
    vertical_neighbors(mesh, K) -> Vector{Vector{Int}}

Vertical neighbor table (#49): for each cell, the layer indices adjacent
to layer k — the same column above/below (k−1, k+1 within the active
range). With a single column per cell the table is trivial, but the
#32 vertical advection and #30 diffusion consume it through this API so
a future nested/overlapping-column layout needs no call-site change.
Returns, for each cell i, a Vector{Int} of length 2K with entries
`[above(1), below(1), above(2), below(2), …]` where `above(1) = 0` (no
layer above the top) and `below(K) = 0`.
"""
function vertical_neighbors(mesh::VoronoiSphereMesh, K::Integer)
    K >= 1 || throw(ArgumentError("need ≥ 1 layer"))
    table = Vector{Vector{Int}}(undef, mesh.n_cells)
    for i in 1:mesh.n_cells
        nb = Vector{Int}(undef, 2K)
        for k in 1:K
            nb[2k - 1] = k == 1 ? 0 : k - 1   # above
            nb[2k] = k == K ? 0 : k + 1       # below
        end
        table[i] = nb
    end
    return table
end

end # module OceanVertical
