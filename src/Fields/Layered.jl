# Layered fields (#28): the vertical dimension as a vector of
# per-layer Fields (user-ratified layout; early-ClimaCore style — one
# Field object per layer, all reusing the #8 Field machinery).
#
# DESIGN.md §3.2: the ocean dycore is layered (z-star vertical, D7).
# A layered prognostic (thickness, tracer) is one Field per layer; a
# field living on interfaces (interface position, vertical flux) is one
# Field per interface (K+1 of them for K layers). Both are
# `LayeredField`s here — the distinction is semantic and documented at
# the use sites.
#
# The container deliberately holds plain `Field`s so every payload-
# preserving kernel (`lincomb!`, `axpy!`, `scale!`, `copyto!`) applies
# per layer without new type plumbing, and `HaloArray` payloads carry
# through unchanged (DESIGN.md §3.3 MPI escape hatch).

module Layered

using ..Fields: Field, CellField, EdgeField, similar_field
import ..Fields: mesh_of, location, eltype, lincomb!, axpy!, scale!

export LayeredField, LayeredCellField, LayeredEdgeField, nlayers

"""
    LayeredField{F <: Field}

The vertical dimension as a vector of per-layer Fields: `layers[k]` is
layer `k`'s horizontal field (k = 1 is the top layer). Interface fields
use the same container with `nlayers + 1` entries (entry 1 = surface,
entry K+1 = bottom); which convention applies is documented at each use
site.

Construction goes through [`LayeredCellField`](@ref),
[`LayeredEdgeField`](@ref), or [`similar`](@ref) — the inner constructor
does not check that the layers share a mesh.
"""
struct LayeredField{F <: Field}
    layers::Vector{F}
end

"""
    LayeredCellField = LayeredField{<:CellField}

Type alias for a layered field whose layers are cell-centered.
"""
const LayeredCellField = LayeredField{<:CellField}

"""
    LayeredEdgeField = LayeredField{<:EdgeField}

Type alias for a layered field whose layers are edge-centered.
"""
const LayeredEdgeField = LayeredField{<:EdgeField}

"""Number of layers (or interfaces) of a layered field."""
nlayers(L::LayeredField) = length(L.layers)

"""Layer (or interface) `k` of a layered field."""
Base.getindex(L::LayeredField, k::Integer) = L.layers[k]
Base.firstindex(L::LayeredField) = 1
Base.lastindex(L::LayeredField) = length(L.layers)

mesh_of(L::LayeredField) = mesh_of(L.layers[1])
location(L::LayeredField) = location(L.layers[1])
Base.eltype(::LayeredField{F}) where {F} = eltype(F)

function _check_layers(Ls::LayeredField...)
    n1 = nlayers(Ls[1])
    all(L -> nlayers(L) == n1, Ls) ||
        throw(ArgumentError("layered fields: layer counts differ"))
    L1 = Ls[1]
    for L in Ls[2:end], (k, f) in enumerate(L.layers)
        mesh_of(f) == mesh_of(L1.layers[k]) ||
            throw(ArgumentError("layered fields: layer $k meshes differ"))
        location(f) == location(L1.layers[k]) ||
            throw(ArgumentError("layered fields: layer $k locations differ"))
        eltype(f) == eltype(L1.layers[k]) &&
            eltype(f) == eltype(L1.layers[1]) ||
            throw(ArgumentError("layered fields: eltypes differ"))
    end
    return nothing
end

# ---------------------------------------------------------- construction

"""
    LayeredCellField([f,] mesh, nlayers, ::Type{FT}=Float64) -> LayeredField

Cell-centered layered field: `nlayers` CellFields, layer 1 = top.
With a callable `f`, entry `(i, k)` is `f(i, k)` (cell index, layer
index); without, zeros.
"""
LayeredCellField(mesh, nlayers::Integer, ::Type{FT} = Float64) where {FT} =
    LayeredField([CellField(mesh, FT) for _ in 1:nlayers])
LayeredCellField(f::F, mesh, nlayers::Integer, ::Type{FT} = Float64) where {F, FT} =
    LayeredField([CellField(i -> f(i, k), mesh, FT) for k in 1:nlayers])

"""
    LayeredEdgeField([f,] mesh, nlayers, ::Type{FT}=Float64) -> LayeredField

Edge-centered layered field (per-layer normal velocities, fluxes);
same layer convention as [`LayeredCellField`](@ref).
"""
LayeredEdgeField(mesh, nlayers::Integer, ::Type{FT} = Float64) where {FT} =
    LayeredField([EdgeField(mesh, FT) for _ in 1:nlayers])
LayeredEdgeField(f::F, mesh, nlayers::Integer, ::Type{FT} = Float64) where {F, FT} =
    LayeredField([EdgeField(e -> f(e, k), mesh, FT) for k in 1:nlayers])

# Interface fields are just `LayeredCellField`s (or LayeredEdgeFields)
# with `K + 1` entries for `K` layers — entry 1 is the surface, entry
# K+1 the bottom. No separate type: the convention is documented at the
# use sites.

# --------------------------------------------------------------- kernels
# Each kernel loops layers and delegates to the #8 Field kernel, so
# payload preservation and the location/mesh checks come for free.

"""
    similar(L::LayeredField) -> LayeredField

Uninitialized layered field with the same per-layer payload types.
"""
Base.similar(L::LayeredField) =
    LayeredField([similar_field(f) for f in L.layers])

"""
    copy(L::LayeredField) -> LayeredField

Layered deep copy (payload types preserved).
"""
Base.copy(L::LayeredField) =
    LayeredField([copyto!(similar_field(f), f) for f in L.layers])

"""
    fill!(L::LayeredField, v) -> LayeredField

Set every layer's values to `v`.
"""
Base.fill!(L::LayeredField, v) = (foreach(f -> fill!(f, v), L.layers); L)

"""
    lincomb!(out::LayeredField, α, A::LayeredField, β, B::LayeredField)
        -> LayeredField

`out.layers[k] = α·A.layers[k] + β·B.layers[k]` for every layer.
"""
function lincomb!(out::LayeredField, α::Number, A::LayeredField, β::Number,
                  B::LayeredField)
    _check_layers(out, A, B)
    for k in eachindex(out.layers)
        lincomb!(out.layers[k], α, A.layers[k], β, B.layers[k])
    end
    return out
end

"""
    axpy!(out::LayeredField, α, X::LayeredField) -> LayeredField

`out.layers[k] .+= α·X.layers[k]` for every layer.
"""
function axpy!(out::LayeredField, α::Number, X::LayeredField)
    _check_layers(out, X)
    for k in eachindex(out.layers)
        axpy!(out.layers[k], α, X.layers[k])
    end
    return out
end

"""
    scale!(L::LayeredField, α) -> LayeredField

`L.layers[k] .*= α` for every layer.
"""
function scale!(L::LayeredField, α::Number)
    foreach(f -> scale!(f, α), L.layers)
    return L
end

"""
    copyto!(dst::LayeredField, src::LayeredField) -> LayeredField

Layer-wise copy.
"""
function Base.copyto!(dst::LayeredField, src::LayeredField)
    _check_layers(dst, src)
    for k in eachindex(dst.layers)
        copyto!(dst.layers[k], src.layers[k])
    end
    return dst
end

end # module Layered
