"""Fields: data at mesh locations.

TRiSK staggering (DESIGN.md §3.3): mass and tracers at cells, normal
velocities on edges, vorticity at vertices. Physics code never touches
`Field.data` directly — operators are the only access path (AGENTS.md
hard rule 3, MPI escape hatch).

The array type is a type parameter, so a future partitioned (MPI) array
can replace the plain `Array` without touching physics code: the halo
machinery (#10) will provide a `HaloArray <: AbstractArray` and fields
are constructed identically.

Design notes (DESIGN.md §3.3, core abstraction 2):

- A `Field` carries its mesh reference, so operators can look up geometry
  without passing the mesh separately.
- `Field` is parametric in *location* as a phantom type argument `L`:
  `CellField` values only combine with `CellField` values, etc. Mixing
  staggers in arithmetic is a bug — make it a type error.
- Construction and arithmetic are type-stable: the full type is
  determined by the constructor's arguments.
"""
module Fields

import LinearAlgebra: dot

export Field, CellField, EdgeField, VertexField,
       CellLoc, EdgeLoc, VertexLoc, FieldLocation,
       location, mesh_of, similar_field, length_of,
       lincomb!, axpy!, similar_state, scale!, rdivide!, dot, valid_indices
"""Location of field data on the mesh (TRiSK staggering)."""
abstract type FieldLocation end

"""Cell staggering: mass, tracers, geopotential (one value per Voronoi cell)."""
struct CellLoc <: FieldLocation end

"""Edge staggering: normal velocities and fluxes (one value per mesh edge)."""
struct EdgeLoc <: FieldLocation end

"""Vertex staggering: vorticity (one value per dual cell / mesh vertex)."""
struct VertexLoc <: FieldLocation end

# number of values per location, derived from the mesh
nvalues(::CellLoc, m) = m.n_cells
nvalues(::EdgeLoc, m) = m.n_edges
nvalues(::VertexLoc, m) = m.n_vertices

"""A field: data array + mesh + staggering location (phantom type).

Type parameters: mesh type `M`, element type `T`, array dimensionality
`N` (kept for future 2D fields, e.g. tracers × layers), concrete array
type `AX <: AbstractArray`, and the `FieldLocation` phantom `L`.

The array type is abstract so a future partitioned (MPI) array can
replace the plain `Array` without touching physics code (DESIGN.md §3.3).
"""
struct Field{M <: Any, T, N, AX <: AbstractArray, L <: FieldLocation}
    data::AX
    mesh::M
end

# convenience type aliases for readability in signatures
const CellField{M, T, N, AX} = Field{M, T, N, AX, CellLoc}
const EdgeField{M, T, N, AX} = Field{M, T, N, AX, EdgeLoc}
const VertexField{M, T, N, AX} = Field{M, T, N, AX, VertexLoc}

# ------------------------------------------------------------- construction

"""
    CellField([f,] mesh, ::Type{FT}=Float64) -> Field

Cell-centered field (mass, tracers): one value per Voronoi cell.
With a callable `f`, entries are initialized as `f(i)` for cell index
`i`; without, zeros.
"""
CellField(mesh, ::Type{FT} = Float64) where {FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, CellLoc}(zeros(FT, mesh.n_cells), mesh)
CellField(f::F, mesh, ::Type{FT} = Float64) where {F, FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, CellLoc}(FT[f(i) for i in 1:mesh.n_cells], mesh)

"""
    EdgeField([f,] mesh, ::Type{FT}=Float64) -> Field

Edge-centered field (normal velocities, fluxes): one value per mesh
edge, positive along `n̂_e` (from `edge_cells[1,e]` to `edge_cells[2,e]`).
With a callable `f`, entries are initialized as `f(i)` for edge index
`i`; without, zeros.
"""
EdgeField(mesh, ::Type{FT} = Float64) where {FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, EdgeLoc}(zeros(FT, mesh.n_edges), mesh)
EdgeField(f::F, mesh, ::Type{FT} = Float64) where {F, FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, EdgeLoc}(FT[f(i) for i in 1:mesh.n_edges], mesh)

"""
    VertexField([f,] mesh, ::Type{FT}=Float64) -> Field

Vertex-centered field (vorticity): one value per dual cell (mesh
vertex). With a callable `f`, entries are initialized as `f(i)` for
vertex index `i`; without, zeros.
"""
VertexField(mesh, ::Type{FT} = Float64) where {FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, VertexLoc}(zeros(FT, mesh.n_vertices), mesh)
VertexField(f::F, mesh, ::Type{FT} = Float64) where {F, FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, VertexLoc}(FT[f(i) for i in 1:mesh.n_vertices], mesh)

# --------------------------------------------------------------- accessors

"""Element type of a field."""
Base.eltype(::Field{M, T, N, AX, L}) where {M, T, N, AX, L} = T
Base.eltype(::Type{Field{M, T, N, AX, L}}) where {M, T, N, AX, L} = T

"""Staggering location of a field, as a `FieldLocation` *instance*."""
location(::Field{M, T, N, AX, L}) where {M, T, N, AX, L} = L()

"""Mesh a field lives on."""
mesh_of(f::Field) = f.mesh

"""Number of values in a field (= number of mesh elements at its location)."""
length_of(f::Field) = nvalues(location(f), f.mesh)

"""
    similar_field(f) -> Field
    similar_field(f, ::Type{FT}) -> Field

Uninitialized field with the same mesh, location, and array type as `f`
(optionally with a different element type `FT`). Allocates like
`Base.similar` — data are arbitrary.
"""
similar_field(f::Field{M, T, N, AX, L}) where {M, T, N, AX, L} =
    Field{M, T, N, Base.similar(f.data) |> typeof, L}(similar(f.data), f.mesh)
similar_field(f::Field{M, T, N, AX, L}, ::Type{FT}) where {M, T, N, AX, FT, L} =
    Field{M, FT, N, Base.similar(f.data, FT) |> typeof, L}(similar(f.data, FT), f.mesh)

"""
    similar_field(f, ::Type{FT}, ::FieldLocation) -> Field

Uninitialized field with the payload type of `f` but a *different*
location (and element type `FT`) — how the allocating TRiSK operators
map between staggers without leaving the payload type (a `HaloArray`
payload stays a `HaloArray`, in global index space).
"""
similar_field(f::Field{M, T, N, AX}, ::Type{FT}, ::L) where {M, T, N, AX, FT, L} =
    Field{M, FT, N, Base.similar(f.data, FT, nvalues(L(), f.mesh)) |> typeof, L}(
        similar(f.data, FT, nvalues(L(), f.mesh)), f.mesh)

"""
    fill!(f::Field, v) -> Field

Set all values of `f` to `v`.
"""
Base.fill!(f::Field, v) = (fill!(f.data, v); f)

# ---------------------------------------------------------------- indexing

Base.length(f::Field) = length(f.data)
Base.size(f::Field) = size(f.data)
Base.eachindex(f::Field) = eachindex(f.data)
Base.getindex(f::Field, i::Integer) = f.data[i]
Base.setindex!(f::Field, v, i::Integer) = f.data[i] = v

# ------------------------------------------------------ similar-type math
#
# All elementwise arithmetic is implemented as explicit loops over the
# payload (never as broadcast on `.data`): a `HaloArray` payload (#10)
# would be silently demoted to a plain `Array` by broadcast, losing the
# exchange metadata. Loops over `eachindex`/`getindex` keep the payload
# type — and in global index space the elementwise result is identical.

for op in (:+, :-)
    @eval function Base.$(op)(a::Field{M, T, N, AX, L},
                              b::Field{M, T, N, AX2, L}) where {M, T, N, AX, AX2, L}
        out = similar_field(a)
        @inbounds for i in eachindex(out.data)
            out.data[i] = $(op)(a.data[i], b.data[i])
        end
        return out
    end
end
function Base.:-(a::Field{M, T, N, AX, L}) where {M, T, N, AX, L}
    out = similar_field(a)
    @inbounds for i in eachindex(out.data)
        out.data[i] = -a.data[i]
    end
    return out
end
for op in (:+, :-, :*, :/)
    @eval function Base.$(op)(a::Field{M, T, N, AX, L},
                              b::Number) where {M, T, N, AX, L}
        out = similar_field(a)
        @inbounds for i in eachindex(out.data)
            out.data[i] = $(op)(a.data[i], b)
        end
        return out
    end
    @eval function Base.$(op)(a::Number,
                              b::Field{M, T, N, AX, L}) where {M, T, N, AX, L}
        out = similar_field(b)
        @inbounds for i in eachindex(out.data)
            out.data[i] = $(op)(a, b.data[i])
        end
        return out
    end
end

"""    h1 * h2 -> Field

Elementwise product of same-location fields (e.g. the thickness flux
`F_e = ĥ_e u_e`): payload type preserved, mixed staggers `MethodError`.
"""
function Base.:*(a::Field{M, T, N, AX, L},
                 b::Field{M, T, N, AX2, L}) where {M, T, N, AX, AX2, L}
    out = similar_field(a)
    @inbounds for i in eachindex(out.data)
        out.data[i] = a.data[i] * b.data[i]
    end
    return out
end

# scalar-valued broadcast-style reductions are deliberately NOT provided;
# dot products between fields belong to the operator module (#9) where the
# quadrature weights (areas, lengths) make them physical inner products.

"""
    dot(a::Field, b::Field) -> Real

Unweighted elementwise inner product of same-location fields (Σ aᵢ bᵢ).
Physical (area-weighted) integrals belong to the operator module (#9).
"""
function dot(a::Field{M, T, N, AX, L}, b::Field{M, T, N, AX2, L}) where {M, T, N, AX, AX2, L}
    s = zero(T)
    @inbounds for i in eachindex(a.data)
        s += a.data[i] * b.data[i]
    end
    return s
end

# ------------------------------------------- stagger-checked state kernels
#
# TimeStepping (DESIGN.md §3.3 function-barrier contract) advances a
# prognostic state by elementwise linear combinations. States made of
# Fields (e.g. the SW dycore's SWState) plug into that protocol through
# the three kernels below: like-location Fields combine elementwise,
# mixed-location combinations are MethodErrors — the same staggering
# discipline as the arithmetic above, but writing into caller-provided
# storage (no allocation), which is what the stepper stages need.

"""
    lincomb!(out::Field, α, A::Field, β, B::Field) -> Field

`out .= α·A + β·B` elementwise. All three fields must share location,
mesh, and element type; mixed staggers are `MethodError`.
"""
function lincomb!(out::Field{M, T, N, AX, L}, α::Number,
                  A::Field{M, T, N, AX2, L}, β::Number,
                  B::Field{M, T, N, AX3, L}) where {M, T, N, AX, AX2, AX3, L}
    @inbounds for i in eachindex(out.data)
        out.data[i] = α * A.data[i] + β * B.data[i]
    end
    return out
end

"""
    axpy!(out::Field, α, X::Field) -> Field

`out .+= α·X` elementwise, same location constraints as [`lincomb!`](@ref).
"""
function axpy!(out::Field{M, T, N, AX, L}, α::Number,
               X::Field{M, T, N, AX2, L}) where {M, T, N, AX, AX2, L}
    @inbounds for i in eachindex(out.data)
        out.data[i] += α * X.data[i]
    end
    return out
end

"""
    scale!(f::Field, α) -> Field

`f .*= α` elementwise, in place (payload type preserved).
"""
function scale!(f::Field{M, T, N, AX, L}, α::Number) where {M, T, N, AX, L}
    @inbounds for i in eachindex(f.data)
        f.data[i] *= α
    end
    return f
end

"""    copyto!(dst::Field, src::Field) -> Field

Elementwise copy between same-location fields on the same mesh (payload
type preserved — a broadcast here would demote a `HaloArray` payload).
"""
function Base.copyto!(dst::Field{M, T, N, AX, L},
                      src::Field{M, T, N, AX2, L}) where {M, T, N, AX, AX2, L}
    @inbounds for i in eachindex(dst.data)
        dst.data[i] = src.data[i]
    end
    return dst
end

"""    copy(f::Field) -> Field

Fresh field with the same payload type and a copy of the values (halo
metadata preserved).
"""
Base.copy(f::Field{M, T, N, AX, L}) where {M, T, N, AX, L} =
    copyto!(similar_field(f), f)

"""
    rdivide!(out::Field, A::Field, B::Field) -> Field

`out .= A ./ B` elementwise, same location constraints as
[`lincomb!`](@ref) (payload type preserved — no broadcast). `out` may
alias `A` or `B` (single loop, per-index read-then-write).
"""
function rdivide!(out::Field{M, T, N, AX, L},
                  A::Field{M, T, N, AX2, L},
                  B::Field{M, T, N, AX3, L}) where {M, T, N, AX, AX2, AX3, L}
    @inbounds for i in eachindex(out.data)
        out.data[i] = A.data[i] / B.data[i]
    end
    return out
end

"""
    valid_indices(f::Field) -> index collection

The slots of `f` that hold meaningful values for reductions: all slots
for a plain array payload; only the region's owned slots for a
`HaloArray` payload (method extended in `Halo`). Integrals and other
reductions over fields must iterate these — summing a partitioned field
blindly would count ghosts (double-counting) and NaN-poisoned slots.
"""
valid_indices(f::Field) = valid_indices(f.data)
valid_indices(a::AbstractArray) = eachindex(a)

"""
    similar_state(f::Field...) -> Tuple

Uninitialized Fields with the same mesh/location/array-type as each
argument — the stepper's scratch-stage constructor for Field-based
states.
"""
similar_state(f::Field) = (similar_field(f),)
similar_state(f::Field, fs::Field...) =
    (similar_field(f), similar_state(fs...)...)

end # module Fields
