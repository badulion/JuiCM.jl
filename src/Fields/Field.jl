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

export Field, CellField, EdgeField, VertexField,
       CellLoc, EdgeLoc, VertexLoc, FieldLocation,
       location, mesh_of, similar_field, length_of

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

for op in (:+, :-)
    @eval Base.$(op)(a::Field{M, T, N, AX, L}, b::Field{M, T, N, AX, L}) where {M, T, N, AX, L} =
        Field{M, T, N, AX, L}($(op)(a.data, b.data), a.mesh)
end
Base.:-(a::Field{M, T, N, AX, L}) where {M, T, N, AX, L} =
    Field{M, T, N, AX, L}(-a.data, a.mesh)
for op in (:+, :-, :*, :/)
    @eval Base.$(op)(a::Field{M, T, N, AX, L}, b::Number) where {M, T, N, AX, L} =
        Field{M, T, N, AX, L}($(op)(a.data, b), a.mesh)
    @eval Base.$(op)(a::Number, b::Field{M, T, N, AX, L}) where {M, T, N, AX, L} =
        Field{M, T, N, AX, L}($(op)(a, b.data), b.mesh)
end
# scalar-valued broadcast-style reductions are deliberately NOT provided;
# dot products between fields belong to the operator module (#9) where the
# quadrature weights (areas, lengths) make them physical inner products.

end # module Fields
