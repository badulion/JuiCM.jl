"""Fields: data at mesh locations.

TRiSK staggering (DESIGN.md §3.3): mass and tracers at cells, normal
velocities on edges, vorticity at vertices. Physics code never touches
`Field.data` directly — operators are the only access path (AGENTS.md
hard rule 3, MPI escape hatch).
"""
module Fields

export Field, CellField, EdgeField, VertexField, CellLoc, EdgeLoc, VertexLoc

"""Location of field data on the mesh (TRiSK staggering)."""
abstract type FieldLocation end
struct CellLoc <: FieldLocation end
struct EdgeLoc <: FieldLocation end
struct VertexLoc <: FieldLocation end

"""A field: data array + mesh + staggering location.

Type parameters: mesh type, element type, dimension, array type. The
array type is abstract so a future partitioned (MPI) array can replace
the plain Array without touching physics code (DESIGN.md §3.3).
"""
struct Field{M <: Any, T, N, AX <: AbstractArray, L <: FieldLocation}
    data::AX
    mesh::M
end

"""    CellField(mesh, FT=Float64) -> Field

Cell-centered scalar field (mass, tracers), one value per Voronoi cell.
"""
CellField(mesh, ::Type{FT} = Float64) where {FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, CellLoc}(zeros(FT, mesh.n_cells), mesh)

"""    EdgeField(mesh, FT=Float64) -> Field

Edge-centered scalar field (normal velocities), one value per mesh edge.
"""
EdgeField(mesh, ::Type{FT} = Float64) where {FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, EdgeLoc}(zeros(FT, mesh.n_edges), mesh)

"""    VertexField(mesh, FT=Float64) -> Field

Vertex-centered scalar field (vorticity), one value per mesh vertex
(triangle circumcenter).
"""
VertexField(mesh, ::Type{FT} = Float64) where {FT} =
    Field{typeof(mesh), FT, 1, Vector{FT}, VertexLoc}(zeros(FT, mesh.n_vertices), mesh)

Base.size(f::Field) = size(f.data)
Base.getindex(f::Field, i::Int) = f.data[i]
Base.setindex!(f::Field, v, i::Int) = f.data[i] = v

# similar-type arithmetic
for op in (:+, :-)
    @eval Base.$(op)(a::Field{M, T, N, AX, L}, b::Field{M, T, N, AX, L}) where {M, T, N, AX, L} =
        Field{M, T, N, AX, L}($(op)(a.data, b.data), a.mesh)
end
for op in (:+, :-, :*, :/)
    @eval Base.$(op)(a::Field{M, T, N, AX, L}, b::Number) where {M, T, N, AX, L} =
        Field{M, T, N, AX, L}($(op)(a.data, b), a.mesh)
    @eval Base.$(op)(a::Number, b::Field{M, T, N, AX, L}) where {M, T, N, AX, L} =
        Field{M, T, N, AX, L}($(op)(a, b.data), b.mesh)
end

end # module Fields
