# Halo update machinery (DESIGN.md §3.2, issue #10).
#
# Single-node first, MPI-ready by construction (DESIGN.md D6). The mesh is
# split into contiguous regions (BFS over cell adjacency); each region owns
# its cells and holds GHOST COPIES of neighboring values at their global
# indices. A halo update fills a region's ghost slots with the owner
# regions' values through a pack → send → unpack pipeline that emulates
# message passing: on the MPI transition, only the "send" step changes
# (buffer copy → MPI_Isend/MPI_Irecv); index lists and unpacking are
# unchanged.
#
# MPI swap point (D6): everything below is defined so that a distributed
# implementation replaces `HaloArray.buf` with a rank-local array plus a
# global↔local index map. `halo_update!`'s pack/unpack index lists and the
# owner-computes rule are already rank-local; the only single-node artifact
# is that "send" is a local buffer copy and the source values live in the
# same `buf` (global addressing). Physics/dycore code never touches any of
# this — it calls `halo_update!` after computing on owned elements.

module Halo

using ..Voronoi: VoronoiSphereMesh
using ..Fields: Field, CellLoc, EdgeLoc, VertexLoc, location, mesh_of

export MeshPartition, partition_mesh, n_regions, region_of, owned_indices,
       ghost_cells, HaloInfo, build_halo_info, HaloArray, halo_array,
       HaloWorld, halo_world, halo_update!, halo_exchange!,
       partitioned_fields, owned_view

# ------------------------------------------------------------- partitioning

"""
    MeshPartition

Contiguous decomposition of a mesh's cells into regions, by breadth-first
search over cell-edge adjacency. Cells are the primary partition unit
(owner of a cell owns the primal region); edge and vertex ownership is
derived by rule below, so every mesh element has exactly one owner region:

- edge `e` is owned by the region of `edge_cells[1, e]`;
- vertex `v` is owned by the region of the first cell in `vertex_cells[v]`.

Derived per region (2-layer halo, sufficient for 3rd-order stencils):
ghost cells = non-owned cells within 2 edge-adjacency steps of an owned
cell; ghost edges = non-owned edges incident to any owned or ghost cell;
ghost vertices = non-owned vertices of any owned or ghost edge.
"""
struct MeshPartition
    n_cells::Int
    owner::Vector{Int}              # cell → region, 1-based
    owned_cells::Vector{Vector{Int}}  # region → owned cell indices (sorted)
    cell_neighbors::Vector{Vector{Int}}  # cell → edge-adjacent cells
end

n_regions(p::MeshPartition) = length(p.owned_cells)
region_of(p::MeshPartition, c::Int) = p.owner[c]
owned_indices(p::MeshPartition, r::Int) = p.owned_cells[r]

"""
    partition_mesh(mesh, n_regions) -> MeshPartition

Split the mesh's cells into `n_regions` contiguous regions by BFS growth
from `n_regions` evenly-spaced seed cells. Deterministic.
"""
function partition_mesh(mesh::VoronoiSphereMesh, n_regions::Int)
    1 ≤ n_regions ≤ mesh.n_cells || throw(ArgumentError("n_regions out of range"))
    # cell adjacency via shared edges
    nbrs = [Int[] for _ in 1:mesh.n_cells]
    for e in 1:mesh.n_edges
        i, j = mesh.edge_cells[1, e], mesh.edge_cells[2, e]
        i != j || continue        # skip boundary/self edges (none on a sphere)
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end
    owner = fill(0, mesh.n_cells)
    # deterministic seeds: greedy — pick the first unassigned cell, BFS out,
    # repeat; targets per region keep sizes balanced
    target = cld(mesh.n_cells, n_regions)
    r = 0
    while r < n_regions
        seed = findfirst(==(0), owner)
        seed === nothing && break
        r += 1
        queue = [seed]
        owner[seed] = r
        size = 1
        head = 1
        while head ≤ length(queue) && size < target && any(==(0), owner)
            c = queue[head]; head += 1
            for n in nbrs[c]
                if owner[n] == 0
                    owner[n] = r
                    push!(queue, n)
                    size += 1
                    size >= target && break
                end
            end
        end
    end
    # leftovers (possible when n_regions does not divide cells): attach to
    # the region of any neighbor
    for c in 1:mesh.n_cells
        if owner[c] == 0
            owner[c] = owner[first(nbrs[c])]
        end
    end
    owned = [Int[] for _ in 1:r]
    for c in 1:mesh.n_cells
        push!(owned[owner[c]], c)
    end
    return MeshPartition(mesh.n_cells, owner, owned, nbrs)
end

# ------------------------------------------------------------ ghost listing

"""
    ghost_cells(p::MeshPartition, mesh, r) -> Vector{Int}

Cells within 2 edge-adjacency steps of region `r`'s owned cells that `r`
does not own (sorted, unique). The 2-layer depth supports 3rd-order
stencils; all current TRiSK operators touch at most 1 layer.
"""
function ghost_cells(p::MeshPartition, mesh::VoronoiSphereMesh, r::Int)
    layer1 = falses(p.n_cells)
    for c in p.owned_cells[r], n in p.cell_neighbors[c]
        layer1[n] = true
    end
    layer2 = copy(layer1)
    for c in 1:p.n_cells
        layer1[c] || continue
        for n in p.cell_neighbors[c]
            layer2[n] = true
        end
    end
    return Int[c for c in 1:p.n_cells if layer2[c] && p.owner[c] != r]
end

# ---------------------------------------------------------------- halo info

"""
    HaloInfo

Precomputed exchange plan for one stagger (cells, edges, or vertices) on a
partitioned mesh. For each region `r` and each source region `s ≠ r`,
`send[r][s]` lists the global indices `r` owns that `s` needs as ghosts,
and `recv[r][s]` lists the same indices from `s`'s perspective — i.e.
`recv[r][s] == send[s][r]` by construction. Owner-computes: a region never
writes another region's owned slots.
"""
struct HaloInfo
    n_global::Int                   # ncells / nedges / nvertices for the stagger
    owner::Vector{Int}              # global index → owner region
    owned::Vector{Vector{Int}}      # region → owned global indices
    send::Vector{Vector{Vector{Int}}}  # send[r][s]: indices r owns that s needs
end

"""
    build_halo_info(p::MeshPartition, mesh, ::CellLoc) -> HaloInfo

Cell-stagger exchange plan: region `r` needs (2-layer) ghost cells from
their owner regions; region `r` sends each neighbor region the ghost cells
that region needs from `r`.
"""
function build_halo_info(p::MeshPartition, mesh::VoronoiSphereMesh, ::CellLoc)
    R = n_regions(p)
    need = [Set{Int}() for _ in 1:R]
    for r in 1:R
        union!(need[r], ghost_cells(p, mesh, r))
    end
    send = [[Int[] for _ in 1:R] for _ in 1:R]
    for r in 1:R, s in 1:R
        r == s && continue
        # s needs from r: owned by r, in s's ghost set
        send[r][s] = sort!(collect(intersect(need[s], Set(p.owned_cells[r]))))
    end
    owner = similar(p.owner)
    owner .= p.owner
    return HaloInfo(p.n_cells, owner, [copy(p.owned_cells[r]) for r in 1:R], send)
end

"""
    build_halo_info(p::MeshPartition, mesh, ::EdgeLoc) -> HaloInfo

Edge-stagger exchange plan. Edge ownership rule: `edge_cells[1, e]`'s
region. A region's ghost edges are the non-owned edges incident to its
owned or ghost cells (1 edge-layer — tangent/normal operators on an owned
edge only read that edge's value; operators on owned cells read edges of
owned cells, all owned by construction).
"""
function build_halo_info(p::MeshPartition, mesh::VoronoiSphereMesh, ::EdgeLoc)
    R = n_regions(p)
    eowner = [p.owner[mesh.edge_cells[1, e]] for e in 1:mesh.n_edges]
    need = [Set{Int}() for _ in 1:R]
    for r in 1:R
        gcells = ghost_cells(p, mesh, r)
        for c in Iterators.flatten((p.owned_cells[r], gcells)), e in mesh.cell_edges[c]
            eowner[e] != r && push!(need[r], e)
        end
    end
    send = [[Int[] for _ in 1:R] for _ in 1:R]
    for r in 1:R, s in 1:R
        r == s && continue
        send[r][s] = sort!(collect(intersect(need[s], Set(e for e in 1:mesh.n_edges if eowner[e] == r))))
    end
    return HaloInfo(mesh.n_edges, eowner,
                    [Int[e for e in 1:mesh.n_edges if eowner[e] == r] for r in 1:R], send)
end

"""
    build_halo_info(p::MeshPartition, mesh, ::VertexLoc) -> HaloInfo

Vertex-stagger exchange plan. Vertex ownership rule: the region of the
first cell in `vertex_cells[v]`. Ghost vertices: non-owned vertices of any
edge in a region's owned-or-ghost edge set (curl on an owned edge reads
that edge's two end vertices; with cell-ghosts at 2 layers, edges are at
1, so vertex needs follow).
"""
function build_halo_info(p::MeshPartition, mesh::VoronoiSphereMesh, ::VertexLoc)
    R = n_regions(p)
    vowner = [p.owner[mesh.vertex_cells[v][1]] for v in 1:mesh.n_vertices]
    # edges a region may touch: edges of owned or ghost cells
    touched = [Set{Int}() for _ in 1:R]
    for r in 1:R
        for c in Iterators.flatten((p.owned_cells[r], ghost_cells(p, mesh, r)))
            union!(touched[r], mesh.cell_edges[c])
        end
    end
    need = [Set{Int}() for _ in 1:R]
    for r in 1:R, e in touched[r], v in (mesh.edge_vertex_pos[e], mesh.edge_vertex_neg[e])
        vowner[v] != r && push!(need[r], v)
    end
    send = [[Int[] for _ in 1:R] for _ in 1:R]
    for r in 1:R, s in 1:R
        r == s && continue
        send[r][s] = sort!(collect(intersect(need[s], Set(v for v in 1:mesh.n_vertices if vowner[v] == r))))
    end
    return HaloInfo(mesh.n_vertices, vowner,
                    [Int[v for v in 1:mesh.n_vertices if vowner[v] == r] for r in 1:R], send)
end

# --------------------------------------------------------------- HaloArray

"""
    HaloArray{T} <: AbstractVector{T}

One region's field payload: a flat array in **global index space** where
the region's owned slots are authoritative and non-owned (ghost) slots are
scratch copies refreshed by [`halo_update!`](@ref). Global indexing keeps
every existing TRiSK operator working unchanged: operators gather over
global indices, and after `halo_update!` the ghost slots hold
owner-computed values.

**MPI swap point (D6):** in the distributed version each region's `buf`
becomes a rank-local array plus a global↔local map; the pack/unpack index
lists in `HaloInfo` and the owner-computes rule are unchanged, and only
the "send" step (a local cross-array copy here) becomes real message
passing. A collection of one `HaloArray` per region lives in a
[`HaloWorld`](@ref).
"""
mutable struct HaloArray{T} <: AbstractVector{T}
    buf::Vector{T}
    info::HaloInfo
    region::Int                    # the region this array represents
end

Base.size(a::HaloArray) = (a.info.n_global,)
Base.getindex(a::HaloArray, i::Integer) = a.buf[i]
Base.setindex!(a::HaloArray, v, i::Integer) = (a.buf[i] = v)
Base.similar(a::HaloArray) = HaloArray(similar(a.buf), a.info, a.region)
Base.similar(a::HaloArray, ::Type{T}) where {T} =
    HaloArray(similar(a.buf, T), a.info, a.region)

"""
    halo_array(info::HaloInfo, region; T=Float64, init=nothing) -> HaloArray

Allocate a `HaloArray` for `region` (zeros, or `init(g)` per global index).
Owned slots are authoritative; ghost slots start as copies of `init` (or
zero) and must be refreshed with [`halo_update!`](@ref) before operators
rely on them in a partitioned workflow.
"""
function halo_array(info::HaloInfo, region::Int; T = Float64, init = nothing)
    buf = init === nothing ? zeros(T, info.n_global) : T[init(g) for g in 1:info.n_global]
    return HaloArray(buf, info, region)
end

"""
    HaloWorld{T}

One `HaloArray` per region plus the shared exchange plan — the full
single-node emulation of a distributed field. Construct with
[`halo_world`](@ref); refresh ghosts with [`halo_update!`](@ref).
"""
struct HaloWorld{T}
    info::HaloInfo
    arrays::Vector{HaloArray{T}}
end

Base.length(w::HaloWorld) = length(w.arrays)
Base.getindex(w::HaloWorld, r::Integer) = w.arrays[r]

"""
    halo_world(info::HaloInfo; T=Float64, init=nothing, poison_ghosts=true)
        -> HaloWorld

Allocate one `HaloArray` per region. `init(g)` seeds every slot; with
`poison_ghosts = true` (the realistic partitioned state) each region's
non-owned slots are then overwritten with `NaN`. Only the 2-layer ghost
set is refreshed by `halo_update!` — deeper non-owned slots stay `NaN`,
which is intentional: any operator read beyond the exchanged stencil
depth fails loudly instead of silently using a stale copy.
"""
function halo_world(info::HaloInfo; T = Float64, init = nothing,
                    poison_ghosts = true)
    R = length(info.owned)
    arrays = Vector{HaloArray{T}}(undef, R)
    for r in 1:R
        a = halo_array(info, r; T = T, init = init)
        if poison_ghosts
            for g in 1:info.n_global
                info.owner[g] != r && (a.buf[g] = NaN)
            end
        end
        arrays[r] = a
    end
    return HaloWorld(info, arrays)
end

"""
    halo_exchange!(arrays::Vector{<:HaloArray}) -> Vector

Exchange ghosts **within a collection** of per-region arrays sharing one
`HaloInfo`: every region's ghost slots are refreshed from the current
owned values of the other arrays in the collection. This is the
primitive the dycore uses per SSPRK stage — stage buffers are fresh
`HaloArray`s, not the model's world arrays, so the exchange must target
exactly the collection about to be read (owner-computes: every value
read was written by its owner in the same stage).
"""
function halo_exchange!(arrays::Vector{A}) where {A <: HaloArray}
    isempty(arrays) && return arrays
    info = arrays[1].info
    R = length(arrays)
    for my in 1:R, s in 1:R
        s == my && continue
        recvlist = info.send[s][my]    # what s owns that I need
        isempty(recvlist) && continue
        dst = arrays[my].buf
        src = arrays[s].buf
        for g in recvlist
            dst[g] = src[g]
        end
    end
    return arrays
end

"""
    halo_update!(w::HaloWorld[, r]) -> HaloWorld

Refresh ghost slots through the emulated pack → send → unpack pipeline.
Without `r`, all regions are updated.

1. **pack**: region `s` copies its owned values needed by `my` into a send
   buffer (index list `send[s][my]`);
2. **send**: on one node this is the buffer handoff (the MPI swap point —
   becomes `MPI_Isend`/`MPI_Irecv` with the same buffers and index lists);
3. **unpack**: region `my` writes the received values into its ghost slots
   at their global indices (the same list, mirrored).

Owner-computes: a region only ever writes its own (non-owned, ghost)
slots, and only from values the owner produced. After the update, a
ghost-slot read returns exactly the value the owner region last wrote to
its owned slot.
"""
function halo_update!(w::HaloWorld, r::Integer)
    info = w.info
    my = Int(r)
    R = length(w.arrays)
    T = eltype(w.arrays[my])
    for s in 1:R
        s == my && continue
        recvlist = info.send[s][my]    # what s owns that I need
        isempty(recvlist) && continue
        # pack on s's side (in MPI: s does this with its local buffer)
        sbuf = Vector{T}(undef, length(recvlist))
        sbuf_src = w.arrays[s].buf
        for (k, g) in enumerate(recvlist)
            sbuf[k] = sbuf_src[g]
        end
        # send (single-node: buffer handoff; MPI swap point)
        rbuf = sbuf
        # unpack into my ghost slots
        dst = w.arrays[my].buf
        for (k, g) in enumerate(recvlist)
            dst[g] = rbuf[k]
        end
    end
    return w
end

halo_update!(w::HaloWorld) = (for r in 1:length(w.arrays)
                                  halo_update!(w, r)
                              end; w)

"""
    halo_exchange!(fields::Vector{<:Field}) -> Vector

Field-level ghost exchange: extracts the `HaloArray` payloads and calls
the array-level [`halo_exchange!`](@ref) on them (all fields must share
one `HaloInfo`). This is the form the dycore uses per SSPRK stage —
stage buffers are fresh Fields wrapping fresh `HaloArray`s, so the
exchange must target exactly the collection about to be read
(owner-computes: every value read was written by its owner in the same
stage).
"""
function halo_exchange!(fields::Vector{<:Field})
    isempty(fields) && return fields
    arrays = [f.data for f in fields]
    halo_exchange!(arrays)
    return fields
end

# ------------------------------------------------------- Field integration
"""
    partitioned_fields(f::Field, info::HaloInfo) -> Vector{Field}

One field per region, each wrapping a fresh `HaloArray` copy of `f`'s
values (non-owned slots poisoned to `NaN` — call `halo_update!` on the
corresponding [`HaloWorld`](@ref) before relying on them).
"""
function partitioned_fields(f::Field{M, T, N, AX, L},
                            info::HaloInfo) where {M, T, N, AX, L}
    R = length(info.owned)
    fields = Vector{Field{M, T, N, HaloArray{T}, L}}(undef, R)
    for r in 1:R
        a = HaloArray(copy(f.data), info, r)
        for g in 1:info.n_global
            info.owner[g] != r && (a.buf[g] = NaN)
        end
        fields[r] = Field{M, T, N, HaloArray{T}, L}(a, f.mesh)
    end
    return fields
end

"""
    owned_view(a::HaloArray) -> SubArray

View of the region's owned slots (for reductions and output).
"""
owned_view(a::HaloArray) = view(a.buf, a.info.owned[a.region])

# Field integration: partitioned payload support for the Field kernels.
# In global index space the elementwise kernels are correct as-is; only
# reductions must skip non-owned slots (ghosts are stale copies, deeper
# slots NaN-poisoned).

"""
    valid_indices(a::HaloArray) -> view

The region's owned slots — the reduction domain for a partitioned
payload (see [`Fields.valid_indices`](@ref)).
"""
valid_indices(a::HaloArray) = owned_view(a)

end # module Halo
