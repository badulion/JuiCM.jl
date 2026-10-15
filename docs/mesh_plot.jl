# Mesh visualization (issue #4): primal Voronoi cells + dual triangulation.
#
# Run from the repo root:
#   julia --project=. docs/mesh_plot.jl
#
# Writes docs/img/mesh_l{level}.png. Makie is a dev-time-only dependency
# (DESIGN.md §6); install it into a temporary environment if not present:
#   julia --project=. -e 'using Pkg; Pkg.add("Makie"); Pkg.add("CairoMakie")'

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using CairoMakie

function lonlat(x)
    θ = acos(clamp(x[3], -1.0, 1.0))          # colatitude
    return rad2deg(atan(x[2], x[1])), rad2deg(π / 2 - θ)
end

function plot_mesh(level::Int; lloyd_iterations::Int = 0)
    mesh = generate_icosahedral_mesh(n_subdivisions=level,
                                     lloyd_iterations=lloyd_iterations)

    fig = Figure(size = (1400, 700))
    # primal cells (Voronoi): cell centers + cell edges
    ax1 = Axis(fig[1, 1]; title = "primal Voronoi cells (level $level)",
               xlabel = "lon [deg]", ylabel = "lat [deg]", aspect = 2)
    for c in 1:mesh.n_cells
        pts = [Point2f(lonlat(mesh.vertex_xyz[:, v]))
               for v in mesh.cell_corners[c]]
        poly!(ax1, pts; color = :transparent, strokecolor = :black, strokewidth = 0.6)
        clon, clat = lonlat(mesh.cell_xyz[:, c])
        scatter!(ax1, [Point2f(clon, clat)]; color = :red, markersize = 3)
    end
    # dual triangulation: vertex positions + edges between adjacent vertices
    ax2 = Axis(fig[1, 2]; title = "dual triangulation (level $level)",
               xlabel = "lon [deg]", ylabel = "lat [deg]", aspect = 2)
    for v in 1:mesh.n_vertices
        pts = [Point2f(lonlat(mesh.cell_xyz[:, c]))
               for c in mesh.vertex_cells[v]]
        poly!(ax2, pts; color = :transparent, strokecolor = :blue, strokewidth = 0.6)
        vlon, vlat = lonlat(mesh.vertex_xyz[:, v])
        scatter!(ax2, [Point2f(vlon, vlat)]; color = :green, markersize = 3)
    end
    # highlight pentagons
    for c in mesh.pentagon_cells
        pts = [Point2f(lonlat(mesh.vertex_xyz[:, v]))
               for v in mesh.cell_corners[c]]
        poly!(ax1, pts; color = (:red, 0.35))
    end

    out = joinpath(@__DIR__, "img", "mesh_l$level.png")
    save(out, fig)
    println("wrote $out")
    return out
end

for lev in (0, 2)
    plot_mesh(lev)
end
