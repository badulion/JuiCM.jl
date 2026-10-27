# Moved unchanged from test/ (issue #75: the fast suite runs ≤ 30 s per
# testset; full-accuracy runs live in validation/). Reference data stay in
# test/reference_data/.

# Variable bathymetry H(x, y) in the ocean dycore (issue #65, DESIGN.md
# D7): the fast mode over topography b = −H, the per-column re-stretch,
# z* regrid target and floor, and the DESIGN.md §7 z* pressure-gradient
# check over a slope.

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using JuiCM.Ocean: ocean_step!
using JuiCM.OceanVertical: layer_thickness, bathymetry_flat,
                           bathymetry_shelf_break
using JuiCM.ShallowWater: williamson_constants
using Test

@testset "ocean bathymetry (#65)" begin
    eos = LinearEOS(alpha = 2.0e-3)

    # at-rest state on depth field Hf with layer temperature T(z_c)
    # (z_c the z* layer-centre height at η = 0)
    function resting_state(m, grid, Hf, Tfun)
        K = nlayers(grid)
        fr = grid.zstar_fractions
        h = layer_thickness(grid, CellField(i -> 0.0, m), Hf)
        T = LayeredCellField((i, k) -> Tfun(m, i, -(fr[k] + fr[k + 1]) / 2 * Hf[i]),
                             m, K)
        S = LayeredCellField((i, k) -> 35.0, m, K)
        return ocean_state(LayeredEdgeField(m, K), h, T, S)
    end
    umax(m, s) = maximum(abs(s.u[k][e]) for k in 1:nlayers(s.u), e in 1:m.n_edges)
    content(m, s, q) = sum(m.cell_area[i] * s.h[k][i] * q[k][i]
                           for k in 1:nlayers(s.h), i in 1:m.n_cells)
    volume(m, s) = sum(m.cell_area[i] * s.h[k][i]
                       for k in 1:nlayers(s.h), i in 1:m.n_cells)
    # dense patch: 5 K colder in |λ| < 0.3, |φ| < 0.3 (all layers)
    patch(m, i, z) = begin
        λ = atan(m.cell_xyz[2, i], m.cell_xyz[1, i])
        φ = asin(clamp(m.cell_xyz[3, i], -1.0, 1.0))
        abs(λ) < 0.3 && abs(φ) < 0.3 ? 15.0 : 20.0
    end



    # shelf (200 m) → deep ocean (1000 m) across λ = 0, ~1300 km wide
    m = generate_icosahedral_mesh(n_subdivisions = 4)
    K = 10
    grid = VerticalGrid(1000.0, K)
    H = bathymetry_shelf_break(m, 200.0, 1000.0; width = 0.2)


    @testset "z* PGF over a slope: stratified rest (DESIGN.md §7)" begin
        # linear stratification T = 20 + 5 K · z/1000 m (N ≈ 0.01 1/s,
        # internal speed √(g′H) ≈ 10 m/s): terrain-following z* layers
        # cross the isotherms on the slope, and the two-term PGF
        # (−∇p̄ − b∇z̄) does not cancel exactly there — the classic
        # σ-coordinate error. Measured (5-day run): max |u| 2–3 mm/s,
        # bounded (no growth). The bound below is 1 cm/s, 1e-3 of the
        # internal wave speed.
        s = resting_state(m, grid, H, (m, i, z) -> 20.0 + 5.0 * z / 1000)
        v0 = volume(m, s)
        c0 = content(m, s, s.T)
        Tlo0, Thi0 = extrema(s.T[k][i] for k in 1:K, i in 1:m.n_cells)
        u_t = Float64[]
        for n in 1:120
            s = ocean_step!(s, grid, eos, (n - 1) * 3600.0, 3600.0;
                            regrid = true, bathymetry = H)
            n % 24 == 0 && push!(u_t, umax(m, s))
        end
        @info "z* PGF error over the slope (max |u| per day)" u_t
        @test maximum(u_t) < 1e-2
        @test u_t[end] < 2 * u_t[1]                 # bounded, not growing
        # volume and heat content conserved to round-off, T bounded
        @test abs(volume(m, s) / v0 - 1) < 1e-12
        @test abs(content(m, s, s.T) / c0 - 1) < 1e-12
        Tlo, Thi = extrema(s.T[k][i] for k in 1:K, i in 1:m.n_cells)
        @test Tlo >= Tlo0 - 1e-10
        @test Thi <= Thi0 + 1e-10
    end

end
