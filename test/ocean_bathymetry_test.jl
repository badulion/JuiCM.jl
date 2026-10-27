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

    @testset "flat bathymetry field == flat-bottom path" begin
        m = generate_icosahedral_mesh(n_subdivisions = 3)
        grid = VerticalGrid(500.0, 6)
        Hf = bathymetry_flat(m, 500.0)
        a = resting_state(m, grid, Hf, patch)
        b = copy(a)
        for n in 1:6
            a = ocean_step!(a, grid, eos, (n - 1) * 3600.0, 3600.0;
                            regrid = true, nu_h = 1e5)
            b = ocean_step!(b, grid, eos, (n - 1) * 3600.0, 3600.0;
                            regrid = true, nu_h = 1e5, bathymetry = Hf)
        end
        @test umax(m, a) > 1e-3                     # non-trivial flow
        @test maximum(abs(a.u[k][e] - b.u[k][e]) for k in 1:6, e in 1:m.n_edges) <
              1e-9 * umax(m, a)
        @test maximum(abs(a.h[k][i] - b.h[k][i]) for k in 1:6, i in 1:m.n_cells) < 1e-9
        @test maximum(abs(a.T[k][i] - b.T[k][i]) for k in 1:6, i in 1:m.n_cells) < 1e-9
        @test maximum(abs(a.eta[i] - b.eta[i]) for i in 1:m.n_cells) < 1e-9
    end

    @testset "simple-z with bathymetry is rejected" begin
        m = generate_icosahedral_mesh(n_subdivisions = 2)
        gz = VerticalGrid(500.0, 4; mode = :simplez)
        Hf = bathymetry_flat(m, 500.0)
        st = resting_state(m, VerticalGrid(500.0, 4), Hf, (m, i, z) -> 20.0)
        @test_throws ArgumentError ocean_step!(st, gz, eos, 0.0, 3600.0;
                                               bathymetry = Hf)
    end

    # shelf (200 m) → deep ocean (1000 m) across λ = 0, ~1300 km wide
    m = generate_icosahedral_mesh(n_subdivisions = 4)
    K = 10
    grid = VerticalGrid(1000.0, K)
    H = bathymetry_shelf_break(m, 200.0, 1000.0; width = 0.2)

    @testset "resting homogeneous ocean over a slope stays at rest" begin
        # uniform density: the layer PGF is exactly −g∇η, and the fast
        # mode over b = −H sees a flat free surface — no force anywhere
        s = resting_state(m, grid, H, (m, i, z) -> 20.0)
        for n in 1:24
            s = ocean_step!(s, grid, eos, (n - 1) * 3600.0, 3600.0;
                            regrid = true, bathymetry = H)
        end
        @test umax(m, s) < 1e-12
        @test maximum(abs(s.eta[i]) for i in 1:m.n_cells) < 1e-10
        # the column follows the terrain: Σ_k h_k = H + η per cell
        @test maximum(abs(sum(s.h[k][i] for k in 1:K) - H[i] - s.eta[i])
                      for i in 1:m.n_cells) < 1e-9
    end


    @testset "z* PGF over a slope: stratified rest, 1 day (smoke)" begin
        # the 5-day bound (DESIGN.md §7, measured 2–3 mm/s) runs in
        # validation/x65_zstar_pgf_stratified.jl (issue #75)
        s = resting_state(m, grid, H, (m, i, z) -> 20.0 + 5.0 * z / 1000)
        v0 = volume(m, s)
        c0 = content(m, s, s.T)
        for n in 1:24
            s = ocean_step!(s, grid, eos, (n - 1) * 3600.0, 3600.0;
                            regrid = true, bathymetry = H)
        end
        @test umax(m, s) < 1e-2
        @test abs(volume(m, s) / v0 - 1) < 1e-12
        @test abs(content(m, s, s.T) / c0 - 1) < 1e-12
    end

    @testset "dense release on the shelf: conservation over a slope" begin
        s = resting_state(m, grid, H, patch)
        v0 = volume(m, s)
        c0 = content(m, s, s.T)
        for n in 1:24
            s = ocean_step!(s, grid, eos, (n - 1) * 1800.0, 1800.0;
                            regrid = true, nu_h = 1e5, bathymetry = H)
        end
        @test umax(m, s) > 1e-2
        @test abs(volume(m, s) / v0 - 1) < 1e-12
        @test abs(content(m, s, s.T) / c0 - 1) < 1e-12
        Tlo, Thi = extrema(s.T[k][i] for k in 1:K, i in 1:m.n_cells)
        @test Tlo >= 15.0 - 1e-10
        @test Thi <= 20.0 + 1e-10
        @test maximum(abs(sum(s.h[k][i] for k in 1:K) - H[i] - s.eta[i])
                      for i in 1:m.n_cells) < 1e-9
        # z* target: every layer is its fraction of the local column
        fr = grid.zstar_fractions
        @test maximum(abs(s.h[k][i] - (fr[k + 1] - fr[k]) * (H[i] + s.eta[i]))
                      for k in 1:K, i in 1:m.n_cells) < 1e-9
    end
end
