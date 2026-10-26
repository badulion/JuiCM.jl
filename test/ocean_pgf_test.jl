# Selectable pressure-gradient schemes (issue #73): :twoterm (#30),
# :adcroft2008 (Adcroft, Hallberg & Harrison 2008, finite volume),
# :mom6 (the same with MOM6's PLM density reconstruction) and :roms
# (Shchepetkin & McWilliams 2003, ROMS prsgrd32).

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using JuiCM.ShallowWater: williamson_constants
using JuiCM.Ocean: ocean_step!
using Test

@testset "ocean pressure-gradient schemes (#73)" begin
    (; g, R) = williamson_constants()
    m = generate_icosahedral_mesh(n_subdivisions = 3)
    K = 8
    ρ0 = 1000.0
    # sloping bathymetry and a tilted free surface; bottom-refined z*
    H = CellField(i -> 3000.0 + 800.0 * m.cell_xyz[3, i], m)
    grid = VerticalGrid(3000.0, K; stretching = u -> 1 - (1 - u)^2)
    interfaces(eta, h) = [CellField(i -> eta[i] - sum(h[j][i] for j in 1:(k - 1); init = 0.0), m)
                          for k in 1:(K + 1)]
    @test PGF_SCHEMES == (:twoterm, :adcroft2008, :mom6, :roms)
    @test_throws ArgumentError pressure_gradient(:nope, layer_thickness(grid,
                     CellField(i -> 0.0, m), H), nothing, nothing, ρ0)

    @testset "uniform density: exactly −g∇η" begin
        eta = CellField(i -> 0.5 * m.cell_xyz[1, i] + 0.2 * m.cell_xyz[2, i], m)
        h = layer_thickness(grid, eta, H)
        z = interfaces(eta, h)
        ρ = LayeredCellField((i, k) -> ρ0, m, K)
        gη(e) = g * (eta[m.edge_cells[2, e]] - eta[m.edge_cells[1, e]]) / (R * m.d_e[e])
        scale = maximum(abs(gη(e)) for e in 1:m.n_edges)
        for sch in (:adcroft2008, :mom6, :roms)
            pg = pressure_gradient(sch, h, z, ρ, ρ0)
            @test maximum(abs(pg[k][e] + gη(e)) for k in 1:K, e in 1:m.n_edges) <
                  1e-10 * scale
        end
    end

    @testset "linear stratification over a slope: rest is exact (mom6, roms)" begin
        # ρ(z) = ρ0 + γ·(−z) with exact layer means, flat surface: the
        # continuum PGF is zero. :mom6 (PLM reconstruction) and :roms
        # (cubic fits) are exact for linear profiles; :adcroft2008
        # (layer-constant density) and :twoterm are not (measured
        # 5.2e-8 m/s² both, against 1.7e-18 for :mom6/:roms).
        eta = CellField(i -> 0.0, m)
        h = layer_thickness(grid, eta, H)
        z = interfaces(eta, h)
        γ = 1e-3
        ρ = LayeredCellField((i, k) -> ρ0 + γ * (-(z[k][i] + z[k + 1][i]) / 2), m, K)
        scale = g * γ / ρ0 * 800.0          # ≈ |b ∇z| of one layer, the cancelled size
        err(sch) = (pg = pressure_gradient(sch, h, z, ρ, ρ0);
                    maximum(abs(pg[k][e]) for k in 1:K, e in 1:m.n_edges))
        @test err(:mom6) < 1e-12 * scale
        @test err(:roms) < 1e-12 * scale
        @test err(:adcroft2008) > 1e-6 * scale   # the documented limitation
    end

    @testset "stratified rest over a slope in ocean_step! (dynamic)" begin
        # linearly stratified ocean at rest over a shelf slope, exact
        # layer means: the continuum stays at rest. Measured at the #45
        # V4 scale (240 km cells, 48 layers, 2.3 days): :twoterm and
        # :adcroft2008 3.9 mm/s, :mom6 and :roms 2e-10 m/s; for a tanh
        # interface (150 m half-width) 0.31 / 0.31 / 0.31 / 0.077 m/s.
        eos = LinearEOS(alpha = 2.0e-3)
        mz = generate_icosahedral_mesh(n_subdivisions = 3)
        Kz = 10
        Hz = CellField(i -> 2000.0 + 1000.0 * mz.cell_xyz[3, i], mz)
        gz = VerticalGrid(2000.0, Kz)
        hz = layer_thickness(gz, CellField(i -> 0.0, mz), Hz)
        zc(i, k) = -(sum(hz[j][i] for j in 1:(k - 1); init = 0.0) + hz[k][i] / 2)
        T = LayeredCellField((i, k) -> 20.0 + 5.0 * zc(i, k) / 3000.0, mz, Kz)
        S = LayeredCellField((i, k) -> 35.0, mz, Kz)
        umax = Dict{Symbol, Float64}()
        for sch in PGF_SCHEMES
            st = ocean_state(LayeredEdgeField(mz, Kz), hz, T, S)
            for n in 1:12
                st = ocean_step!(st, gz, eos, (n - 1) * 3600.0, 3600.0;
                                 regrid = true, bathymetry = Hz, pgf = sch)
            end
            umax[sch] = maximum(abs(st.u[k][e]) for k in 1:Kz, e in 1:mz.n_edges)
        end
        @info "spurious |u| over the slope after 12 h" umax
        @test umax[:mom6] < 1e-8
        @test umax[:roms] < 1e-8
        @test umax[:twoterm] > 1e4 * umax[:mom6]       # the σ-coordinate error
    end

    @testset "every scheme runs in ocean_step! and conserves" begin
        eos = LinearEOS(alpha = 2.0e-3)
        mz = generate_icosahedral_mesh(n_subdivisions = 3)
        Hz = CellField(i -> 1000.0 + 300.0 * mz.cell_xyz[3, i], mz)
        gz = VerticalGrid(1000.0, 6)
        hz = layer_thickness(gz, CellField(i -> 0.0, mz), Hz)
        lon(i) = atan(mz.cell_xyz[2, i], mz.cell_xyz[1, i])
        T = LayeredCellField((i, k) -> abs(lon(i)) < 0.3 ? 15.0 : 20.0, mz, 6)
        S = LayeredCellField((i, k) -> 35.0, mz, 6)
        vol(s) = sum(mz.cell_area[i] * s.h[k][i] for k in 1:6, i in 1:mz.n_cells)
        heat(s) = sum(mz.cell_area[i] * s.h[k][i] * s.T[k][i] for k in 1:6, i in 1:mz.n_cells)
        @test_throws ArgumentError ocean_step!(ocean_state(LayeredEdgeField(mz, 6), hz, T, S),
                                               gz, eos, 0.0, 1800.0; pgf = :nope)
        for sch in PGF_SCHEMES
            st = ocean_state(LayeredEdgeField(mz, 6), hz, T, S)
            v0 = vol(st); c0 = heat(st)
            for n in 1:6
                st = ocean_step!(st, gz, eos, (n - 1) * 1800.0, 1800.0;
                                 regrid = true, bathymetry = Hz, nu_h = 1e5, pgf = sch)
            end
            @test maximum(abs(st.u[k][e]) for k in 1:6, e in 1:mz.n_edges) > 1e-2
            @test abs(vol(st) / v0 - 1) < 1e-12
            @test abs(heat(st) / c0 - 1) < 1e-12
        end
    end
end
