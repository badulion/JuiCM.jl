# Smoke tests for the full-accuracy runs that live in validation/ (issue
# #75): the same code paths on a coarse mesh and a short time, checking that
# they run, conserve and stay bounded. The published-accuracy criteria are
# checked by `julia --project=test validation/runtests.jl`.

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using JuiCM.ShallowWater: williamson_constants
using JuiCM.ShallowWater.WilliamsonICs: case5_velocity, case5_height, case5_mountain
using LinearAlgebra
using Test

@testset "validation smoke tests (#75)" begin
    (; R, g) = williamson_constants()
    lonlat(x) = (atan(x[2], x[1]), asin(clamp(x[3], -1, 1)))

    @testset "V1 smoke: Williamson case 5, 1 day on 642 cells" begin
        m = generate_icosahedral_mesh(n_subdivisions = 3)
        h0 = CellField(i -> case5_height(lonlat(m.cell_xyz[:, i])...), m)
        b = CellField(i -> case5_mountain(lonlat(m.cell_xyz[:, i])...), m)
        u0 = EdgeField(m)
        for e in 1:m.n_edges
            λ, θ = lonlat(m.edge_x[:, e])
            u, v = case5_velocity(λ, θ)
            s = sqrt(max(1 - m.edge_x[3, e]^2, 1e-15))
            λ̂ = [-m.edge_x[2, e], m.edge_x[1, e], 0.0] ./ s
            θ̂ = [-m.edge_x[3, e] * m.edge_x[1, e] / s,
                  -m.edge_x[3, e] * m.edge_x[2, e] / s, s]
            u0[e] = u * dot(λ̂, m.edge_normal[:, e]) + v * dot(θ̂, m.edge_normal[:, e])
        end
        st = sw_state(h0, u0)
        M0 = sw_mass(st)
        st = sw_integrate!(st, 0.0, 86400.0, 48; topography = b)
        @test abs(sw_mass(st) / M0 - 1) < 1e-12
        @test all(isfinite(st.h[i]) && 3500 < st.h[i] < 6100 for i in 1:m.n_cells)
    end

    eos = LinearEOS(alpha = 2.0e-3)

    @testset "V3 smoke: walled lock exchange, 1 day on 642 cells" begin
        m = generate_icosahedral_mesh(n_subdivisions = 3)
        grid = VerticalGrid(200.0, 4)
        ch = walled_channel(m; n_long = 12, n_wide = 6)
        gp = 0.5
        st, λg = walled_lock_state(m, grid, eos, ch; g_prime = gp, lock_frac = 0.25)
        ΔT = gp / (g * eos.alpha)
        r = walled_lock_exchange!(st, grid, eos, ch, 20.0, ΔT, λg;
                                  n_steps = 24, Δt = 3600.0, nu_h = 1e5)
        @test r.xf[2][end] > r.xf[2][1]               # the front moves east
        @test maximum(abs(b / r.buoyancy[1] - 1) for b in r.buoyancy) < 1e-12
        Tlo, Thi = extrema(r.state.T[k][i] for k in 1:4, i in 1:m.n_cells)
        @test Tlo >= 20.0 - ΔT - 1e-10 && Thi <= 20.0 + 1e-10
    end

    @testset "z* regrid smoke: lock channel, 12 h on 642 cells" begin
        m = generate_icosahedral_mesh(n_subdivisions = 3)
        H, K, gp = 2e5, 6, 4e-4
        grid = VerticalGrid(H, K)
        ΔT = gp / (g * eos.alpha)
        st, _ = lock_channel_state(m, grid, eos; g_prime = gp)
        r = gravity_current_benchmark!(st, grid, eos, 20.0, ΔT; n_steps = 96,
                                       save_every = 24, frac = 0.1, regrid = true)
        s = r.state
        @test all(isfinite(s.u[k][e]) for k in 1:K, e in 1:m.n_edges)
        @test maximum(abs(b / r.buoyancy[1] - 1) for b in r.buoyancy) ≤ 1e-10
        tgt = layer_thickness(grid, s.eta)
        @test maximum(abs(s.h[k][i] - tgt[k][i]) for k in 1:K, i in 1:m.n_cells) < 1e-9 * H
    end
end
