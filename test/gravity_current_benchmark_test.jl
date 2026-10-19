# Gravity-current benchmark module (issue #40): IC builders, diagnostics,
# driver plumbing, and the TRiSK vector Laplacian the dissipation uses.
# Quantitative IC/diagnostic checks only — the front-speed (Froude)
# rung with long runs lives in the gravity-current validation test.

using JuiCM, Test, Random
using LinearAlgebra: dot

@testset "gravity current benchmark (#40)" begin
    m = generate_icosahedral_mesh(n_subdivisions = 4)   # config of record
    H0, K, gp = 2e5, 10, 4e-4
    grid = VerticalGrid(H0, K)
    eos = LinearEOS()
    (; g, R) = JuiCM.ShallowWater.williamson_constants()
    ΔT = gp / (g * eos.alpha)
    T_amb = 20.0
    d̄ = sum(m.d_e) / length(m.d_e)
    lonlat(i) = (atan(m.cell_xyz[2, i], m.cell_xyz[1, i]),
                 asin(clamp(m.cell_xyz[3, i], -1.0, 1.0)))
    in_lock(i, λl, φl) = (p = lonlat(i); abs(p[1]) < λl && abs(p[2]) < φl)
    Ā = 4π / m.n_cells

    @testset "lock_exchange_state default lock" begin
        st, gp_r = lock_exchange_state(m, grid, eos; g_prime = gp)
        @test gp_r == gp
        λl, φl = 2.5 * d̄, π / 8                      # documented defaults
        dense = [i for i in 1:m.n_cells if st.T[1][i] < T_amb - ΔT / 2]
        @test length(dense) > 0
        # the dense set is exactly the predicate set, full depth
        @test Set(dense) == Set(i for i in 1:m.n_cells if in_lock(i, λl, φl))
        @test all(st.T[k][i] ≈ T_amb - ΔT for k in 1:K, i in dense)
        # dense volume [m³]: discrete value exact; vs the continuous patch
        # R²·H0·(2λl)(2 sin φl) within half a boundary-cell ring
        # (centre-inclusion bias; ≈ −3.5 cells at nsub = 4)
        V = H0 * R^2 * sum(m.cell_area[i] for i in dense)
        Vc = H0 * R^2 * 2λl * 2sin(φl)
        perim = 2 * (2λl + 2φl)
        @test abs(V - Vc) <= H0 * R^2 * perim * d̄ / 2
        @test abs(V - Vc) <= 5 * H0 * R^2 * Ā
        # L [m] keyword is the physical half-length
        stL, _ = lock_exchange_state(m, grid, eos; g_prime = gp, L = λl * R)
        @test all(stL.T[1][i] == st.T[1][i] for i in 1:m.n_cells)
        @test_throws ArgumentError lock_exchange_state(m, grid, eos;
                                       g_prime = gp, L = 1e5, λ_lock = 0.1)
        @test_throws ArgumentError lock_exchange_state(m, grid, eos;
                                       g_prime = gp, λ_lock = 0.0)
    end

    sc, _ = lock_channel_state(m, grid, eos; g_prime = gp)
    λl, φw = 2.5 * d̄, 2.0 * d̄

    @testset "lock_channel_T agrees with lock_channel_state" begin
        Tc = lock_channel_T(m, λl, φw, ΔT, T_amb)
        @test all(sc.T[k][i] == Tc[i] for k in 1:K, i in 1:m.n_cells)
        # φ band applied (no cell outside |φ| < φw is dense)
        @test all(Tc[i] == T_amb for i in 1:m.n_cells if abs(lonlat(i)[2]) >= φw)
        @test all(Tc[i] == T_amb for i in 1:m.n_cells if abs(lonlat(i)[1]) >= λl)
        @test count(i -> Tc[i] < T_amb, 1:m.n_cells) > 0
    end

    @testset "front_position on the IC" begin
        for nbins in (200, 100)
            xf = front_position(sc.T, sc.h; T_amb = T_amb, ΔT = ΔT,
                                nbins = nbins)
            @test abs(xf - λl) <= 2π / nbins
        end
        # the threshold comes from the passed IC ΔT, not from the
        # field's current extrema: with 4ΔT the lock reads f = 0.25
        λc, f = dense_fraction_profile(sc.T, sc.h; T_amb = T_amb, ΔT = ΔT)
        @test maximum(f) ≈ 1.0
        @test minimum(f) ≈ 0.0 atol = 1e-12
        @test front_position(sc.T, sc.h; T_amb = T_amb, ΔT = 4ΔT) == 0.0
    end

    @testset "interface_monotone" begin
        @test interface_monotone(sc.T, sc.h; T_amb = T_amb, ΔT = ΔT)
        # hand-perturbed: a detached dense blob on the axis east of the lock
        st2, _ = lock_channel_state(m, grid, eos; g_prime = gp)
        i_b = argmin([let (λ, φ) = lonlat(i)
                          abs(λ - (λl + 6d̄)) + abs(φ) end for i in 1:m.n_cells])
        for k in 1:K
            st2.T[k][i_b] = T_amb - ΔT
        end
        @test !interface_monotone(st2.T, st2.h; T_amb = T_amb, ΔT = ΔT)
        # ... also on the west side
        st3, _ = lock_channel_state(m, grid, eos; g_prime = gp)
        i_w = argmin([let (λ, φ) = lonlat(i)
                          abs(λ + (λl + 6d̄)) + abs(φ) end for i in 1:m.n_cells])
        for k in 6:K                                  # bottom half only
            st3.T[k][i_w] = T_amb - ΔT
        end
        @test !interface_monotone(st3.T, st3.h; T_amb = T_amb, ΔT = ΔT)
    end

    @testset "buoyancy_content on the IC" begin
        B = buoyancy_content(sc.T, sc.h; T_amb = T_amb)
        A_lock = sum(m.cell_area[i] for i in 1:m.n_cells if in_lock(i, λl, φw))
        @test B ≈ ΔT * H0 * A_lock rtol = 1e-12
        @test buoyancy_content(sc.T, sc.h; T_amb = T_amb - ΔT) ≈
              -ΔT * H0 * (4π - A_lock) rtol = 1e-10
    end

    @testset "front_speed least squares" begin
        t = collect(0.0:450.0:9000.0)
        x = 0.1 .+ 3e-6 .* t
        @test front_speed(x, t) ≈ 3e-6 rtol = 1e-12
        x2 = copy(x); x2[t .> 4000] .= 0.0
        @test front_speed(x2, t; window = (0.0, 4000.0)) ≈ 3e-6 rtol = 1e-12
        @test_throws ArgumentError front_speed(x, t[1:end-1])
    end

    @testset "driver: config-of-record plumbing, 2 steps" begin
        r = gravity_current_benchmark!(sc, grid, eos, T_amb, ΔT; n_steps = 2)
        @test r.t == [0.0, 450.0, 900.0]
        @test length(r.xf) == 3 && length(r.buoyancy) == 3
        @test r.state isa OceanState && r.state !== sc
        @test all(isfinite, r.buoyancy)
        @test r.buoyancy[end] ≈ r.buoyancy[1] rtol = 1e-8   # FCT conserves
        @test abs(r.xf[end] - λl) <= 2 * 2π / 200            # no jump in 15 min
    end

    @testset "TRiSK laplacian" begin
        ip(a, b) = sum(m.l_e[e] * m.d_e[e] * a[e] * b[e] for e in 1:m.n_edges)
        Random.seed!(40)
        for _ in 1:3
            u = EdgeField(e -> randn(), m)
            L = laplacian(u)
            D = divergence(u); ζ = curl(u)
            q = ip(u, L)
            @test q <= 0
            # exact adjoint identity: ⟨u,∇²u⟩ = −Σ A_i D_i² − Σ A_v ζ_v²
            ref = -sum(m.cell_area[i] * D[i]^2 for i in 1:m.n_cells) -
                  sum(m.vertex_area[v] * ζ[v]^2 for v in 1:m.n_vertices)
            @test q ≈ ref rtol = 1e-10
        end
        # rigid rotation u = ẑ×r: continuous ∇²u = −2u (unit sphere)
        u = EdgeField(e -> dot([-m.edge_x[2, e], m.edge_x[1, e], 0.0],
                               m.edge_normal[:, e]), m)
        L = laplacian(u)
        @test ip(u, L) / ip(u, u) ≈ -2.0 atol = 2e-3      # −1.9994 at nsub 4
        @test maximum(abs(L[e]) for e in 1:m.n_edges) > 0.5  # not in kernel
        r = [L[e] / u[e] for e in 1:m.n_edges if abs(u[e]) > 0.3]
        @test all(-3.0 .< r .< -0.5)                      # documented scatter
    end
end
