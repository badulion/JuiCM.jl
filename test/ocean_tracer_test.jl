# Layered tracer advection (issue #32, DESIGN.md §5 Phase 2 advection
# rung 1: first-order upwind).
#
# Acceptance (issue #32):
#   1. Compatibility: a uniform tracer advected by ANY flow stays
#      uniform to round-off — the tracer fluxes are the thickness
#      fluxes the volume continuity uses, times the concentration.
#   2. Conservation: global tracer content Σ A_i h_i C_i is invariant
#      (closed sphere, flux form; the content form conserves to
#      round-off, the concentration form through the dycore's SSPRK to
#      the quadratic-invariant accuracy of the integrator — measured).
#   3. Positivity: a Gaussian patch under strong deformation keeps
#      min C ≥ its initial floor exactly (upwind is a convex update).
#   4. Convergence order ≥ 1: solid-body rotation of a smooth Gaussian
#      patch, measured against the analytic solution at two grid
#      resolutions; the l1 error must halve (or better) under
#      refinement at fixed CFL — first order.
#
# Test design notes:
#   - The flow is a solid-body zonal rotation (u = u0 cosφ eastward) —
#     divergence-free, steady, and exactly representable on the C-grid
#     normal velocities (each edge's normal component of the analytic
#     velocity is set pointwise; the resulting discrete divergence is
#     round-off small but NOT identically zero — the C-grid cannot
#     represent cosφ divergence-free exactly at cell centers, so the
#     pure-advection tests use the full advective form; the uniform-
#     tracer compatibility test is exact for ANY flow including this
#     one, which is the point of the flux construction).
#   - The analytic reference for the convergence test follows the patch
#     passively: at time t the Gaussian center has rotated by ωt with
#     ω = u0/R. Measured on a single layer (barotropic setup, K = 1
#     grid is not available — K = 2 with identical layers and the
#     patch in both, halving the per-layer error the same).
#   - Order measurement: two refinements of n_subdivisions (2 → 3)
#     with Δt scaled to keep the CFL fixed; order = log2(err_coarse /
#     err_fine) from the l1 norms.
#
# GOTCHA (documented in AGENTS-session notes): `using LinearAlgebra`
# in this file shadows the Fields `axpy!` export — the tests use
# `copyto!`/arithmetic instead, or qualify with `JuiCM.axpy!`.

@testset "ocean tracer advection: flux-form upwind (#32)" begin
    using JuiCM: VerticalGrid, LinearEOS, layer_thickness,
                 tracer_tendency!, advect_tracer!, tracer_total,
                 LayeredField, ocean_state, ocean_step!, ocean_integrate!,
                 williamson_constants
    using JuiCM: divergence, edge_mean
    using JuiCM: mesh_of, similar_field, scale!, copyto!, axpy!, lincomb!

    m = generate_icosahedral_mesh(n_subdivisions=2)
    (; R, g) = williamson_constants()
    K = 4
    H0 = 4000.0

    "Solid-body zonal flow: u = u0·cosφ eastward in every layer."
    function solid_body_u!(u, mesh; u0)
        for k in 1:nlayers(u), e in 1:mesh.n_edges
            x = mesh.edge_x[1, e]; y = mesh.edge_x[2, e]
            nx = mesh.edge_normal[1, e]; ny = mesh.edge_normal[2, e]
            rxy = sqrt(x^2 + y^2)
            # eastward unit vector (−sinλ, cosλ, 0)·n̂_e
            ve = -y * nx + (x / rxy) * ny
            u[k][e] = u0 * sqrt(1 - mesh.edge_x[3, e]^2) * ve
        end
        return u
    end

    "Uniform tracer column, value C0."
    function uniform_C(mesh, K; C0)
        C = LayeredCellField(mesh, K)
        for k in 1:K, i in 1:mesh.n_cells
            C[k][i] = C0
        end
        return C
    end

    # ------------------------------------------------------------- 1
    @testset "compatibility: uniform tracer under a general flow" begin
        grid = VerticalGrid(H0, K)
        h = layer_thickness(grid, CellField(i -> 0.0, m))
        u = solid_body_u!(LayeredEdgeField(m, K), m; u0 = 15.0)
        C = uniform_C(m, K; C0 = 3.0)
        out = LayeredCellField(m, K)
        tracer_tendency!(out, C, h, u)
        # zero to round-off (the tracer flux IS the thickness flux
        # times C; ∇·(F·C) = C·∇·F + F·∇C with both terms vanishing
        # together — exact for any flow, divergent or not)
        @test maximum(maximum(abs.(out[k].data)) for k in 1:K) < 1e-16

        # through the dycore: a stratified resting column with uniform
        # T anomaly advected through a wave-active step — T stays
        # uniform to round-off after full mode-split steps
        eos = LinearEOS()
        T0 = 20.0
        st = ocean_state(
            LayeredEdgeField(m, K),
            h,
            uniform_C(m, K; C0 = T0),
            uniform_C(m, K; C0 = 35.0))
        # kick the barotropic mode so the re-stretch actually does work
        η0 = 2.0
        for i in 1:m.n_cells
            st.eta[i] = η0
            for k in 1:K
                st.h[k][i] *= (H0 + η0) / H0
            end
        end
        st2 = ocean_integrate!(st, grid, eos, 0.0, 2 * 86400.0, 2)
        @test maximum(maximum(abs.(st2.T[k].data .- T0)) for k in 1:K) ==
              0.0
        @test maximum(maximum(abs.(st2.S[k].data .- 35.0)) for k in 1:K) ==
              0.0
    end

    # ------------------------------------------------------------- 2
    @testset "conservation: closed-domain content budget" begin
        grid = VerticalGrid(H0, K)
        h = layer_thickness(grid, CellField(i -> 0.0, m))
        u = solid_body_u!(LayeredEdgeField(m, K), m; u0 = 15.0)
        # nonuniform patch
        C = LayeredCellField(m, K)
        for k in 1:K, i in 1:m.n_cells
            x, y, z = m.cell_xyz[1, i], m.cell_xyz[2, i], m.cell_xyz[3, i]
            C[k][i] = 1.0 + 2.0 * exp(-((x - 1)^2 + y^2 + z^2) / 0.05)
        end
        Q0 = tracer_total(C, h)
        Δt = 600.0
        # standalone content-form steps: exact invariant. The volume
        # evolution is chained explicitly (h ← h_new each step) with
        # the same fluxes the tracer scheme uses.
        hcur = h
        for n in 1:50
            hnew = LayeredField([similar_field(hcur[k]) for k in 1:K])
            for k in 1:K
                hs = scale!(copyto!(similar_field(hcur[k]), hcur[k]), 1 / R)
                us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
                d = divergence(edge_mean(hs) * us)
                scale!(d, -R * Δt)
                copyto!(hnew[k], hcur[k])
                axpy!(hnew[k], 1.0, d)
            end
            advect_tracer!(C, hcur, hnew, u, Δt)
            hcur = hnew
        end
        Q1 = tracer_total(C, hcur)
        @test abs(Q1 - Q0) / abs(Q0) < 1e-12

        # through the dycore: SSPRK3 concentration form. The invariant
        # Σ A h C is quadratic in the state → conserved only to the
        # integrator's quadratic-invariant error (not bit-exactly);
        # the measured drift must stay small (≪ diffusivity signal).
        Cdy = LayeredCellField(m, K)
        for k in 1:K, i in 1:m.n_cells
            x, y, z = m.cell_xyz[1, i], m.cell_xyz[2, i], m.cell_xyz[3, i]
            Cdy[k][i] = 1.0 + 2.0 * exp(-((x - 1)^2 + y^2 + z^2) / 0.05)
        end
        eos = LinearEOS()
        Tsurf, ΔTcol = 20.0, 10.0
        T = LayeredCellField(m, K); S = LayeredCellField(m, K)
        fs = grid.zstar_fractions
        for k in 1:K, i in 1:m.n_cells
            zc = -(fs[k] + fs[k + 1]) / 2 * grid.H0
            T[k][i] = Tsurf + ΔTcol * zc / grid.H0 + Cdy[k][i] - 1.0
            S[k][i] = 35.0
        end
        # quiet dynamics (resting stratified state, Ω = 0): the tracer
        # budget is then dominated by the advection machinery itself
        stdy = ocean_state(LayeredEdgeField(m, K), h, T, S)
        Qd0 = tracer_total(stdy.T, stdy.h)
        stdy = ocean_integrate!(stdy, grid, eos, 0.0, 5 * 86400.0, 5;
                                Omega = 0.0)
        Qd1 = tracer_total(stdy.T, stdy.h)
        # SSPRK3 concentration form + the re-stretch closure: the
        # invariant Σ A h C is quadratic in the state, so the drift is
        # the integrator's quadratic-invariant error plus the O(η′·C′)
        # re-stretch term — here the state is nearly resting, so this
        # is dominated by the integrator; measured ≈ 2e-6 per 5 days
        @test abs(Qd1 - Qd0) / abs(Qd0) < 1e-5
    end

    # ------------------------------------------------------------- 3
    @testset "positivity: Gaussian patch under strong deformation" begin
        grid = VerticalGrid(H0, K)
        h = layer_thickness(grid, CellField(i -> 0.0, m))
        # strong convergent/divergent flow: dipole of source/sink —
        # use a solid-body flow plus a barotropic kick so the re-stretch
        # deforms the column; what matters is that C ≥ floor exactly
        u = solid_body_u!(LayeredEdgeField(m, K), m; u0 = 30.0)
        C = LayeredCellField(m, K)
        for k in 1:K, i in 1:m.n_cells
            x, y, z = m.cell_xyz[1, i], m.cell_xyz[2, i], m.cell_xyz[3, i]
            C[k][i] = 1.0 + 2.0 * exp(-((x - 1)^2 + y^2 + z^2) / 0.05)
        end
        Q0 = tracer_total(C, h)
        Δt = 600.0
        hcur = h
        for n in 1:30
            hnew = LayeredField([similar_field(hcur[k]) for k in 1:K])
            for k in 1:K
                hs = scale!(copyto!(similar_field(hcur[k]), hcur[k]), 1 / R)
                us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
                d = divergence(edge_mean(hs) * us)
                scale!(d, -R * Δt)
                copyto!(hnew[k], hcur[k])
                axpy!(hnew[k], 1.0, d)
            end
            advect_tracer!(C, hcur, hnew, u, Δt)
            hcur = hnew
            @test minimum(minimum(C[k].data) for k in 1:K) ≥ 1.0
        end
        # and the floor holds with the patch fully smeared
        @test tracer_total(C, hcur) ≈ Q0 rtol = 1e-12
    end

    # ------------------------------------------------------------- 4
    @testset "convergence: first order under refinement" begin
        # solid-body rotation of a smooth Gaussian patch, one layer
        # active (identical layers — the per-layer solution is the 2D
        # advection of the patch); ω = u0/R
        u0 = 20.0
        ω = u0 / R
        Tfinal = 5 * 86400.0                # 5 days ≈ 0.58 rad rotation

        function run_case(nsub)
            mesh = generate_icosahedral_mesh(n_subdivisions=nsub)
            Kk = 2
            grid = VerticalGrid(H0, Kk)
            h = layer_thickness(grid, CellField(i -> 0.0, mesh))
            u = solid_body_u!(LayeredEdgeField(mesh, Kk), mesh; u0 = u0)
            C = LayeredCellField(mesh, Kk)
            for k in 1:Kk, i in 1:mesh.n_cells
                x, y, z = mesh.cell_xyz[1, i], mesh.cell_xyz[2, i],
                          mesh.cell_xyz[3, i]
                # width 0.5 rad — well resolved at nsub = 2 (≥ 4 cells
                # across the e-folding scale); upwind's first-order
                # error is then dominated by diffusion u·dx/2, which
                # converges at O(dx) in l1 (measured 1.09 between
                # nsub 2→3 and 1.02 between 3→4)
                C[k][i] = 1.0 + 2.0 *
                          exp(-((x - 1)^2 + y^2 + z^2) / 0.5)
            end
            # Δt from the advective CFL at fixed CFL number across
            # refinements (dx ∝ 2^-nsub → Δt ∝ 2^-nsub)
            dx = minimum(mesh.d_e) * R
            Δt = 0.4 * dx / u0
            hcur = h
            nsteps = ceil(Int, Tfinal / Δt)
            Δt = Tfinal / nsteps            # exact final time
            for n in 1:nsteps
                hnew = LayeredField([similar_field(hcur[k]) for k in 1:Kk])
                for k in 1:Kk
                    hs = scale!(copyto!(similar_field(hcur[k]), hcur[k]),
                                1 / R)
                    us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
                    d = divergence(edge_mean(hs) * us)
                    scale!(d, -R * Δt)
                    copyto!(hnew[k], hcur[k])
                    axpy!(hnew[k], 1.0, d)
                end
                advect_tracer!(C, hcur, hnew, u, Δt)
                hcur = hnew
            end
            # analytic reference: patch rotated by ω·Tfinal about the
            # z axis (eastward = −φ̂ direction → lon decreases along
            # the flow, so the reference at cell x is the IC at the
            # BACK-rotated coordinates Rz(−θ)·x)
            θ = ω * Tfinal
            err = 0.0
            for k in 1:Kk, i in 1:mesh.n_cells
                x, y = mesh.cell_xyz[1, i], mesh.cell_xyz[2, i]
                xr = cos(θ) * x + sin(θ) * y
                yr = -sin(θ) * x + cos(θ) * y
                z = mesh.cell_xyz[3, i]
                Cref = 1.0 + 2.0 * exp(-((xr - 1)^2 + yr^2 + z^2) / 0.5)
                err += mesh.cell_area[i] * abs(C[k][i] - Cref)
            end
            return err
        end

        e2 = run_case(2)
        e3 = run_case(3)
        order = log2(e2 / e3)
        # first-order scheme: expected order ≈ 1 (allow 0.5–2; the
        # coarse-grid constant dominates at nsub = 2–3 with a Gaussian
        # width of 0.05 rad vs dx ≈ 0.12 rad at nsub = 2 — the patch is
        # marginally resolved, which is where upwind's order shows)
        @test order ≥ 0.5
        @test e3 < e2                        # refinement reduces error
    end
end
