# Flux-corrected tracer advection (issue #36, DESIGN.md §5 Phase 2
# advection rung 2: higher-order positive-definite — the production
# scheme; #32's upwind stays as the reference/debug scheme).
#
# Acceptance (issue #36):
#   1. Convergence order ≥ 1.5: solid-body rotation of a smooth Gaussian
#      patch, measured against the analytic solution at two grid
#      refinements. Δt is REFINED as dx² (not the fixed-CFL Δt ∝ dx of
#      the #32 test) — with the explicit Euler content update the
#      temporal error is O(Δt), which at fixed CFL converges as O(dx)
#      and would mask the scheme's O(dx²) spatial operator (measured:
#      order 0.69 at fixed CFL, 1.98 with Δt ∝ dx²). The Δt ∝ dx²
#      protocol isolates the spatial order, which is what the issue
#      criterion (and the "must beat the #32 baseline" comparison)
#      targets; the #32 test's fixed-CFL protocol remains in place for
#      the upwind scheme.
#   2. Positivity: FCT keeps min C ≥ floor EXACTLY (0 to round-off
#      tolerance) on a deformational flow where the UNLIMITED
#      high-order donor-gradient scheme goes negative — the limiter is
#      demonstrably doing real work (the unlimited minimum is reported).
#   3. Conservation: global tracer content Σ A_i h_i C_i invariant to
#      round-off under limited fluxes (the limiter scales SHARED edge
#      fluxes, so the flux form stays exact).
#   4. Comparison (reported, not asserted): upwind vs FCT l1 error on
#      the same 1-step rotation — the PR carries the table.
#
# Test design notes:
#   - Solid-body flow identical to the #32 test (same helper shape);
#     the deformational flow uses a nondivergent spherical-cellular
#     streamfunction ψ = U0·R·sin(3λ)·cos²φ, velocities from
#     u = n̂×∇ψ (uλ = −ψ_φ/R, uφ = ψ_λ/(R cosφ)) — with the WRONG
#     sign/derivative pairing the test flow itself drives h negative
#     and the tracer blows up (documented probe-debug finding; the
#     checks below run with h bounded ≥ 1500 m so the scheme, not the
#     flow, is what's exercised).
#   - FCT unit-scales h and u by 1/R internally (same scaled-flux
#     convention as the dycore; see TracerFCT.jl header).
#
# GOTCHA (documented in AGENTS-session notes): `using LinearAlgebra`
# in this file shadows the Fields `axpy!` export — the tests use
# `copyto!`/arithmetic instead, or qualify with `JuiCM.axpy!`.

@testset "ocean tracer advection: FCT, second-order positive-definite (#36)" begin
    using JuiCM: VerticalGrid, layer_thickness, williamson_constants,
                 fct_advect_tracer!, advect_tracer!
    using JuiCM: divergence, divergence!, edge_mean, cell_gradient
    using JuiCM: mesh_of, similar_field, scale!, copyto!, axpy!, fill!
    using JuiCM.TracerFCT: _fct_fluxes, _apply_fct!
    using JuiCM: LayeredField

    (; R) = williamson_constants()
    H0 = 4000.0
    K = 4
    Kk = 2                              # two-layer barotropic test setup

    "Solid-body zonal flow: u = u0·cosφ eastward. NOTE: the normal
    component uses the FULL unit vector (−y/rxy, x/rxy) — folding cosφ
    = rxy into the amplitude gives u0·rxy·ve exactly (the #32 test
    helper omits the first /rxy, making its flow u0·cos²φ; harmless
    there because its reference is the same flow, wrong here where the
    analytic reference is rigid rotation)."
    function solid_body_u!(u, mesh; u0)
        for k in 1:nlayers(u), e in 1:mesh.n_edges
            x = mesh.edge_x[1, e]; y = mesh.edge_x[2, e]
            nx = mesh.edge_normal[1, e]; ny = mesh.edge_normal[2, e]
            rxy = sqrt(x^2 + y^2)
            ve = -(y / rxy) * nx + (x / rxy) * ny
            u[k][e] = u0 * sqrt(1 - mesh.edge_x[3, e]^2) * ve
        end
        return u
    end

    "Nondivergent cellular flow from ψ = U0·R·sin(3λ)·cos²φ (u = n̂×∇ψ)."
    function cellular_u!(u, mesh; U0)
        for k in 1:nlayers(u), e in 1:mesh.n_edges
            λ = atan(mesh.edge_x[2, e], mesh.edge_x[1, e])
            φ = asin(clamp(mesh.edge_x[3, e], -1.0, 1.0))
            nx = mesh.edge_normal[1, e]; ny = mesh.edge_normal[2, e]
            nz = mesh.edge_normal[3, e]
            tλ = [-sin(λ), cos(λ), 0.0]
            tφ = [-sin(φ)*cos(λ), -sin(φ)*sin(λ), cos(φ)]
            dψdλ = 3U0 * R * cos(3λ) * cos(φ)^2
            dψdφ = -2U0 * R * sin(3λ) * cos(φ) * sin(φ)
            uλ = -dψdφ / R
            uφ = dψdλ / (R * cos(φ))
            u[k][e] = uλ * (tλ[1]*nx + tλ[2]*ny + tλ[3]*nz) +
                      uφ * (tφ[1]*nx + tφ[2]*ny + tφ[3]*nz)
        end
        return u
    end

    "h update from the shared volume flux (one Euler content step)."
    function step_h!(hcur, hnew, u, Δt)
        for k in 1:nlayers(hcur)
            hs = scale!(copyto!(similar_field(hcur[k]), hcur[k]), 1 / R)
            us = scale!(copyto!(similar_field(u[k]), u[k]), 1 / R)
            d = divergence(edge_mean(hs) * us)
            scale!(d, -R * Δt)
            copyto!(hnew[k], hcur[k])
            axpy!(hnew[k], 1.0, d)
        end
        return hnew
    end

    "Upwind reference error on the SAME Δt ∝ dx² protocol (for the
    comparison in testset 1 — both schemes get the same small time
    steps, so the difference is purely spatial)."
    function upwind_err_ref(nsub; u0, Tfinal)
        mesh = generate_icosahedral_mesh(n_subdivisions=nsub)
        grid = VerticalGrid(H0, Kk)
        h = layer_thickness(grid, CellField(i -> 0.0, mesh))
        u = solid_body_u!(LayeredEdgeField(mesh, Kk), mesh; u0 = u0)
        C = LayeredCellField(mesh, Kk)
        for k in 1:Kk, i in 1:mesh.n_cells
            x, y, z = mesh.cell_xyz[1, i], mesh.cell_xyz[2, i],
                      mesh.cell_xyz[3, i]
            C[k][i] = 1.0 + 2.0 * exp(-((x - 1)^2 + y^2 + z^2) / 2.0)
        end
        dx = minimum(mesh.d_e) * R
        Δt = 0.4 * dx^2 / (u0 * R)
        nsteps = ceil(Int, Tfinal / Δt)
        Δt = Tfinal / nsteps
        hcur = h
        for n in 1:nsteps
            hnew = LayeredField([similar_field(hcur[k]) for k in 1:Kk])
            step_h!(hcur, hnew, u, Δt)
            advect_tracer!(C, hcur, hnew, u, Δt)
            hcur = hnew
        end
        θ = (u0 / R) * Tfinal
        err = 0.0
        for k in 1:Kk, i in 1:mesh.n_cells
            x, y = mesh.cell_xyz[1, i], mesh.cell_xyz[2, i]
            xr = cos(θ) * x + sin(θ) * y
            yr = -sin(θ) * x + cos(θ) * y
            z = mesh.cell_xyz[3, i]
            Cref = 1.0 + 2.0 * exp(-((xr - 1)^2 + yr^2 + z^2) / 2.0)
            err += mesh.cell_area[i] * abs(C[k][i] - Cref)
        end
        return err
    end

    # ------------------------------------------------------------- 1
    @testset "convergence: order ≥ 1.5 (spatial, Δt ∝ dx²)" begin
        u0 = 20.0
        Tfinal = 5 * 86400.0

        function run_case(nsub)
            mesh = generate_icosahedral_mesh(n_subdivisions=nsub)
            grid = VerticalGrid(H0, Kk)
            h = layer_thickness(grid, CellField(i -> 0.0, mesh))
            u = solid_body_u!(LayeredEdgeField(mesh, Kk), mesh; u0 = u0)
            C = LayeredCellField(mesh, Kk)
            for k in 1:Kk, i in 1:mesh.n_cells
                x, y, z = mesh.cell_xyz[1, i], mesh.cell_xyz[2, i],
                          mesh.cell_xyz[3, i]
                # width √2 rad — well resolved even at nsub = 2
                C[k][i] = 1.0 + 2.0 *
                          exp(-((x - 1)^2 + y^2 + z^2) / 2.0)
            end
            # Δt ∝ dx²: explicit-Euler temporal error O(Δt) then
            # converges as O(dx²), isolating the spatial operator
            # (module-header rationale; fixed CFL gives order ≈ 1)
            dx = minimum(mesh.d_e) * R
            Δt = 0.4 * dx^2 / (u0 * R)
            nsteps = ceil(Int, Tfinal / Δt)
            Δt = Tfinal / nsteps            # exact final time
            hcur = h
            for n in 1:nsteps
                hnew = LayeredField([similar_field(hcur[k]) for k in 1:Kk])
                step_h!(hcur, hnew, u, Δt)
                fct_advect_tracer!(C, hcur, hnew, u, Δt)
                hcur = hnew
            end
            # analytic reference: patch rotated by ω·Tfinal, back-rotated
            # coordinates Rz(−θ)·x (same convention as the #32 test)
            θ = (u0 / R) * Tfinal
            err = 0.0
            for k in 1:Kk, i in 1:mesh.n_cells
                x, y = mesh.cell_xyz[1, i], mesh.cell_xyz[2, i]
                xr = cos(θ) * x + sin(θ) * y
                yr = -sin(θ) * x + cos(θ) * y
                z = mesh.cell_xyz[3, i]
                Cref = 1.0 + 2.0 *
                       exp(-((xr - 1)^2 + yr^2 + z^2) / 2.0)
                err += mesh.cell_area[i] * abs(C[k][i] - Cref)
            end
            return err
        end

        e2 = run_case(2)
        e3 = run_case(3)
        order = log2(e2 / e3)
        # measured (see PR): upwind 0.82 vs FCT 1.98 on the same setup —
        # the second-order spatial operator is real
        @test order ≥ 1.5
        @test e3 < e2
        # and FCT beats the #32 baseline on the same grids (l1)
        up2 = upwind_err_ref(2; u0, Tfinal)
        up3 = upwind_err_ref(3; u0, Tfinal)
        @test e2 < up2
        @test e3 < up3
        println("FCT convergence: e2=", round(e2, digits=4),
                " e3=", round(e3, digits=4), " order=", round(order, digits=2),
                "  (upwind: ", round(up2, digits=4), " / ",
                round(up3, digits=4), ")")
    end

    # ------------------------------------------------------------- 2
    @testset "positivity: limiter vs unlimited high-order" begin
        mesh = generate_icosahedral_mesh(n_subdivisions=2)
        grid = VerticalGrid(H0, Kk)
        h = layer_thickness(grid, CellField(i -> 0.0, mesh))
        u = cellular_u!(LayeredEdgeField(mesh, Kk), mesh; U0 = 8.0)
        # strip tracer with ZERO background: any undershoot is a
        # negative concentration
        C0 = LayeredCellField(mesh, Kk)
        for k in 1:Kk, i in 1:mesh.n_cells
            z = mesh.cell_xyz[3, i]
            C0[k][i] = abs(z) < 0.1 ? 1.0 : 0.0
        end
        Δt = 86400.0 / 24
        nsteps = 96
        hmin_run = Inf
        # --- unlimited high-order (donor-gradient, NO limiter) ---
        hcur = deepcopy(h); Cu = deepcopy(C0)
        for n in 1:nsteps
            hnew = LayeredField([similar_field(hcur[k]) for k in 1:Kk])
            step_h!(hcur, hnew, u, Δt)
            for k in 1:Kk
                hs = scale!(copyto!(similar_field(hcur[k]), hcur[k]), 1/R)
                us = scale!(copyto!(similar_field(u[k]), u[k]), 1/R)
                F_lo = similar_field(us); F_hi = similar_field(us)
                _fct_fluxes(F_lo, F_hi, Cu[k], hs, us)
                Q = copyto!(similar_field(Cu[k]), hcur[k] * Cu[k])
                dQ = divergence!(similar_field(Cu[k]), F_hi)
                scale!(dQ, -R*Δt); axpy!(dQ, 1.0, Q)
                for i in 1:mesh.n_cells
                    Cu[k][i] = dQ[i] / hnew[k][i]
                end
            end
            hcur = hnew
        end
        unlimited_min = minimum(minimum(Cu[k].data) for k in 1:Kk)
        println("unlimited high-order min C = ", round(unlimited_min, digits=5),
                "  (negative ⇒ the limiter is doing real work)")
        # --- FCT ---
        hcur = deepcopy(h); Cf = deepcopy(C0)
        for n in 1:nsteps
            hnew = LayeredField([similar_field(hcur[k]) for k in 1:Kk])
            step_h!(hcur, hnew, u, Δt)
            hmin_run = min(hmin_run, minimum(minimum(hnew[k].data) for k in 1:Kk))
            fct_advect_tracer!(Cf, hcur, hnew, u, Δt)
            hcur = hnew
        end
        fct_min = minimum(minimum(Cf[k].data) for k in 1:Kk)
        println("FCT min C = ", round(fct_min, digits=6),
                "   (min h along the way = ", round(hmin_run, digits=1), " m)")
        # the limiter is demonstrably needed...
        @test unlimited_min < 0.0
        # ...and it works: positivity exact (to round-off)
        @test fct_min ≥ 0.0
        @test fct_min ≥ -1e-12
        # and no new maxima: bounded by the IC max
        @test maximum(maximum(Cf[k].data) for k in 1:Kk) ≤ 1.0 + 1e-12
    end

    # ------------------------------------------------------------- 3
    @testset "conservation: round-off under limited fluxes" begin
        mesh = generate_icosahedral_mesh(n_subdivisions=2)
        grid = VerticalGrid(H0, Kk)
        h = layer_thickness(grid, CellField(i -> 0.0, mesh))
        u = cellular_u!(LayeredEdgeField(mesh, Kk), mesh; U0 = 8.0)
        C = LayeredCellField(mesh, Kk)
        for k in 1:Kk, i in 1:mesh.n_cells
            x, y, z = mesh.cell_xyz[1, i], mesh.cell_xyz[2, i],
                      mesh.cell_xyz[3, i]
            C[k][i] = 1.0 + exp(-((x - 1)^2 + y^2 + z^2) / 0.5)
        end
        Δt = 86400.0 / 24
        m0 = sum(sum(h.layers[k].data .* C.layers[k].data .* mesh.cell_area)
                 for k in 1:Kk)
        hcur = h
        for n in 1:24
            hnew = LayeredField([similar_field(hcur[k]) for k in 1:Kk])
            step_h!(hcur, hnew, u, Δt)
            fct_advect_tracer!(C, hcur, hnew, u, Δt)
            hcur = hnew
        end
        m1 = sum(sum(hcur.layers[k].data .* C.layers[k].data .* mesh.cell_area)
                 for k in 1:Kk)
        @test abs(m1 - m0) / m0 < 1e-12
    end

    # ------------------------------------------------------------- 4
    @testset "compatibility: uniform tracer stays uniform" begin
        mesh = generate_icosahedral_mesh(n_subdivisions=2)
        grid = VerticalGrid(H0, Kk)
        h = layer_thickness(grid, CellField(i -> 0.0, mesh))
        u = cellular_u!(LayeredEdgeField(mesh, Kk), mesh; U0 = 8.0)
        C = LayeredCellField(mesh, Kk)
        for k in 1:Kk, i in 1:mesh.n_cells
            C[k][i] = 3.0
        end
        Δt = 86400.0 / 24
        hcur = h
        for n in 1:10
            hnew = LayeredField([similar_field(hcur[k]) for k in 1:Kk])
            step_h!(hcur, hnew, u, Δt)
            fct_advect_tracer!(C, hcur, hnew, u, Δt)
            hcur = hnew
        end
        # donor-gradient reconstruction of a constant is exact; β
        # scaling of a zero antidiffusive flux is a no-op — uniform C
        # must be preserved to round-off under ANY flow
        @test maximum(maximum(abs.(C[k].data .- 3.0)) for k in 1:Kk) < 1e-12
    end
end
