# Regression tests for the #40 review defects in the layered ocean
# driver (src/Dycore/Ocean/Dynamics.jl):
#
#   A. Tracer/volume compatibility of the operator-split FCT tracer
#      step: the tracer transport must use the time-integrated layer
#      mass fluxes that actually moved the thickness (slow SSPRK stages
#      + fast barotropic substeps), so that (i) a constant tracer stays
#      constant to round-off and (ii) the global content Σ A h C is
#      conserved to round-off. Before the fix the FCT was driven with
#      the end-of-step velocity and the start-of-step h; the mismatch
#      leaked through the surface interface (w_1 ≠ 0). Measured before
#      the fix on this fixture: relative content drift ≈ 1e-4 (see the
#      PR/issue record); after: ≈ 1e-15.
#   B. Collapsed-layer quadratic drag must stay finite and dissipative
#      when an SSPRK stage hands it a zero/negative thickness.
#   C. The minimum-thickness floor must not leave h ≤ 0 in a starved
#      column (headroom ≤ deficit).

@testset "ocean dynamics: #40 review fixes" begin
    using JuiCM: ocean_state, ocean_step!, ocean_integrate!,
                 ocean_tendency!, ocean_volume, LinearEOS, VerticalGrid,
                 layer_thickness, tracer_total, LayeredCellField,
                 LayeredEdgeField, CellField, EdgeField, nlayers,
                 williamson_constants

    m = generate_icosahedral_mesh(n_subdivisions=2)
    (; R) = williamson_constants()
    K = 4
    H0 = 4000.0

    "Eastward solid-body normal component u0·cosφ·(ê·n̂) at edge e."
    function solid_body(mesh, e, u0)
        x = mesh.edge_x[1, e]; y = mesh.edge_x[2, e]
        nx = mesh.edge_normal[1, e]; ny = mesh.edge_normal[2, e]
        rxy = sqrt(x^2 + y^2)
        return u0 * sqrt(1 - mesh.edge_x[3, e]^2) * (-y * nx + (x / rxy) * ny)
    end

    """
    Column-consistent (Σh − H0 = η) stratified state with a Gaussian
    free-surface bump (the fast mode moves water), a barotropic
    solid-body flow and a baroclinic shear (the slow fluxes move water
    between layers). `Tfun(k, i)` sets T; S is uniform.
    """
    function active_state(grid; Tfun, u0 = 0.4, ushear = 0.3, bump = 1.5)
        eta = CellField(i -> begin
            x, y, z = m.cell_xyz[1, i], m.cell_xyz[2, i], m.cell_xyz[3, i]
            bump * exp(-((x - 1)^2 + y^2 + z^2) / 0.1)
        end, m)
        h = layer_thickness(grid, eta)
        T = LayeredCellField(m, K); S = LayeredCellField(m, K)
        for k in 1:K, i in 1:m.n_cells
            T[k][i] = Tfun(k, i)
            S[k][i] = 35.0
        end
        shear = [1.5, 0.5, -0.5, -1.5]
        u = LayeredEdgeField(m, K)
        ubar = EdgeField(e -> solid_body(m, e, u0), m)
        for k in 1:K, e in 1:m.n_edges
            # meridional-ish shear pattern so the anomaly flux diverges
            u[k][e] = ubar[e] + ushear * shear[k] *
                      solid_body(m, e, 1.0) * m.edge_x[3, e]
        end
        return ocean_state(u, h, T, S, eta, ubar)
    end

    eos = LinearEOS()
    Δt = 1800.0
    nsteps = 6

    # ------------------------------------------------------------ A(i)
    @testset "constant tracer stays constant (flow + active floor)" begin
        grid = VerticalGrid(H0, K)
        st = active_state(grid; Tfun = (k, i) -> 12.5)
        # thin layer 2 at a cluster of cells (column sum unchanged:
        # the water goes to layer 3) so the floor binds
        for i in 1:m.n_cells
            if m.cell_xyz[3, i] > 0.8
                d = st.h[2][i] - 5.0
                st.h[2][i] -= d
                st.h[3][i] += d
            end
        end
        st2 = ocean_integrate!(st, grid, eos, 0.0, nsteps * Δt, nsteps;
                               min_h_frac = 0.01)
        dT = maximum(maximum(abs(st2.T[k][i] - 12.5) for i in 1:m.n_cells)
                     for k in 1:K)
        dS = maximum(maximum(abs(st2.S[k][i] - 35.0) for i in 1:m.n_cells)
                     for k in 1:K)
        @test dT < 1e-11
        @test dS < 1e-11
        # the floor actually acted
        @test minimum(st2.h[2][i] for i in 1:m.n_cells) ≥ 0.01 * H0 - 1e-9
    end

    # ----------------------------------------------------------- A(ii)
    @testset "global tracer content conserved to round-off" begin
        grid = VerticalGrid(H0, K)
        Tfun(k, i) = begin
            x, y, z = m.cell_xyz[1, i], m.cell_xyz[2, i], m.cell_xyz[3, i]
            20.0 - 2.0 * k + 3.0 * exp(-((x - 0.7)^2 + (y - 0.7)^2 + z^2) / 0.2)
        end
        st = active_state(grid; Tfun = Tfun)
        Q0 = tracer_total(st.T, st.h)
        V0 = ocean_volume(grid, st)
        for n in 1:nsteps
            st = ocean_step!(st, grid, eos, (n - 1) * Δt, Δt)
        end
        Q1 = tracer_total(st.T, st.h)
        drift = abs(Q1 - Q0) / abs(Q0)
        @info "FCT-split tracer content drift" drift
        @test drift ≤ 1e-10
        @test abs(ocean_volume(grid, st) - V0) / V0 < 1e-12
        # state stays column-consistent (the precondition of the flux
        # bookkeeping): Σ h − H0 = η
        @test maximum(abs(sum(st.h[k][i] for k in 1:K) - H0 - st.eta[i])
                      for i in 1:m.n_cells) < 1e-9
    end

    # -------------------------------------------------------------- B
    @testset "drag finite and dissipative at h ≤ 0 edges" begin
        grid = VerticalGrid(H0, K)
        st = active_state(grid; Tfun = (k, i) -> 20.0 - k)
        # starve layer K at two cells: exactly zero and negative
        st.h[K][1] = 0.0
        st.h[K][2] = -10.0
        out_d = JuiCM.TimeStepping._stage(st)
        out_0 = JuiCM.TimeStepping._stage(st)
        ocean_tendency!(out_d, st; grid = grid, eos = eos, c_drag = 2.5e-3)
        ocean_tendency!(out_0, st; grid = grid, eos = eos, c_drag = 0.0)
        @test all(isfinite(out_d.u[k][e]) for k in 1:K, e in 1:m.n_edges)
        # the drag increment opposes u on every edge (no anti-drag)
        @test all((out_d.u[k][e] - out_0.u[k][e]) * st.u[k][e] ≤ 0
                  for k in 1:K, e in 1:m.n_edges)
        # and it acts at the starved edges
        e1 = first(m.cell_edges[1])
        @test out_d.u[K][e1] != out_0.u[K][e1]
    end

    # -------------------------------------------------------------- C
    @testset "floor handles a starved column" begin
        floor! = JuiCM.Ocean._apply_thickness_floor!
        h = LayeredCellField(m, K); T = LayeredCellField(m, K)
        S = LayeredCellField(m, K); eta = CellField(i -> 0.0, m)
        h_min = 0.01 * H0
        for k in 1:K, i in 1:m.n_cells
            h[k][i] = H0 / K
            T[k][i] = 10.0 + k
            S[k][i] = 35.0
        end
        # column 1: starved, positive total (headroom < deficit)
        for (k, v) in enumerate((30.0, -5.0, 10.0, 20.0))
            h[k][1] = v
        end
        # column 2: non-positive total
        for (k, v) in enumerate((1.0, -3.0, 0.0, 1.0))
            h[k][2] = v
        end
        # column 3: ordinary clip with enough headroom
        h[2][3] = 1.0; h[1][3] += H0 / K - 1.0
        col(i) = sum(h[k][i] for k in 1:K)
        content(i) = sum(h[k][i] * T[k][i] for k in 1:K)
        c1, q1 = col(1), content(1)
        c3, q3 = col(3), content(3)
        stats = Dict{Symbol, Float64}()
        floor!(h, (T, S), eta, h_min, H0; stats = stats)
        @test all(h[k][i] > 0 for k in 1:K, i in 1:m.n_cells)
        # positive-total starved column: column volume and content
        # conserved (homogenized), every layer at the column mean
        @test col(1) ≈ c1 rtol = 1e-14
        @test content(1) ≈ q1 rtol = 1e-14
        @test all(h[k][1] ≈ c1 / K for k in 1:K)
        # non-positive column: refilled to h_min, the added volume is
        # recorded and η follows (column stays consistent)
        @test all(h[k][2] == h_min for k in 1:K)
        @test stats[:added_volume_columns] == 1
        @test stats[:homogenized_columns] == 1
        @test eta[2] ≈ K * h_min - H0
        # ordinary column: exact column/content conservation, h ≥ h_min
        @test col(3) ≈ c3 rtol = 1e-14
        @test content(3) ≈ q3 rtol = 1e-14
        @test minimum(h[k][3] for k in 1:K) ≥ h_min - 1e-12
    end

    # -------------------------------------------------------------- D
    @testset "floor is opt-in and scale-checked" begin
        grid = VerticalGrid(H0, K)
        st = active_state(grid; Tfun = (k, i) -> 20.0 - k)
        st.h[2][5] = 1.0; st.h[3][5] += H0 / K - 1.0   # thin, no floor
        # default: no floor — the thin layer is not clipped to 1% H0
        st2 = ocean_step!(st, grid, eos, 0.0, 60.0)
        @test st2.h[2][5] < 0.01 * H0
        # an h_min that cannot fit the column is rejected
        @test_throws ArgumentError ocean_step!(st, grid, eos, 0.0, 60.0;
                                               min_h_frac = 0.3)
    end
end
