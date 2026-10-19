# z* regridding in `ocean_step!` (issue #62, DESIGN.md D7).
#
#   A. `_remap_edge_columns!` (conservative MC-limited PLM momentum
#      remap): uniform velocity preserved, column Σ ĥ u conserved to
#      round-off, no new extrema per edge column.
#   B. `fct_advect_tracer!(…; vertical_order = 2)`: content exact, and
#      a linear vertical profile is remapped EXACTLY in the interior
#      (the PLM + Courant-corrected face value is the swept-region mean
#      of a linear profile), which first-order upwind is not.
#   C. `ocean_step!(…; regrid = true)` with active flow: layers sit on
#      the z* target after every step, tracer content conserved to
#      round-off, constant tracer stays constant, volume conserved.
#   D. Reduced-length lock-channel run (configuration of record,
#      nsub = 4, K = 10) with regrid: finite, T inside its initial
#      bounds, buoyancy drift ≤ 1e-10, the front propagates.

using JuiCM, Test

@testset "ocean z* regrid (#62)" begin
    using JuiCM: ocean_state, ocean_step!, ocean_integrate!, ocean_volume,
                 LinearEOS, VerticalGrid, layer_thickness, tracer_total,
                 LayeredCellField, LayeredEdgeField, CellField, EdgeField,
                 fct_advect_tracer!, williamson_constants
    remap! = JuiCM.Ocean._remap_edge_columns!

    m = generate_icosahedral_mesh(n_subdivisions = 2)
    K = 4
    H0 = 4000.0
    hm(h, k, e) = 0.5 * (h[k][m.edge_cells[1, e]] + h[k][m.edge_cells[2, e]])

    # ---------------------------------------------------------------- A
    @testset "momentum remap: conservative and monotone" begin
        grid = VerticalGrid(H0, K)
        # source: perturbed Lagrangian layers with the same column total
        hsrc = LayeredCellField((i, k) -> H0 / K * (1 + 0.3 * sin(i + 2k)),
                                m, K)
        for i in 1:m.n_cells
            tot = sum(hsrc[k][i] for k in 1:K)
            for k in 1:K
                hsrc[k][i] *= H0 / tot
            end
        end
        hdst = layer_thickness(grid, CellField(i -> 0.0, m))
        # uniform velocity is preserved exactly
        u = LayeredEdgeField((e, k) -> 1.7, m, K)
        remap!(u, hsrc, hdst)
        @test maximum(abs(u[k][e] - 1.7) for k in 1:K, e in 1:m.n_edges) <
              1e-12
        # column momentum Σ ĥ u conserved, no new extrema
        u = LayeredEdgeField((e, k) -> cos(0.3e + k), m, K)
        mom0 = [sum(hm(hsrc, k, e) * u[k][e] for k in 1:K) for e in 1:m.n_edges]
        umax0 = [maximum(u[k][e] for k in 1:K) for e in 1:m.n_edges]
        umin0 = [minimum(u[k][e] for k in 1:K) for e in 1:m.n_edges]
        remap!(u, hsrc, hdst)
        mom1 = [sum(hm(hdst, k, e) * u[k][e] for k in 1:K) for e in 1:m.n_edges]
        @test maximum(abs.(mom1 .- mom0)) < 1e-12 * H0
        @test all(maximum(u[k][e] for k in 1:K) <= umax0[e] + 1e-12
                  for e in 1:m.n_edges)
        @test all(minimum(u[k][e] for k in 1:K) >= umin0[e] - 1e-12
                  for e in 1:m.n_edges)
        # identity remap (source == target) leaves u untouched (up to
        # the (ĥu)/ĥ rounding)
        u2 = LayeredEdgeField((e, k) -> sin(0.7e - k), m, K)
        u2c = copy(u2)
        remap!(u2, hdst, hdst)
        @test maximum(abs(u2[k][e] - u2c[k][e]) for k in 1:K,
                      e in 1:m.n_edges) < 1e-15
    end

    # ---------------------------------------------------------------- B
    @testset "FCT vertical_order = 2: exact linear remap, content exact" begin
        Kv = 6
        h = 500.0
        # downward displacement δ_k of interface k of the target relative
        # to the source (surface and bottom fixed); non-uniform, so
        # first-order upwind is NOT exact on a linear profile
        δ = [0.0, 120.0, 60.0, -40.0, 90.0, 30.0, 0.0]
        h_old = LayeredCellField((i, k) -> h, m, Kv)
        h_new = LayeredCellField((i, k) -> h + δ[k + 1] - δ[k], m, Kv)
        u0 = LayeredEdgeField((e, k) -> 0.0, m, Kv)
        # linear in depth: C(z) = c_i + 0.5 (z/h + 0.5), z downward
        Cini(i, k) = 10.0 + 0.5 * k + 0.01 * (i % 7)
        # exact target-layer mean: the source profile averaged over
        # [z_k + δ_k, z_{k+1} + δ_{k+1}]
        exact(i, k) = Cini(i, k) + 0.5 * (δ[k] + δ[k + 1]) / (2h)
        C2 = LayeredCellField(Cini, m, Kv)
        C1 = LayeredCellField(Cini, m, Kv)
        q0 = tracer_total(C2, h_old)
        fct_advect_tracer!(C2, h_old, h_new, u0, 600.0; vertical_order = 2)
        fct_advect_tracer!(C1, h_old, h_new, u0, 600.0)
        # target layers 1–4: every donor (interfaces 2–5: layers 2, 3, 3,
        # 5) has a two-sided MC slope, which is exact for a linear profile
        err2 = maximum(abs(C2[k][i] - exact(i, k)) for k in 1:4,
                       i in 1:m.n_cells)
        err1 = maximum(abs(C1[k][i] - exact(i, k)) for k in 1:4,
                       i in 1:m.n_cells)
        @test err2 < 1e-12
        @test err1 > 1e-3             # upwind: O(δ/h · slope) error
        @test abs(tracer_total(C2, h_new) / q0 - 1) < 1e-14
        # no new extrema in any column
        @test all(minimum(C2[k][i] for k in 1:Kv) >= Cini(i, 1) - 1e-12 &&
                  maximum(C2[k][i] for k in 1:Kv) <= Cini(i, Kv) + 1e-12
                  for i in 1:m.n_cells)
        @test_throws ArgumentError fct_advect_tracer!(C2, h_old, h_new, u0,
                                                      600.0; vertical_order = 3)
    end

    # ---------------------------------------------------------------- C
    "Eastward solid-body normal component u0·cosφ·(ê·n̂) at edge e."
    function solid_body(e, u0)
        x = m.edge_x[1, e]; y = m.edge_x[2, e]
        nx = m.edge_normal[1, e]; ny = m.edge_normal[2, e]
        rxy = sqrt(x^2 + y^2)
        return u0 * sqrt(1 - m.edge_x[3, e]^2) * (-y * nx + (x / rxy) * ny)
    end
    # column-consistent stratified state with a free-surface bump, a
    # barotropic solid-body flow and a baroclinic shear (the slow fluxes
    # move water between layers — the regrid then has work to do)
    function active_state(grid; Tfun)
        eta = CellField(i -> begin
            x, y, z = m.cell_xyz[1, i], m.cell_xyz[2, i], m.cell_xyz[3, i]
            1.5 * exp(-((x - 1)^2 + y^2 + z^2) / 0.1)
        end, m)
        h = layer_thickness(grid, eta)
        T = LayeredCellField(Tfun, m, K)
        S = LayeredCellField((i, k) -> 35.0, m, K)
        shear = [1.5, 0.5, -0.5, -1.5]
        ubar = EdgeField(e -> solid_body(e, 0.4), m)
        u = LayeredEdgeField((e, k) -> ubar[e] + 0.3 * shear[k] *
                                       solid_body(e, 1.0) * m.edge_x[3, e],
                             m, K)
        return ocean_state(u, h, T, S, eta, ubar)
    end
    eos = LinearEOS()
    Δt = 1800.0
    nsteps = 6

    @testset "regridded steps: z* target, content, constancy" begin
        grid = VerticalGrid(H0, K)
        Tfun(i, k) = begin
            x, y, z = m.cell_xyz[1, i], m.cell_xyz[2, i], m.cell_xyz[3, i]
            20.0 - 2.0 * k + 3.0 * exp(-((x - 0.7)^2 + (y - 0.7)^2 + z^2) / 0.2)
        end
        st = active_state(grid; Tfun = Tfun)
        Q0 = tracer_total(st.T, st.h)
        S0 = tracer_total(st.S, st.h)
        V0 = ocean_volume(grid, st)
        Tlo, Thi = extrema(st.T[k][i] for k in 1:K, i in 1:m.n_cells)
        maxdev = 0.0
        for n in 1:nsteps
            st = ocean_step!(st, grid, eos, (n - 1) * Δt, Δt; regrid = true)
            tgt = layer_thickness(grid, st.eta)
            maxdev = max(maxdev, maximum(abs(st.h[k][i] - tgt[k][i])
                                         for k in 1:K, i in 1:m.n_cells))
        end
        @test maxdev == 0.0           # h is SET to the target
        drift = abs(tracer_total(st.T, st.h) - Q0) / abs(Q0)
        @info "z* regrid tracer content drift" drift
        @test drift ≤ 1e-12
        # S is uniform: stays uniform (constant tracer), content exact
        @test maximum(abs(st.S[k][i] - 35.0) for k in 1:K, i in 1:m.n_cells) <
              1e-11
        @test abs(tracer_total(st.S, st.h) / S0 - 1) ≤ 1e-12
        @test abs(ocean_volume(grid, st) - V0) / V0 < 1e-12
        # column consistency Σ h − H0 = η
        @test maximum(abs(sum(st.h[k][i] for k in 1:K) - H0 - st.eta[i])
                      for i in 1:m.n_cells) < 1e-9
        # FCT + bound: T stays in its initial range
        @test all(Tlo - 1e-10 <= st.T[k][i] <= Thi + 1e-10
                  for k in 1:K, i in 1:m.n_cells)
        @test all(isfinite(st.u[k][e]) for k in 1:K, e in 1:m.n_edges)
        # the regrid actually moved u relative to the Lagrangian path
        st_l = ocean_step!(active_state(grid; Tfun = Tfun), grid, eos, 0.0, Δt)
        st_r = ocean_step!(active_state(grid; Tfun = Tfun), grid, eos, 0.0, Δt;
                           regrid = true)
        @test maximum(abs(st_l.u[k][e] - st_r.u[k][e])
                      for k in 1:K, e in 1:m.n_edges) > 0.0
        # default stays Lagrangian: regrid = false is the #30 path
        st_d = ocean_step!(active_state(grid; Tfun = Tfun), grid, eos, 0.0, Δt;
                           regrid = false)
        @test all(st_d.u[k][e] == st_l.u[k][e] for k in 1:K, e in 1:m.n_edges)
        @test all(st_d.h[k][i] == st_l.h[k][i] for k in 1:K, i in 1:m.n_cells)
        # integrate! forwards the keyword
        st_i = ocean_integrate!(active_state(grid; Tfun = Tfun), grid, eos,
                                0.0, Δt, 1; regrid = true)
        @test all(st_i.h[k][i] == st_r.h[k][i] for k in 1:K, i in 1:m.n_cells)
        @test all(st_i.u[k][e] == st_r.u[k][e] for k in 1:K, e in 1:m.n_edges)
        # the tracer remap is the FCT vertical flux: :upwind is refused
        @test_throws ArgumentError ocean_step!(active_state(grid; Tfun = Tfun),
                                               grid, eos, 0.0, Δt; regrid = true,
                                               tracer_advection = :upwind)
    end

    # ---------------------------------------------------------------- D
    @testset "lock channel with regrid (reduced length, 3 d)" begin
        mL = generate_icosahedral_mesh(n_subdivisions = 4)  # config of record
        HL, KL, gp = 2e5, 10, 4e-4
        gridL = VerticalGrid(HL, KL)
        (; g) = williamson_constants()
        ΔT = gp / (g * eos.alpha)
        T_amb = 20.0
        st, _ = lock_channel_state(mL, gridL, eos; g_prime = gp)
        r = gravity_current_benchmark!(st, gridL, eos, T_amb, ΔT;
                                       n_steps = 576, save_every = 48,
                                       frac = 0.1, regrid = true)
        s = r.state
        @test all(isfinite(s.u[k][e]) for k in 1:KL, e in 1:mL.n_edges)
        umax = maximum(abs(s.u[k][e]) for k in 1:KL, e in 1:mL.n_edges)
        Tlo, Thi = extrema(s.T[k][i] for k in 1:KL, i in 1:mL.n_cells)
        bdrift = maximum(abs(b / r.buoyancy[1] - 1) for b in r.buoyancy)
        @info "lock channel, z* regrid, 3 d" umax Tlo Thi bdrift r.xf[end]
        @test umax < 30.0
        @test Tlo >= T_amb - ΔT - 1e-10
        @test Thi <= T_amb + 1e-10
        @test bdrift ≤ 1e-10
        # layers on the z* target at the end
        tgt = layer_thickness(gridL, s.eta)
        @test maximum(abs(s.h[k][i] - tgt[k][i])
                      for k in 1:KL, i in 1:mL.n_cells) < 1e-9 * HL
        # the 10 % front propagates east of the lock edge
        @test r.xf[end] > r.xf[1]
    end
end
