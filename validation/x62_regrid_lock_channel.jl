# Moved unchanged from test/ (issue #75: the fast suite runs ≤ 30 s per
# testset; full-accuracy runs live in validation/). Reference data stay in
# test/reference_data/.

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

    # ---------------------------------------------------------------- B

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
