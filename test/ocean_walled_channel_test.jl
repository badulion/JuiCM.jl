# Walled-channel lock exchange (issue #40, rung V3) and the dycore pieces it
# needs: solid no-flux edges, free-slip wall vorticity, wall-cell kinetic
# energy, and the complete-tendency barotropic/baroclinic split.

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using JuiCM.TRiSK: potential_vorticity, curl
using JuiCM.Ocean: ocean_step!
using JuiCM.TRiSK: edge_mean
using JuiCM.ShallowWater: williamson_constants
using Random
using Test

@testset "walled channel (#40, rung V3)" begin
    eos = LinearEOS(alpha = 2.0e-3)
    (; g, R) = williamson_constants()

    # ------------------------------------------------------------- walls
    @testset "closed edges: no flux, regions isolated, buoyancy exact" begin
        m = generate_icosahedral_mesh(n_subdivisions = 3)
        grid = VerticalGrid(200.0, 4)
        ch = walled_channel(m; n_long = 12, n_wide = 6)
        @test !isempty(ch.closed_edges)
        @test all(ch.inside[m.edge_cells[1, e]] != ch.inside[m.edge_cells[2, e]]
                  for e in ch.closed_edges)
        st, λg = walled_lock_state(m, grid, eos, ch; g_prime = 0.5)
        ΔT = 0.5 / (g * eos.alpha)
        b0 = buoyancy_content(st.T, st.h; T_amb = 20.0)
        for n in 1:6
            st = ocean_step!(st, grid, eos, (n - 1) * 3600.0, 3600.0;
                             Omega = 0.0, regrid = true, nu_h = 1e5,
                             closed_edges = ch.closed_edges)
        end
        # no normal flow through any wall, in any layer or the fast mode
        @test all(st.u[k][e] == 0.0 for k in 1:4, e in ch.closed_edges)
        @test all(st.ubar[e] == 0.0 for e in ch.closed_edges)
        # the outside ocean is uniform and isolated: it stays at rest
        outside_edges = [e for e in 1:m.n_edges
                         if !ch.inside[m.edge_cells[1, e]] &&
                            !ch.inside[m.edge_cells[2, e]]]
        @test maximum(abs(st.u[k][e]) for k in 1:4, e in outside_edges) < 1e-12
        @test all(st.T[k][i] == 20.0 for k in 1:4, i in 1:m.n_cells
                  if !ch.inside[i])
        # the flow inside is non-trivial, buoyancy content exact
        @test maximum(abs(st.u[k][e]) for k in 1:4, e in 1:m.n_edges) > 1e-3
        b = buoyancy_content(st.T, st.h; T_amb = 20.0)
        @test abs(b / b0 - 1) < 1e-12
        # T stays within the initial bounds (monotone transport)
        Tlo, Thi = extrema(st.T[k][i] for k in 1:4, i in 1:m.n_cells)
        @test Tlo >= 20.0 - ΔT - 1e-10
        @test Thi <= 20.0 + 1e-10
    end

    @testset "free-slip wall vorticity and wall-cell kinetic energy" begin
        m = generate_icosahedral_mesh(n_subdivisions = 3)
        ch = walled_channel(m; n_long = 12, n_wide = 6)
        wv = wall_vertices(m, ch.closed_edges)
        @test wall_vertices(m, nothing) === nothing
        @test issorted(wv) && allunique(wv)
        rng = MersenneTwister(4040)
        u = EdgeField(e -> randn(rng), m)
        h = CellField(i -> 1.0 + 0.1 * m.cell_xyz[3, i], m)
        # ζ_v = 0 on wall vertices (Adcroft & Marshall 1998), unchanged
        # elsewhere
        q0 = potential_vorticity(u, h)
        q1 = potential_vorticity(u, h; free_slip_vertices = wv)
        @test all(q1[v] == 0.0 for v in wv)
        wset = Set(wv)
        @test all(q1[v] == q0[v] for v in 1:m.n_vertices if !(v in wset))
        @test any(q0[v] != 0.0 for v in wv)
        # the free-slip Laplacian stays negative semi-definite
        for trial in 1:5
            ur = EdgeField(e -> randn(rng), m)
            L = laplacian(ur; free_slip_vertices = wv)
            @test sum(m.l_e[e] * m.d_e[e] * ur[e] * L[e] for e in 1:m.n_edges) <= 0
        end
        # wall cells average u² over their OPEN edges only: a uniform
        # |u| = c on open edges (0 on the walls) gives the same K as an
        # interior cell with |u| = c on ALL edges, c²·Σ_e(A_e/4)/A_i
        # (≈ c²; the spherical kite areas sum to A_i only approximately);
        # without the wall list the closed edges dilute it
        c = 0.7
        cl = Set(ch.closed_edges)
        uc = EdgeField(e -> e in cl ? 0.0 : c, m)
        Kw = kinetic_energy(uc; closed_edges = ch.closed_edges)
        K0 = kinetic_energy(uc)
        wallcells = unique([m.edge_cells[s, e] for e in ch.closed_edges for s in 1:2])
        Kint = kinetic_energy(EdgeField(e -> c, m))   # interior reference
        @test all(abs(Kw[i] - Kint[i]) < 1e-12 for i in wallcells)
        @test all(K0[i] < Kint[i] - 1e-3 for i in wallcells)
    end

    # ----------------------------------------------------- the mode split
    @testset "split: layers' depth-mean velocity equals ū" begin
        # the pre-#40 split removed only the pressure-force depth-mean, so
        # the Coriolis/Bernoulli depth-mean accumulated in u′ as momentum
        # without mass (measured 3 m/s in this channel). Now the complete
        # depth-mean forces the fast mode and u′ stays depth-mean free.
        m = generate_icosahedral_mesh(n_subdivisions = 3)
        grid = VerticalGrid(200.0, 6)
        ch = walled_channel(m; n_long = 12, n_wide = 6)
        st, _ = walled_lock_state(m, grid, eos, ch; g_prime = 0.5)
        for n in 1:8
            st = ocean_step!(st, grid, eos, (n - 1) * 3600.0, 3600.0;
                             Omega = 0.0, regrid = true, nu_h = 1e5,
                             closed_edges = ch.closed_edges)
        end
        hm = [edge_mean(st.h[k]) for k in 1:6]
        worst = 0.0
        for e in 1:m.n_edges
            num = sum(hm[k][e] * st.u[k][e] for k in 1:6)
            den = sum(hm[k][e] for k in 1:6)
            worst = max(worst, abs(num / den - st.ubar[e]))
        end
        @test worst < 1e-12
        @test maximum(abs(st.u[k][e]) for k in 1:6, e in 1:m.n_edges) > 1e-2
    end

    @testset "momentum remap: any interface displacement (overlap form)" begin
        # thin layers + a large vertical excursion: target interfaces
        # displaced by up to ~3 source layers. The swept-flux form of #62
        # (one flux per interface) is only valid for |d| ≤ ĥ_donor; the
        # overlap remap must stay conservative and bounded for any d.
        m = generate_icosahedral_mesh(n_subdivisions = 2)
        Kc = 12
        hsrc = LayeredCellField((i, k) -> k <= 6 ? 2.0 : 30.0, m, Kc)
        hdst = LayeredCellField((i, k) -> 16.0, m, Kc)   # same total 192
        rng = MersenneTwister(62)
        u = LayeredEdgeField((e, k) -> randn(rng), m, Kc)
        hmean(h, k, e) = 0.5 * (h[k][m.edge_cells[1, e]] + h[k][m.edge_cells[2, e]])
        mom0 = [sum(hmean(hsrc, k, e) * u[k][e] for k in 1:Kc) for e in 1:m.n_edges]
        lo0 = [minimum(u[k][e] for k in 1:Kc) for e in 1:m.n_edges]
        hi0 = [maximum(u[k][e] for k in 1:Kc) for e in 1:m.n_edges]
        JuiCM.Ocean._remap_edge_columns!(u, hsrc, hdst)
        mom1 = [sum(hmean(hdst, k, e) * u[k][e] for k in 1:Kc) for e in 1:m.n_edges]
        @test maximum(abs.(mom1 .- mom0)) < 1e-11 * 192
        @test all(lo0[e] - 1e-12 <= u[k][e] <= hi0[e] + 1e-12
                  for k in 1:Kc, e in 1:m.n_edges)
    end

    # ------------------------------------------------------- rung V3
end
