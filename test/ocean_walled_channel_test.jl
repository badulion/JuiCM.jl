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
    @testset "V3: lock-exchange front Froude number (Shin et al. 2004)" begin
        # Configuration (scale-free problem: only F = U/√(g′H) matters):
        # closed channel 60 × 24 mean cell spacings on the
        # n_subdivisions = 5 mesh (d̄_e ≈ 240 km), full-depth lock at the
        # west end (15 % of the length), H = 200 m, g′ = 0.5 m/s²
        # (c0 = √(g′H) = 10 m/s), 40 z* layers, no rotation, shock-
        # sensor viscosity cap ν_h = 4.8e5 m²/s (grid Reynolds number
        # U·dx/ν = 2.5 with U = c0/2 — the Ilıcak et al. 2012 / MPAS-O
        # lock-exchange setting). Measured sensitivity (docs/ and the
        # PR record): F rises with layer count (10/20/40/60 →
        # 0.43/0.48/0.494/0.498 on the 10 % contour at this width) and is
        # converged in dx and Δt.
        m = generate_icosahedral_mesh(n_subdivisions = 5, lloyd_iterations = 2)
        H, K, gp = 200.0, 40, 0.5
        grid = VerticalGrid(H, K)
        ch = walled_channel(m; n_long = 60, n_wide = 24)
        st, λg = walled_lock_state(m, grid, eos, ch; g_prime = gp,
                                   lock_frac = 0.15)
        ΔT = gp / (g * eos.alpha)
        c0 = sqrt(gp * H)
        de = sum(m.d_e) / length(m.d_e)
        days = 8.0
        Δt0 = 0.25 * de * R / (1.5 * c0)
        n_steps = ceil(Int, days * 86400 / Δt0)
        Δt = days * 86400 / n_steps
        r = walled_lock_exchange!(st, grid, eos, ch, 20.0, ΔT, λg;
                                  n_steps = n_steps, Δt = Δt,
                                  fracs = (0.02, 0.1, 0.25), nu_h = 4.8e5)
        s = r.state
        # slumping-phase front Froude number on three contours
        Fs = [front_speed(r.xf[j], r.t; window = (1.5 * 86400, days * 86400)) / c0
              for j in 1:3]
        umax = maximum(abs(s.u[k][e]) for k in 1:K, e in 1:m.n_edges)
        bdrift = maximum(abs(b / r.buoyancy[1] - 1) for b in r.buoyancy)
        Tlo, Thi = extrema(s.T[k][i] for k in 1:K, i in 1:m.n_cells)
        x_end = r.xf[2][end]
        @info "V3 walled lock exchange" F_2pct = Fs[1] F_10pct = Fs[2] F_25pct = Fs[3] umax bdrift x_end
        # Shin, Dalziel & Linden (2004), full-depth lock: F = 0.48–0.50
        # (Benjamin's energy-conserving front: 0.5). Acceptance (issue
        # #40, ratified 2026-10-16): within ~10 % of the published value.
        ref = Dict{String, Float64}()
        for ln in eachline(joinpath(@__DIR__, "reference_data",
                                    "shin2004_lock_exchange.csv"))
            startswith(ln, "#") && continue
            f = split(ln, ",")
            ref[f[1]] = parse(Float64, f[2])
        end
        Fref_lo = ref["front_froude_full_depth_lo"]
        Fref_hi = ref["front_froude_full_depth_hi"]
        tol = ref["tolerance_relative"]
        for F in Fs
            @test (1 - tol) * Fref_lo <= F <= (1 + tol) * Fref_hi
        end
        # the front speed itself: Benjamin's (1968) energy-conserving
        # value F = 0.5 for the full-depth lock, to within 5 % on the
        # nose contours (stricter than the ratified 10 % band above)
        @test abs(Fs[1] / 0.5 - 1) < 0.05
        @test abs(Fs[2] / 0.5 - 1) < 0.05
        # a sharp front: the contours move together (a wedge-shaped,
        # smeared current gives a slow 25 % contour)
        @test maximum(Fs) - minimum(Fs) < 0.03
        # front position vs the published constant-speed curve (from
        # the gate at t = 0), within 10 %
        x_ref = (Fref_lo + Fref_hi) / 2 * c0 * days * 86400
        @test abs(x_end / x_ref - 1) < tol
        # buoyancy conserved to round-off, monotone transport, bounded
        @test bdrift < 1e-12
        @test Tlo >= 20.0 - ΔT - 1e-10
        @test Thi <= 20.0 + 1e-10
        @test umax < 2 * c0
    end
end
