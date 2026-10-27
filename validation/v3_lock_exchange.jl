# Moved unchanged from test/ (issue #75: the fast suite runs ≤ 30 s per
# testset; full-accuracy runs live in validation/). Reference data stay in
# test/reference_data/.

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


    # ----------------------------------------------------- the mode split


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
        for ln in eachline(joinpath(@__DIR__, "..", "test", "reference_data",
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
