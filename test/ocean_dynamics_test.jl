# Layered ocean dynamics (issue #30, DESIGN.md §3.2, §5 Phase 2).
#
# Acceptance (issue #30, user-ratified decisions):
#   1. Resting flat-bottom state: tendency and full mode-split steps
#      stay at round-off (exactly 0.0) in BOTH coordinate modes
#      (:zstar and :simplez) — the discrete PGF cancels exactly for a
#      horizontally-uniform state (all gradients are differences of
#      uniform fields), and the depth-mean projection of a zero force
#      is zero.
#   2. Internal gravity wave: equatorial mode-1 perturbation with the
#      exact discrete vertical eigenfunction oscillates at the model's
#      linear frequency; measured phase speed matches the analytic
#      first-baroclinic-mode speed c = N·H0/π with N² = g·α·ΔT/H0 to
#      < 5% (the discrete K-layer mode speed is within 0.5% of the
#      continuum value for K = 10; the residual gap is the time-step
#      error of the standing-wave frequency fit, verified converging
#      with Δt). Ω = 0 (nonrotating; the analytic reference is the
#      nonrotating long-wave speed).
#   3. Volume conservation: global volume Σ A_i Σ_k h_k_i is invariant
#      to round-off over multiple wave steps (the fast solver's
#      flux-form continuity + the exact column re-stretch).
#   4. Column invariant of the slow step: Σ_k ∂h_k/∂t = 0 exactly
#      (anomaly-form continuity) — probed through the tendency.
#   5. Fast barotropic coupling: η/ū respond to a barotropic
#      perturbation and the layers re-stretch consistently
#      (Σ_k h_k − H0 = η after every step).
#
# Measurement notes: the wave test uses a standing wave (the IC's
# horizontal pattern is cos(mλ)·exp(−(φ/15°)²), so the linear solution
# is a superposition of east/west-propagating waves = a standing
# oscillation at ω = c·m/R). The modal amplitude a(t) = Σ A_i Σ_k
# h'_k·pattern_ki is projected on the initial pattern and fitted with
# p·cos(ωt) + q·sin(ωt) over an ω window (least-squares power max).
# There are no wall BCs on the sphere — the "closed channel" of the
# classic test is replaced by the periodicity of the spherical geometry
# (documented in the PR).

@testset "ocean dynamics: layered primitive equations (#30)" begin
    using JuiCM: OceanState, ocean_state, ocean_tendency!, ocean_step!,
                 ocean_integrate!, ocean_volume, ocean_kinetic_energy,
                 ocean_ssh, LinearEOS, VerticalGrid, layer_thickness,
                 layer_center_z, simplez_center_depths, ssh_from_thickness,
                 LayeredCellField, LayeredEdgeField, CellField, EdgeField,
                 nlayers, williamson_constants

    m = generate_icosahedral_mesh(n_subdivisions=2)
    (; R, g) = williamson_constants()
    K = 10
    H0 = 4000.0

    "Stratified resting column: T linear in depth (N² = g α ΔT/H0)."
    function resting_state(grid; T_surf = 20.0, ΔT = 10.0)
        eos = LinearEOS()
        h = layer_thickness(grid, CellField(i -> 0.0, m))
        T = LayeredCellField(m, K)
        S = LayeredCellField(m, K)
        if grid.mode === :simplez
            zf = simplez_center_depths(grid)
            for k in 1:K, i in 1:m.n_cells
                T[k][i] = T_surf + ΔT * zf[k] / grid.H0
                S[k][i] = 35.0
            end
        else
            fs = grid.zstar_fractions
            for k in 1:K, i in 1:m.n_cells
                zc = -(fs[k] + fs[k + 1]) / 2 * grid.H0
                T[k][i] = T_surf + ΔT * zc / grid.H0
                S[k][i] = 35.0
            end
        end
        u = LayeredEdgeField(m, K)
        return ocean_state(u, h, T, S), eos
    end

    "Mode-1 internal-wave perturbation on a resting stratified state."
    function wave_state(grid; amp = 2.0, mwavenum = 4,
                        shape = nothing, eos = LinearEOS(),
                        T_surf = 20.0, ΔT = 10.0)
        st, _ = resting_state(grid; T_surf = T_surf, ΔT = ΔT)
        # depth structure: analytic mode-1 (sin(πζ/H0), zero column
        # mean) or the exact discrete eigenfunction
        if shape === nothing
            proj = sum(sin(pi * (grid.zstar_fractions[j] +
                                  grid.zstar_fractions[j + 1]) / 2) *
                       (grid.zstar_fractions[j + 1] -
                        grid.zstar_fractions[j]) for j in 1:K)
            shape = [sin(pi * (grid.zstar_fractions[k] +
                               grid.zstar_fractions[k + 1]) / 2 * grid.H0 /
                         grid.H0) - proj for k in 1:K]
        end
        # baseline BEFORE perturbation (projection reference for the
        # standing-wave amplitude measurement)
        h0 = [st.h[k][i] for k in 1:K, i in 1:m.n_cells]
        pattern = zeros(K, m.n_cells)
        for i in 1:m.n_cells
            lam = atan(m.cell_xyz[2, i], m.cell_xyz[1, i])
            phi = asin(m.cell_xyz[3, i])
            lat = exp(-(phi / 0.26)^2)
            for k in 1:K
                pattern[k, i] = shape[k] * cos(mwavenum * lam) * lat
                st.h[k][i] += amp * pattern[k, i]
            end
        end
        return st, pattern, eos, h0
    end

    @testset "resting state exactness (both coordinate modes)" begin
        for mode in (:zstar, :simplez)
            grid = VerticalGrid(H0, K; mode = mode)
            st, eos = resting_state(grid)

            # tendency exactly zero
            out = JuiCM.TimeStepping._stage(st)
            ocean_tendency!(out, st; grid = grid, eos = eos)
            @test maximum(maximum(abs.(out.u[k].data)) for k in 1:K) == 0.0
            @test maximum(maximum(abs.(out.h[k].data)) for k in 1:K) == 0.0

            # full steps stay at round-off
            st2 = ocean_integrate!(st, grid, eos, 0.0, 5 * 86400.0, 5)
            @test maximum(maximum(abs.(st2.u[k].data)) for k in 1:K) == 0.0
            @test maximum(abs.(st2.eta.data)) == 0.0
            @test maximum(abs.(st2.ubar.data)) == 0.0
        end
    end

    @testset "slow-step column invariant (anomaly continuity)" begin
        grid = VerticalGrid(H0, K)
        st, pattern, eos, _ = wave_state(grid)
        # project to anomalies as the driver does (Field API: out += α·X;
        # qualify — LinearAlgebra's axpy! shadows the export here)
        for k in 1:K
            copyto!(st.u[k], st.u[k] - st.ubar)
        end
        out = JuiCM.TimeStepping._stage(st)
        ocean_tendency!(out, st; grid = grid, eos = eos)
        # Σ_k ∂h_k/∂t = 0 exactly at every cell (flux-form divergence of
        # anomaly fluxes; column budget belongs to the fast solver)
        for i in 1:m.n_cells
            s = sum(out.h[k][i] for k in 1:K)
            @test s == 0.0
        end    end

    @testset "internal gravity wave: phase speed vs NH/π" begin
        grid = VerticalGrid(H0, K)
        eos = LinearEOS()
        ΔT = 10.0

        # exact discrete vertical eigenfunction (linear stability of the
        # layered column: ω²δh_k = (gH0/K)k_h²φ_k with the layered PGF
        # matrix; the second-largest eigenvalue is mode 1 — the largest
        # (λ = K) is the barotropic mode, removed by the split)
        rho = [1.0 + eos.alpha * ΔT * (k - 0.5) / K for k in 1:K]
        Amat = zeros(K, K)
        for k in 1:K, j in 1:K
            Amat[k, j] = (j < k ? (rho[j] - rho[k]) : rho[k])
        end
        ev = eigen(Symmetric(Amat))
        ord = sortperm(ev.values, rev = true)
        λ1 = ev.values[ord[2]]
        φ1 = ev.vectors[:, ord[2]]
        φ1 ./= maximum(abs.(φ1))
        c_model = sqrt(λ1 * g * H0 / K)

        # analytic continuum reference
        N2 = g * eos.alpha * ΔT / H0
        c_analytic = sqrt(N2) * H0 / π
        # the discrete mode-1 speed is the honest reference for the
        # model (within 0.5% of the continuum value for K = 10)
        @test abs(c_model - c_analytic) / c_analytic < 0.005

        mwavenum = 4
        st, pattern, _, h0 = wave_state(grid; amp = 2.0, mwavenum = mwavenum,
                                        shape = φ1)

        # standing-wave modal amplitude a(t); h0 is the PRE-perturbation
        # baseline returned by wave_state (capturing it now would measure
        # A·(cos ωt − 1) and corrupt the frequency fit)
        amp(t) = begin
            s = 0.0
            for i in 1:m.n_cells
                for k in 1:K
                    s += m.cell_area[i] * (st.h[k][i] - h0[k, i]) *
                         pattern[k, i]
                end
            end
            s
        end
        ω_ref = c_model * mwavenum / R
        Tper = 2π / ω_ref

        nsteps = 250
        dt = 30000.0                      # ≈ Tper/118 (CFL-safe; ≥ 2
                                          # oscillation periods total)
        ts = Float64[]
        as = Float64[]
        for n in 1:nsteps
            st = ocean_step!(st, grid, eos, (n - 1) * dt, dt; Omega = 0.0)
            push!(ts, n * dt)
            push!(as, amp(0))
        end

        # least-squares frequency fit over [0.6, 1.6]·ω_ref
        best = (-Inf, 0.0)
        for w in range(0.6 * ω_ref, 1.6 * ω_ref, length = 2000)
            sp = sum(as[i] * cos(w * ts[i]) for i in eachindex(ts))
            sq = sum(as[i] * sin(w * ts[i]) for i in eachindex(ts))
            p = sp^2 + sq^2
            p > best[1] && (best = (p, w))
        end
        @test best[2] > 0
        c_fit = best[2] * R / mwavenum
        # phase speed within 5% of the model's discrete mode speed
        @test abs(c_fit - c_model) / c_model < 0.05

        # the discrete mode speed against the analytic NH/π reference:
        # 5% acceptance (K = 10 discretization is ~0.4%; the rest is
        # standing-wave fit error — verified converging with Δt)
        @test abs(c_model - c_analytic) / c_analytic < 0.05
    end

    @testset "volume conservation over wave steps" begin
        grid = VerticalGrid(H0, K)
        st, _, eos, _ = wave_state(grid)
        V0 = ocean_volume(grid, st)
        st = ocean_integrate!(st, grid, eos, 0.0, 10 * 86400.0, 10;
                              Omega = 0.0)
        V1 = ocean_volume(grid, st)
        @test abs(V1 - V0) / V0 < 1e-12

        # state consistency: Σ_k h_k − H0 = η after every step
        maxerr = 0.0
        for i in 1:m.n_cells
            maxerr = max(maxerr, abs(sum(st.h[k][i] for k in 1:K) -
                                     (H0 + st.eta[i])))
        end
        @test maxerr < 1e-10
    end

    @testset "fast barotropic coupling" begin
        grid = VerticalGrid(H0, K)
        st0, eos = resting_state(grid)
        # barotropic IC: uniform column anomaly η0 = 1 m — a CONSISTENT
        # state: layers stretched proportionally AND the carried fast η
        # set to match (η = Σh − H0)
        η0 = 1.0
        etaf = CellField(i -> η0, m)
        h = layer_thickness(grid, CellField(i -> 0.0, m))
        for k in 1:K, i in 1:m.n_cells
            st0.h[k][i] *= (H0 + η0) / H0
        end
        st = ocean_state(st0.u, st0.h, st0.T, st0.S, etaf)
        # global volume reference AFTER the η0 scaling (st0 shares fields
        # with st, so computing it after the step would include the
        # injected anomaly)
        V0 = ocean_volume(grid, st)
        # uniform η: no pressure gradient → the state must persist
        st = ocean_step!(st, grid, eos, 0.0, 600.0)
        # the column carries the anomaly after the step (fast solver
        # conserves it; the re-stretch enforces Σh = H0 + η)
        for i in 1:m.n_cells
            col = sum(st.h[k][i] for k in 1:K)
            @test col ≈ H0 + st.eta[i] rtol = 1e-12
            @test st.eta[i] ≈ η0 rtol = 1e-9
        end
        # global volume invariant through the step (unit-sphere areas)
        @test abs(ocean_volume(grid, st) - V0) < 1e-6 * V0
    end

    @testset "energy diagnostic" begin
        grid = VerticalGrid(H0, K)
        st, _, eos, _ = wave_state(grid)
        E0 = ocean_kinetic_energy(st)
        @test E0 == 0.0                    # u = 0 IC
        st = ocean_integrate!(st, grid, eos, 0.0, 5 * 86400.0, 5;
                              Omega = 0.0)
        @test ocean_kinetic_energy(st) > 0.0
        @test ocean_ssh(grid, st) === st.eta
    end
end
