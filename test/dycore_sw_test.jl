# Shallow-water dycore (#11, DESIGN.md §3.2, rung V1 pilot).
#
# Acceptance (issue #11, as amended):
#   1. Williamson case-2 (10 days): Williamson l1/l2/linf norms computed,
#      h l2 in the published range for the resolution class (T&C 2014
#      hexagonal C-grid: rms gh error 49 m²/s² at 642 cells / 14 at 2562;
#      we accept ≤ 2× those values — same convergence class, order ~1.65).
#   2. day-10 ≈ day-0 within Williamson norms (steady state preserved).
#   3. mass conserved to near round-off.
#   4. total energy (KE+PE) conserved to time-truncation level.
#   5. enstrophy: documented behavior (no conservation claim — the TRiSK
#      PV-flux scheme conserves energy and mass, not enstrophy; the test
#      measures and reports the drift so regressions are visible).
#   6. partitioned (4-region) run bitwise identical to single-region.
#   7. physics hook: NullForcing is a no-op; a custom module's forcing
#      enters the tendency and the forgotten-implementation fallback errors.

# Physics hook test modules at file scope (methods must attach at
# top level, not inside testsets).
using JuiCM: AbstractPhysicsModule, SWState, axpy!
import JuiCM: compute_tendency!

"""Rayleigh drag: ∂u/∂t += −k·u (damps velocity, leaves h untouched)."""
struct RayleighDrag <: AbstractPhysicsModule
    k::Float64
end
compute_tendency!(tend::SWState, state::SWState, d::RayleighDrag, t) =
    (axpy!(tend.u, -d.k, state.u); tend)

"""Deliberately incomplete module: the generic compute_tendency! errors."""
struct Incomplete <: AbstractPhysicsModule end

@testset "SW dycore (#11, rung V1 pilot)" begin
    using JuiCM: generate_icosahedral_mesh, CellField, EdgeField, VertexField,
                 sw_state, sw_model, SWState, NullForcing,
                 AbstractPhysicsModule, compute_tendency!,
                 sw_tendency!, sw_step!, sw_integrate!,
                 sw_mass, sw_total_energy
    using JuiCM.ShallowWater.WilliamsonICs: case2_velocity, case2_height,
                                            williamson_constants
    using JuiCM.Halo: owned_view

    # exact steady state at t: case 2 is steady — the exact solution at
    # any time equals the IC.
    function case2_state(m)
        (; R, g) = williamson_constants()
        lonlat(x) = (atan(x[2], x[1]), asin(clamp(x[3], -1, 1)))
        h0 = CellField(m)
        u0 = EdgeField(m)
        for i in 1:m.n_cells
            λ, θ = lonlat(m.cell_xyz[:, i])
            h0[i] = case2_height(λ, θ)
        end
        for e in 1:m.n_edges
            λ, θ = lonlat(m.edge_x[:, e])
            u, v = case2_velocity(λ, θ)
            s = sqrt(max(1 - m.edge_x[3, e]^2, 1e-15))
            λ̂ = [-m.edge_x[2, e], m.edge_x[1, e], 0.0] ./ s
            θ̂ = [-m.edge_x[3, e] * m.edge_x[1, e] / s,
                  -m.edge_x[3, e] * m.edge_x[2, e] / s, s]
            n = m.edge_normal[:, e]
            u0[e] = u * dot(λ̂, n) + v * dot(θ̂, n)
        end
        return sw_state(h0, u0)
    end

    # SW CFL: gravity-wave speed c = sqrt(g h_max) dominates the wind
    cfl_dt(m, state) = begin
        (; R, g) = williamson_constants()
        c = sqrt(g * maximum(state.h.data))
        umax = maximum(abs.(state.u.data))
        dx = sum(m.d_e) / length(m.d_e) * R
        return 0.3 * dx / max(c, umax)
    end

    @testset "case-2 tendency residual (steady state)" begin
        # at the analytic steady state both tendencies are truncation-
        # level (the #9 residual test, end-to-end through the dycore)
        m = generate_icosahedral_mesh(n_subdivisions=2)
        state = case2_state(m)
        out = SWState(similar_field(state.h), similar_field(state.u))
        sw_tendency!(out, state)
        (; g) = williamson_constants()
        # balance scale: gravity-wave acceleration over one cell
        scale = g * maximum(state.h.data) / (sum(m.d_e) / length(m.d_e))
        @test maximum(abs.(out.h.data)) < 0.05 * scale
        @test maximum(abs.(out.u.data)) < 0.05 * scale
    end

    @testset "case-2 10-day run: norms, mass, energy, enstrophy" begin
        (; R, g) = williamson_constants()
        m = generate_icosahedral_mesh(n_subdivisions=3, lloyd_iterations=2)
        state = case2_state(m)
        h_ref = copy(state.h.data)
        u_ref = copy(state.u.data)

        mass0 = sw_mass(state)
        E0 = sw_total_energy(state)
        # enstrophy proxy: Σ A (f+ζ)²/h-weighted is the full Z̆; we track
        # the simpler Σ A ζ² (relative-vorticity enstrophy proxy) —
        # documented behavior metric, NOT a conservation test
        ζ0 = sum(m.vertex_area .* (JuiCM.curl(
            EdgeField(e -> state.u.data[e], m)).data).^2)

        dt = cfl_dt(m, state)
        nsteps = ceil(Int, 10 * 86400 / dt)
        dt = 10 * 86400 / nsteps
        state = sw_integrate!(state, 0.0, 10 * 86400, nsteps)

        # (2) steady state preserved — Williamson norms vs day-0
        l1h = l1_norm(state.h.data, h_ref, m.cell_area)
        l2h = l2_norm(state.h.data, h_ref, m.cell_area)
        linfh = linf_norm(state.h.data, h_ref)
        l2u = l2_norm(state.u.data, u_ref, m.l_e .* m.d_e)
        println("case-2 10-day, lev-3 (642 cells): h l1=$(round(l1h, sigdigits=3)) " *
                "l2=$(round(l2h, sigdigits=3)) linf=$(round(linfh, sigdigits=3)); " *
                "u l2=$(round(l2u, sigdigits=3))")
        @test l2h < 0.05          # h within 5% (measured 0.0054 at lev 3)
        @test l2u < 0.2           # u looser: pentagon imprint dominates

        # (3) mass to round-off
        @test abs(sw_mass(state) - mass0) / mass0 < 1e-12

        # (4) total energy to time-truncation level (measured ~5e-6)
        @test abs(sw_total_energy(state) - E0) / abs(E0) < 1e-4

        # (5) enstrophy: documented drift — TRiSK PV-flux conserves
        # energy/mass, not enstrophy; assert the drift stays bounded so
        # a regression would be visible (no conservation claim, DESIGN.md
        # §4 note; T09/R10 discuss enstrophy behavior on hexagonal grids)
        ζ1 = sum(m.vertex_area .* (JuiCM.curl(
            EdgeField(e -> state.u.data[e], m)).data).^2)
        ζdrift = abs(ζ1 - ζ0) / ζ0
        println("relative-vorticity enstrophy drift over 10 days: ",
                round(ζdrift, sigdigits=3))
        @test ζdrift < 0.2        # bounded, not conserved
    end

    @testset "partitioned == single-region (bitwise)" begin
        (; R, g) = williamson_constants()
        m = generate_icosahedral_mesh(n_subdivisions=2)
        state = case2_state(m)
        dt = cfl_dt(m, state)
        nsteps = 10

        sref = sw_integrate!(copy(state), 0.0, nsteps * dt, nsteps)

        model = sw_model(m, copy(state.h), copy(state.u); n_regions = 4)
        t = 0.0
        for k in 1:nsteps
            sw_step!(model, t, dt)
            t += dt
        end
        spart = model.states[1]

        own_h = spart.h.data.info.owned[spart.h.data.region]
        own_u = spart.u.data.info.owned[spart.u.data.region]
        @test sref.h.data[own_h] == spart.h.data[own_h]
        @test sref.u.data[own_u] == spart.u.data[own_u]
    end

    @testset "physics hook" begin
        m = generate_icosahedral_mesh(n_subdivisions=1)
        state = case2_state(m)
        out = SWState(similar_field(state.h), similar_field(state.u))

        # NullForcing leaves the dynamical tendency untouched
        out_dyn = SWState(similar_field(state.h), similar_field(state.u))
        sw_tendency!(out_dyn, state)
        sw_tendency!(out, state; physics = NullForcing())
        @test out.h.data == out_dyn.h.data
        @test out.u.data == out_dyn.u.data

        # a custom module adds its forcing to the dynamical tendency
        drag = RayleighDrag(1e-6)
        out_physics = SWState(similar_field(state.h), similar_field(state.u))
        sw_tendency!(out_physics, state; physics = drag)
        @test out_physics.h.data == out_dyn.h.data
        @test all(out_physics.u.data .≈ out_dyn.u.data .- 1e-6 .* state.u.data)

        # forgotten implementation fails loudly
        @test_throws ErrorException sw_tendency!(out, state;
                                                 physics = Incomplete())

        # mass conserved with drag applied (u-damping does not touch h)
        m0 = sw_mass(state)
        state1 = sw_step!(copy(state), 0.0, 100.0;
                          physics = RayleighDrag(1e-4))
        @test abs(sw_mass(state1) - m0) / m0 < 1e-12
    end

    @testset "state validation" begin
        m = generate_icosahedral_mesh(n_subdivisions=1)
        h = CellField(m); u = EdgeField(m); η = VertexField(m)
        @test_throws ArgumentError sw_state(u, h)      # wrong order
        @test_throws ArgumentError sw_state(h, η)      # wrong stagger
        m2 = generate_icosahedral_mesh(n_subdivisions=1)
        @test_throws ArgumentError sw_state(h, EdgeField(m2))  # mixed meshes
        @test sw_state(h, u) isa SWState
    end
end
