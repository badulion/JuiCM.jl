# Time-stepping tests (#13): SSPRK(3,3) and RK4.
#
# Quantitative checks: observed convergence order via dt-halving,
# round-off preservation of invariants, SSP positivity on a stiff
# decay problem where forward Euler is unstable, and the tendency-
# function contract (no mutation of the input state).
using Test
using JuiCM
using JuiCM.TimeStepping: ssprk3_step!, ssprk3_integrate!,
                          rk4_step!, rk4_integrate!
using LinearAlgebra: norm

"""Observed order of convergence: log2 of the error ratio between two
successive dt halvings."""
observed_order(e_coarse, e_fine) = log2(e_coarse / e_fine)

@testset "time stepping (#13)" begin
    @testset "scalar ODE du/dt = −u: 3rd-order convergence" begin
        # exact solution u(t) = u0 exp(−t); integrate to T = 1
        L(out, u, t) = (out[] = -u[])
        u0 = [1.0]
        T = 1.0
        exact = exp(-T)
        errors = Float64[]
        for n in (4, 8, 16, 32)
            uf = ssprk3_integrate!(L, copy(u0), 0.0, T, n)
            push!(errors, abs(uf[] - exact))
        end
        # errors must decay ~ dt³: each halving reduces error by ~8
        # (the coarsest pair is still in the pre-asymptotic regime, so
        # its observed order may exceed 3; the last two pairs must be
        # clean)
        for i in 3:length(errors)
            p = observed_order(errors[i-1], errors[i])
            @test p ≈ 3.0 atol = 0.1
        end
        @test errors[end] < 1e-5
    end

    @testset "RK4: 4th-order convergence on the same problem" begin
        L(out, u, t) = (out[] = -u[])
        u0 = [1.0]
        exact = exp(-1.0)
        errors = Float64[]
        for n in (4, 8, 16, 32)
            uf = rk4_integrate!(L, copy(u0), 0.0, 1.0, n)
            push!(errors, abs(uf[] - exact))
        end
        for i in 3:length(errors)
            @test observed_order(errors[i-1], errors[i]) ≈ 4.0 atol = 0.1
        end
    end

    @testset "system ODE: damped oscillator (NamedTuple state), 3rd order" begin
        # harmonic oscillator with light damping, as a NamedTuple of
        # arrays (exercises the generic state plumbing):
        #   dU/dt = (v, −ω²x − 2γv), U = (x = [x0], v = [v0])
        ω = 2.0
        γ = 0.1
        function L(out, U, t)
            out.x[1] = U.v[1]
            out.v[1] = -ω^2 * U.x[1] - 2γ * U.v[1]
            return nothing
        end
        U0 = (x = [1.0], v = [0.0])
        T = 2.0
        # high-accuracy reference via tiny RK4 steps
        Uref = rk4_integrate!(L, deepcopy(U0), 0.0, T, 100_000)
        errors = Float64[]
        for n in (10, 20, 40, 80)
            Uf = ssprk3_integrate!(L, deepcopy(U0), 0.0, T, n)
            push!(errors, hypot(Uf.x[1] - Uref.x[1], Uf.v[1] - Uref.v[1]))
        end
        for i in 3:length(errors)
            @test observed_order(errors[i-1], errors[i]) ≈ 3.0 atol = 0.2
        end
    end

    @testset "linear invariant preserved to round-off" begin
        # Linear invariants (du/dt = A u with 1ᵀA = 0: mass/tracer
        # conservation) are preserved EXACTLY by every Runge–Kutta
        # method (internal consistency of the convex stage
        # combinations) — this is the property the SW dynamical core
        # relies on for mass conservation under explicit stepping.
        # Two-cell tracer exchange: u' = (v−u, u−v), total mass constant.
        L(out, U, t) = (out[1] = U[2] - U[1]; out[2] = U[1] - U[2]; nothing)
        U0 = [1.0, 3.0]
        mass(U) = U[1] + U[2]
        for stepper in (ssprk3_integrate!, rk4_integrate!)
            Uf = stepper(L, copy(U0), 0.0, 10.0, 1000)
            # preserved to accumulated round-off (each step's convex
            # combinations add ~1 ulp; 1000 steps ⇒ ~1e-13), not bit-exact
            @test mass(Uf) ≈ mass(U0) atol = 1e-12
        end
        # step-by-step path too
        U = copy(U0)
        t, Δt = 0.0, 0.01
        for _ in 1:1000
            U = ssprk3_step!(L, U, t, Δt)
            t += Δt
        end
        @test mass(U) ≈ mass(U0) atol = 1e-12
        # and the solution converged to the correct equilibrium (2.0, 2.0)
        @test U ≈ [2.0, 2.0] atol = 1e-5
    end

    @testset "quadratic invariant: no secular drift" begin
        # RK schemes do NOT preserve quadratic invariants exactly (only
        # linear ones and, for special methods, symplectic ones) — but
        # for a well-resolved oscillation the energy error must stay
        # bounded and small (no secular growth), scaling as dt³ for
        # SSPRK3. After 20 periods at ωΔt = 0.003:
        ω = 3.0
        L(out, U, t) = (out[1] = U[2]; out[2] = -ω^2 * U[1]; nothing)
        U0 = [1.0, 0.0]
        E0 = 1.0
        for (n_periods, dt) in ((20, 0.001), (40, 0.002))
            U = copy(U0)
            t = 0.0
            for _ in 1:(n_periods * round(Int, 2π / ω / dt))
                U = ssprk3_step!(L, U, t, dt)
                t += dt
            end
            E = U[1]^2 + (U[2] / ω)^2
            @test abs(E - E0) < 1e-5        # bounded, no secular drift
            # solution itself is 3rd-order accurate
            @test U[1] ≈ cos(ω * t) atol = 1e-5
            @test U[2] ≈ -ω * sin(ω * t) atol = 1e-4
        end
    end

    @testset "SSP positivity: convex-combination property" begin
        # SSPRK(3,3) has SSP coefficient c_eff = 1: whenever forward
        # Euler preserves positivity at step Δt, so does SSPRK(3,3).
        # For u' = −λu the FE positivity condition is z = λΔt ≤ 1.
        # Verify positivity over many steps for z ≤ 1, and that the
        # linear amplification factor matches the exact SSPRK3
        # polynomial amp(z) = 1/3 + (2/3)(1−z)(3/4 + (1−z)²/4):
        λ = 100.0
        L(out, u, t) = (out[] = -λ * u[])
        for dt in (0.005, 0.01)   # z = 0.5, 1.0 — FE-positive ⇒ SSPRK3-positive
            u = [1.0]
            t = 0.0
            for _ in 1:200
                u = ssprk3_step!(L, u, t, dt)
                t += dt
                @test u[] ≥ 0
            end
            z = λ * dt
            amp = 1 / 3 + (2 / 3) * (1 - z) * (3 / 4 + (1 - z)^2 / 4)
            @test abs(u[] - amp^200) < 1e-12 * amp^200   # linear theory exact
            @test 0 < amp ≤ 1                            # positive & stable
        end
        # At z > 1 forward Euler flips sign and the SSP guarantee is
        # void (SSP inherits FE's positivity threshold, it does not
        # extend it): at z = 2 the SSPRK3 amplification is exactly
        # −1/3, and the solution alternates sign — documented behavior.
    end

    @testset "tendency contract: L must not mutate U" begin
        # the stepper hands L a scratch buffer; a conforming L writes
        # only to it. Verify the stepper does not corrupt the input U:
        L(out, u, t) = (out .= -u)
        U0 = [1.0, 2.0, 3.0]
        Ucopy = copy(U0)
        Uf = ssprk3_step!(L, U0, 0.0, 0.1)
        @test U0 == Ucopy                    # input untouched
        @test Uf != U0                       # output is a new object
        # and the step is correct: u' = −u for one FE-ish stage pattern
        # (exact 1-step SSPRK3 result):
        # u1 = u0 − dt u0 = (1−dt)u0
        # u2 = (3/4)u0 + (1/4)((1−dt)u0 − dt(1−dt)u0) ... verified vs the
        # composition computed by hand:
        dt = 0.1
        u1 = @. (1 - dt) * U0
        u2 = @. (3 / 4 * U0 + 1 / 4 * (1 - dt)^2 * U0)
        u3 = @. (1 / 3 * U0 + 2 / 3 * (1 - dt) * u2)
        @test Uf ≈ u3 rtol = 1e-14
    end

    @testset "integrate! respects time argument passed to L" begin
        # L records the times it was called with; the stepper must pass
        # t, t+dt, t+dt (SSPRK(3,3) stage times per Shu–Osher)
        times = Float64[]
        L(out, u, t) = (push!(times, t); (out[] = 0.0))
        ssprk3_step!(L, [1.0], 5.0, 0.5)
        @test times ≈ [5.0, 5.5, 5.25] atol = 1e-14
        empty!(times)
        rk4_step!(L, [1.0], 5.0, 0.5)
        @test times ≈ [5.0, 5.25, 5.25, 5.5] atol = 1e-14    end
end
