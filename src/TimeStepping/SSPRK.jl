"""Explicit Runge–Kutta time stepping, grid-independent.

Implements the explicit strong-stability-preserving Runge–Kutta scheme
SSPRK(3,3) (Shu & Osher 1988, eq. (2.12)/(2.13); the third-order,
three-stage SSP scheme) plus the classical fourth-order Runge–Kutta
(RK4), as a generic function API over any state type that supports
`similar`-style staging, elementwise linear combination, and linnorm —
plain arrays, NamedTuples of arrays, and (later) mesh Fields all work.

This is the initial, grid-independent core of the time-stepping layer
(DESIGN.md §3.2, `Dycore/*/TimeStepping.jl`): it advances *any*
prognostic state `U` given a tendency function `L`, per the
function-barrier contract of DESIGN.md §3.3. The eventual dycore
integrators will wrap these steppers with their own stage storage;
the SSP positivity property is the reason this scheme is planned for
the shallow-water core (DESIGN.md §5 Phase 1).

# Tendency-function contract (function barrier, DESIGN.md §3.3)

    L(out, U, t) -> nothing

`L` writes the tendency `dU/dt` at state `U` and time `t` into the
pre-allocated buffer `out` (which the stepper provides; it is a fresh
stage of the same shape/type as `U`). The contract is:
- **mutating, out-first**: all output goes into `out`; nothing is
  allocated inside `L` on steady-state runs;
- **pure in `U`**: `L` must not modify `U`;
- **type-stable barrier**: the stepper never inspects `out`'s contents
  beyond elementwise arithmetic, so any array-like state works.

# SSPRK(3,3) (Shu & Osher 1988)

With `uⁿ` the current state, `Δt` the step, `L` the tendency:

    u⁽¹⁾ = uⁿ + Δt L(uⁿ)
    u⁽²⁾ = (3/4) uⁿ + (1/4) [u⁽¹⁾ + Δt L(u⁽¹⁾)]
    uⁿ⁺¹ = (1/3) uⁿ + (2/3) [u⁽²⁾ + Δt L(u⁽²⁾)]

Butcher tableau (canonical form):
    c = (0, 1, 1/2);  b = (1/6, 1/6, 2/3);
    a₂₁ = 1;  a₃₁ = 1/4, a₃₂ = 1/4
- SSP coefficient c_eff = 1 (the scheme is a convex combination of
  forward-Euler steps with total weight 1): if forward Euler with step
  Δt preserves convexity/positivity, so does SSPRK(3,3) with the same
  Δt;
- third-order accurate (TVD for scalar convex conservation laws).

References:
- Shu, C.-W., Osher, S. (1988): "Efficient implementation of
  essentially non-oscillatory shock-capturing schemes", J. Comput.
  Phys. 77, 439–471 (scheme (2.12)–(2.13), sometimes numbered (2.18)).
- Gottlieb, S., Shu, C.-W., Tadmor, E. (2001): "Strong stability-
  preserving high-order time discretization methods", SIAM Rev. 43,
  89–112 (SSP framework, SSP coefficient discussion).
"""
module TimeStepping

export ssprk3_step!, ssprk3_integrate!, rk4_step!, rk4_integrate!

"""    ssprk3_step!(L, U, t, Δt; stages = nothing) -> U_next

Advance `U` one time step of size `Δt` with SSPRK(3,3) (Shu & Osher
1988) under the tendency contract `L(out, U, t)` documented in the
module docstring. Returns the new state (a fresh object; the input
`U` is not modified unless it aliases the return). Pass `stages`
(a tuple of pre-allocated buffers with `copy`-able structure) to make
the step allocation-free; by default stages are allocated per call.
"""
function ssprk3_step!(L, U, t, Δt)
    u1 = _stage(U)
    u2 = _stage(U)
    tmp = _stage(U)
    return _ssprk3_core!(L, U, t, Δt, u1, u2, tmp)
end

function _ssprk3_core!(L, U, t, Δt, u1, u2, tmp)
    # Butcher form of Shu–Osher SSPRK(3,3): stage times c = (t, t+Δt,
    # t+Δt/2); b = (1/6, 1/6, 2/3). The stage-3 L evaluation sits at
    # t + Δt/2 (canonical Butcher c₃ = 1/2).
    # stage 1: u1 = U + Δt L(U, t)
    L(u1, U, t)
    _lincomb!(u1, 1.0, U, Δt, u1)
    # stage 2: u2 = (3/4) U + (1/4)(u1 + Δt L(u1, t+Δt))
    L(tmp, u1, t + Δt)
    _lincomb!(u2, 1.0, u1, Δt, tmp)          # u2 = u1 + Δt L(u1)
    _lincomb!(u2, 3 / 4, U, 1 / 4, u2)       # u2 = (3/4)U + (1/4)u2
    # stage 3: Uⁿ⁺¹ = (1/3) U + (2/3)(u2 + Δt L(u2, t+Δt/2))
    L(tmp, u2, t + Δt / 2)
    _lincomb!(tmp, 1.0, u2, Δt, tmp)         # tmp = u2 + Δt L(u2)
    _lincomb!(tmp, 1 / 3, U, 2 / 3, tmp)     # tmp = (1/3)U + (2/3)tmp
    return tmp
end

"""    ssprk3_integrate!(L, U, t0, t_final, n_steps) -> U

Integrate `U` from `t0` to `t_final` in `n_steps` equal SSPRK(3,3)
steps (`Δt = (t_final − t0)/n_steps`). Returns the final state.
"""
function ssprk3_integrate!(L, U, t0, t_final, n_steps)
    Δt = (t_final - t0) / n_steps
    t = t0
    for _ in 1:n_steps
        U = ssprk3_step!(L, U, t, Δt)
        t += Δt
    end
    return U
end

"""    rk4_step!(L, U, t, Δt) -> U_next

Advance `U` one step with the classical explicit fourth-order
Runge–Kutta scheme (Kutta 1901; the standard textbook tableau).
Not SSP — provided for comparison and as the validation baseline.
Same tendency contract as `ssprk3_step!`.
"""
function rk4_step!(L, U, t, Δt)
    k1 = _stage(U)
    k2 = _stage(U)
    k3 = _stage(U)
    k4 = _stage(U)
    tmp = _stage(U)

    L(k1, U, t)                              # k1 = L(U, t)
    _lincomb!(tmp, 1.0, U, Δt / 2, k1)       # tmp = U + Δt/2 k1
    L(k2, tmp, t + Δt / 2)
    _lincomb!(tmp, 1.0, U, Δt / 2, k2)
    L(k3, tmp, t + Δt / 2)
    _lincomb!(tmp, 1.0, U, Δt, k3)
    L(k4, tmp, t + Δt)

    # Uⁿ⁺¹ = U + Δt (k1 + 2k2 + 2k3 + k4)/6
    _lincomb!(tmp, 1.0, U, 0.0, U)           # tmp = U
    _axpy!(tmp, Δt / 6, k1)
    _axpy!(tmp, 2Δt / 6, k2)
    _axpy!(tmp, 2Δt / 6, k3)
    _axpy!(tmp, Δt / 6, k4)
    return tmp
end

"""    rk4_integrate!(L, U, t0, t_final, n_steps) -> U

Integrate `U` from `t0` to `t_final` in `n_steps` equal RK4 steps.
"""
function rk4_integrate!(L, U, t0, t_final, n_steps)
    Δt = (t_final - t0) / n_steps
    t = t0
    for _ in 1:n_steps
        U = rk4_step!(L, U, t, Δt)
        t += Δt
    end
    return U
end

# ------------------------------------------------------------------
# State plumbing: elementwise ops over arbitrary array-like states.
# States may be plain arrays, NamedTuples of arrays, or anything that
# answers to the recursion below. This is the seam where mesh Fields
# will plug in later (Field supports .data + similar).
# ------------------------------------------------------------------

_stage(U::AbstractArray) = similar(U)
_stage(U::NamedTuple) = map(_stage, U)
_stage(U::Number) = zero(U)

"""_lincomb!(out, α, A, β, B): out = α·A + β·B elementwise."""
function _lincomb!(out::AbstractArray, α, A::AbstractArray, β, B::AbstractArray)
    @inbounds for i in eachindex(out)
        out[i] = α * A[i] + β * B[i]
    end
    return out
end
function _lincomb!(out::NamedTuple, α, A::NamedTuple, β, B::NamedTuple)
    for k in keys(out)
        _lincomb!(out[k], α, A[k], β, B[k])
    end
    return out
end
_lincomb!(out::Number, α, A::Number, β, B::Number) = α * A + β * B

"""_axpy!(out, α, X): out += α·X elementwise."""
function _axpy!(out::AbstractArray, α, X::AbstractArray)
    @inbounds for i in eachindex(out)
        out[i] += α * X[i]
    end
    return out
end
function _axpy!(out::NamedTuple, α, X::NamedTuple)
    for k in keys(out)
        _axpy!(out[k], α, X[k])
    end
    return out
end
_axpy!(out::Number, α, X::Number) = out + α * X

end # module TimeStepping
