"""Williamson et al. (1992) shallow-water test cases: analytic initial
conditions.

Grid-independent reference states for the validation ladder (DESIGN.md
§4, rung V1): every field is an analytic function of longitude and
latitude (or 3D position) evaluated at arbitrary points, so the
conditions can be sampled on any mesh — or on pure quadrature grids —
before and independently of the dynamical core.

Formulas follow Williamson, Drake, Hack, Jakob, Swarztrauber (1992),
"A standard test set for numerical approximations to the shallow water
equations in spherical geometry", J. Comput. Phys. 102, 211–224 —
cross-verified against the STSWM v2.0 Fortran source (`init.f`,
`anlytc.f`; Jakob is a co-author of the paper) and, for all constants
and the Case 6 formulas, independently against ClimaCore.jl's
`examples/sphere/shallow_water.jl`.

Conventions:
- longitudes λ ∈ [0, 2π) increasing **eastward**, latitudes
  θ ∈ [−π/2, π/2] increasing northward (the Williamson/STSWM convention)
- `u` = zonal (eastward) wind component, `v` = meridional wind component
  at the given point
- `h` = fluid thickness (geopotential divided by g); Case 5 also exposes
  the mountain height `b`, and its total depth `h + b` equals the
  Case 2 profile with its own h₀

Implemented cases ( Williamson et al. 1992 section numbers):
- **Case 2** — global steady-state nonlinear zonal geostrophic flow
  (eqs. 95–97): exact steady solution of the nonlinear SWE
- **Case 5** — zonal flow over an isolated mountain (eqs. 129–134):
  Case 2-style balance over conical terrain; run 15 days, no analytic
  time solution (reference simulations only)
- **Case 6** — Rossby–Haurwitz wavenumber-4 wave (eqs. 137–144): exact
  solution of the nondivergent barotropic vorticity equation, drifting
  slowly eastward; the analytic continuation shifts longitude by
  Δλ(t) = [R(3+R)ω − 2Ω]/[(1+R)(2+R)] · t (STSWM `anlytc.f`)
"""
module WilliamsonICs

export williamson_constants, case2_velocity, case2_height,
       case5_velocity, case5_height, case5_mountain,
       case6_velocity, case6_height, case6_phase_shift

"""Williamson test-case physical constants (all cases, Williamson et
al. 1992 §2; identical values in STSWM and ClimaCore):

- `R` — planet radius, 6.37122×10⁶ m
- `Omega` — planetary rotation rate, 7.292×10⁻⁵ s⁻¹
- `g` — gravitational acceleration, 9.80616 m s⁻²
"""
williamson_constants() = (R = 6.37122e6, Omega = 7.292e-5, g = 9.80616)

# ------------------------------------------------------------------
# Case 2: steady-state nonlinear zonal geostrophic flow (eqs. 95–97)
# ------------------------------------------------------------------

"""    case2_velocity(lon, lat; u0 = 2π·R/(12 days), alpha = 0) -> (u, v)

Case 2 horizontal velocity (Williamson eq. (95) with the tilt rotation
of eqs. (96)–(97); STSWM `init.f` case 2). With tilt angle `alpha`
(rotation axis tilted by `alpha` from the North Pole toward longitude
0):

    u = u0 (cosθ cosα + cosλ sinθ sinα)
    v = −u0 sinλ sinα

Default `u0 = 2πR/(12 days)` (one circuit in 12 days); `alpha = 0`
gives the untilted zonal flow `u = u0 cosθ, v = 0`. The flow is in
geostrophic balance with `case2_height` and is **steady**: the exact
solution at any time equals the initial condition.
"""
function case2_velocity(lon, lat; u0 = nothing, alpha = 0.0)
    (; R) = williamson_constants()
    isnothing(u0) && (u0 = 2π * R / (12 * 86400))
    u = u0 * (cos(lat) * cos(alpha) + cos(lon) * sin(lat) * sin(alpha))
    v = -u0 * sin(lon) * sin(alpha)
    return (u, v)
end

"""    case2_height(lon, lat; h0 = 2.94e4/g, u0, alpha) -> h

Case 2 thickness (Williamson eqs. (96)–(97); STSWM `init.f` case 2).
The rotated latitude θ′ appearing in the geopotential is defined by

    sinθ′ = −cosλ cosθ sinα + sinθ cosα

and

    g h = g h₀ − (a Ω u₀ + u₀²/2) sin²θ′

with `g h₀ = 2.94×10⁴ m² s⁻²` (so h₀ ≈ 2997.9 m at the poles). With
`alpha = 0` this reduces to `gh = gh₀ − (aΩu₀ + u₀²/2) sin²θ`. Together
with `case2_velocity` this is an exact **steady** solution of the
nonlinear shallow-water equations on the sphere.
"""
function case2_height(lon, lat; u0 = nothing, alpha = 0.0)
    (; R, Omega, g) = williamson_constants()
    isnothing(u0) && (u0 = 2π * R / (12 * 86400))
    sin_theta_prime = -cos(lon) * cos(lat) * sin(alpha) + sin(lat) * cos(alpha)
    gh = 2.94e4 - (R * Omega * u0 + u0^2 / 2) * sin_theta_prime^2
    return gh / g
end

# ------------------------------------------------------------------
# Case 5: zonal flow over an isolated mountain (eqs. 129–134)
# ------------------------------------------------------------------

"""    case5_mountain(lon, lat; b0 = 2000, R_mountain = π/9,
                   center = (3π/2, π/6)) -> b

Case 5 terrain height (Williamson eq. (133); STSWM `init.f` case 5;
identical form to Ringler et al. 2010 eq. (80)):

    b = b₀ (1 − r/R)   for r < R,   0 otherwise

with **r the Euclidean distance in (λ, θ) coordinate space** (not
great-circle distance — confirmed identically in STSWM `init.f` and
Ringler et al. 2010):

    r² = min(R², (λ − λc)² + (θ − θc)²)

Constants: b₀ = 2000 m, R = π/9, mountain center
(λc, θc) = (3π/2, π/6) = (270°E, 30°N). Ringler 2010 prints λc = −π/2,
which is the same meridian (−π/2 + 2π = 3π/2).
"""
function case5_mountain(lon, lat; b0 = 2000.0, R_mountain = π / 9,
                        center = (3π / 2, π / 6))
    λc, θc = center
    # unwrap the longitude difference to the shortest arc so that, e.g.,
    # λ = −π/2 and λ = 3π/2 (the same meridian, printed differently in
    # Williamson/STSWM vs Ringler et al. 2010) both hit the mountain
    dλ = mod(lon - λc + π, 2π) - π
    # Euclidean distance in coordinate space, per Williamson eq. (133);
    # clamp to the mountain radius (min(R², ·) in the paper)
    r2 = min(R_mountain^2, dλ^2 + (lat - θc)^2)
    r = sqrt(r2)
    return b0 * (1 - r / R_mountain)
end

"""    case5_velocity(lon, lat; u0 = 20, alpha = 0) -> (u, v)

Case 5 initial velocity (Williamson eq. (129)): zonal geostrophic flow
as Case 2 with u₀ = 20 m s⁻¹ and no tilt:

    u = u0 cosθ,    v = 0
"""
function case5_velocity(lon, lat; u0 = 20.0, alpha = 0.0)
    return (u0 * cos(lat) * cos(alpha) + u0 * cos(lon) * sin(lat) * sin(alpha),
            -u0 * sin(lon) * sin(alpha))
end

"""    case5_height(lon, lat; u0 = 20, alpha = 0) -> h

Case 5 initial *total* depth (Williamson eqs. (130)–(132); STSWM
`init.f` case 5): the Case 2 geopotential profile with the Case 5
constants, `h₀ = 5960` m total depth at rest and `u₀ = 20` m s⁻¹:

    g (h + b) = g h₀ − (a Ω u₀ + u₀²/2) sin²θ′

The value returned is the **fluid thickness above the terrain**,

    h = [g h₀ − (a Ω u₀ + u₀²/2) sin²θ′]/g − b(λ, θ)

with `b` from `case5_mountain`. The total depth `h + b(λ, θ)` equals
the analytic profile exactly (tested to round-off).
"""
function case5_height(lon, lat; u0 = 20.0, alpha = 0.0)
    (; R, Omega, g) = williamson_constants()
    sin_theta_prime = -cos(lon) * cos(lat) * sin(alpha) + sin(lat) * cos(alpha)
    total_depth =
        (g * 5960 - (R * Omega * u0 + u0^2 / 2) * sin_theta_prime^2) / g
    return total_depth - case5_mountain(lon, lat)
end

# ------------------------------------------------------------------
# Case 6: Rossby–Haurwitz wavenumber-4 wave (eqs. 137–144)
# ------------------------------------------------------------------

# Case 6 constants (STSWM init.f; confirmed by ClimaCore):
#   R = 4 (zonal wavenumber), K = ω = 7.848×10⁻⁶ s⁻¹, h₀ = 8000 m
_rh_wavenumber() = 4
_rh_K() = 7.848e-6
_rh_h0() = 8000.0

"""    case6_velocity(lon, lat; t = 0) -> (u, v)

Case 6 Rossby–Haurwitz velocity (Williamson eqs. (137)–(138); STSWM
`anlytc.f` case 6, R = 4, K = 7.848×10⁻⁶ s⁻¹):

    u = a ω cosθ + a K cos^(R−1)θ (R sin²θ − cos²θ) cos(R(λ − Δλ(t)))
    v = −a K R cos^(R−1)θ sinθ sin(R(λ − Δλ(t)))

where `Δλ(t) = [R(3+R)ω − 2Ω]/[(1+R)(2+R)] · t` is the eastward phase
drift (STSWM `anlytc.f`; ≈ 0.213 rad/day for the standard constants —
see `case6_phase_shift`). At t = 0, Δλ = 0.
"""
function case6_velocity(lon, lat; t = 0.0)
    (; R, Omega) = williamson_constants()
    a = R
    m = _rh_wavenumber()
    K = _rh_K()
    dλ = case6_phase_shift(t)
    c = cos(lat)
    s = sin(lat)
    u = a * K * c +
        a * K * c^(m - 1) * (m * s^2 - c^2) * cos(m * (lon - dλ))
    v = -a * K * m * c^(m - 1) * s * sin(m * (lon - dλ))
    return (u, v)
end

"""    case6_height(lon, lat; t = 0) -> h

Case 6 Rossby–Haurwitz geopotential/thickness (Williamson eqs.
(139)–(144); STSWM `anlytc.f`/`init.f` case 6):

    g h = g h₀ + a² [A(θ) + B(θ) cos(R(λ−Δλ)) + C(θ) cos(2R(λ−Δλ))]

with the latitude-dependent factors (θ = latitude, R = 4,
K = ω = 7.848×10⁻⁶ s⁻¹):

    A = ω(2Ω+ω)cos²θ/2 + K²cos^(2R)θ[(R+1)cos²θ + (2R²−R−2) − 2R²/cos²θ]/4
    B = 2(Ω+ω)K/[(R+1)(R+2)] · cos^Rθ [(R²+2R+2) − (R+1)²cos²θ]
    C = K²cos^(2R)θ [(R+1)cos²θ − (R+2)]/4

**Poles:** the bracketed terms with cos⁻²θ in A are finite in the limit
because cos^(2R−2)θ → 0 dominates; the implementation evaluates A via
the algebraically identical grouping
`K²cos^(2R)θ·(−2R²)/cos²θ = −(K²R²/2)·cos^(2R−2)θ·(...)`, which is
well-defined (zero) at the poles — see the docstring note in the code.
"""
function case6_height(lon, lat; t = 0.0)
    (; R, Omega) = williamson_constants()
    a = R
    g = williamson_constants().g
    m = _rh_wavenumber()
    K = _rh_K()
    h0 = _rh_h0()
    ω = K
    dλ = case6_phase_shift(t)
    c = cos(lat)
    c2 = c^2
    # guard cos²θ in the denominator of A: rewrite so no division by zero
    # A's last bracket: (K²/4)cos^{2R}θ · (−2R²/cos²θ) = −(K²R²/2) cos^{2R−2}θ
    cos2Rm2 = m >= 1 ? c^(2m - 2) : zero(c)   # = 0 at poles for R=4
    A = (ω * (2Omega + ω) * c2) / 2 +
        (K^2 * c^(2m) * ((m + 1) * c2 + (2m^2 - m - 2))) / 4 -
        (K^2 * m^2 / 2) * cos2Rm2
    B = (2 * (Omega + ω) * K) / ((m + 1) * (m + 2)) * c^m *
        ((m^2 + 2m + 2) - (m + 1)^2 * c2)
    C = (K^2 * c^(2m) * ((m + 1) * c2 - (m + 2))) / 4
    gh = g * h0 + a^2 * (A + B * cos(m * (lon - dλ)) +
                         C * cos(2m * (lon - dλ)))
    return gh / g
end

"""    case6_phase_shift(t) -> Δλ

Eastward phase drift of the Rossby–Haurwitz wave (STSWM `anlytc.f`
case 6, verified verbatim):

    Δλ(t) = [R(3+R)ω − 2Ω]/[(1+R)(2+R)] · t

With R = 4, ω = 7.848×10⁻⁶ s⁻¹, Ω = 7.292×10⁻⁵ s⁻¹:
ν = (28ω − 2Ω)/30 ≈ 2.463×10⁻⁶ rad s⁻¹ > 0 — the pattern translates
**eastward at ≈ 0.213 rad/day** (one circuit in ≈ 29.5 days). The mean
flow itself rotates eastward at ω ≈ 0.678 rad/day, so the wave drifts
*westward relative to the mean flow* (the classic Rossby–Haurwitz
behavior); the STSWM (λ − Δλ) convention with Δλ > 0 fixes the
absolute motion unambiguously.
"""
function case6_phase_shift(t)
    (; Omega) = williamson_constants()
    m = _rh_wavenumber()
    ω = _rh_K()
    return (m * (3 + m) * ω - 2 * Omega) / ((1 + m) * (2 + m)) * t
end

end # module WilliamsonICs
