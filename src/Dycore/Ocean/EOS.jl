# Ocean equation of state hook (issue #30, DESIGN.md §3.2, §5 Phase 4).
#
# The pressure-gradient density enters the layered momentum equation
# through a pluggable EOS (the function-barrier seam that Phase 4's
# full seawater EOS — SeawaterPolynomials.jl — plugs into without
# touching the dycore, per DESIGN.md §5). For #30 the concrete EOS is
# linear in temperature (user-ratified 2026-10-16):
#
#     ρ(T) = ρ0·(1 − α·(T − T_ref))
#
# with thermal expansion α [1/K] and reference temperature T_ref [°C or
# K — only differences matter]. Salinity is accepted in the interface
# (prognostic field, no dynamics in #30) so the Phase-4 signature is
# already the final one.
#
# Reference: the standard linearized EOS used in layered/density-space
# models (e.g. the MIPOM/MOM-class linear EOS; see also Roquet et al.
# 2015, §2a, on linearization limits of seawater EOS).

module OceanEOS

using ..Fields: Field, CellField
using ..Layered: LayeredCellField, nlayers

export AbstractOceanEOS, LinearEOS, ocean_density, ocean_density!

"""
    AbstractOceanEOS

Equation-of-state interface (DESIGN.md §3.2, the pluggable-EOS hook):
pure data plus one kernel, [`ocean_density!`](@ref). Concrete EOS types
carry their coefficients; Phase 4's SeawaterPolynomials-based EOS
implements the same single method.

Constructors take no mesh — density is evaluated onto caller-provided
fields, keeping the EOS payload-free and threadable.
"""
abstract type AbstractOceanEOS end

"""
    LinearEOS(ρ0 = 1025.0; alpha = 2.0e-4, T_ref = 10.0) -> LinearEOS

Linear-in-temperature EOS (user-ratified for #30):

    ρ(T) = ρ0·(1 − α·(T − T_ref))

- `ρ0` [kg/m³] — reference (mean) density
- `alpha` [1/K] — thermal expansion coefficient (positive: warmer =
  lighter)
- `T_ref` [K or °C] — reference temperature; only differences matter

Salinity is part of the [`ocean_density!`](@ref) signature but ignored
here (no salinity dynamics in #30).
"""
struct LinearEOS{FT} <: AbstractOceanEOS
    """Reference density [kg/m³]."""
    ρ0::FT
    """Thermal expansion coefficient [1/K]."""
    alpha::FT
    """Reference temperature [K or °C]."""
    T_ref::FT
end
LinearEOS(ρ0::Real = 1025.0; alpha::Real = 2.0e-4, T_ref::Real = 10.0) =
    LinearEOS(Float64(ρ0), Float64(alpha), Float64(T_ref))

"""
    ocean_density!(ρ::LayeredCellField, T::LayeredCellField,
                   S::LayeredCellField, eos::AbstractOceanEOS) -> LayeredCellField

Evaluate the EOS layer-by-layer into the preallocated `ρ` (the dycore
hot path — allocation-free). `S` must be provided for signature
stability across EOS implementations; [`LinearEOS`](@ref) ignores it.

Returns `ρ` for chaining. The allocating convenience
[`ocean_density`](@ref) wraps this.
"""
function ocean_density!(ρ::LayeredCellField, T::LayeredCellField,
                        S::LayeredCellField, eos::LinearEOS)
    (; ρ0, alpha, T_ref) = eos
    K = nlayers(T)
    for k in 1:K
        tk = T[k]
        ρk = ρ[k]
        for i in eachindex(ρk)
            ρk[i] = ρ0 * (1 - alpha * (tk[i] - T_ref))
        end
    end
    return ρ
end

"""
    ocean_density(T::LayeredCellField, S::LayeredCellField,
                  eos::AbstractOceanEOS) -> LayeredCellField

Allocating form of [`ocean_density!`](@ref) — tests and setup code.
"""
function ocean_density(T::LayeredCellField, S::LayeredCellField,
                       eos::AbstractOceanEOS)
    return ocean_density!(similar(T), T, S, eos)
end

end # module OceanEOS
