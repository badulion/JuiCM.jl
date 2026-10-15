"""Normalized error norms for spherical scalar fields.

Implements the area-weighted normalized error norms of Williamson et al.
(1992), eqs. (80)–(82) — the standard quantitative measure for the
Williamson shallow-water test suite and the validation ladder (DESIGN.md
§4, rungs V0/V1). The same definitions appear as eqs. (74)–(76) of
Ringler et al. (2010).

The implementation is deliberately **grid-independent**: a field is just
a vector of point values together with a vector of associated area
weights (e.g. Voronoi cell areas, quadrature weights of a lat-lon ring
sum, or exact equal-area weights of a Fibonacci point set). No mesh
types are required, so the norms can be exercised — and validated —
before the grid exists.

Definitions (Williamson et al. 1992, eqs. (80)–(82)): with `I[f]` the
area-weighted global mean

    I[f] = Σⱼ f(j) A(j) / Σⱼ A(j)

the normalized norms of a computed field `f` against a reference `f_ref`
are

    l1   = I[|f − f_ref|] / I[|f_ref|]
    l2   = sqrt( I[(f − f_ref)²] / I[f_ref²] )
    linf = max |f − f_ref| / max |f_ref|

All three are dimensionless and zero iff `f ≡ f_ref` (assuming the
reference is not identically zero). `linf` uses the unweighted max of
the pointwise error, normalized by the unweighted max of the reference.

References:
- Williamson, D.L., Drake, J.B., Hack, J.J., Jakob, R., Swarztrauber,
  P.N. (1992): "A standard test set for numerical approximations to the
  shallow water equations in spherical geometry", J. Comput. Phys. 102,
  211–224, eqs. (80)–(82).
- Ringler, T.G., Thuburn, J., Klemp, J.B., Skamarock, W.C. (2010),
  J. Comput. Phys. 229, 3065–3090, eqs. (74)–(76).
"""
module ErrorNorms

export l1_norm, l2_norm, linf_norm, error_norms

# Denominator guards: the norms are undefined for an identically-zero
# reference field; we raise instead of returning NaN so a broken test
# setup fails loudly rather than silently comparing against NaN.
_denom_zero_error(name) =
    error("ErrorNorms: reference field has zero $name — normalized norm undefined")

"""    l1_norm(f, f_ref, area) -> Real

Normalized area-weighted l1 norm (Williamson et al. 1992, eq. (80)):

    l1 = Σⱼ |fⱼ − f_ref,j| Aⱼ / Σⱼ |f_ref,j| Aⱼ

`area` must have the same length as the fields; weights need not sum to
anything in particular (the ratio makes the normalization cancel).
"""
function l1_norm(f, f_ref, area)
    length(f) == length(f_ref) == length(area) ||
        throw(DimensionMismatch("f, f_ref, area must have equal length"))
    num = sum(abs(f[j] - f_ref[j]) * area[j] for j in eachindex(f))
    den = sum(abs(f_ref[j]) * area[j] for j in eachindex(f))
    den == 0 && _denom_zero_error("weighted mean magnitude")
    return num / den
end

"""    l2_norm(f, f_ref, area) -> Real

Normalized area-weighted l2 (root-mean-square) norm (Williamson et al.
1992, eq. (81)):

    l2 = sqrt( Σⱼ (fⱼ − f_ref,j)² Aⱼ / Σⱼ f_ref,j² Aⱼ )
"""
function l2_norm(f, f_ref, area)
    length(f) == length(f_ref) == length(area) ||
        throw(DimensionMismatch("f, f_ref, area must have equal length"))
    num = sum(abs2(f[j] - f_ref[j]) * area[j] for j in eachindex(f))
    den = sum(abs2(f_ref[j]) * area[j] for j in eachindex(f))
    den == 0 && _denom_zero_error("weighted mean square")
    return sqrt(num / den)
end

"""    linf_norm(f, f_ref) -> Real

Normalized maximum (infinity) norm (Williamson et al. 1992, eq. (82)):

    linf = maxⱼ |fⱼ − f_ref,j| / maxⱼ |f_ref,j|

Unweighted: the max is over point values, not area-weighted means.
"""
function linf_norm(f, f_ref)
    length(f) == length(f_ref) ||
        throw(DimensionMismatch("f and f_ref must have equal length"))
    num = maximum(abs(f[j] - f_ref[j]) for j in eachindex(f))
    den = maximum(abs(f_ref[j]) for j in eachindex(f))
    den == 0 && _denom_zero_error("maximum magnitude")
    return num / den
end

"""    error_norms(f, f_ref, area) -> NamedTuple

All three normalized error norms at once:
`(l1 = l1_norm(...), l2 = l2_norm(...), linf = linf_norm(...))`.
"""
function error_norms(f, f_ref, area)
    return (l1 = l1_norm(f, f_ref, area),
            l2 = l2_norm(f, f_ref, area),
            linf = linf_norm(f, f_ref))
end

end # module ErrorNorms
