# Error-norm tests (#7).
#
# The norms are grid-independent, so the test "grids" here are pure point
# sets with quadrature weights:
#   * lat-lon rings with Gauss–Legendre nodes in μ = sin(lat) and
#     equispaced longitudes — exact for all spherical harmonics with
#     total degree ≤ 2·n_lat − 1 and zonal wavenumber |m| ≤ n_lon/2
#     (Gauss–Legendre is exact for polynomials in μ up to degree
#     2·n_lat − 1; the longitude trapezoid is exact for Fourier modes).
#   * a Fibonacci point set with equal-area weights (uniform coverage,
#     not exact — used for the qualitative/scaling checks).
using Test
using JuiCM
using JuiCM.ErrorNorms: l1_norm, l2_norm, linf_norm, error_norms
using LinearAlgebra: norm

"""Gauss–Legendre nodes and weights on [-1,1] with n points (Newton's
method on the Legendre recurrence; standard textbook construction)."""
function gauss_legendre(n)
    μ = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    for i in 1:n
        # initial guess (Chebyshev-like)
        x = cos(π * (i - 0.25) / (n + 0.5))
        # Newton iteration on P_n(x) using the three-term recurrence
        for _ in 1:100
            p0, p1 = 1.0, x
            for k in 1:n-1
                p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
            end
            # derivative: P_n' = n (x P_n − P_{n−1}) / (x² − 1)
            dp = n * (x * p1 - p0) / (x^2 - 1)
            dx = -p1 / dp
            x += dx
            abs(dx) < 1e-15 && break
        end
        p0, p1 = 1.0, x
        for k in 1:n-1
            p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
        end
        dp = n * (x * p1 - p0) / (x^2 - 1)
        μ[i] = x
        w[i] = 2 / ((1 - x^2) * dp^2)
    end
    return μ, w
end

"""lat-lon quadrature grid: n_lat Gauss–Legendre rings × n_lon
equispaced longitudes. Returns longitudes, latitudes, and area weights
A_j = w_j · R² (with R = 1: weights sum to 4π, the full sphere area).
Exact for all spherical harmonics with total degree ≤ 2·n_lat − 1 and
zonal wavenumber |m| ≤ n_lon/2 (Gauss–Legendre is exact for polynomials
in μ = sin(lat) up to degree 2·n_lat − 1; the equispaced longitude
trapezoid is exact for Fourier modes)."""
function latlon_grid(n_lat, n_lon; R = 1.0)
    μ, wμ = gauss_legendre(n_lat)
    lons = collect(range(0, 2π; length = n_lon + 1))[1:(end-1)]
    lats = asin.(μ)
    w = [wμ[i] * R^2 * 2π / n_lon for i in 1:n_lat, _ in 1:n_lon]
    # vec(w) is column-major: j = (l-1)·n_lat + i (lat varies fastest
    # within each longitude block). Build the coordinate vectors to
    # match that layout exactly and verify the alignment.
    lon_v = Float64[]
    lat_v = Float64[]
    for l in 1:n_lon, i in 1:n_lat
        push!(lon_v, lons[l])
        push!(lat_v, lats[i])
    end
    @assert length(lon_v) == length(lat_v) == length(w)
    for l in 1:n_lon   # spot-check the (lon, lat, w) alignment
        j = (l - 1) * n_lat + 1
        @assert lon_v[j] == lons[l] && lat_v[j] == lats[1]
        @assert w[j] == wμ[1] * R^2 * 2π / n_lon
    end
    return lon_v, lat_v, w
end

"""Fibonacci sphere point set with equal-area weights."""
function fibonacci_sphere(n)
    pts = Vector{NTuple{3,Float64}}(undef, n)
    φ = (sqrt(5) - 1) / 2
    for i in 0:(n-1)
        z = 1 - 2 * (i + 0.5) / n
        r = sqrt(1 - z^2)
        θ = 2π * i * φ
        pts[i+1] = (r * cos(θ), r * sin(θ), z)
    end
    return pts
end

@testset "error norms (#7)" begin
    @testset "self-comparison is exactly zero" begin
        n = 64
        pts = fibonacci_sphere(n)
        area = fill(4π / n, n)
        f = [sin(3p[1]) + p[2]^2 + cos(4p[3]) for p in pts]
        @test l1_norm(f, f, area) == 0
        @test l2_norm(f, f, area) == 0
        @test linf_norm(f, f) == 0
        norms = error_norms(f, f, area)
        @test norms == (l1 = 0.0, l2 = 0.0, linf = 0.0)
    end

    @testset "area weights integrate the sphere" begin
        # lat-lon Gauss–Legendre grid: Σ w = 4π (unit sphere), and with
        # radius R² scaling, 4πR² — the total surface area (issue #7).
        for R in (1.0, 6.37122e6)
            lon, lat, w = latlon_grid(24, 48; R = R)
            @test sum(w) ≈ 4π * R^2 rtol = 1e-12
        end
        # equal-area Fibonacci weights sum to 4π as well
        n = 500
        pts = fibonacci_sphere(n)
        @test sum(fill(4π / n, n)) ≈ 4π rtol = 1e-12
    end

    @testset "quadrature grid is exact for low-degree spherical harmonics" begin
        lon, lat, w = latlon_grid(24, 48)
        # spherical harmonics up to total degree 4 (well within the
        # exactness envelope: degree ≤ 2·24−1 = 47, |m| ≤ 24)
        f(λ, θ) = sin(θ) + (sin(θ)^2 - 1 / 3) - sin(θ)^2 * cos(2λ) +
                  (1 - 3sin(θ)^2) * cos(3λ)
        # all terms have zero spherical mean: sinθ is odd under θ → −θ,
        # sin²θ − 1/3 is the degree-2 Legendre pattern (I[sin²θ] = 1/3,
        # so it is mean-free), and each cos(mλ) factor is zonal-mean-free
        @test sum(w[j] * f(lon[j], lat[j]) for j in eachindex(w)) /
              sum(w) ≈ 0 atol = 1e-13
        # and the norm quadrature handles a nonzero-mean field exactly:
        # I[cos²θ cos²λ] = I[cos²θ]·I[cos²λ] = (2/3)·(1/2) = 1/3
        # (∫cos³θdθ = 4/3 over [-π/2,π/2] ⇒ I[cos²θ] = 2π·(4/3)/4π = 2/3)
        @test sum(w[j] * cos(lat[j])^2 * cos(lon[j])^2 for j in eachindex(w)) /
              sum(w) ≈ 1 / 3 rtol = 1e-12
        # hence I[3 + cos²θ cos²λ] = 3 + 1/3
        g(λ, θ) = 3 + cos(θ)^2 * cos(λ)^2
        @test sum(w[j] * g(lon[j], lat[j]) for j in eachindex(w)) /
              sum(w) ≈ 3 + 1 / 3 rtol = 1e-12
    end

    @testset "manufactured error field: analytically predicted norms" begin
        lon, lat, w = latlon_grid(24, 48)
        # reference field, nowhere zero so all denominators are safe
        f_ref = [2 + cos(lon[j]) * cos(lat[j]) for j in eachindex(lon)]
        # error field e = f − f_ref = 0.5 (constant)
        f = [f_ref[j] + 0.5 for j in eachindex(lon)]
        @test l1_norm(f, f_ref, w) ≈ 0.5 / 2.0 rtol = 1e-12   # I[|e|]/I[|f_ref|] = 0.5/2
        # l2 denominator: I[f_ref²] = I[4 + 4cosλcosθ + cos²λcos²θ]
        #   = 4 + 0 + I[cos²λcos²θ] = 4 + 1/3 (I[cos²λcos²θ] verified above)
        @test l2_norm(f, f_ref, w) ≈ 0.5 / sqrt(4 + 1 / 3) rtol = 1e-12
    end

    @testset "manufactured error field: direct quadrature comparison" begin
        lon, lat, w = latlon_grid(24, 48)
        f_ref = [2 + cos(lon[j]) * cos(lat[j]) for j in eachindex(lon)]
        f = [f_ref[j] + 0.25 * sin(3lon[j]) * lat[j] for j in eachindex(lon)]
        e = [f[j] - f_ref[j] for j in eachindex(lon)]
        I(x) = sum(w[j] * x[j] for j in eachindex(w)) / sum(w)
        @test l1_norm(f, f_ref, w) ≈ I(abs.(e)) / I(abs.(f_ref)) rtol = 1e-13
        @test l2_norm(f, f_ref, w) ≈ sqrt(I(abs2.(e)) / I(abs2.(f_ref))) rtol = 1e-13
        @test linf_norm(f, f_ref) ≈ maximum(abs.(e)) / maximum(abs.(f_ref)) rtol = 1e-13
        # all norms scale linearly in the error magnitude: (c·e vs f_ref)
        for c in (0.1, 3.0, -7.0)
            fc = f_ref .+ c .* e
            @test l1_norm(fc, f_ref, w) ≈ abs(c) * l1_norm(f, f_ref, w) rtol = 1e-12
            @test l2_norm(fc, f_ref, w) ≈ abs(c) * l2_norm(f, f_ref, w) rtol = 1e-12
            @test linf_norm(fc, f_ref) ≈ abs(c) * linf_norm(f, f_ref) rtol = 1e-12
        end
    end

    @testset "norm algebra: triangle-type inequality" begin
        n = 300
        pts = fibonacci_sphere(n)
        w = fill(4π / n, n)
        f = [exp(sin(2p[2]) + cos(3p[3])) for p in pts]
        g = [1 + p[1]^3 + sin(4p[2]) for p in pts]
        h = [cos(5p[1] * p[2]) + 2 for p in pts]
        # |f−g| ≤ |f−h| + |h−g| pointwise ⇒ the same for l1
        @test l1_norm(f, g, w) ≤ l1_norm(f, h, w) + l1_norm(h, g, w) + 1e-14
        # l2 of a sum: ||e1+e2||₂ ≤ ||e1||₂ + ||e2||₂ (weighted Cauchy–Schwarz)
        e1 = [f[j] - h[j] for j in 1:n]
        e2 = [h[j] - g[j] for j in 1:n]
        l2e1 = sqrt(sum(w[j] * e1[j]^2 for j in 1:n) / sum(w[j] * (g[j])^2 for j in 1:n))
        # raw (unnormalized) weighted l2 via norms against a zero field is
        # not available (denominator guard), so verify Cauchy–Schwarz directly
        @test sqrt(sum(w[j] * (e1[j] + e2[j])^2 for j in 1:n)) ≤
              sqrt(sum(w[j] * e1[j]^2 for j in 1:n)) +
              sqrt(sum(w[j] * e2[j]^2 for j in 1:n)) + 1e-12
    end

    @testset "radii: norms are dimensionless (R-independence)" begin
        lon, lat, w = latlon_grid(24, 48)
        f_ref = [1 + sin(2lat[j]) for j in eachindex(lon)]
        f = [f_ref[j] + 0.1 * cos(4lon[j]) for j in eachindex(lon)]
        @test error_norms(f, f_ref, w).l1 ≈
              error_norms(f, f_ref, w .* (6.37122e6)^2).l1 rtol = 1e-12
        @test error_norms(f, f_ref, w).l2 ≈
              error_norms(f, f_ref, w .* (6.37122e6)^2).l2 rtol = 1e-12
    end

    @testset "quantitative: truncation error decays geometrically" begin
        # band-limited-smooth field f = Σ_k a_k cos(kλ) P_k-like series with
        # coefficients a_k = 2^{-k}; truncating at order N leaves an error
        # dominated by the tail Σ_{k>N} 2^{-k} = 2^{-N} ⇒ each doubling of
        # the truncation order must (at least) square the reduction... here
        # geometric: err(2N)/err(N) ≈ 2^{-N} — verify err decreases by
        # a large factor, quantitatively consistent with the coefficient tail.
        lon, lat, w = latlon_grid(24, 48)
        coeffs = [2.0^(-k) for k in 1:40]
        full = [sum(coeffs[k] * cos(k * lon[j]) * cos(k * lat[j]) for k in 1:40)
                for j in eachindex(lon)]
        trunc_err(N) = l2_norm([sum(coeffs[k] * cos(k * lon[j]) * cos(k * lat[j])
                                    for k in 1:N) for j in eachindex(lon)],
                               full, w)
        e8, e16, e32 = trunc_err(8), trunc_err(16), trunc_err(32)
        # tail bound: error(N) ≤ 2^{-N}·(number of retained modes weight)
        # strictly quantitative: error(16)/error(8) ≤ 1/8 and error(32)/error(16) ≤ 1/8
        @test e16 < e8 / 8
        @test e32 < e16 / 8
        # and the truncation at 40 (full series) is exactly zero
        @test trunc_err(40) == 0
    end

    @testset "error handling" begin
        @test_throws DimensionMismatch l1_norm([1.0, 2.0], [1.0], [1.0, 1.0])
        @test_throws DimensionMismatch l2_norm([1.0, 2.0], [1.0], [1.0, 1.0])
        @test_throws DimensionMismatch linf_norm([1.0, 2.0], [1.0])
        # identically-zero reference: normalized norms are undefined
        @test_throws ErrorException l1_norm([1.0, 1.0], [0.0, 0.0], [1.0, 1.0])
        @test_throws ErrorException l2_norm([1.0, 1.0], [0.0, 0.0], [1.0, 1.0])
        @test_throws ErrorException linf_norm([1.0, 1.0], [0.0, 0.0])
    end
end
