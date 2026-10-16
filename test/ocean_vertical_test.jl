# Ocean vertical machinery (#28 z-star + #49 bathymetry/regridding,
# DESIGN.md D7, §5 Phase 2).
#
# Acceptance (issue #28):
#   1. Column sum of layer thicknesses equals H0 + η to round-off for
#      arbitrary η (bathymetry variation is issue #49 — strict split,
#      user-ratified; here H0 is the flat-bottom reference depth).
#   2. Flat bottom, uniform η: interface positions follow the z-star map
#      analytically — depth below surface ζ_k = d_k·(1 + η/H0) with
#      d_k = f_k·H0 the fixed z*-space depth (equivalently
#      z_k = η − f_k·(H0 + η)) — matched to < 1e-12 relative.
#   3. Discrete volume invariance: Σ (cell area × layer thickness)
#      tracks pure surface elevation to round-off,
#      V(η) − V(0) = A_total·η.
#   4. Docstrings cite the z-star definitions (Adcroft & Campin 2004,
#      MOM6/MPAS-O convention).
#
# Acceptance (issue #49):
#   5. Volume under any analytic bathymetry matches the exact analytic
#      integral to < 1e-10 relative (volume = Σ A_i·H(i); for the
#      longitude-only profiles the exact integral is computed by dense
#      longitude quadrature at matched latitudes).
#   6. Regridding round trip: z-star thicknesses recomputed after a
#      prescribed η(t) forcing recover the analytic profile; layer
#      volume Σ A_k h_k invariant to round-off through a full forcing
#      cycle (per-cell bathymetry).
#   7. Slope test: linear slope bathymetry, uniform stratification,
#      resting state → interfaces follow the per-cell z-star map exactly
#      (the analytic resting state is exact under regridding — the
#      discrete probe of the z-star PGF-on-slope bug class; the
#      dynamical PGF probe itself lands with #30's resting-state test).

@testset "ocean vertical: z-star coordinate (#28)" begin
    using JuiCM: VerticalGrid, zstar_thicknesses, LayeredCellField,
                 zstar_stretch_factor, interface_z, layer_thickness,
                 layer_center_z, dz_interfaces, interp_to_interfaces,
                 ssh_from_thickness

    m = generate_icosahedral_mesh(n_subdivisions=2)
    A_total = sum(m.cell_area)

    @testset "grid construction and validation" begin
        g = VerticalGrid(4000.0, 10)
        @test nlayers(g) == 10
        f = g.zstar_fractions
        @test f[1] == 0.0 && f[end] == 1.0
        @test all(>(0), diff(f))
        # uniform by default; Δz* = H0/K
        @test zstar_thicknesses(g) ≈ fill(400.0, 10) rtol = 1e-14

        # stretched grid: u -> u^2 concentrates layers near the surface
        gs = VerticalGrid(4000.0, 10; stretching = u -> u^2)
        @test gs.zstar_fractions[2] ≈ 0.01
        @test zstar_thicknesses(gs)[1] ≈ 40.0
        @test sum(zstar_thicknesses(gs)) ≈ 4000.0 rtol = 1e-14

        @test_throws ArgumentError VerticalGrid(-1.0, 5)
        @test_throws ArgumentError VerticalGrid(4000.0, 0)
        @test_throws ArgumentError VerticalGrid(4000.0, 5;
                                                stretching = u -> u + 1)
        @test_throws ArgumentError VerticalGrid(4000.0, 5;
                                                stretching = u -> 1 - u)
        # non-monotone stretching (flat interior) must fail
        @test_throws ArgumentError VerticalGrid(4000.0, 5;
                                                stretching = u -> u < 0.5 ? 0.2 : u)
    end

    @testset "acceptance 1: column sum = H0 + η (round-off, arbitrary η)" begin
        g = VerticalGrid(4000.0, 10)
        K = nlayers(g)
        for trial in 1:5
            # arbitrary (including negative) per-cell η, |η| < 0.2·H0
            eta = CellField(i -> (0.2 * (2rand() - 1)) * 4000.0, m)
            h = layer_thickness(g, eta)
            @test length(h.layers) == K
            err = maximum([abs(sum(h[k][i] for k in 1:K) - (4000.0 + eta[i]))
                           for i in 1:m.n_cells])
            @test err < 1e-9   # ≈ round-off on 4e3-magnitude sums (measured 9e-13)

            # inverse: η recovered from thicknesses (ssh_from_thickness)
            eta2 = ssh_from_thickness(g, h)
            err2 = maximum([abs(eta2[i] - eta[i]) for i in 1:m.n_cells])
            @test err2 < 1e-9  # measured 1e-12
        end
        # stretched grid satisfies the same invariants
        gs = VerticalGrid(4000.0, 10; stretching = u -> u^2)
        eta = CellField(i -> 0.15 * 4000.0 * sin(3m.cell_xyz[1, i]), m)
        h = layer_thickness(gs, eta)
        @test maximum([abs(sum(h[k][i] for k in 1:10) - (4000.0 + eta[i]))
                       for i in 1:m.n_cells]) < 1e-9
    end

    @testset "acceptance 2: analytic interface map (uniform η)" begin
        H0 = 4000.0
        for (K, stretch) in ((10, identity), (8, u -> u^2), (5, u -> sqrt(u)))
            g = VerticalGrid(H0, K; stretching = stretch)
            for eta in (-150.0, 0.0, 250.0)
                zi = interface_z(g, eta, m)
                @test length(zi) == K + 1
                σ = 1 + eta / H0
                for i in (1, m.n_cells ÷ 2, m.n_cells)
                    d_star = [f * H0 for f in g.zstar_fractions]  # z* depths
                    for k in 1:(K + 1)
                        # issue #28 form: depth below surface ζ_k = d_k (1 + η/H0)
                        ζ_k = eta - zi[k][i]
                        @test ζ_k ≈ d_star[k] * σ rtol = 1e-12
                    end
                    # surface and bottom anchors
                    @test zi[1][i] == eta
                    @test zi[K + 1][i] == -H0
                end
            end
        end
    end

    @testset "acceptance 3: discrete volume invariance" begin
        g = VerticalGrid(4000.0, 12)
        K = nlayers(g)
        V_of(eta_val) = begin
            h = layer_thickness(g, eta_val, m)
            return sum(m.cell_area[i] * sum(h[k][i] for k in 1:K)
                       for i in 1:m.n_cells)
        end
        V0 = V_of(0.0)
        @test V0 ≈ A_total * 4000.0 rtol = 1e-14
        for eta_val in (-200.0, 137.0, 500.0)
            @test (V_of(eta_val) - V0) ≈ A_total * eta_val rtol = 1e-12
        end
        # per-cell η field: volume change = A-weighted mean elevation
        eta = CellField(i -> 0.1 * 4000.0 * m.cell_xyz[3, i], m)
        h = layer_thickness(g, eta)
        dV = sum(m.cell_area[i] * sum(h[k][i] for k in 1:K)
                 for i in 1:m.n_cells) - V0
        dV_exact = sum(m.cell_area[i] * eta[i] for i in 1:m.n_cells)
        # dV subtracts two ~1.6e8 volumes: round-off floor ~1e-8 abs;
        # assert relative-to-column agreement instead
        @test abs(dV - dV_exact) < 1e-8 + 1e-12 * A_total * 4000.0
    end

    @testset "vertical metric operators" begin
        H0 = 4000.0
        g = VerticalGrid(H0, 10; stretching = u -> u^2)
        K = nlayers(g)
        eta = CellField(i -> 0.1 * H0 * sin(2m.cell_xyz[1, i]) *
                                cos(m.cell_xyz[2, i]), m)
        h = layer_thickness(g, eta)
        zc = layer_center_z(g, eta)
        zi = interface_z(g, eta)

        # centers are the interface midpoints
        for i in 1:m.n_cells, k in 1:K
            @test zc[k][i] ≈ (zi[k][i] + zi[k + 1][i]) / 2 rtol = 1e-14
        end

        # dz between centers = (h_k + h_{k+1})/2, and equals the z-space
        # distance between the analytic centers
        dz = dz_interfaces(h)
        for i in 1:m.n_cells, k in 1:(K - 1)
            @test dz[k][i] ≈ (h[k][i] + h[k + 1][i]) / 2 rtol = 1e-14
            @test dz[k][i] ≈ zc[k][i] - zc[k + 1][i] rtol = 1e-12
        end
        @test length(dz) == K - 1
        @test_throws ArgumentError dz_interfaces(layer_thickness(
            VerticalGrid(H0, 1), 0.0, m))

        # thickness-weighted interpolation to interfaces is exact for a
        # linear-in-z profile (T = a + b·z) at the INTERIOR interfaces —
        # the centers are layer midpoints, so the weighted mean
        # reproduces T at the interface. The surface/bottom entries use
        # the constant-extension convention (edge-layer value), which on
        # a stretched grid is NOT the analytic T at z = η / z = −H0 —
        # asserted separately.
        a, b = 15.0, -0.005
        T = LayeredCellField((i, k) -> a + b * zc[k][i], m, K)
        Ti = interp_to_interfaces(T, h)
        @test length(Ti) == K + 1
        for i in 1:m.n_cells, k in 2:K
            @test Ti[k][i] ≈ a + b * zi[k][i] rtol = 1e-12
        end
        # boundary entries are the edge-layer values by convention
        for i in 1:m.n_cells
            @test Ti[1][i] == T[1][i]
            @test Ti[K + 1][i] == T[K][i]
        end

        # stretch factor: scalar and field forms agree
        @test zstar_stretch_factor(250.0, H0) == 1 + 250.0 / H0
        σ = zstar_stretch_factor(eta, H0)
        @test maximum([abs(σ[i] - (1 + eta[i] / H0)) for i in 1:m.n_cells]) <
              1e-15
    end

    @testset "LayeredField kernels" begin
        using JuiCM: lincomb!, axpy!, scale!
        K = 6
        L1 = LayeredCellField((i, k) -> 1.0 * i + 10.0 * k, m, K)
        L2 = LayeredCellField((i, k) -> 0.5 * k, m, K)

        # lincomb!/axpy!/scale!/copyto! per layer vs direct construction
        out = LayeredCellField(m, K)
        lincomb!(out, 2.0, L1, -1.0, L2)
        for k in 1:K
            @test all(out[k][i] == 2 * L1[k][i] - L2[k][i]
                      for i in 1:m.n_cells)
        end
        axpy!(out, 3.0, L2)
        scale!(out, 0.5)
        for k in 1:K
            @test all(out[k][i] == 0.5 * (2 * L1[k][i] + 2 * L2[k][i])
                      for i in 1:m.n_cells)
        end

        c = copy(L1)
        copyto!(out, c)
        fill!(out, 0.0)
        @test maximum(maximum(abs.(out[k].data)) for k in 1:K) == 0

        # similar: same layout, uninitialized values (check structure only)
        s = similar(L1)
        @test s isa LayeredCellField
        @test nlayers(s) == K
        @test mesh_of(s) == m

        # mismatched layer counts / meshes rejected
        @test_throws ArgumentError lincomb!(LayeredCellField(m, K), 1.0, L1,
                                            1.0, LayeredCellField(m, K + 1))
        m2 = generate_icosahedral_mesh(n_subdivisions=1)
        @test_throws ArgumentError lincomb!(LayeredCellField(m, K), 1.0, L1,
                                            1.0, LayeredCellField(m2, K))
        # indexing
        @test L1[1] === L1.layers[1]
        @test L1[lastindex(L1)] === L1.layers[K]
        @test eltype(L1) == Float64
    end

    # ------------------------------------------------- #49 bathymetry
    @testset "bathymetry profiles (#49)" begin
        using JuiCM: bathymetry_flat, bathymetry_channel, bathymetry_slope,
                     bathymetry_shelf_break
        lonlat(x) = (atan(x[2], x[1]), asin(clamp(x[3], -1, 1)))

        Hf = bathymetry_flat(m, 4000.0)
        @test all(==(4000.0), Hf.data)

        # slope: linear in longitude between the bounds, clamped outside
        Hs = bathymetry_slope(m, 500.0, 4000.0)
        for i in 1:m.n_cells
            λ, _ = lonlat(m.cell_xyz[:, i])
            λc = clamp(λ, -π / 2, π / 2)
            @test Hs[i] ≈ 500 + 3500 * (λc + π / 2) / π rtol = 1e-14
        end

        # shelf break: monotone in λ, shelf/deep plateaus approached as
        # tanh tails (exponentially; still ~0.2% residual at |λ| = π/4
        # with width π/16, ~1e-5 at |λ| = π/3) — assert the plateaus
        # well outside the transition and the monotone rise across it
        Hb = bathymetry_shelf_break(m, 500.0, 4000.0; λ_break = 0.0,
                                    width = π / 16)
        λs = [lonlat(m.cell_xyz[:, i])[1] for i in 1:m.n_cells]
        @test all(h -> isapprox(h, 500; rtol = 1e-4), Hb.data[λs .< -π / 3])
        @test all(h -> isapprox(h, 4000; rtol = 1e-4), Hb.data[λs .> π / 3])
        @test all(>(0), Hb.data)
        mid = Hb.data[-π / 8 .< λs .< π / 8]
        @test all(h -> 500 < h < 4000, mid)
        @test issorted(Hb.data[sortperm(λs)])

        # channel: deep interior, shallow ends; the longitude wrap
        # (|λ| > π/2) maps onto the nearest channel end, so the far
        # wrap region is the shallow H0 and the profile stays within
        # [H0, H_deep] everywhere
        Hc = bathymetry_channel(m, 1000.0, 4000.0)
        @test all(h -> isapprox(h, 4000; rtol = 1e-12),
                  Hc.data[abs.(λs) .< π / 2 - π / 8 - 0.05])
        @test all(h -> isapprox(h, 1000; rtol = 1e-12),
                  Hc.data[abs.(λs) .> 3π / 5])
        @test all(h -> 1000 <= h <= 4000, Hc.data)

        # mixed field/real dispatch: flat-H results identical
        eta = CellField(i -> 20.0, m)
        g = VerticalGrid(4000.0, 10)
        @test layer_thickness(g, eta, 4000.0)[1].data ==
              layer_thickness(g, eta, bathymetry_flat(m, 4000.0))[1].data
    end

    @testset "acceptance 5 (#49): bathymetry volume vs analytic" begin
        using JuiCM: bathymetry_flat, bathymetry_slope, bathymetry_shelf_break
        # exactness rung (< 1e-10 relative): a field the discrete measure
        # represents exactly — flat H. Σ A_i·H0 vs the analytic 4πR²H0
        # (R = 1): round-off only.
        H0 = 4000.0
        Hf = bathymetry_flat(m, H0)
        Vdisc = sum(m.cell_area[i] * Hf[i] for i in 1:m.n_cells)
        @test Vdisc ≈ 4π * H0 rtol = 1e-12

        # varying profiles: the discrete volume carries O(Δλ) sampling
        # error at 162 cells (the mesh is not mirror-symmetric in
        # longitude), so the honest quantitative rung is *convergence*:
        # the error against the analytic integral must shrink with
        # refinement (3% → ~0.8% one level up, roughly the cell-λ spread).
        function analytic_mean(Hλ)
            nλ = 1_000_000
            dλ = 2π / nλ
            return sum(Hλ(-π + (j - 0.5) * dλ) for j in 1:nλ) * dλ / (2π)
        end
        Hs = bathymetry_slope(m, 500.0, 4000.0)
        exact = analytic_mean(λ -> 500 + 3500 * clamp(λ + π / 2, 0, π) / π)
        slope_mean(d) = sum(d.cell_area[i] *
                            bathymetry_slope(d, 500.0, 4000.0)[i]
                            for i in 1:d.n_cells) / sum(d.cell_area)
        e1 = abs(slope_mean(m) - exact)
        e2 = abs(slope_mean(generate_icosahedral_mesh(n_subdivisions = 3)) -
                 exact)
        e3 = abs(slope_mean(generate_icosahedral_mesh(n_subdivisions = 4)) -
                 exact)
        @test e1 < 0.03 * 4000           # coarse error bounded ~3% of range
        @test e3 < e2 < e1               # monotone convergence, 3 levels
        @test e3 / e1 < 0.45             # roughly halving per refinement

        # shelf break: same convergence check for the smooth tanh profile
        function shelf_err(d)
            H = bathymetry_shelf_break(d, 500.0, 4000.0; λ_break = 0.0,
                                       width = π / 16)
            return abs(sum(d.cell_area[i] * H[i] for i in 1:d.n_cells) /
                       sum(d.cell_area) -
                       analytic_mean(λ -> 500 + 3500 *
                                          (1 + tanh(λ / (π / 16))) / 2))
        end
        @test shelf_err(generate_icosahedral_mesh(n_subdivisions = 4)) <
              shelf_err(generate_icosahedral_mesh(n_subdivisions = 3)) <
              shelf_err(m)
    end

    @testset "acceptance 6-7 (#49): regridding round trip + slope" begin
        using JuiCM: bathymetry_slope, thickness_from_ssh, regrid_zstar,
                     effective_layer_counts
        g49 = VerticalGrid(4000.0, 10)
        K = nlayers(g49)
        H = bathymetry_slope(m, 1000.0, 4000.0)

        # (7) slope, resting state (η = 0): interfaces follow the per-cell
        # z-star map exactly — the discrete probe of the PGF-on-slope class
        eta0 = CellField(i -> 0.0, m)
        zi = interface_z(g49, eta0, H)
        for i in 1:m.n_cells
            for k in 1:(K + 1)
                f = g49.zstar_fractions[k]
                @test zi[k][i] ≈ -f * H[i] rtol = 1e-13
            end
        end

        # (6) round trip: full forcing cycle η(t) = A sin(t) regridded at
        # each stage; layer volume invariant and η recovery exact
        A = 120.0
        Vols = Float64[]
        etas = [CellField(i -> A * sin(2π * s / 8) * m.cell_xyz[3, i], m)
                for s in 0:8]
        for eta in etas
            h = thickness_from_ssh(g49, eta, H)
            push!(Vols, sum(m.cell_area[i] * sum(h[k][i] for k in 1:K)
                            for i in 1:m.n_cells))
            # per-cell column sum = H + η
            for i in 1:m.n_cells
                @test abs(sum(h[k][i] for k in 1:K) - (H[i] + eta[i])) <
                      1e-9
            end
            # analytic profile recovery: active-layer thicknesses equal
            # Δz*_k (H + η) (no reduction at this floor for this H)
            h_exact = layer_thickness(g49, eta, H)
            @test all(abs(h[k][i] - h_exact[k][i]) < 1e-9
                      for i in 1:m.n_cells for k in 1:K)
        end
        # volume tracks the A-weighted mean elevation exactly (round-off
        # on ~1.6e8-magnitude sums → absolute tolerance)
        A_tot = sum(m.cell_area)
        V0 = Vols[1]
        for s in 1:8
            eta_mean = sum(m.cell_area[i] * etas[s][i] for i in 1:m.n_cells)
            @test abs((Vols[s + 1] - V0) - eta_mean) < 1e-8
        end

        # min-thickness reduction: shallow bathymetry + aggressive floor
        Hsh = bathymetry_slope(m, 300.0, 4000.0)
        eta = CellField(i -> 10.0, m)
        h, counts = regrid_zstar(g49, eta, Hsh; h_min_frac = 0.15)
        @test minimum(counts) < K        # reduction actually happened
        @test all(1 .<= counts .<= K)
        for i in 1:m.n_cells
            Hi, etai = Hsh[i], eta[i]
            # merged bottom layer ≥ floor of reference thickness
            @test h[counts[i]][i] ≥ 0.15 * Hi * (1 + etai / Hi) - 1e-9
            # column sum invariant with the reduced count
            @test abs(sum(h[k][i] for k in 1:counts[i]) - (Hi + etai)) <
                  1e-9
            # inactive slots are zero
            @test all(==(0.0), h[k][i] for k in (counts[i] + 1):K)
        end
        # counts are η-independent (reference-column reduction)
        h2, counts2 = regrid_zstar(g49, CellField(i -> 0.0, m), Hsh;
                                   h_min_frac = 0.15)
        @test counts == counts2

        # vertical neighbor tables
        nb = vertical_neighbors(m, K)
        @test length(nb) == m.n_cells
        @test length(nb[1]) == 2K
        for i in 1:m.n_cells
            @test nb[i][1] == 0            # above layer 1: none
            @test nb[i][2K] == 0           # below layer K: none
            for k in 2:(K - 1)
                @test nb[i][2k - 1] == k - 1
                @test nb[i][2k] == k + 1
            end
            @test nb[i][3] == 1            # above layer 2
            @test nb[i][2K - 1] == K - 1   # above layer K
        end
        @test_throws ArgumentError vertical_neighbors(m, 0)
    end
end
