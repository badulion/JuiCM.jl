using Test
using Statistics
using JuiCM
using JuiCM.Icosahedron: icosahedron, subdivide
using JuiCM.Voronoi: spherical_circumcenter, spherical_polygon_area,
                     generate_icosahedral_mesh
using LinearAlgebra
using Random

@testset "JuiCM" verbose = true begin
    @testset "package loads" begin
        @test JuiCM isa Module
    end

    @testset "icosahedron construction" begin
        ico = icosahedron()
        @test size(ico.xyz) == (3, 12)
        @test size(ico.edge_vertices) == (2, 30)
        @test size(ico.face_vertices) == (3, 20)
        # all vertices on the unit sphere
        @test all(norm(ico.xyz[:, i]) ≈ 1 for i in 1:12)
        # all edges equal length: unit-sphere icosahedron edge = 2/√(φ²+1)
        ℓ = 2 / sqrt(MathConstants.golden^2 + 1)
        for e in 1:30
            a, b = ico.edge_vertices[1, e], ico.edge_vertices[2, e]
            @test norm(ico.xyz[:, a] .- ico.xyz[:, b]) ≈ ℓ atol = 1e-12
        end
        # every vertex has degree 5
        for v in 1:12
            deg = count(e -> v in ico.edge_vertices[:, e], 1:30)
            @test deg == 5
        end
        # faces consistently oriented (outward normals)
        for f in 1:20
            i, j, k = ico.face_vertices[:, f]
            n = cross(ico.xyz[:, j] .- ico.xyz[:, i], ico.xyz[:, k] .- ico.xyz[:, i])
            @test dot(n, ico.xyz[:, i]) > 0
        end
    end

    @testset "icosahedron subdivision" begin
        ico = icosahedron()
        s1 = subdivide(ico)
        # V → V + E = 12 + 30 = 42; E → 2E + 3F = 60 + 60 = 120; F → 80
        @test size(s1.xyz, 2) == 42
        @test size(s1.edge_vertices, 2) == 120
        @test size(s1.face_vertices, 2) == 80
        # Euler characteristic V − E + F = 2
        @test size(s1.xyz, 2) - size(s1.edge_vertices, 2) + size(s1.face_vertices, 2) == 2
        # all vertices on unit sphere
        @test all(norm(s1.xyz[:, i]) ≈ 1 for i in 1:42)
        # orientation preserved
        for f in 1:80
            i, j, k = s1.face_vertices[:, f]
            n = cross(s1.xyz[:, j] .- s1.xyz[:, i], s1.xyz[:, k] .- s1.xyz[:, i])
            @test dot(n, s1.xyz[:, i]) > 0
        end
        s2 = subdivide(s1)
        @test size(s2.xyz, 2) == 162
        @test size(s2.xyz, 2) - size(s2.edge_vertices, 2) + size(s2.face_vertices, 2) == 2
    end

    @testset "spherical circumcenter" begin
        # circumcenter of an octant triangle = direction (1,1,1)/√3
        a = [1.0, 0, 0]; b = [0.0, 1, 0]; c = [0.0, 0, 1]
        cc = spherical_circumcenter(a, b, c)
        @test cc ≈ normalize([1.0, 1, 1]) atol = 1e-12
        @test norm(cc) ≈ 1
        # equidistant from all three corners
        for p in (a, b, c)
            @test dot(cc, p) ≈ dot(cc, a) atol = 1e-12
        end
    end

    @testset "spherical polygon area (Girard)" begin
        # octant: area = 4π/8
        octant = [[1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]
        @test spherical_polygon_area(octant) ≈ π / 2 atol = 1e-12
        ll(lon, lat) = [cosd(lat) * cosd(lon), cosd(lat) * sind(lon), sind(lat)]
        # hemisphere: equatorial decagon
        hemi = [ll(k * 36, 0) for k in 0:9]
        @test spherical_polygon_area(hemi) ≈ 2π atol = 1e-12
        # cap of colatitude 30°, inscribed n-gon → converges to cap area
        # (chord-vs-arc deficit is O(h²): tolerance reflects 72-gon accuracy)
        cap = [ll(k * 30, 60) for k in 0:11]
        @test spherical_polygon_area(cap) < 2π * (1 - cosd(30))   # inscribed polygon is smaller
        cap_fine = [ll(k * 5, 60) for k in 0:71]
        @test abs(spherical_polygon_area(cap_fine) - 2π * (1 - cosd(30))) < 1e-3
        # orientation: clockwise gives negative area
        @test spherical_polygon_area(reverse(octant)) ≈ -π / 2 atol = 1e-12
    end

    @testset "icosahedral Voronoi mesh invariants (#3)" begin
        for lev in (0, 1, 2)
            m = generate_icosahedral_mesh(n_subdivisions=lev)

            # counts: 10·4^lev + 2 cells, 30·4^lev edges, 20·4^lev vertices
            @test m.n_cells == 10 * 4^lev + 2
            @test m.n_edges == 30 * 4^lev
            @test m.n_vertices == 20 * 4^lev

            # areas: primal cells and dual cells both sum to 4π
            @test sum(m.cell_area) ≈ 4π rtol = 1e-12
            @test sum(m.vertex_area) ≈ 4π rtol = 1e-12

            # exactly 12 pentagons (the icosahedron vertices)
            @test length(m.pentagon_cells) == 12
            @test all(c -> length(m.cell_corners[c]) == 5, m.pentagon_cells)
            @test all(c -> length(m.cell_corners[c]) == 6,
                      setdiff(1:m.n_cells, m.pentagon_cells))

            # Euler characteristic: V − E + F = 2
            @test m.n_vertices - m.n_edges + m.n_cells == 2

            # connectivity: each cell edge list is closed under traversal
            for c in 1:m.n_cells
                @test length(m.cell_edges[c]) == length(m.cell_corners[c])
            end
            # each edge's two cells are distinct
            for e in 1:m.n_edges
                @test m.edge_cells[1, e] != m.edge_cells[2, e]
            end

            # corner weights Σ_v R_{i,v} = 1 per cell (T09 Eq 13)
            for c in 1:m.n_cells
                @test sum(m.corner_weight[c]) ≈ 1 rtol = 1e-10
                @test all(w -> 0 ≤ w ≤ 1, m.corner_weight[c])
            end
        end

        # edge geometry: n̂_e tangent to sphere at x_e, t̂_e = x̂_e × n̂_e,
        # orthonormal frame; t̂_e points neg → pos vertex
        m = generate_icosahedral_mesh(n_subdivisions=1)
        for e in 1:m.n_edges
            @test norm(m.edge_x[:, e]) ≈ 1 atol = 1e-12
            @test norm(m.edge_normal[:, e]) ≈ 1 atol = 1e-12
            @test norm(m.edge_tangent[:, e]) ≈ 1 atol = 1e-12
            @test abs(dot(m.edge_normal[:, e], m.edge_tangent[:, e])) < 1e-12
            @test abs(dot(m.edge_x[:, e], m.edge_normal[:, e])) < 1e-12
            vpos = m.edge_vertex_pos[e]
            vneg = m.edge_vertex_neg[e]
            @test dot(m.vertex_xyz[:, vpos] .- m.vertex_xyz[:, vneg],
                      m.edge_tangent[:, e]) > 0
        end

        # tangential reconstruction weights w_{e,e'} (T09 Eq 33):
        # the three paper identities must hold to round-off.
        for lev in (0, 1)
            m = generate_icosahedral_mesh(n_subdivisions=lev)

            # (a) T09 Eq 17: for cell i, edge e' ∈ EC(i), vertex v ∈ VC(i)
            # not an endpoint of e': Σ_{e∈EVC(v,i)} w̄_{e,e'} t^e_v = R_{i,v} n_i^{e'}
            # with w̄ the bare weight (w / (l_{e'}/d_e)).
            worst = 0.0
            for i in 1:m.n_cells
                n = length(m.cell_edges[i])
                for (kp, eprime) in enumerate(m.cell_edges[i])
                    for k in 1:n
                        v = m.cell_corners[i][k]
                        (m.edge_vertex_pos[eprime] == v ||
                         m.edge_vertex_neg[eprime] == v) && continue
                        lhs = 0.0
                        for e in (m.cell_edges[i][k], m.cell_edges[i][mod1(k - 1, n)])
                            jdx = findfirst(==(eprime), m.ecp_edges[e])
                            jdx === nothing && continue
                            wbare = m.ecp_weights[e][jdx] / (m.l_e[eprime] / m.d_e[e])
                            lhs += wbare * (m.edge_vertex_pos[e] == v ? 1.0 : -1.0)
                        end
                        n_out = i == m.edge_cells[1, eprime] ? 1.0 : -1.0
                        worst = max(worst, abs(lhs - m.corner_weight[i][k] * n_out))
                    end
                end
            end
            @test worst < 1e-12

            # (b) T09 Eq 14: dual-divergence interpolation identity
            # Σ_{e∈EV(v)} t^e_v d_e F⊥_e = Σ_{i∈CV(v)} R_{i,v} Σ_{e'∈EC(i)} n_i^{e'} l_{e'} F_{e'}
            # for arbitrary edge fluxes F, with F⊥_e = Σ w_{e,e'} F_{e'}.
            rng = MersenneTwister(20261015)
            F = randn(rng, m.n_edges)
            worst = 0.0
            for v in 1:m.n_vertices
                lhs = 0.0
                for e in m.vertex_edges[v]
                    Fp = sum(m.ecp_weights[e][k] * F[m.ecp_edges[e][k]]
                             for k in 1:length(m.ecp_edges[e]))
                    lhs += (m.edge_vertex_pos[e] == v ? 1.0 : -1.0) * m.d_e[e] * Fp
                end
                rhs = 0.0
                for i in m.vertex_cells[v]
                    kv = findfirst(==(v), m.cell_corners[i])
                    flux = sum((i == m.edge_cells[1, e] ? 1.0 : -1.0) * m.l_e[e] * F[e]
                               for e in m.cell_edges[i])
                    rhs += m.corner_weight[i][kv] * flux
                end
                worst = max(worst, abs(lhs - rhs))
            end
            @test worst < 1e-11

            # (c) T09 Eq 39 (antisymmetry of bare weights): energy conservation
            worst = 0.0
            for e in 1:m.n_edges
                for (k, ep) in enumerate(m.ecp_edges[e])
                    jdx = findfirst(==(e), m.ecp_edges[ep])
                    jdx === nothing && continue
                    wbare = m.ecp_weights[e][k] / (m.l_e[ep] / m.d_e[e])
                    wbare2 = m.ecp_weights[ep][jdx] / (m.l_e[e] / m.d_e[ep])
                    worst = max(worst, abs(wbare + wbare2))
                end
            end
            @test worst < 1e-11
        end

        # Lloyd smoothing improves (or preserves) minimum cell quality
        m0 = generate_icosahedral_mesh(n_subdivisions=2, lloyd_iterations=0)
        m1 = generate_icosahedral_mesh(n_subdivisions=2, lloyd_iterations=3)
        @test minimum(m1.cell_quality) ≥ minimum(m0.cell_quality) - 1e-9
    end

    @testset "mesh quality suite (#4, rung V0.5)" begin
        using JuiCM: mesh_summary, adjacency_consistent, kite_uniformity

        for (lev, lloyd) in ((0, 0), (1, 0), (2, 2))
            m = generate_icosahedral_mesh(n_subdivisions=lev,
                                          lloyd_iterations=lloyd)

            # summary aggregates are consistent with the mesh
            s = mesh_summary(m)
            @test s.n_cells == m.n_cells
            @test s.total_area ≈ 4π rtol = 1e-12
            @test s.n_pentagons == 12
            @test 0 < s.min_quality ≤ 1
            @test s.min_quality ≤ s.mean_quality

            # full adjacency consistency (V0.5: every edge 2 cells, dual
            # graph consistent, polygons close)
            @test adjacency_consistent(m)

            # kite uniformity in bounds; icosahedral family ≥ 0.8 for
            # these levels (0.85 at level 0, improving under Lloyd)
            @test kite_uniformity(m) > 0.8
        end

        # SCVT convergence reproducibility: mesh generation is deterministic
        # (no RNG), so two identical calls give identical meshes.
        a = generate_icosahedral_mesh(n_subdivisions=1, lloyd_iterations=2)
        b = generate_icosahedral_mesh(n_subdivisions=1, lloyd_iterations=2)
        @test a.cell_xyz == b.cell_xyz
        @test a.cell_area == b.cell_area
        @test a.cell_quality == b.cell_quality

        # level-1 icosphere is already an SCVT fixed point: Lloyd does not
        # move any center (centers coincide with cell centroids by the
        # subdivision symmetry)
        m0 = generate_icosahedral_mesh(n_subdivisions=1, lloyd_iterations=0)
        m3 = generate_icosahedral_mesh(n_subdivisions=1, lloyd_iterations=3)
        @test m0.cell_xyz ≈ m3.cell_xyz atol = 1e-12

        # level 2 is not at the fixed point; Lloyd strictly improves both
        # the mean and the minimum cell quality (V0.5: SCVT convergence)
        m0 = generate_icosahedral_mesh(n_subdivisions=2, lloyd_iterations=0)
        m3 = generate_icosahedral_mesh(n_subdivisions=2, lloyd_iterations=3)
        @test sum(m3.cell_quality) / m3.n_cells >
              sum(m0.cell_quality) / m0.n_cells
        @test minimum(m3.cell_quality) > minimum(m0.cell_quality)
    end

    @testset "fields: type-stable staggering API (#8)" begin
        using JuiCM.Fields: FieldLocation, CellLoc, EdgeLoc, VertexLoc,
                            location, mesh_of, similar_field, length_of

        m = generate_icosahedral_mesh(n_subdivisions=1)

        # construction: correct counts, element types, locations
        h = CellField(m)
        u = EdgeField(m)
        η = VertexField(m)
        @test length(h) == m.n_cells
        @test length(u) == m.n_edges
        @test length(η) == m.n_vertices
        @test eltype(h) == Float64
        @test location(h) == CellLoc()
        @test location(u) == EdgeLoc()
        @test location(η) == VertexLoc()
        @test location(h) isa FieldLocation
        @test mesh_of(h) === m

        # float type selection + callable initialization
        h32 = CellField(m, Float32)
        @test eltype(h32) == Float32
        hc = CellField(i -> sin(1.0), m)
        @test hc[1] == sin(1.0)
        uc = EdgeField(e -> Float64(e) / 10, m)
        @test uc[10] == 1.0

        # indexing reads/writes at the right location
        h[1] = 3.5
        @test h[1] == 3.5
        u[m.n_edges] = -1.25
        @test u[m.n_edges] == -1.25

        # type-stable construction and arithmetic
        @inferred CellField(m)
        @inferred EdgeField(m)
        @inferred VertexField(m)
        @inferred CellField(i -> 1.0, m)
        @inferred similar_field(h)
        @inferred similar_field(h, Float32)
        @inferred h + h
        @inferred h - h
        @inferred 2.0 * h
        @inferred h / 2
        @inferred -h

        # similar-type arithmetic preserves type and mesh
        s = h + h
        @test s isa Field && location(s) == CellLoc()
        @test mesh_of(s) === m
        @test s.data == 2 .* h.data

        # mixing staggers is a type error (TRiSK staggering is not mixable)
        @test_throws MethodError h + u
        @test_throws MethodError h + η
        @test_throws MethodError u + η

        # similar_field: same type & mesh, fresh storage
        f2 = similar_field(h)
        @test location(f2) == CellLoc() && mesh_of(f2) === m
        @test f2.data !== h.data
        f32 = similar_field(h, Float32)
        @test eltype(f32) == Float32

        # fill!
        fill!(f2, 2.5)
        @test all(==(2.5), f2.data)
    end

    @testset "TRiSK operators (#9, rung V0)" begin
        using JuiCM.TRiSK: divergence, divergence!, gradient, curl,
                           tangential_flux, edge_mean, kinetic_energy,
                           pv_flux
        using JuiCM.ErrorNorms: l2_norm

        m = generate_icosahedral_mesh(n_subdivisions=3)

        # --- divergence: discrete compatibility Σ A_i (∇·F)_i = 0 (round-off)
        F = EdgeField(e -> sin(m.edge_x[1, e]) + cos(2m.edge_x[2, e]), m)
        divF = divergence(F)
        @test abs(sum(m.cell_area .* divF.data)) < 1e-13

        # --- divergence: point source in one cell is exact per-neighbor.
        # With F routed out of cell c0 into each neighbor through their
        # shared edge (F_e = −l_e-normalized), div = −l_e/A_j on the
        # neighbor side and +Σl_e/A_i on c0.  Here we check the simpler
        # exact statement: F = 1 on a single edge e0 gives
        # div_i = ±l_e0/A_i on the two flanking cells, 0 elsewhere.
        F1 = EdgeField(m); fill!(F1, 0.0); F1.data[17] = 1.0
        d1 = divergence(F1)
        i1, j1 = m.edge_cells[1, 17], m.edge_cells[2, 17]
        for c in 1:m.n_cells
            expected = c == i1 ? m.l_e[17] / m.cell_area[c] :
                       c == j1 ? -m.l_e[17] / m.cell_area[c] : 0.0
            @test d1.data[c] ≈ expected rtol = 1e-14
        end

        # --- gradient of a linear field: 2nd-order accurate toward c·x̂.
        # ∇h on edge e ≈ c·n̂_e · sin(d)/d correction → order h²; check
        # the error against the exact edge projection shrinks O(h²).
        c = [0.3, -0.5, 0.8]
        lin = CellField(i -> dot(c, view(m.cell_xyz, :, i)), m)
        glin = gradient(lin)
        gerr = [abs(glin.data[e] - dot(c, view(m.edge_normal, :, e)))
                for e in 1:m.n_edges]
        @test maximum(gerr) < 0.05 * norm(c)

        # --- curl ∘ gradient = 0 exactly (round-off), any cell field
        @test all(abs.(curl(gradient(lin)).data) .< 1e-14)

        # --- div-grad adjointness (R10 product rule, A_e = l_e d_e):
        # Σ A_i h_i (∇·F)_i + Σ A_e (∇h)_e F_e = 0 exactly.
        G = EdgeField(e -> exp(m.edge_x[3, e]), m)
        h2 = CellField(i -> 1 + m.cell_xyz[2, i]^2, m)
        lhs = sum(m.cell_area .* h2.data .* divergence(G).data)
        gF = gradient(h2)
        rhs = sum(m.l_e .* m.d_e .* gF.data .* G.data)
        @test lhs + rhs ≈ 0 atol = 1e-12

        # --- solid-body vorticity: 2nd-order convergence (1.91/1.98
        # measured at levels 3/4 — assert < 0.05 relative at lev 3).
        U0 = 0.7
        usp = EdgeField(e -> begin
            x = view(m.edge_x, :, e)
            U0 * (-x[2] * m.edge_normal[1, e] + x[1] * m.edge_normal[2, e])
        end, m)
        ζ = curl(usp)
        ζerr = maximum(abs.(ζ.data .- 2U0 .* m.vertex_xyz[3, :]))
        @test ζerr < 0.05 * 2U0

        # --- edge_mean preserves constants on both staggers
        @test all(==(1.0), edge_mean(CellField(i -> 1.0, m)).data)
        @test all(==(1.0), edge_mean(VertexField(v -> 1.0, m)).data)

        # --- case-2 steady state: q̂ F⊥ + ∇(Φ+K) residual is
        # truncation-level and 2nd-order (measured 0.0104 → 0.0056
        # median from lev 3 to 4; here lev 3 only, threshold 0.05).
        Ωe = 7.292e-5; gh0 = 2.94e4; g = 9.80616; a = 6.37122e6
        u0ms = 2π * a / (12 * 86400)
        U0r = u0ms / a
        C = Ωe * U0r + U0r^2 / 2
        Φ = CellField(i -> gh0 / a^2 - C * m.cell_xyz[3, i]^2, m)
        uc2 = EdgeField(e -> begin
            x = view(m.edge_x, :, e)
            U0r * sqrt(1 - x[3]^2) *
                dot([-x[2], x[1], 0.0] ./ max(sqrt(1 - x[3]^2), 1e-12),
                    view(m.edge_normal, :, e))
        end, m)
        ηc2 = curl(uc2)
        hstar = CellField(i -> (gh0 - (a*Ωe*u0ms + u0ms^2/2) * m.cell_xyz[3, i]^2) / g / a, m)
        qc2 = VertexField(v -> begin
            num = 0.0; den = 0.0
            for cc in m.vertex_cells[v]
                nck = 1 / length(m.cell_corners[cc])
                num += nck * hstar.data[cc]
                den += nck
            end
            (2Ωe * m.vertex_xyz[3, v] + ηc2.data[v]) / (num / den)
        end, m)
        Fh = edge_mean(hstar)
        Fu = EdgeField(m); Fu.data .= Fh.data .* uc2.data
        Fp = tangential_flux(Fu)
        q̂ = edge_mean(qc2)
        Cor = EdgeField(m); Cor.data .= Fp.data .* q̂.data
        K = kinetic_energy(uc2)
        PhiK = CellField(m); PhiK.data .= Φ.data .+ K.data
        resid = Cor.data .+ gradient(PhiK).data
        scale = maximum(abs.(PhiK.data)) * 2 * (Ωe + U0r)  # analytic balance scale
        @test median(abs.(resid)) / maximum(abs.(gradient(PhiK).data)) < 0.05

        # --- pv_flux: on a uniform-PV field the symmetric mean reduces to
        # the plain tangential flux times q̃ (R10 Eq 49 = Eq 24 with the
        # PV mean inserted; weights applied to NORMAL fluxes, once)
        quni = VertexField(v -> 0.37, m)
        Qp = pv_flux(quni, Fu)
        Fp2 = tangential_flux(Fu)
        @test Qp.data ≈ 0.37 .* Fp2.data rtol = 1e-13

        # --- rung V0: flux-form advection by solid-body rotation,
        # SSPRK3 in time, centered edge tracer; convergence ≥ 1.5.
        for (lev_hi, lev_lo) in ((4, 3),)            mhi = generate_icosahedral_mesh(n_subdivisions=lev_hi)
            mlo = generate_icosahedral_mesh(n_subdivisions=lev_lo)
            adv_err(m) = begin
                U0a = 0.7; T = 0.4; β = 12.0
                x0 = [cos(0.6), 0.0, sin(0.6)]
                c_, s_ = cos(U0a * T), sin(U0a * T)
                xT = [c_*x0[1] - s_*x0[2], s_*x0[1] + c_*x0[2], x0[3]]
                u = EdgeField(e -> begin
                    x = view(m.edge_x, :, e)
                    U0a * (-x[2] * m.edge_normal[1, e] + x[1] * m.edge_normal[2, e])
                end, m)
                h0 = CellField(i -> exp(-β * (1 - dot(view(m.cell_xyz, :, i), x0))), m)
                dt = 0.15 * sum(m.d_e) / length(m.d_e) / U0a
                nst = ceil(Int, T / dt); dt = T / nst
                ec1, ec2 = m.edge_cells[1, :], m.edge_cells[2, :]
                ud = u.data
                L(out, hh, t) = begin
                    fill!(out, 0.0)
                    for e in 1:m.n_edges
                        f = ud[e] * (hh[ec1[e]] + hh[ec2[e]]) / 2 * m.l_e[e]
                        out[ec1[e]] -= f
                        out[ec2[e]] += f
                    end
                    out ./= m.cell_area
                end
                hv = copy(h0.data)
                t = 0.0
                for _ in 1:nst
                    hv = ssprk3_step!(L, hv, t, dt)
                    t += dt
                end
                href = [exp(-β * (1 - dot(view(m.cell_xyz, :, i), xT)))
                        for i in 1:m.n_cells]
                l2_norm(hv, href, m.cell_area)
            end
            ehi = adv_err(mhi)
            elo = adv_err(mlo)
            order = log(elo / ehi) / log(2)
            @test order ≥ 1.5
        end
    end

    @testset "halo machinery (#10)" begin
        using JuiCM.Fields: CellLoc, EdgeLoc, VertexLoc
        using JuiCM.Halo: MeshPartition, partition_mesh, n_regions,
                          region_of, owned_indices, ghost_cells,
                          build_halo_info, halo_array, halo_world,
                          halo_update!, partitioned_fields, owned_view

        # elements of stagger L that a cell c's operators touch
        mesh_touched(p, m, ::CellLoc, c) = (c,)
        mesh_touched(p, m, ::EdgeLoc, c) = m.cell_edges[c]
        mesh_touched(p, m, ::VertexLoc, c) =
            (v for e in m.cell_edges[c]
             for v in (m.edge_vertex_pos[e], m.edge_vertex_neg[e]))

        m = generate_icosahedral_mesh(n_subdivisions=2)
        p = partition_mesh(m, 4)

        # --- partition invariants: every element owned exactly once,
        # regions non-empty, region_of agrees with owned lists
        @test n_regions(p) == 4
        own_count = zeros(Int, m.n_cells)
        for r in 1:4, c in p.owned_cells[r]
            own_count[c] += 1
        end
        @test all(==(1), own_count)
        @test all(!isempty, p.owned_cells)
        @test all(r -> all(c -> region_of(p, c) == r, p.owned_cells[r]), 1:4)
        @test sum(length, p.owned_cells) == m.n_cells

        # --- pentagon cells: owned exactly once (the #10 acceptance case)
        @test all(c -> own_count[c] == 1, m.pentagon_cells)

        # --- ghost sets: 2-layer, exclude owned, consistent for all regions
        for r in 1:4
            gc = ghost_cells(p, m, r)
            @test allunique(gc) && issorted(gc)
            @test all(c -> p.owner[c] != r, gc)
            @test length(gc) > 0
        end

        # --- halo info: send lists are exact owner↔ghost mirrors,
        # sorted, unique, disjoint from receiver's owned set
        for (L, n) in ((CellLoc(), m.n_cells), (EdgeLoc(), m.n_edges),
                       (VertexLoc(), m.n_vertices))
            info = build_halo_info(p, m, L)
            info2 = build_halo_info(p, m, L)
            @test info.n_global == n
            for r in 1:4, s in 1:4
                r == s && continue
                @test info.send[r][s] == info2.send[r][s]  # deterministic
                @test allunique(info.send[r][s]) && issorted(info.send[r][s])
                @test all(g -> info.owner[g] == r, info.send[r][s])
            end
        end

        # --- constant field: owner-computes exchange preserves the
        # constant exactly (round-off = exact equality here) on all
        # three staggers and all regions (acceptance item 1)
        for (L, n) in ((CellLoc(), m.n_cells), (EdgeLoc(), m.n_edges),
                       (VertexLoc(), m.n_vertices))
            info = build_halo_info(p, m, L)
            w = halo_world(info; init = g -> 3.14159, poison_ghosts = false)
            # perturb owned values per region, then exchange
            for r in 1:4, g in info.owned[r]
                w[r].buf[g] += r
            end
            halo_update!(w)
            for r in 1:4, s in 1:4
                r == s && continue
                # every index exchanged with s must equal s's (owner) value
                for g in info.send[s][r]
                    @test w[r].buf[g] === w[s].buf[g]
                end
            end
            # semantic coverage: every NON-OWNED element that operators on
            # owned elements touch is covered by the exchange
            # (cells: 2-layer ghosts; edges: edges of owned∪ghost cells;
            # vertices: ends of touched edges)
            for r in 1:4
                oc = Set(p.owned_cells[r])
                gc = Set(ghost_cells(p, m, r))
                touched = Set{Int}()
                for c in Iterators.flatten((oc, gc))
                    union!(touched, mesh_touched(p, m, L, c))
                end
                for g in touched
                    info.owner[g] == r && continue
                    s = info.owner[g]
                    @test g in info.send[s][r]
                end
            end
        end

        # --- ghost accuracy: after exchange, every 2-layer ghost slot is
        # exactly the owner's value (copies, not interpolation) —
        # "halo error consistent with operator order" = 0 by design
        ic = build_halo_info(p, m, CellLoc())
        wc = halo_world(ic; init = g -> sin(m.cell_xyz[1, g]) + 0.3 * cos(2 * m.cell_xyz[2, g]))
        for r in 1:4, c in p.owned_cells[r]
            wc[r].buf[c] += 10 * r
        end
        halo_update!(wc)
        for r in 1:4, g in ghost_cells(p, m, r)
            @test wc[r].buf[g] == wc[p.owner[g]].buf[g]
        end

        # --- poisoning: un-refreshed ghosts are NaN (fail loudly on
        # stale reads); after halo_update! all 2-layer ghosts are finite
        wp = halo_world(ic; init = g -> 1.0, poison_ghosts = true)
        @test any(isnan, wp[2].buf)
        halo_update!(wp, 2)
        @test all(g -> !isnan(wp[2].buf[g]), ghost_cells(p, m, 2))

        # --- pentagon halos correct: perturb values AT the 12 pentagon
        # cells in their owner arrays, exchange, check every region's
        # ghost copy of a pentagon cell matches the owner
        wc2 = halo_world(ic; init = g -> 0.0, poison_ghosts = false)
        for c in m.pentagon_cells
            wc2[p.owner[c]].buf[c] = 7.0
        end
        halo_update!(wc2)
        for r in 1:4, c in m.pentagon_cells
            p.owner[c] == r && continue
            c in Set(ghost_cells(p, m, r)) && @test wc2[r].buf[c] == 7.0
        end

        # --- TRiSK operators run unchanged on HaloArray payloads and
        # agree with the global computation on owned cells
        ie = build_halo_info(p, m, EdgeLoc())
        we = halo_world(ie; init = g -> sin(m.edge_x[1, g]), poison_ghosts = false)
        halo_update!(we)
        u_h = Field{typeof(m), Float64, 1, typeof(we[1]), EdgeLoc}(we[1], m)
        d_h = divergence(u_h)
        u_g = EdgeField(e -> we[1].buf[e], m)
        d_g = divergence(u_g)
        @test maximum(abs.(d_h.data[p.owned_cells[1]] .- d_g.data[p.owned_cells[1]])) == 0

        # --- partitioned_fields: one field per region, NaN-poisoned
        # ghosts, same values on owned slots as the source field
        h0 = CellField(i -> m.cell_xyz[3, i]^2, m)
        fs = partitioned_fields(h0, ic)
        @test length(fs) == 4
        @test all(r -> all(i -> fs[r].data[i] == h0.data[i], p.owned_cells[r]), 1:4)
        @test any(isnan, fs[1].data.buf)   # some non-owned slot poisoned

        # --- owned_view covers exactly the owned slots
        @test length(owned_view(wp[3])) == length(p.owned_cells[3])
        @test all(g -> wp[3].buf[g] == wp[3].buf[g], owned_indices(p, 3))
    end

    include("error_norms_test.jl")
    include("williamson_ics_test.jl")
    include("time_stepping_test.jl")
    include("dycore_sw_test.jl")
    include("ocean_vertical_test.jl")
    include("ocean_dynamics_test.jl")
    include("ocean_tracer_test.jl")
    include("ocean_tracer_fct_test.jl")
    include("ocean_fct_guard_test.jl")
    include("ocean_dynamics_fixes_test.jl")
    include("gravity_current_benchmark_test.jl")
    include("ocean_regrid_test.jl")
    include("ocean_walled_channel_test.jl")
    include("ocean_coriolis_test.jl")
    include("ocean_bathymetry_test.jl")
    include("ocean_barotropic_test.jl")
    include("ocean_cross_terms_test.jl")
    include("ocean_pgf_test.jl")
    include("validation_smoke_test.jl")
end
