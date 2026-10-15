using Test
using JuiCM
using JuiCM.Icosahedron: icosahedron, subdivide
using JuiCM.Voronoi: spherical_circumcenter, spherical_polygon_area,
                     generate_icosahedral_mesh
using LinearAlgebra
using Random

@testset "JuiCM" begin
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
end
