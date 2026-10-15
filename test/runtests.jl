using Test
using JuiCM
using JuiCM.Icosahedron: icosahedron, subdivide
using JuiCM.Voronoi: spherical_circumcenter, spherical_polygon_area
using LinearAlgebra

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
end
