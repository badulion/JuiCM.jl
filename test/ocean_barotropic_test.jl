# Regression tests for issue #69: the barotropic (fast) mode of
# `ocean_step!` must evolve at full speed and rotate with the same
# Coriolis parameter as the layers.
#
# Before the fix the filtered fast state was the time-mean over
# [t, t + Δt] — centred on t + Δt/2 but stored at t + Δt — so every
# barotropic process ran at half speed (measured: a passive tracer in a
# steady barotropic flow moved at 0.497 U), and the fast mode always
# rotated with Earth's Ω whatever `Omega`/`f0` said.

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using JuiCM.ShallowWater: williamson_constants
using JuiCM.Fields: similar_field
using LinearAlgebra
using Test

@testset "ocean barotropic mode (#69)" begin
    (; R) = williamson_constants()
    m = generate_icosahedral_mesh(n_subdivisions = 4)
    K = 2
    grid = VerticalGrid(1000.0, K)
    h = layer_thickness(grid, CellField(i -> 0.0, m))
    S = LayeredCellField((i, k) -> 35.0, m, K)
    lon(p) = atan(p[2], p[1])
    lat(p) = asin(clamp(p[3], -1.0, 1.0))
    east(e) = (x = m.edge_x[:, e]; v = [-x[2], x[1], 0.0]; v / max(norm(v), 1e-12))
    north(e) = (x = m.edge_x[:, e]; cross(x / norm(x), east(e)))

    @testset "passive tracer moves with the barotropic flow" begin
        # non-rotating, uniform density, steady solid-body flow U cos φ
        # in every layer (u′ = 0): a tracer patch must be carried at U
        eos = LinearEOS(alpha = 0.0)
        U = 10.0
        uf(e) = U * dot(east(e), m.edge_normal[:, e]) * norm(m.edge_x[1:2, e])
        T = LayeredCellField((i, k) -> (p = m.cell_xyz[:, i];
                                        abs(lon(p)) < 0.3 && abs(lat(p)) < 0.3 ? 1.0 : 0.0),
                             m, K)
        Δt = 1800.0
        n = 48
        for regrid in (false, true)
            st = ocean_state(LayeredEdgeField((e, k) -> uf(e), m, K), h, T, S,
                             CellField(i -> 0.0, m), EdgeField(uf, m))
            cx(st) = begin
                w = 0.0; x = 0.0
                for k in 1:K, i in 1:m.n_cells
                    c = m.cell_area[i] * st.h[k][i] * st.T[k][i]
                    w += c; x += c * lon(m.cell_xyz[:, i])
                end
                x / w
            end
            x0 = cx(st)
            for j in 1:n
                st = ocean_step!(st, grid, eos, (j - 1) * Δt, Δt;
                                 f0 = 0.0, regrid = regrid)
            end
            ratio = (cx(st) - x0) * R / (U * n * Δt)
            @test abs(ratio - 1) < 0.02            # 0.497 before #69
        end
    end

    @testset "barotropic inertial oscillation on an f-plane" begin
        # uniform density, every layer u = U east (pure barotropic),
        # f-plane f0: ū must turn clockwise at f. The filter centred on
        # t + Δt damps a slow oscillation by ≈ (fΔt)²/24 per step only
        # (≈ 0.92 after two periods at fΔt = 0.18; measured 0.88).
        eos = LinearEOS(alpha = 2.0e-3)
        T = LayeredCellField((i, k) -> 20.0, m, K)
        U = 0.1
        f0 = 1.0e-4
        uf(e) = U * dot(east(e), m.edge_normal[:, e])
        st = ocean_state(LayeredEdgeField((e, k) -> uf(e), m, K), h, T, S,
                         CellField(i -> 0.0, m), EdgeField(uf, m))
        band = [e for e in 1:m.n_edges if abs(m.edge_x[3, e]) < 0.2]
        proj(f, dir) = sum(f[e] * dot(dir(e), m.edge_normal[:, e]) for e in band) /
                       sum(dot(dir(e), m.edge_normal[:, e])^2 for e in band)
        Δt = 1800.0
        n = 64                                     # ft = 11.52, ~1.8 periods
        for j in 1:n
            st = ocean_step!(st, grid, eos, (j - 1) * Δt, Δt; f0 = f0)
        end
        ue = proj(st.ubar, east)
        un = proj(st.ubar, north)
        ft = f0 * n * Δt
        # phase of (ue, un) = (U cos ft, −U sin ft): clockwise at f
        dphase = mod(atan(-un, ue) - ft + π, 2π) - π
        @test abs(dphase) < 0.25                   # ≈ 0.13 rad; f/2 would be ≈ 2.4
        @test 0.8 < hypot(ue, un) / U < 1.0
        # the layers carry the same rotation (u′ stays zero)
        @test maximum(abs(st.u[k][e] - st.ubar[e]) for k in 1:K, e in 1:m.n_edges) <
              1e-10
    end

    @testset "fast mode honours Omega and f0" begin
        f = coriolis_parameter(m; f0 = 3.0e-5)
        @test all(f[v] == 3.0e-5 for v in 1:m.n_vertices)
        f2 = coriolis_parameter(m; Omega = 1.0e-5)
        @test all(f2[v] ≈ 2e-5 * m.vertex_xyz[3, v] for v in 1:m.n_vertices)
        # sw_tendency! uses the supplied Coriolis parameter
        hs = CellField(i -> 1000.0, m)
        us = EdgeField(e -> dot(east(e), m.edge_normal[:, e]), m)
        o0 = sw_tendency!(SWState(similar_field(hs), similar_field(us)),
                          SWState(hs, us); coriolis = coriolis_parameter(m; f0 = 0.0))
        o1 = sw_tendency!(SWState(similar_field(hs), similar_field(us)),
                          SWState(hs, us); coriolis = coriolis_parameter(m; f0 = 1e-4))
        band = [e for e in 1:m.n_edges if abs(m.edge_x[3, e]) < 0.2]
        d = [o1.u[e] - o0.u[e] for e in band]
        ex = [-1e-4 * dot(north(e), m.edge_normal[:, e]) for e in band]
        @test norm(d .- ex) / norm(ex) < 0.05
    end
end
