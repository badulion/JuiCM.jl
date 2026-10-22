# Regression test for issue #70: the slow momentum terms must act on the
# full velocity u′ + ū, so a baroclinic velocity anomaly is carried by
# the barotropic flow (the ū × u′ cross terms −ζ̄u′⊥ − ζ′ū⊥ − ∇(ū·u′)).
# Before the fix the slow tendency saw u′ alone: a baroclinic anomaly
# in a steady barotropic flow did not move (measured 0.00 U).

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using JuiCM.ShallowWater: williamson_constants
using JuiCM.TRiSK: edge_mean
using LinearAlgebra
using Test

@testset "ocean split cross terms (#70)" begin
    (; R, g) = williamson_constants()
    m = generate_icosahedral_mesh(n_subdivisions = 4)
    K = 2
    grid = VerticalGrid(1000.0, K)
    eos = LinearEOS(alpha = 0.0)          # uniform density: u′ is passive
    lon(p) = atan(p[2], p[1])
    lat(p) = asin(clamp(p[3], -1.0, 1.0))
    east(e) = (x = m.edge_x[:, e]; v = [-x[2], x[1], 0.0]; v / max(norm(v), 1e-12))
    # steady non-rotating solid-body flow U cos φ, balanced by
    # g η = U² cos² φ / 2 (centrifugal; minus its mean)
    U = 10.0
    ubf(e) = U * dot(east(e), m.edge_normal[:, e]) * norm(m.edge_x[1:2, e])
    eta = CellField(i -> U^2 * (m.cell_xyz[1, i]^2 + m.cell_xyz[2, i]^2) / (2g) -
                         U^2 / (3g), m)
    h = layer_thickness(grid, eta)
    T = LayeredCellField((i, k) -> 20.0, m, K)
    S = LayeredCellField((i, k) -> 35.0, m, K)
    # smooth baroclinic anomaly ±ε east (Gaussian, e-folding 0.35 rad ≈
    # 4.7 cells; depth-mean free). A sharp-edged 8-cell patch is carried
    # at 0.84 U — the grid-scale part of a step lags (second-order
    # dispersion): 0.93 U at 2.7 cells, 0.97 U at 4.7 cells, robust to
    # Δt, amplitude and regrid (measured)
    ε = 0.5
    gauss(e) = exp(-(lon(m.edge_x[:, e])^2 + lat(m.edge_x[:, e])^2) / 0.35^2)
    u = LayeredEdgeField((e, k) -> ubf(e) + gauss(e) * (k == 1 ? ε : -ε) *
                                             dot(east(e), m.edge_normal[:, e]),
                         m, K)
    st = ocean_state(u, h, T, S, eta, EdgeField(ubf, m))
    Ae = [m.l_e[e] * m.d_e[e] for e in 1:m.n_edges]
    function xc(st)
        hm = [edge_mean(st.h[k]) for k in 1:K]
        w = 0.0; x = 0.0
        for k in 1:K, e in 1:m.n_edges
            c = Ae[e] * hm[k][e] * (st.u[k][e] - st.ubar[e])^2
            w += c; x += c * lon(m.edge_x[:, e])
        end
        x / w
    end
    x0 = xc(st)
    Δt = 1800.0
    n = 48
    for j in 1:n
        st = ocean_step!(st, grid, eos, (j - 1) * Δt, Δt; f0 = 0.0, regrid = true)
    end
    ratio = (xc(st) - x0) * R / (U * n * Δt)
    @info "baroclinic anomaly carried by ū" ratio
    @test abs(ratio - 1) < 0.05               # 0.97; 0.00 before #70
    # the background barotropic flow itself stays steady
    @test sum(st.ubar[e] * ubf(e) for e in 1:m.n_edges) /
          sum(ubf(e)^2 for e in 1:m.n_edges) > 0.99
end
