# Regression test for issue #66: the slow-mode (baroclinic) PV flux
# −q̂F⊥ of `ocean_tendency!` must carry the physical Coriolis
# acceleration −f k̂ × u. Before the fix the scaled-variable PV flux was
# never multiplied by R, so the term was 1/R ≈ 1.6e-7 too weak (measured
# ratio to −f·U: 1.6e-7) and the baroclinic layers did not rotate.

using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh
using JuiCM.ShallowWater: williamson_constants
using LinearAlgebra
using Test

@testset "ocean slow-mode Coriolis (#66)" begin
    (; Omega) = williamson_constants()
    m = generate_icosahedral_mesh(n_subdivisions = 4)
    K = 2
    grid = VerticalGrid(1000.0, K)
    eos = LinearEOS(alpha = 2.0e-3)
    east(e) = (x = m.edge_x[:, e]; v = [-x[2], x[1], 0.0]; v / norm(v))
    north(e) = (x = m.edge_x[:, e]; cross(x / norm(x), east(e)))
    # baroclinic zonal flow u_1 = +U, u_2 = −U (depth-mean free), uniform
    # density, flat surface: no pressure force, uniform kinetic energy,
    # so the slow tendency is the Coriolis term −f U north (+ the
    # relative-vorticity part, U/(R tan φ)-small next to f)
    U = 0.1
    u = LayeredEdgeField((e, k) -> (k == 1 ? U : -U) *
                                   dot(east(e), m.edge_normal[:, e]), m, K)
    h = layer_thickness(grid, CellField(i -> 0.0, m))
    T = LayeredCellField((i, k) -> 20.0, m, K)
    S = LayeredCellField((i, k) -> 35.0, m, K)
    st = ocean_state(u, h, T, S)
    out = copy(st)
    ocean_tendency!(out, st; grid = grid, eos = eos, projection = :full)
    band = [e for e in 1:m.n_edges if 0.6 < m.edge_x[3, e] < 0.8]   # 37°–53°N
    for k in 1:K
        sgn = k == 1 ? 1.0 : -1.0
        expect = [-2Omega * m.edge_x[3, e] * sgn * U *
                  dot(north(e), m.edge_normal[:, e]) for e in band]
        got = [out.u[k][e] for e in band]
        ratio = dot(got, expect) / dot(expect, expect)
        @test abs(ratio - 1) < 0.05
        # edge-wise agreement, not only on average
        @test norm(got .- expect) / norm(expect) < 0.1
    end
end
