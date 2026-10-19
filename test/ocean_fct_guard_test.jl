# FCT tracer guards (issue #40 follow-up to #36): drained-cell recovery,
# vertical-flux consistency, and the conservative column bound.
#
# Regression targets (all measured on the pre-fix code):
#   1. Drained cell: a layer whose h_new collapses to 1e-6 m (below
#      h_neglect) used to come out with C = Q (the content, not a
#      concentration): C = 20 → 2e-5. It must keep a concentration
#      inside the initial bounds.
#   2. Vertical consistency: the interface flux seen from the layer
#      above and from the layer below must be the SAME number, or the
#      column content Σ_k h_k C_k is not conserved (pre-fix: centered
#      value on one side, upwind on the other → a stratified column
#      with a vertical exchange lost ~17% of its content in one step).
#   3. Conservation of the bound: the post-vertical clip redistributes
#      clipped content within the column; Σ A h C is conserved to
#      round-off whenever the column can absorb it.
#
# The states are hand-built: u = 0 (no horizontal transport, so h^horiz
# = h_old exactly) and h_new chosen with an unchanged column total
# (surface interface velocity w_1 = 0), so the only transport is the
# vertical exchange diagnosed from the continuity residual.

@testset "ocean FCT guards: drained cells and conservative bound (#40)" begin
    using JuiCM: similar_field, LayeredField

    mesh = generate_icosahedral_mesh(n_subdivisions=2)
    n = mesh.n_cells
    K = 3
    Δt = 3600.0

    content(C, h) = sum(sum(mesh.cell_area[i] * h[k][i] * C[k][i]
                            for i in 1:n) for k in 1:K)

    "Stratified (nonlinearly in k), horizontally varying tracer in
    [8, 22]."
    function strat_C()
        C = LayeredCellField(mesh, K)
        Ck = (22.0, 19.0, 10.0)
        for k in 1:K, i in 1:n
            C[k][i] = Ck[k] - (mesh.cell_xyz[3, i] + 1)
        end
        return C
    end

    function uniform_h(H)
        h = LayeredCellField(mesh, K)
        for k in 1:K, i in 1:n
            h[k][i] = H
        end
        return h
    end

    Cbounds(C) = (minimum(minimum(C[k][i] for i in 1:n) for k in 1:K),
                  maximum(maximum(C[k][i] for i in 1:n) for k in 1:K))

    u = LayeredEdgeField(mesh, K)              # zero velocity

    # ------------------------------------------------------------- 1
    @testset "drained cell keeps a bounded concentration" begin
        for hd in (1.0e-6, 0.0, -1.0e-3)       # tiny, zero, negative
            C = strat_C()
            Cmin0, Cmax0 = Cbounds(C)
            h_old = uniform_h(1000.0)
            h_new = deepcopy(h_old)
            i0 = 7
            # layer 1 drains into layer 2 (column total unchanged)
            h_new[1][i0] = hd
            h_new[2][i0] = 2000.0 - hd
            C1_before = C[1][i0]
            m0 = content(C, h_old)
            fct_advect_tracer!(C, h_old, h_new, u, Δt)
            Cmin1, Cmax1 = Cbounds(C)
            @test all(isfinite, (Cmin1, Cmax1))
            @test Cmin1 ≥ Cmin0 - 1e-12
            @test Cmax1 ≤ Cmax0 + 1e-12
            # the vanished cell holds a concentration from its local
            # bounds (pre-fix: C = Q = 2e-5 for hd = 1e-6)
            @test abs(C[1][i0] - C1_before) ≤ 4.0
            # layer 2 received layer 1's water: a mix of the two
            @test min(C1_before, C[2][i0]) - 1e-12 ≤ C[2][i0]
            # content conserved to round-off
            @test abs(content(C, h_new) - m0) / m0 < 1e-12
        end
    end

    # ------------------------------------------------------------- 2
    @testset "vertical exchange conserves column content" begin
        C = strat_C()
        Cmin0, Cmax0 = Cbounds(C)
        h_old = uniform_h(1000.0)
        h_new = deepcopy(h_old)
        # moderate exchange in every column: layer 2 thins, 1 and 3 thicken
        for i in 1:n
            s = 100.0 * (1 + mesh.cell_xyz[1, i]) # 0..200 m
            h_new[1][i] += s / 2
            h_new[2][i] -= s
            h_new[3][i] += s / 2
        end
        m0 = content(C, h_old)
        fct_advect_tracer!(C, h_old, h_new, u, Δt)
        @test abs(content(C, h_new) - m0) / m0 < 1e-12
        Cmin1, Cmax1 = Cbounds(C)
        @test Cmin1 ≥ Cmin0 - 1e-12
        @test Cmax1 ≤ Cmax0 + 1e-12
    end

    # ------------------------------------------------------------- 3
    @testset "over-drain of a thin cell: clip is conservative" begin
        # a thin (but not vanished) middle layer whose outflow volume
        # exceeds its own thickness: the upwind donor over-drains it,
        # Q < 0 before the clip. The clip must move the deficit within
        # the column, not delete it.
        C = strat_C()
        Cmin0, Cmax0 = Cbounds(C)
        h_old = uniform_h(1000.0)
        for i in 1:n
            h_old[2][i] = 5.0
        end
        h_new = deepcopy(h_old)
        i0 = 11
        # layer 2 receives 995 m from layer 1 and passes 995 m on to
        # layer 3 (downward interface velocities w_2 = w_3 > 0): the
        # outflow volume 995 m ≫ h = 5 m
        h_new[1][i0] = 5.0
        h_new[2][i0] = 5.0                     # unchanged thickness
        h_new[3][i0] = 1000.0 + 995.0
        m0 = content(C, h_old)
        resid = Ref(NaN)
        fct_advect_tracer!(C, h_old, h_new, u, Δt; clip_residual = resid)
        Cmin1, Cmax1 = Cbounds(C)
        @test Cmin1 ≥ Cmin0 - 1e-12
        @test Cmax1 ≤ Cmax0 + 1e-12
        @test abs(content(C, h_new) - m0) / m0 < 1e-12
        # the column absorbed the clipped content: nothing dropped
        @test resid[] == 0.0
    end
end
