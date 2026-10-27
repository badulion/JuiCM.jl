# JuiCM validation suite (issue #75; DESIGN.md §4 validation ladder).
#
# Full-accuracy benchmark runs against published reference values: the
# rungs of the validation ladder. Separate from `Pkg.test()`, which is the
# fast suite (every testset ≤ 30 s; each rung keeps a smoke test there).
# Run from the repository root with the test environment:
#
#     julia --project=test validation/runtests.jl
#
# Reference data: test/reference_data/ (provenance README there).
# AGENTS.md rule 2 applies: never weaken or delete a validation test.

using Test
using Statistics
using Random
using LinearAlgebra
using JuiCM
using JuiCM.Voronoi: generate_icosahedral_mesh

@testset "JuiCM validation" verbose = true begin
    for f in sort(filter(f -> endswith(f, ".jl") && f != "runtests.jl",
                         readdir(@__DIR__)))
        include(joinpath(@__DIR__, f))
    end
end
