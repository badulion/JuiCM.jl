# Ocean pressure-gradient schemes

The layered ocean uses z* layers, which follow the terrain. Wherever density
surfaces cross those sloping layers, the horizontal pressure-gradient force (PGF)
is the difference of two large, nearly cancelling terms. This is the classic
σ-coordinate error and DESIGN.md §7 lists it as a risk. JuiCM offers four PGF
discretizations. You select one with the `pgf` keyword on `ocean_step!`,
`ocean_integrate!` and `ocean_tendency!` (benchmark drivers forward it through
`step_kwargs`):

```julia
ocean_step!(state, grid, eos, t, Δt; bathymetry = H, regrid = true, pgf = :roms)
```

`PGF_SCHEMES` lists the options. The same scheme is used in the slow stages and
in the barotropic-forcing evaluations.

| `pgf` | Scheme | Source |
|---|---|---|
| `:twoterm` (default) | −∇p̄_k − b_k∇z̄_k from the layer-mean pressure and the layer-centre height | #30 |
| `:adcroft2008` | Finite-volume contour integral of the pressure anomaly around each layer control volume, with analytic integration and density constant within each layer | Adcroft, Hallberg & Harrison (2008), *Ocean Modelling* 22, 106–113 |
| `:mom6` | The same finite-volume integral with MOM6's piecewise-linear (PLM) density reconstruction within layers | MOM6 `MOM_PressureForce_FV.F90`, `PLM_functions.F90` (`RECONSTRUCT_FOR_PRESSURE`, PLM, boundary extrapolation) |
| `:roms` | Density-Jacobian scheme with monotonized cubic fits | Shchepetkin & McWilliams (2003), *JGR* 108(C3), 3090; ROMS `prsgrd32.h` |

Every scheme reproduces −g∇η exactly when density is uniform. The module header
of `src/Dycore/Ocean/PressureGradient.jl` gives the formulas. Two adaptations to
the icosahedral mesh:

- **`:roms` stencil.** ROMS uses neighbouring columns on a structured grid. Here
  the extra horizontal neighbour of a cell is the one across its edge *opposite*
  the force edge, which is exact on hexagons.
- **`:mom6` reconstruction.** MOM6 reconstructs T and S. With a linear EOS,
  reconstructing ρ instead gives the same result.

## Measured behaviour (issue #73)

The test case is an ocean at rest over a uniform slope (s* = 4.2e-4), on cells
240 km apart with 48 z* layers refined towards the bottom, run for 2.3 days. The
water is at rest, so any velocity is spurious; it is compared with the Nof speed
of rung V4, 2.4 m/s.

| Stratification | `:twoterm` | `:adcroft2008` | `:mom6` | `:roms` |
|---|---|---|---|---|
| Linear ρ(z) | 3.9 mm/s | 3.9 mm/s | 2e-10 m/s | 2e-10 m/s |
| tanh interface, 150 m half-width | 0.32 m/s | 0.31 m/s | 0.31 m/s | 0.077 m/s |
| Sharp two-layer step | 3.3 m/s | 3.4 m/s | 3.4 m/s | 2.7 m/s |

**Linear stratification.** `:mom6` and `:roms` are exact for linear profiles, so
a linearly stratified ocean stays at rest to round-off. Without reconstruction
(`:adcroft2008`), the finite-volume integral does no better than `:twoterm`.

**Sharp interfaces.** For interfaces that are sharp relative to the layer
spacing, ROMS's monotonized cubic fits are the best of the four. A perfectly
sharp density step is a representation error, not a PGF error, and no scheme
fixes it. Each layer that the interface cuts stores the mean density of its
dense and light parts, and that mean differs between neighbouring columns
because the layers slope.
